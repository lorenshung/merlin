"""Integer-domain rounding is removed only with a closed integral image proof."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.integral_float_rounding import fold_integral_float_rounding


def source(*, integer=True, divisor=1.0):
    typ = "i64" if integer else "f32"
    cast = "%f=arith.sitofp %a : i64 to f32" if integer else "%f=arith.mulf %a,%one : f32"
    return parse_mlir_text(f"""module {{func.func @forward(%x:tensor<4x{typ}>) -> tensor<4xf32> {{
      %e=tensor.empty() : tensor<4xf32>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%x:tensor<4x{typ}>) outs(%e:tensor<4xf32>) {{
        ^bb0(%a:{typ},%seed:f32):
        %one=arith.constant 1.0 : f32
        {cast}
        %c=arith.constant {divisor} : f32
        %q=arith.divf %f,%c : f32
        %v=math.trunc %q : f32
        linalg.yield %v : f32
      }} -> tensor<4xf32>
      return %r : tensor<4xf32>
    }} }}""")


@pytest.mark.parametrize("divisor", [1.0, -1.0, 0.5, -0.25])
def test_every_rounded_integer_stays_integral_under_exact_binary_rescaling(divisor):
    module = source(divisor=divisor)
    assert fold_integral_float_rounding(module) == 1
    module.verify()
    assert not any(op.name == "math.trunc" for op in module.walk())


@pytest.mark.parametrize("options", [dict(integer=False), dict(divisor=2.0), dict(divisor=3.0), dict(divisor=0.0)])
def test_fractional_and_unproved_images_keep_the_rounding(options):
    module = source(**options)
    before = str(module)
    assert fold_integral_float_rounding(module) == 0 and str(module) == before
