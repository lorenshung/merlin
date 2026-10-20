"""Only a complete finite rational zero-threshold identity is exposed."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.quantized_zero_threshold import normalize
from merlin.xdsl_dialects.lowering.contraction_coverage import is_compare_select_relu


def source(alpha=0, scale=1, unknown=False):
    return parse_mlir_text(f"""module {{func.func @forward(%input:tensor<3xf32>) -> tensor<3xf32> {{
      %e=tensor.empty() : tensor<3xf32>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%input:tensor<3xf32>) outs(%e:tensor<3xf32>) {{
        ^bb0(%x:f32,%unused:f32):
        %zero=arith.constant 0.0 : f32
        %alpha=arith.constant {float(alpha)} : f32
        %scale=arith.constant {float(scale)} : f32
        %exp={('"fixture.unknown"(%x) : (f32)->f32' if unknown else "math.expm1 %x : f32")}
        %negative=arith.mulf %alpha,%exp : f32
        %positive=arith.mulf %x,%scale : f32
        %p=arith.cmpf oge,%x,%zero : f32
        %out=arith.select %p,%positive,%negative : f32
        linalg.yield %out : f32
      }} -> tensor<3xf32>
      return %r : tensor<3xf32>
    }} }}""")


def test_exact_finite_zero_coefficient_eliminates_the_transcendental_branch():
    module = source()
    assert normalize(module) == 1
    assert sum(is_compare_select_relu(op) for op in module.walk()) == 1
    assert all(op.name != "math.expm1" for op in module.walk())


@pytest.mark.parametrize("options", [dict(alpha=1), dict(scale=2), dict(unknown=True)])
def test_a_different_function_or_unknown_primitive_is_not_a_native_threshold(options):
    module = source(**options)
    original = str(module)
    assert normalize(module) == 0
    assert str(module) == original
