"""Integer-domain rounding is removed only with a closed integral image proof."""

import pytest
from mlir_fixtures import assert_unchanged, pointwise

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.integral_float_rounding import fold_integral_float_rounding


def source(*, integer=True, divisor=1.0):
    typ = "i64" if integer else "f32"
    cast = "%f=arith.sitofp %a : i64 to f32" if integer else "%f=arith.mulf %a,%one : f32"
    body = f"""%one=arith.constant 1.0 : f32
      {cast}
      %c=arith.constant {divisor} : f32
      %q=arith.divf %f,%c : f32
      %v=math.trunc %q : f32"""
    return pointwise([f"tensor<4x{typ}>"], "tensor<4xf32>", body)


@pytest.mark.parametrize("divisor", [1.0, -1.0, 0.5, -0.25])
def test_every_rounded_integer_stays_integral_under_exact_binary_rescaling(divisor):
    module = source(divisor=divisor)
    assert fold_integral_float_rounding(module) == (2 if divisor == 1 else 1)
    module.verify()
    assert not any(op.name == "math.trunc" for op in module.walk())


@pytest.mark.parametrize("options", [dict(integer=False), dict(divisor=2.0), dict(divisor=3.0), dict(divisor=0.0)])
def test_fractional_and_unproved_images_keep_the_rounding(options):
    assert_unchanged(source(**options), fold_integral_float_rounding)


@pytest.mark.parametrize("escape", [False, True])
def test_unit_remainder_preserves_observable_negative_zero(escape):
    module = parse_mlir_text(f"""module {{func.func @forward(%x:i64) -> f32 {{
      %f=arith.sitofp %x : i64 to f32
      %one=arith.constant 1.0 : f32
      %r="arith.remf"(%f,%one) : (f32,f32)->f32
      %sub=arith.subf %f,%r : f32
      return {"%r" if escape else "%sub"} : f32
    }} }}""")
    assert fold_integral_float_rounding(module) == int(not escape)
    assert any(getattr(getattr(op, "op_name", None), "data", op.name) == "arith.remf" for op in module.walk()) is escape


def test_an_overflowing_half_conversion_cannot_justify_a_zero_remainder():
    module = parse_mlir_text("""module {func.func @forward(%x:i64) -> f16 {
      %f=arith.sitofp %x : i64 to f16
      %one=arith.constant 1.0 : f16
      %r="arith.remf"(%f,%one) : (f16,f16)->f16
      %sub=arith.subf %f,%r : f16
      return %sub : f16
    }}""")
    assert fold_integral_float_rounding(module) == 0
