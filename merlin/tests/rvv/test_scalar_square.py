"""Constant-square rewrite and its arithmetic edge cases."""

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.scalar_square import canonicalize_scalar_squares


def _module(exponent="2.0", dtype="f32"):
    return parse_mlir_text(f"""module {{
      func.func @forward(%x: {dtype}) -> {dtype} {{
        %e = arith.constant {exponent} : {dtype}
        %y = math.powf %x, %e : {dtype}
        return %y : {dtype}
      }}
    }}""")


@pytest.mark.parametrize("dtype", ["f32", "f64"])
def test_square_preserves_operand_uses_and_provenance(dtype):
    from xdsl.dialects.builtin import StringAttr

    module = _module(dtype=dtype)
    power = next(op for op in module.walk() if op.name == "math.powf")
    power.attributes["prov.region_id"] = StringAttr("normalization")
    assert canonicalize_scalar_squares(module) == 1
    module.verify()
    square = next(op for op in module.walk() if op.name == "arith.mulf")
    assert square.lhs is square.rhs
    assert square.attributes["prov.region_id"].data == "normalization"
    assert not any(op.name == "math.powf" for op in module.walk())
    assert canonicalize_scalar_squares(module) == 0


def test_other_and_dynamic_exponents_are_retained():
    assert canonicalize_scalar_squares(_module("3.0")) == 0
    module = parse_mlir_text("""module {
      func.func @forward(%x: f32, %e: f32) -> f32 {
        %y = math.powf %x, %e : f32
        return %y : f32
      }
    }""")
    assert canonicalize_scalar_squares(module) == 0


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_square_numeric_special_values_and_random_inputs(dtype):
    rng = np.random.default_rng(947)
    values = np.concatenate(
        (rng.normal(size=10000), [-0.0, 0.0, np.inf, -np.inf, np.nan, np.finfo(dtype).tiny, np.finfo(dtype).max])
    ).astype(dtype)
    with np.errstate(over="ignore", invalid="ignore"):
        expected = np.power(values, dtype(2))
        actual = values * values
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(np.signbit(actual), np.signbit(expected))


def test_build_preparation_applies_square_rewrite(tmp_path):
    from merlin.runtime.backends.zephyr_model import _prepare_model_mlir
    from merlin.xdsl_dialects._common import text

    source = tmp_path / "model.mlir"
    source.write_text(text(_module()))
    prepared = _prepare_model_mlir(source, tmp_path)
    module = parse_mlir_text(prepared.read_text())
    module.verify()
    assert any(op.name == "arith.mulf" for op in module.walk())
    assert not any(op.name == "math.powf" for op in module.walk())
