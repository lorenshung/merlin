# ruff: noqa: E501 -- MLIR fixtures retain one operation per line
"""Literal division retains an exact integer numerator and a rational readout."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.integer_components import rewrite_integer_components


# The literal reciprocal is carried as a normalized signed-byte significand
# (scaled by 2**6 so it fills the int8 operand), and the single exact readout
# owns the binary exponent: the common exponent is therefore 6 lower than the
# unnormalized one. The rational denominator (3 for 1/3) is unchanged.
@pytest.mark.parametrize("divisor,exponent,denominator", [(2.0, -9, 1), (-2.0, -9, 1), (3.0, -8, 3), (0.5, -7, 1)])
def test_literal_division_has_a_complete_numerator_producer(divisor, exponent, denominator):
    module = parse_mlir_text(f"""module {{
      func.func @forward(%x: tensor<4xf16>) -> tensor<4xf16> {{
        %c = arith.constant {divisor} : f16
        %s = tensor.splat %c : tensor<4xf16>
        %e = tensor.empty() : tensor<4xf16>
        %r = linalg.generic {{indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}}
          ins(%x, %s : tensor<4xf16>, tensor<4xf16>) outs(%e : tensor<4xf16>) {{
        ^bb0(%xv: f16, %dv: f16, %unused: f16):
          %q = arith.divf %xv, %dv : f16
          linalg.yield %q : f16
        }} -> tensor<4xf16>
        return %r : tensor<4xf16>
      }}
    }}""")
    bounds, proofs = rewrite_floating_components(module, [(-2, 7)])
    assert len(proofs) == 1 and proofs[0]["source_data_arithmetic"]
    assert proofs[0]["common_exponent"] == exponent
    assert not any(op.name == "arith.divf" for op in module.walk())
    plans = rewrite_integer_components(module, limb_attribute="test.limb", known_bounds=bounds, input_bounds=[0])
    assert plans
    module.verify()
