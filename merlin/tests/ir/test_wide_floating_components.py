"""Finite IEEE exponent spans use exact bit-field packing, never float scaling."""

from fractions import Fraction

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.fixed_point_ir import finite_grid
from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.integer_components import rewrite_integer_components


def test_extreme_bfloat_payload_has_full_integer_producers():
    values = [2.0**-133, -(2.0**127), 0.0]
    exponent, bound = finite_grid(values)
    assert exponent == -133 and bound == 1 << 260
    for value in values:
        assert Fraction.from_float(value) / Fraction(2) ** exponent == int(
            Fraction.from_float(value) / Fraction(2) ** exponent
        )
    module = parse_mlir_text("""module {
      func.func @forward(%x: tensor<3xbf16>) -> tensor<3xbf16> {
        %e = tensor.empty() : tensor<3xbf16>
        %r = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>], iterator_types=["parallel"]}
          ins(%x : tensor<3xbf16>) outs(%e : tensor<3xbf16>) {
        ^bb0(%a:bf16,%b:bf16):
          %v=arith.addf %a,%a : bf16
          linalg.yield %v : bf16
        } -> tensor<3xbf16>
        return %r : tensor<3xbf16>
      }}""")
    bounds, proofs = rewrite_floating_components(module, [(exponent, bound)])
    assert proofs[0]["packing"] == "exact_ieee_bit_fields"
    assert proofs[0]["reconstruction"] == "bounded_i1024"
    assert not any(op.name in {"arith.fptosi", "arith.mulf"} for op in module.walk())
    assert all(
        op.operands[0].type.width.data <= 128 for op in module.walk() if op.name in {"arith.divui", "arith.remui"}
    )
    plans = rewrite_integer_components(module, limb_attribute="test.limb", known_bounds=bounds, input_bounds=[0])
    assert plans and max(entry["index"] for plan in plans for entry in plan["inputs"]) > 6
    module.verify()
