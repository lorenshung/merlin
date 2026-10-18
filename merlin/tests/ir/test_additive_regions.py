"""Closed reductions carry data work, and never grant a conversion-only placement."""

import pytest
from xdsl.dialects.builtin import i128

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.additive_regions import positive_zero_seed, retype_additive
from merlin.llvmlower.floating_components import rewrite_floating_components
from merlin.llvmlower.integer_components import rewrite_integer_components


def reduction(seed="0.0", operation="addf"):
    return parse_mlir_text(
        """builtin.module {
    func.func @forward(%x: tensor<2x3xf16>) -> tensor<2xf32> {
      %c = arith.constant SEED : f32
      %z = tensor.splat %c : tensor<2xf32>
      %r = linalg.generic {
        indexing_maps = [affine_map<(d0,d1)->(d0,d1)>,affine_map<(d0,d1)->(d0)>],
        iterator_types = ["parallel","reduction"]
      } ins(%x : tensor<2x3xf16>) outs(%z : tensor<2xf32>) {
      ^bb0(%a: f16, %b: f32):
        %w = arith.extf %a : f16 to f32
        %v = arith.OP %w, %b : f32
        linalg.yield %v : f32
      } -> tensor<2xf32>
      func.return %r : tensor<2xf32>
    }}""".replace("SEED", seed).replace("OP", operation)
    )


def test_additive_reduction_emits_integer_sum_and_valid_output_rank_maps():
    m = reduction()
    bounds, proofs = rewrite_floating_components(m, [(-12, 5120)])
    plans = rewrite_integer_components(m, limb_attribute="test.limb", input_bounds=[0], known_bounds=bounds)
    m.verify()
    assert any(p["source_data_arithmetic"] for p in proofs)
    assert plans
    assert not any(op.name == "arith.addf" for op in m.walk())
    for op in m.walk():
        if op.name == "linalg.generic":
            assert all(a.data.num_dims == len(op.get_iterator_types()) for a in op.get_indexing_maps())
    producers = [op for op in m.walk() if "test.limb" in op.attributes]
    assert any(any(it.data.value == "reduction" for it in op.get_iterator_types()) for op in producers)


@pytest.mark.parametrize("seed", ["-0.0", "1.0"])
def test_nonpositive_zero_seed_is_not_replaced(seed):
    m = reduction(seed=seed)
    op = next(op for op in m.walk() if op.name == "linalg.generic")
    assert not positive_zero_seed(op.outputs[0])
    assert retype_additive(op, {}, i128, floating=True) is None


def test_unknown_data_operation_refuses():
    m = reduction(operation="divf")
    op = next(op for op in m.walk() if op.name == "linalg.generic")
    assert retype_additive(op, {}, i128, floating=True) is None
