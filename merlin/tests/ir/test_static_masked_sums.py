"""Index-selected integer prefixes are exact masked contractions, not host scans."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.integer_components import rewrite_integer_components
from merlin.llvmlower.static_masked_sums import expose_static_masked_sums
from merlin.targetgen.native_unit_policy import construction_evidence


def source(dtype="i64", *, seed="0", predicate="ule", flags=""):
    return parse_mlir_text(f"""module {{ func.func @forward(%x:tensor<2x3x{dtype}>) -> tensor<2x3x{dtype}> {{
      %z=arith.constant {seed} : {dtype}
      %empty=tensor.splat %z : tensor<2x3x{dtype}>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0,d1,d2)->(d2,d1)>,
            affine_map<(d0,d1,d2)->(d0,d1)>],iterator_types=["parallel","parallel","reduction"]}}
        ins(%x:tensor<2x3x{dtype}>) outs(%empty:tensor<2x3x{dtype}>) {{
        ^bb0(%a:{dtype},%acc:{dtype}):
          %p=linalg.index 0 : index
          %k=linalg.index 2 : index
          %cond=arith.cmpi {predicate},%k,%p : index
          %v=arith.select %cond,%a,%z : {dtype}
          %s=arith.{"addf" if dtype == "f32" else "addi"} %acc,%v {flags} : {dtype}
          linalg.yield %s : {dtype}
        }} -> tensor<2x3x{dtype}>
      func.return %r : tensor<2x3x{dtype}>
    }} }}""")


@pytest.mark.parametrize("dtype", ["i8", "i32", "i64"])
@pytest.mark.parametrize("predicate,weights", [("ule", [1, 1, 0, 1]), ("ult", [0, 1, 0, 0]), ("uge", [1, 0, 1, 1])])
def test_mask_is_a_runtime_data_contraction_with_static_coefficients(dtype, predicate, weights):
    module = source(dtype, predicate=predicate)
    assert expose_static_masked_sums(module) == 1
    module.verify()
    table = next(
        op
        for op in module.walk()
        if op.name == "arith.constant"
        and hasattr(op.value, "get_values")
        and str(op.results[0].type.element_type) == "i8"
    )
    assert list(table.value.get_values()) == weights
    assert construction_evidence(module)["status"] == "eligible"
    plans = rewrite_integer_components(module, limb_attribute="test.limb", input_bounds=[1 << (int(dtype[1:]) - 1)])
    assert plans
    module.verify()


@pytest.mark.parametrize("dtype,seed,flags", [("f32", "0.0", ""), ("i64", "1", ""), ("i64", "0", "overflow<nsw>")])
def test_float_order_nonzero_seed_and_overflow_promises_remain_unchanged(dtype, seed, flags):
    module = source(dtype, seed=seed, flags=flags)
    original = str(module)
    assert expose_static_masked_sums(module) == 0
    assert str(module) == original
