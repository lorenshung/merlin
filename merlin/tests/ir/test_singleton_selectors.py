"""Full-domain compare cells prove singleton index fills without using input data."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.singleton_selectors import fold_singleton_selectors
from merlin.targetgen.native_unit_policy import construction_evidence


def source(dtype="i8", seed="-128", cmp="cmpi sgt", eq="cmpi eq", *, extent=1, nan=False, tie=True):
    guard = (
        """%nan=arith.cmpf uno,%xv,%xv : TYPE
               %p=arith.ori %nan,%base : i1"""
        if nan
        else "%p=arith.ori %base,%false : i1"
    )
    expression = "%base=arith.ori %gt,%tied : i1" if tie else "%base=arith.ori %gt,%false : i1"
    return parse_mlir_text(f"""module {{
      func.func @forward(%x:tensor<3x{extent}x{dtype}>) -> tensor<3xi64> {{
        %s=arith.constant {seed} : {dtype}
        %vseed=tensor.splat %s : tensor<3x{dtype}>
        %large=arith.constant 9223372036854775807 : i64
        %iseed=tensor.splat %large : tensor<3xi64>
        %v,%r=linalg.generic {{indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,
          affine_map<(d0,d1)->(d0)>,affine_map<(d0,d1)->(d0)>],iterator_types=["parallel","reduction"]}}
          ins(%x:tensor<3x{extent}x{dtype}>) outs(%vseed,%iseed:tensor<3x{dtype}>,tensor<3xi64>) {{
          ^bb0(%xv:{dtype},%acc:{dtype},%idx:i64):
            %i=linalg.index 1 : index
            %j=arith.index_cast %i : index to i64
            %gt=arith.{cmp}, %xv,%acc : {dtype}
            %eq=arith.{eq}, %xv,%acc : {dtype}
            %lt=arith.cmpi slt,%j,%idx : i64
            %tied=arith.andi %eq,%lt : i1
            %false=arith.constant false
            {expression}
            {guard.replace("TYPE", dtype)}
            %out=arith.select %p,%xv,%acc : {dtype}
            %index=arith.select %p,%j,%idx : i64
            linalg.yield %out,%index : {dtype},i64
        }} -> (tensor<3x{dtype}>,tensor<3xi64>)
        func.return %r : tensor<3xi64>
      }}
    }}""")


@pytest.mark.parametrize(
    "dtype,seed,cmp,eq",
    [
        ("i8", "-128", "cmpi sgt", "cmpi eq"),
        ("i32", "2147483647", "cmpi slt", "cmpi eq"),
        ("i64", "-9223372036854775808", "cmpi sgt", "cmpi eq"),
        ("i64", "0", "cmpi ugt", "cmpi eq"),
        ("i1", "0", "cmpi ugt", "cmpi eq"),
    ],
)
def test_complete_integer_domain_including_tied_endpoints_proves_zero(dtype, seed, cmp, eq):
    module = source(dtype, seed, cmp, eq)
    original = str(module)
    assert construction_evidence(module)["status"] == "eligible"
    assert str(module) == original
    assert fold_singleton_selectors(module) == 1
    module.verify()
    returned = next(op for op in module.walk() if op.name == "func.return").operands[0]
    assert returned.owner.name == "tensor.splat"
    assert returned.owner.operands[0].owner.value.value.data == 0
    assert not any(op.name == "linalg.generic" for op in module.walk())


@pytest.mark.parametrize("dtype", ["f16", "bf16", "f32", "f64"])
def test_ordered_float_cells_and_nan_cell_are_all_covered(dtype):
    module = source(
        dtype,
        "0xFF800000"
        if dtype == "f32"
        else "0xFFF0000000000000"
        if dtype == "f64"
        else "0xFF80"
        if dtype == "bf16"
        else "0xFC00",
        "cmpf ogt",
        "cmpf oeq",
        nan=True,
    )
    assert fold_singleton_selectors(module) == 1
    module.verify()


@pytest.mark.parametrize("extent,tie,seed", [(2, True, "-128"), (1, False, "-128"), (1, True, "0")])
def test_unproved_predicate_or_larger_reduction_is_not_folded(extent, tie, seed):
    module = source(extent=extent, tie=tie, seed=seed)
    original = str(module)
    assert fold_singleton_selectors(module) == 0
    assert str(module) == original


def test_float_nan_cell_prevents_an_ordered_only_identity_proof():
    module = source("f32", "0xFF800000", "cmpf ogt", "cmpf oeq", nan=False)
    original = str(module)
    assert fold_singleton_selectors(module) == 0
    assert str(module) == original


def test_unknown_data_arithmetic_cannot_be_licensed_by_singleton_shape():
    module = source()
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    from xdsl.dialects import arith

    generic.body.block.insert_op_before(
        arith.MuliOp(generic.body.block.args[0], generic.body.block.args[0]), generic.body.block.last_op
    )
    original = str(module)
    assert fold_singleton_selectors(module) == 0
    assert str(module) == original
