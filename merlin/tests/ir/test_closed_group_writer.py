"""Supplied complete-source writer contracts never route by shape alone."""

from dataclasses import replace

import pytest
from test_ordered_fma_groups import contraction, module, narrow
from xdsl.dialects import arith, builtin, func, memref
from xdsl.ir import Block

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.closed_group_writer import (
    ClosedGroupWriterContract,
    install_closed_group_writers,
    source_function_semantic_sha256,
)
from merlin.llvmlower.ordered_bf16_group_binding import SourceExactGroupPreparation
from merlin.xdsl_dialects._common import text


def prepared(tmp_path, *, emit_c_interface=False):
    block = Block(arg_types=[builtin.TensorType(builtin.bf16, [2, 3, 4]), builtin.TensorType(builtin.bf16, [2, 4, 5])])
    left, right = contraction(block, *block.args), contraction(block, *block.args)
    a, b = narrow(block, left.results[0]), narrow(block, right.results[0])
    source = tmp_path / "source.mlir"
    original = module(block, [a.results[0], b.results[0]])
    if emit_c_interface:
        original.body.block.first_op.attributes["llvm.emit_c_interface"] = builtin.UnitAttr()
    source.write_text(text(original))
    prepare = SourceExactGroupPreparation()
    path = prepare(source, tmp_path / "prepared")
    ir = parse_mlir_text(path.read_text())
    records = prepare.receipt["records"]
    functions = {op.sym_name.data: op for op in ir.body.block.ops if isinstance(op, func.FuncOp)}
    return ir, prepare.receipt, records, functions


def contract(function, symbol="external_group"):
    return ClosedGroupWriterContract(
        source_function_semantic_sha256(function),
        symbol,
        "1" * 64,
        "2" * 64,
        "exact_endpoint",
        "3" * 64,
        64,
        True,
        True,
        True,
        True,
        "rne_returned_values",
        "native_functional_screen",
    )


def test_same_shape_selected_and_unselected_source_bodies_stay_distinct(tmp_path):
    ir, receipt, records, functions = prepared(tmp_path)
    selected, unchanged = (functions[record["symbol"]] for record in records)
    old_body = text(unchanged)
    result = install_closed_group_writers(ir, receipt, {records[0]["symbol"]: contract(selected)})
    assert result["source_groups"] == result["physical_writers"] == 1
    assert text(unchanged) == old_body
    assert not any(op.name == "math.fma" for op in selected.walk())
    assert any(op.name == "math.fma" for op in unchanged.walk())
    assert result["fresh_writer_report"][0]["calls"] == 1
    assert result["fresh_writer_report"][0]["argument_ranks"] == [3, 3, 3]
    ir.verify()


def test_equal_complete_semantics_and_abi_share_one_physical_writer(tmp_path):
    ir, receipt, records, functions = prepared(tmp_path)
    contracts = {record["symbol"]: contract(functions[record["symbol"]]) for record in records}
    result = install_closed_group_writers(ir, receipt, contracts)
    assert result["source_groups"] == 2 and result["physical_writers"] == 1
    assert result["fresh_writer_report"][0]["calls"] == 2


def test_selected_source_writer_roundtrips_through_normal_frontend(tmp_path):
    ir, receipt, records, functions = prepared(tmp_path)
    install_closed_group_writers(ir, receipt, {records[0]["symbol"]: contract(functions[records[0]["symbol"]])})
    roundtrip = parse_mlir_text(text(ir))
    roundtrip.verify()
    assert any(op.name == "bufferization.to_buffer" for op in roundtrip.walk())
    assert any(op.name == "bufferization.to_tensor" for op in roundtrip.walk())
    assert any(op.name == "memref.alloc" for op in roundtrip.walk())


@pytest.mark.parametrize(
    "field,value",
    [
        ("source_semantic_sha256", "0" * 64),
        ("fully_writes_result", False),
        ("preserves_inputs", False),
        ("borrowed_buffers", False),
        ("complete_source_fallback", False),
        ("fenv_policy", "unproved"),
        ("allocation_alignment", 3),
        ("numerical_witness_sha256", "bad"),
    ],
)
def test_missing_source_effect_or_endpoint_obligation_refuses_transactionally(tmp_path, field, value):
    ir, receipt, records, functions = prepared(tmp_path)
    before = text(ir)
    selected = functions[records[0]["symbol"]]
    with pytest.raises(ValueError):
        install_closed_group_writers(ir, receipt, {records[0]["symbol"]: replace(contract(selected), **{field: value})})
    assert text(ir) == before


def test_source_mutation_after_binding_cannot_install_same_shape_provider(tmp_path):
    ir, receipt, records, functions = prepared(tmp_path)
    selected = functions[records[0]["symbol"]]
    supplied = contract(selected)
    constant = next(op for op in selected.walk() if isinstance(op, arith.ConstantOp))
    constant.properties["value"] = builtin.FloatAttr(0.25, builtin.f32)
    ir.verify()
    before = text(ir)
    with pytest.raises(ValueError, match="retained function changed"):
        install_closed_group_writers(ir, receipt, {records[0]["symbol"]: supplied})
    assert text(ir) == before


def test_same_physical_symbol_with_different_numerical_policy_refuses(tmp_path):
    ir, receipt, records, functions = prepared(tmp_path)
    first, second = records
    before = text(ir)
    with pytest.raises(ValueError, match="unequal source, ABI or proof"):
        install_closed_group_writers(
            ir,
            receipt,
            {
                first["symbol"]: contract(functions[first["symbol"]]),
                second["symbol"]: replace(
                    contract(functions[second["symbol"]]), numerical_policy="bounded_adjacent_endpoint"
                ),
            },
        )
    assert text(ir) == before


def test_provenance_does_not_choose_policy_but_numerical_properties_remain_bound(tmp_path):
    ir, receipt, records, functions = prepared(tmp_path)
    selected = functions[records[0]["symbol"]]
    pin = source_function_semantic_sha256(selected)
    selected.attributes["prov.fqn"] = builtin.StringAttr("arbitrary_source_name")
    assert source_function_semantic_sha256(selected) == pin
    selected.attributes["strictfp"] = builtin.UnitAttr()
    assert source_function_semantic_sha256(selected) != pin


def test_planned_borrowed_symbol_collision_refuses_before_any_body_replacement(tmp_path):
    ir, receipt, records, functions = prepared(tmp_path)
    before = text(ir)
    with pytest.raises(ValueError, match="planned declaration"):
        install_closed_group_writers(
            ir,
            receipt,
            {
                records[0]["symbol"]: contract(functions[records[0]["symbol"]]),
                records[1]["symbol"]: contract(functions[records[1]["symbol"]], "external_group_borrowed"),
            },
        )
    assert text(ir) == before


def test_duplicated_source_receipt_cannot_claim_extra_call_coverage(tmp_path):
    ir, receipt, records, functions = prepared(tmp_path)
    before = text(ir)
    receipt["records"].append(records[0])
    receipt["source_groups"] += 1
    with pytest.raises(ValueError, match="distinct binding count"):
        install_closed_group_writers(ir, receipt, {records[0]["symbol"]: contract(functions[records[0]["symbol"]])})
    assert text(ir) == before


@pytest.mark.parametrize("name", ["external_group", "external_group_borrowed"])
def test_non_function_top_level_symbol_collision_refuses_transactionally(tmp_path, name):
    ir, receipt, records, functions = prepared(tmp_path)
    ir.body.block.add_op(
        memref.GlobalOp.get(builtin.StringAttr(name), builtin.MemRefType(builtin.f32, [1]), builtin.UnitAttr())
    )
    ir.verify()
    before = text(ir)
    with pytest.raises(ValueError, match="source or planned declaration"):
        install_closed_group_writers(ir, receipt, {records[0]["symbol"]: contract(functions[records[0]["symbol"]])})
    assert text(ir) == before
