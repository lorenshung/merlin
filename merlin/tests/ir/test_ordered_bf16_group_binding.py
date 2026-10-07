"""Normal preparation binds distinct source calls and detects valid-IR mutation."""

import hashlib

import pytest
from test_ordered_fma_groups import contraction, module, narrow, simple
from xdsl.dialects import arith, builtin, func
from xdsl.ir import Block

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.ordered_bf16_group_binding import SourceExactGroupPreparation, verify_group_call_coverage
from merlin.xdsl_dialects._common import text


def test_normal_preparation_routes_same_shape_source_instances_independently(tmp_path):
    block = Block(arg_types=[builtin.TensorType(builtin.bf16, [2, 3, 4]), builtin.TensorType(builtin.bf16, [2, 4, 5])])
    left = contraction(block, *block.args)
    right = contraction(block, *block.args)
    a, b = narrow(block, left.results[0]), narrow(block, right.results[0])
    source = tmp_path / "source.mlir"
    source.write_text(text(module(block, [a.results[0], b.results[0]])))
    pin = hashlib.sha256(source.read_bytes()).hexdigest()
    prepare = SourceExactGroupPreparation(expected_source_sha256=pin)
    result = prepare(source, tmp_path / "normal_prepared")
    coverage = verify_group_call_coverage(result, prepare.receipt)
    assert coverage == dict(groups=2, actual_calls=2, source_contractions=2, target_product_provider_installed=False)
    records = prepare.receipt["records"]
    assert records[0]["endpoint_types"] == records[1]["endpoint_types"]
    assert records[0]["binding_sha256"] != records[1]["binding_sha256"]
    assert records[0]["symbol"] != records[1]["symbol"]
    assert all(record["implementation_kind"] == "ordinary_source_cpu" for record in records)
    assert hashlib.sha256(source.read_bytes()).hexdigest() == pin


def test_preparation_refuses_changed_original_file_before_output(tmp_path):
    source = tmp_path / "source.mlir"
    source.write_text(text(simple()[0]))
    prepare = SourceExactGroupPreparation(expected_source_sha256="0" * 64)
    with pytest.raises(ValueError, match="source changed"):
        prepare(source, tmp_path / "normal_prepared")
    assert not (tmp_path / "normal_prepared").exists()


@pytest.mark.parametrize("mutation", ["coefficient", "extra_call", "receipt", "enclosing_numeric"])
def test_bound_source_call_coverage_refuses_mutation(tmp_path, mutation):
    source = tmp_path / "source.mlir"
    source.write_text(text(simple()[0]))
    prepare = SourceExactGroupPreparation()
    result = prepare(source, tmp_path / "normal_prepared")
    ir = parse_mlir_text(result.read_text())
    symbol = prepare.receipt["records"][0]["symbol"]
    declaration = next(
        operation
        for operation in ir.body.block.ops
        if isinstance(operation, func.FuncOp) and operation.sym_name.data == symbol
    )
    if mutation == "coefficient":
        constant = next(
            operation
            for operation in declaration.walk()
            if isinstance(operation, arith.ConstantOp) and operation.value == builtin.FloatAttr(0.125, builtin.f32)
        )
        constant.properties["value"] = builtin.FloatAttr(0.25, builtin.f32)
        ir.verify()
        result.write_text(text(ir))
    elif mutation == "extra_call":
        call = next(
            operation
            for operation in ir.walk()
            if isinstance(operation, func.CallOp) and operation.callee.root_reference.data == symbol
        )
        call.parent.insert_op_before(call.clone(), call)
        ir.verify()
        result.write_text(text(ir))
    elif mutation == "enclosing_numeric":
        ir.attributes["strictfp"] = builtin.UnitAttr()
        result.write_text(text(ir))
    else:
        prepare.receipt["records"][0]["binding_data"]["source_operations"] += 1
    with pytest.raises(ValueError, match="changed"):
        verify_group_call_coverage(result, prepare.receipt)
