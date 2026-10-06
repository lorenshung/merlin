from pathlib import Path

from merlin.frontends.linalg_mlir import parse_mlir_text as parse_mlir
from merlin.llvmlower.insert_slice_destination import rewrite_module
from merlin.xdsl_dialects._common import text


def fixture():
    return Path(__file__).with_name("insert_destination_fixture.mlir").read_text()


def test_exact_scalar_body_and_static_slice_are_preserved():
    module = parse_mlir(fixture())
    before = next(op for op in module.walk() if op.name == "linalg.generic")
    body_before = [op.name for op in before.body.block.ops]
    rows = rewrite_module(module)
    assert len(rows) == 1 and rows[0]["offsets"] == [1, 1]
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    assert [op.name for op in generic.body.block.ops] == body_before
    assert generic.outputs[0].owner.name == "tensor.extract_slice"
    assert generic.outputs[0].owner.source.owner.name == "linalg.fill"
    parse_mlir(text(module)).verify()


def test_live_producer_value_elsewhere_is_not_moved():
    source = (
        fixture()
        .replace("-> tensor<4x5xi8> {", "-> (tensor<4x5xi8>,tensor<2x3xi8>) {")
        .replace("return %result : tensor<4x5xi8>", "return %result,%q : tensor<4x5xi8>,tensor<2x3xi8>")
    )
    module = parse_mlir(source)
    before = text(module)
    assert rewrite_module(module) == []
    assert text(module) == before


def test_reading_uninitialized_output_or_index_sensitive_body_refused():
    source = fixture().replace("linalg.yield %y : i8", "%sum = arith.addi %y,%unused : i8\n  linalg.yield %sum : i8")
    assert rewrite_module(parse_mlir(source)) == []
    source = fixture().replace("%y = arith.fptosi", "%index = linalg.index 0 : index\n  %y = arith.fptosi")
    assert rewrite_module(parse_mlir(source)) == []
