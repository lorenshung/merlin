"""Transfer screens compare observed SSA layouts and canonical dtypes, never spellings or defaults."""

from __future__ import annotations

from merlin.common import mlir_query as mq
from merlin.targetgen.access_observations import type_layout
from merlin.targetgen.application_graph import _program_graph
from merlin.targetgen.software_spec import screen_transfer_contract

_PROGRAM = """"builtin.module"() ({
  "func.func"() <{function_type = (tensor<2x8xi8>, memref<4xi32>, memref<4xi32, strided<[2]>>) -> (), sym_name = "f"}> ({
  ^bb0(%t: tensor<2x8xi8>, %m: memref<4xi32>, %s: memref<4xi32, strided<[2]>>):
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()
"""


def _spec(status="reviewed"):
    return {
        "transfer_contracts": {
            "status": status,
            "operand_load": {
                "from": "host",
                "to": "accelerator",
                "copy": {"dtype": "int8", "layout": "row_major_contiguous"},
                "evidence": {"status": "reviewed", "basis": "fixture"},
            },
        }
    }


def _screen(spec, **observed):
    return screen_transfer_contract(spec, source_placement="host", destination_placement="accelerator", **observed)


def test_value_layout_is_read_from_the_type(tmp_path):
    path = tmp_path / "p.mlir"
    path.write_text(_PROGRAM)
    module = mq.parse(str(path))
    graph = _program_graph(module, "0" * 64)
    arguments = [argument for block in graph["blocks"] for argument in block["arguments"]]
    layouts = {argument["type"]: argument["layout"] for argument in arguments}
    assert layouts["tensor<2x8xi8>"] == "row_major_contiguous"
    assert layouts["memref<4xi32>"] == "row_major_contiguous"
    # An explicit strided layout is not assumed away.
    assert [value for key, value in layouts.items() if "strided" in key] == [None]
    func = next(op for op in mq.walk(module) if mq.op_name(op) == "func.func")
    assert type_layout(func.regions[0].blocks[0].args[0].type) == "row_major_contiguous"


def test_mlir_and_registry_spellings_of_one_format_match():
    row_major = {"operand_layout": "row_major_contiguous", "result_layout": "row_major_contiguous"}
    admitted = _screen(_spec(), operand_dtype="i8", result_dtype="i8", **row_major)
    assert admitted["status"] == "admitted" and admitted["matching_declarations"] == ["operand_load"]
    # A different format is still refused, and unknown layouts stay unresolved rather than matched.
    assert _screen(_spec(), operand_dtype="i32", result_dtype="i32", **row_major)["status"] == "unsupported"
    assert _screen(_spec(), operand_dtype="i8", result_dtype="i8")["status"] == "unknown"
    assert _screen(_spec("unreviewed"), operand_dtype="i8", result_dtype="i8", **row_major)["status"] == "unknown"
