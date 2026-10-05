"""The granted OOT kit numbers source operations exactly as whole-model admission does."""

from copy import deepcopy
import hashlib
import json

from merlin.targetgen.oot_starterkit.plan import (
    main, source_operation_inventory, validate_mixed_program_plan,
)


SOURCE = '''builtin.module {
  func.func @forward() -> tensor<2xi32> {
    %0 = tensor.empty() : tensor<2xi32>
    %1 = arith.constant dense<[3, 7]> : tensor<2xi32>
    func.return %1 : tensor<2xi32>
  }
}'''
LOWERED = '''builtin.module {
  llvm.func @kernel(%0: !llvm.ptr) {
    %1 = llvm.mlir.constant(0 : i64) : i64
    %2 = llvm.getelementptr %0[%1] {merlin.global_task = 0 : i64, merlin.source_op_index = 1 : i64} : (!llvm.ptr, i64) -> !llvm.ptr, i32
    llvm.return
  }
}'''


def _buffer():
    return {
        "tensors": {"result": {"shape": [2], "dtype": "i32", "role": "output"}},
        "kernel_abi": {"kind": "whole_program", "args": [{"tensor": "result", "access": "write"}],
                       "outputs": ["result"]},
        "params": {"global_program_plan": {
            "schema": "mixed_program_plan_v1",
            "source_sha256": hashlib.sha256(SOURCE.encode()).hexdigest(),
            "source_op_count": 2,
            "tasks": [{"task_index": 0, "kind": "host", "source_op_indices": [0, 1],
                       "instruction_start": 1, "instruction_end": 2, "reads": [], "writes": ["result"]}],
            "schedule_instruction_count": 3,
            "prologue_instruction_range": [0, 1], "epilogue_instruction_range": [2, 3],
            "entry_bindings": [],
            "source_values": [{"op_index": 1, "result_index": 0, "tensor": "result"}],
            "output_bindings": ["result"], "compiler_temporaries": [],
        }},
    }


def test_exact_direct_source_inventory_includes_init_and_pins_bytes():
    inventory = source_operation_inventory(SOURCE)
    assert inventory["source_op_count"] == 2
    assert [row["operation"] for row in inventory["operations"]] == ["tensor.empty", "arith.constant"]
    assert inventory["returns"][0]["source"] == {"op_index": 1, "result_index": 0}
    assert inventory["source_sha256"] != source_operation_inventory(SOURCE + "\n")["source_sha256"]
    renamed = source_operation_inventory(SOURCE.replace("@forward", "@model_entry"))
    assert renamed["entry"] == "model_entry"
    assert renamed["source_op_count"] == 2


def test_public_preflight_accepts_structural_plan_without_claiming_execution():
    result = validate_mixed_program_plan(SOURCE, _buffer(), LOWERED)
    assert result["ok"], result["findings"]
    assert result["scope"] == "public structural preflight only"
    cb = _buffer()
    del cb["params"]["global_program_plan"]["compiler_temporaries"]
    assert validate_mixed_program_plan(SOURCE, cb, LOWERED)["ok"]


def test_public_preflight_rejects_misnumbering_duplicate_region_and_crossing():
    cb = _buffer()
    cb["params"]["global_program_plan"]["tasks"][0]["source_op_indices"] = [1, 1]
    assert not validate_mixed_program_plan(SOURCE, cb)["ok"]
    cb = _buffer()
    cb["params"]["global_program_plan"]["tasks"][0]["source_op_indices"] = [1]
    assert "ownership" in " ".join(validate_mixed_program_plan(SOURCE, cb)["findings"])
    cb = _buffer()
    cb["params"]["global_program_plan"]["tasks"][0]["reads"] = ["missing"]
    assert "absent tensor" in " ".join(validate_mixed_program_plan(SOURCE, cb)["findings"])


def test_public_preflight_rejects_byte_drift_and_wrong_lowered_owner():
    cb = _buffer()
    assert not validate_mixed_program_plan(SOURCE + "\n", cb)["ok"]
    assert not validate_mixed_program_plan(SOURCE, cb, LOWERED.replace("merlin.global_task = 0", "merlin.global_task = 1"))["ok"]


def test_public_preflight_reads_declared_physical_shape_without_claiming_encoding_proof():
    cb = _buffer()
    cb["tensors"]["result"]["shape"] = [1, 2]
    cb["params"]["storage_encodings"] = {"result": {
        "schema": "grouped_axes_storage_v1", "logical_shape": [2], "physical_shape": [1, 2],
        "dtype": "i32", "axis_groups": [[], [0]], "strides_elements": [2, 1],
        "storage_elements": 2, "offset_elements": 0,
    }}
    assert validate_mixed_program_plan(SOURCE, cb)["ok"]
    cb["params"]["storage_encodings"]["result"]["logical_shape"] = [3]
    assert not validate_mixed_program_plan(SOURCE, cb)["ok"]


def test_cli_inventory_and_validation(tmp_path, capsys):
    source = tmp_path / "capsule.interface.mlir"
    source.write_text(SOURCE)
    cb = tmp_path / "command_buffer.json"
    cb.write_text(json.dumps(_buffer()))
    lowered = tmp_path / "lowered.mlir"
    lowered.write_text(LOWERED)
    assert main(["inventory", "--source", str(source)]) == 0
    assert json.loads(capsys.readouterr().out)["source_op_count"] == 2
    assert main(["validate", "--source", str(source), "--command-buffer", str(cb),
                 "--lowered-mlir", str(lowered)]) == 0
    assert json.loads(capsys.readouterr().out)["ok"]
    bad = deepcopy(_buffer())
    bad["params"]["global_program_plan"]["source_op_count"] = 1
    cb.write_text(json.dumps(bad))
    assert main(["validate", "--source", str(source), "--command-buffer", str(cb)]) == 1
