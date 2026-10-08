"""Contract tests without Torch or a simulator installation."""

import json

from merlin.targetgen.core_aten_bounded_runner import (
    SavedResponseAdapter,
    atomic_json,
    bounded_loader_source,
    bundle_admission_reason,
    case_digest,
    grade_portable_outputs,
    semantic_observation,
)


def test_numeric_success_does_not_claim_complete_semantics():
    evidence = {"numeric_verdict": {"status": "pass"}, "lane": "host"}
    observed = semantic_observation({"status": "pass"}, evidence)
    assert observed["status"] == "ungradable"
    assert observed["evidence"] == evidence
    assert "aliases" in observed["reason"]


def test_failure_and_device_evidence_are_preserved():
    evidence = {"lane": "device", "executed_instructions": 7}
    observed = semantic_observation({"status": "execution_failed", "reason": "missing receipt"}, evidence)
    assert observed == {"status": "execution_failed", "reason": "missing receipt", "evidence": evidence}


def test_loader_preserves_layout_and_seed():
    case = {
        "overload": "aten.alias.default",
        "rng_seed": 123,
        "arguments": {
            "args": {
                "kind": "tuple",
                "items": [
                    {
                        "kind": "tensor",
                        "dtype": "float32",
                        "shape": [2],
                        "stride": [2],
                        "storage_offset": 1,
                        "requires_grad": False,
                        "values": [1, 2],
                    }
                ],
            },
            "kwargs": {},
        },
        "expected": {"kind": "tensor"},
    }
    source = bounded_loader_source(case)
    compile(source, "loader.py", "exec")
    assert "as_strided(shape, stride, offset)" in source
    assert "torch.manual_seed(123)" in source
    assert 'complex(_decode(value["real"]), _decode(value["imag"]))' in source


def test_atomic_response_and_case_identity(tmp_path):
    path = tmp_path / "nested" / "response.json"
    document = {"schema_version": 1, "results": {"case": {"status": "capture_unavailable"}}}
    atomic_json(path, document)
    assert json.loads(path.read_text()) == document
    assert not path.with_suffix(".json.tmp").exists()
    assert SavedResponseAdapter(document).execute({}) == document["results"]
    assert case_digest({"a": 1, "b": 2}) == case_digest({"b": 2, "a": 1})
    assert case_digest({"a": 1}) != case_digest({"a": 2})


def test_portable_nonfinite_oracle_uses_same_comparator_without_mutation():
    import struct

    case = {
        "expected": {
            "kind": "tensor",
            "dtype": "float32",
            "shape": [2],
            "values": [{"kind": "float", "value": "nan"}, {"kind": "float", "value": "inf"}],
        },
        "comparison": "torch_close",
        "comparison_parameters": {"rtol": 1e-5, "atol": 1e-6, "equal_nan": True},
    }
    batch = {
        "cases": [
            {
                "case_id": "case",
                "overload": "aten.alias.default",
                "case": case,
                "status": "bundled",
                "output_indices": [0],
                "output_abi": [{"dtype": "f32", "shape": [2]}],
            }
        ]
    }
    before = json.dumps(batch, sort_keys=True)
    assert (
        grade_portable_outputs(batch, [struct.pack("<ff", float("nan"), float("inf"))])["cases"]["case"]["status"]
        == "pass"
    )
    assert grade_portable_outputs(batch, [struct.pack("<ff", 1, float("inf"))])["cases"]["case"]["status"] == "mismatch"
    assert json.dumps(batch, sort_keys=True) == before


def test_bundle_refusal_reports_native_abi_reason():
    assert (
        bundle_admission_reason({"capture_meta": {"input_abi": [{"dtype": "bf16", "shape": [2]}]}})
        == "ValueError: unsupported batch ABI dtype 'bf16'"
    )
    assert bundle_admission_reason({"capture_meta": {"output_abi": [{"dtype": "f32", "shape": [2]}]}}) is None


def test_function_and_capture_abi_counts_must_agree():
    source = """builtin.module {
  func.func @forward(%arg0: tensor<2xf32>) -> tensor<2xf32> {
    func.return %arg0 : tensor<2xf32>
  }
}"""
    abi = {"dtype": "f32", "shape": [2]}
    assert "argument ABI" in bundle_admission_reason(
        {"capture_meta": {"input_abi": [abi, abi], "output_abi": [abi]}}, source
    )
    assert "result ABI" in bundle_admission_reason(
        {"capture_meta": {"input_abi": [abi], "output_abi": [abi, abi]}}, source
    )


def test_same_conversion_roles_admit_mutation_and_user_results():
    source = """builtin.module {
  func.func @forward(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    func.return %arg0, %arg0 : tensor<2xf32>, tensor<2xf32>
  }
}"""
    abi = {"dtype": "f32", "shape": [2]}
    meta = {
        "input_abi": [abi],
        "output_abi": [abi],
        "result_contract": {"results": [{"role": "user_input_mutation"}, {"role": "user_output"}]},
    }
    assert bundle_admission_reason({"capture_meta": meta}, source) is None
    meta["result_contract"]["results"].pop()
    assert "result ABI" in bundle_admission_reason({"capture_meta": meta}, source)
