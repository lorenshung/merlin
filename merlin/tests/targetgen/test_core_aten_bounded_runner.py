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


def test_full_success_exports_observation_and_device_receipt():
    boundary = dict(
        status="executed",
        output={"values": [7]},
        post_arguments={},
        mutated_arguments=[],
        output_input_aliases=[],
        execution_kind="simulator",
        evidence={"boundary": "same_conversion_exported_program"},
    )
    evidence = {"lane": "device", "routing": {"execution_evidence": {"executed_instructions": 7}}}
    result = semantic_observation({"status": "pass", "semantic_scope": "full", "observation": boundary}, evidence)
    assert result["status"] == "executed"
    assert result["output"] == boundary["output"]
    assert result["evidence"]["boundary"] == boundary["evidence"]["boundary"]
    assert result["evidence"]["routing"] == evidence["routing"]


def test_admission_uses_same_conversion_mutation_result_roles():
    source = """builtin.module {
  func.func @forward(%arg0: tensor<2xf32>) -> (tensor<2xf32>, tensor<2xf32>) {
    func.return %arg0, %arg0 : tensor<2xf32>, tensor<2xf32>
  }
}"""
    abi = {"dtype": "f32", "shape": [2]}
    capture = {
        "capture_meta": {
            "input_abi": [abi],
            "output_abi": [abi],
            "result_contract": {"results": [{"role": "user_input_mutation"}, {"role": "user_output"}]},
        }
    }
    assert bundle_admission_reason(capture, source) is None


def test_secondary_numeric_grade_does_not_apply_full_boundary():
    import struct

    case = {
        "expected": {"kind": "tensor", "dtype": "float32", "shape": [1], "values": [2.0]},
        "comparison": "torch_close",
    }
    record = dict(
        case_id="case",
        overload="aten.alias.default",
        case=case,
        status="bundled",
        output_indices=[0],
        output_abi=[{"dtype": "f32", "shape": [1]}],
        semantic_boundary={"post_indices": [1]},
    )
    batch = {"cases": [record]}
    assert grade_portable_outputs(batch, [struct.pack("<f", 2.0)])["cases"]["case"]["status"] == "pass"
    assert record["semantic_boundary"] == {"post_indices": [1]}


def test_secondary_numeric_grade_uses_runtime_dynamic_shape():
    import struct

    case = {
        "expected": {"kind": "tensor", "dtype": "float32", "shape": [2], "values": [2.0, 3.0]},
        "comparison": "torch_close",
    }
    batch = {
        "cases": [
            dict(
                case_id="case",
                overload="aten.alias.default",
                case=case,
                status="bundled",
                output_indices=[0],
                output_abi=[{"dtype": "f32", "shape": [2], "declared_shape": [-1]}],
            )
        ]
    }
    result = grade_portable_outputs(batch, [struct.pack("<ff", 2.0, 3.0)], output_shapes=[[2]])
    assert result["cases"]["case"]["status"] == "pass"


def test_failure_family_merges_locations_but_separates_runtime_symbols():
    from merlin.targetgen.core_aten_bounded_runner import failure_family

    first = failure_family("compile_lowering_failed", "/tmp/a.o: undefined reference to `helper'", "aten.alias.default")
    second = failure_family(
        "compile_lowering_failed", "/tmp/b.o: undefined reference to `helper'", "aten.clone.default"
    )
    other = failure_family("compile_lowering_failed", "/tmp/b.o: undefined reference to `other'", "aten.clone.default")
    assert first == second
    assert first != other
    assert first[1] == "Merlin"


def test_summary_handles_families_present_only_in_baseline(tmp_path):
    import importlib.util
    from argparse import Namespace

    from merlin.common.paths import repo_root

    script = repo_root() / "build_tools/scripts/run_bounded_core_aten_suite.py"
    spec = importlib.util.spec_from_file_location("bounded_measurement_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    baseline = tmp_path / "baseline" / "scalar"
    baseline.mkdir(parents=True)
    old = dict(
        case_count=1,
        status_counts={"ungradable": 1},
        cases={"case": {"status": "ungradable", "reason": "byte readback cannot observe aliases"}},
    )
    (baseline / "report.json").write_text(json.dumps(old))
    current = dict(
        case_count=1,
        campaign_complete=True,
        status_counts={"pass": 1},
        cases={"case": {"status": "pass", "evidence": {"lane": "host", "numeric_verdict": {"status": "pass"}}}},
    )
    case = dict(case_id="case", overload="aten.alias.default", partition_assignment={})
    args = Namespace(output=tmp_path / "current", baseline=tmp_path / "baseline", label="scalar")
    module.summarize(args, {"selected_cases": [case]}, current, {}, {})
    result = json.loads((args.output / args.label / "summary.json").read_text())
    assert result["failure_families"] == []
    assert result["delta_by_family"]["legacy numeric-only boundary"] == {"baseline": 1, "current": 0, "delta": -1}
    assert result["full_semantic_passed_count"] == 1
    assert result["numeric_only_passed_count"] == 1


def test_evaluate_only_forwards_from_controller_to_pinned_torch(tmp_path, monkeypatch):
    import importlib.util
    import sys
    from types import SimpleNamespace

    from merlin.common.paths import repo_root

    script = repo_root() / "build_tools/scripts/run_bounded_core_aten_suite.py"
    spec = importlib.util.spec_from_file_location("bounded_evaluation_runner", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"selected_cases": []}))
    pinned = tmp_path / "pinned-python"
    arguments = [
        str(script),
        "--suite",
        str(suite),
        "--output",
        str(tmp_path / "output"),
        "--repo",
        str(tmp_path),
        "--m2m-dir",
        str(tmp_path / "m2m"),
        "--m2m-python",
        str(pinned),
        "--label",
        "scalar",
        "--evaluate-only",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    monkeypatch.setattr(module, "sys", SimpleNamespace(argv=arguments, executable="controller-python", modules={}))
    monkeypatch.setattr(module, "stamp", lambda _: {})

    def unavailable(name):
        assert name == "torch"
        raise ModuleNotFoundError("No module named 'torch'", name="torch")

    monkeypatch.setattr(module, "__import__", unavailable, raising=False)
    commands = []
    monkeypatch.setattr(module.subprocess, "run", lambda argv, **kwargs: commands.append(argv))
    module.main()
    assert len(commands) == 1
    assert commands[0][0] == str(pinned)
    assert "--evaluate-only" in commands[0]
