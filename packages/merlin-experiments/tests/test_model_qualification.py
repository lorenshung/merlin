"""Coarse offline compiler qualification preserves the real process and artifact boundaries."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import yaml
from merlin_experiments import model_qualification as Q


def _bundle(root: Path):
    root.mkdir()
    for name, raw in {
        "model.mlir": b"module {}\n",
        "weights.safetensors": b"fixture",
        "weights.safetensors.manifest.json": b"{}",
        "inputs.npz": b"fixture",
        "input_order.json": b"{}",
        "golden.npy": b"fixture",
    }.items():
        (root / name).write_bytes(raw)
    return root


def _package(root: Path):
    root.mkdir()
    tool = root / "compiler.py"
    tool.write_text(
        "#!/usr/bin/env python3\n"
        "import json,sys\nfrom pathlib import Path\nimport xdsl\n"
        "for arg in sys.argv:\n"
        "    if arg.startswith('--emit-command-buffer='):\n"
        "        Path(arg.partition('=')[2]).write_text(json.dumps({'commands':[], 'declined':{'reason':'fixture'}}))\n"
        "print('module {}')\n"
    )
    tool.chmod(0o755)
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": "fixture",
        "package_id": "fixture",
        "language": "python",
        "integrity_exempt": False,
        "authoring": {"mode": "agent_generated_from_rtl_facts", "author": "fixture", "generated_by_agent": True},
        "entrypoints": {"tool": "compiler.py"},
        "commands": {
            name: {"argv": ["{tool}", "{input_mlir}"]}
            for name in ("parse", "lower_interface_to_target", "lower_target_to_llvm")
        },
    }
    manifest["commands"]["emit_command_buffer"] = {
        "argv": ["{tool}", "--emit-command-buffer={output_json}", "{input_mlir}"]
    }
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    return root


def test_offline_process_boundary_preserves_interpreter_and_refuses_scope_upgrade(tmp_path, monkeypatch):
    if not all(shutil.which(tool) for tool in ("bwrap", "prlimit", "taskset")):
        pytest.skip("bounded local compiler tools are unavailable")
    monkeypatch.setenv("PYTHONPATH", ":".join(str(Path(path).absolute()) for path in __import__("sys").path if path))
    bundle, package = _bundle(tmp_path / "capture"), _package(tmp_path / "compiler")
    output = tmp_path / "qualification"
    result = Q.qualify(
        bundle=bundle,
        package=package,
        target="fixture",
        output=output,
        timeout_seconds=20,
        stage_timeout_seconds=5,
        memory_gib=2,
        lower_native=True,
    )
    assert result["status"] == "completed"
    statuses = {row["entrypoint"]: row["status"] for row in result["observations"]["compiler_observations"]}
    assert statuses == {
        "parse": "accepted",
        "lower_interface_to_target": "unchanged",
        "emit_command_buffer": "declined",
        "lower_target_to_llvm": "unqualified",
    }
    assert all(row["canonical_ir_changed"] is False for row in result["observations"]["compiler_observations"])
    command_row = next(
        row for row in result["observations"]["compiler_observations"] if row["entrypoint"] == "emit_command_buffer"
    )
    assert command_row["command_buffer"]["status"] == "declined"
    assert command_row["command_buffer"]["commands_count"] == 0
    assert result["observations"]["model_routes"][0]["accelerator"]["status"] == "declined"
    assert result["observations"]["model_routes"][0]["status"] == "observed_decline"
    assert result["whole_workload_validation_verified"] is False
    assert result["observations"]["full_native_lowering_verified"] is False
    assert result["observations"]["native_lowerings"][0]["status"] == "failed"
    assert result["observations"]["workflow"]["application_validation_blockers"]
    assert result["observations"]["runtime"]["target_executed"] is False
    assert json.loads((output / "qualification.json").read_bytes()) == result
    assert (output / "README.md").is_file()
    assert (output / "qualification.json").stat().st_mode & 0o222 == 0
    assert output.stat().st_mode & 0o222 == 0
    with pytest.raises(ValueError, match="fresh"):
        Q.qualify(bundle=bundle, package=package, target="fixture", output=output)


def test_multi_program_roster_cannot_escape_or_become_one_forward(tmp_path):
    root = tmp_path / "session"
    root.mkdir()
    _bundle(root / "prefix")
    _bundle(root / "recurrent")
    document = {
        "version": 2,
        "programs": [
            {"name": "prefix", "bundle": "prefix", "steps": 1},
            {"name": "recurrent", "bundle": "recurrent", "steps": 10},
        ],
        "provenance": {"full_checkpoint": False, "synthetic_inputs": True},
    }
    (root / "session_contract.yaml").write_text(yaml.safe_dump(document))
    report = Q.inspect_workflow(root)
    assert len(report["programs"]) == 2
    assert report["application_validation_blockers"]
    document["programs"][1]["bundle"] = "../elsewhere"
    (root / "session_contract.yaml").write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="escapes"):
        Q.inspect_workflow(root)


def test_command_observation_requires_an_explicit_nonempty_route(tmp_path):
    command = tmp_path / "commands.json"
    command.write_text(json.dumps({"commands": []}))
    assert Q._command_observation(command)["status"] == "invalid"
    command.write_text(json.dumps({"commands": [{"opcode": "MATMUL"}], "declined": {"reason": "no"}}))
    assert Q._command_observation(command)["status"] == "invalid"
    command.write_text(json.dumps({"commands": [{"opcode": "MATMUL"}]}))
    assert Q._command_observation(command)["status"] == "emitted"


def test_conditional_ssa_edge_becomes_transfer_only_after_reviewed_placements(monkeypatch):
    from merlin_experiments.phase1 import model_routes

    monkeypatch.setattr(model_routes, "_graph_totality", lambda *_: ({}, []))
    monkeypatch.setattr(model_routes, "_source_complete", lambda *_: True)

    def obligation(operation_id, selected):
        row = {
            "operation_ids": [operation_id],
            "status": "resolved",
            "precision": {
                "status": "resolved",
                "numerical_contracts": {
                    "host": {"status": "resolved"},
                    "accelerator": {"status": "resolved"},
                },
            },
            "host_admission": {"status": "admitted", "reviewed": True},
            "accelerator_admission": {"status": "unsupported", "reviewed": True},
        }
        if selected == "accelerator":
            row["host_admission"]["status"] = "unsupported"
            row["accelerator_admission"]["status"] = "admitted"
        elif selected is None:
            row["host_admission"]["reviewed"] = False
        return row

    application = {
        "capture_sha256": "exact",
        "capture_receipt": {"status": "verified_materialized", "source_closure_verified": True},
        "n_mlir_operations": 2,
        "completeness": {
            "source_trace": {},
            "operation_obligations": [obligation("op:0", "host"), obligation("op:1", None)],
            "transfer_obligations": [{"producer_operation_id": "op:0", "consumer_operation_id": "op:1"}],
        },
    }
    summary, blockers = model_routes._ledger_observation(application, "exact")
    assert summary["conditional_ssa_edges"]["status"] == "pending_placement"
    assert summary["conditional_ssa_edges"]["required_crossing"] == 0
    assert not any("typed transfer lowering" in reason for reason in blockers)

    application["completeness"]["operation_obligations"][1] = obligation("op:1", "host")
    summary, blockers = model_routes._ledger_observation(application, "exact")
    assert summary["conditional_ssa_edges"]["same_lane"] == 1
    assert not any("typed transfer lowering" in reason for reason in blockers)

    application["completeness"]["operation_obligations"][1] = obligation("op:1", "accelerator")
    summary, blockers = model_routes._ledger_observation(application, "exact")
    assert summary["conditional_ssa_edges"]["required_crossing"] == 1
    assert any("typed transfer lowering" in reason for reason in blockers)


def test_unobserved_oot_route_is_unresolved_not_a_compiler_decline():
    from merlin_experiments.phase1.model_routes import summarize_model_routes

    routes = summarize_model_routes({"programs": [{"name": "model", "model_sha256": "exact"}]}, None, [], [])
    assert routes[0]["status"] == "unresolved"
    assert routes[0]["accelerator"]["status"] == "not_emitted"
    assert routes[0]["whole_model_compiler_verified"] is False
