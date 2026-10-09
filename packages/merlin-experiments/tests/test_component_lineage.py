"""Exercise real invocation storage around a harmless native author control.

Other admission facets are isolated with private test monkeypatches. This is not
a qualified model session, target runtime or experimental compiler lineage.
"""

import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import component_lineage as L
from merlin_experiments.phase1.component_origin import FreshCompilerOrigin
from merlin_experiments.phase1.providers import codex_agent as CA
from merlin_experiments.phase2 import component_launch as CL
from merlin_experiments.phase2 import transcript_audit
from merlin_experiments.phase2.contracts import StageGateError

from merlin.common import invocation_record


def test_actual_call_record_path_binds_native_control_descendant(tmp_path, monkeypatch):
    candidate, stage = tmp_path / "candidate", tmp_path / "stage"
    candidate.mkdir()
    stage.mkdir()
    (candidate / "compiler.py").write_text("initial independent control\n")
    source = tmp_path / "source-origin.json"
    source.write_text('{"scope":"synthetic facet control"}')
    origin = FreshCompilerOrigin(
        None, candidate, "1" * 64, candidate, "1" * 64, source, "1" * 64, source, "1" * 64, "{}", (), object()
    )
    monkeypatch.setattr(FreshCompilerOrigin, "verify", lambda *args, **kwargs: {})
    receipts = stage / "receipts.jsonl"
    receipts.write_text('{"scope":"synthetic private broker control"}\n')
    policy = SimpleNamespace(
        target_experiment=SimpleNamespace(),
        receipt_path=receipts,
        build_registry=lambda: (),
        verify_receipts=lambda *args, **kwargs: {"control": "isolated"},
    )
    authority = SimpleNamespace(seed=candidate, binding={}, validate_candidate=lambda *args: {})
    qualification = SimpleNamespace(compiler_origin=origin)
    inputs = CL.ComponentLaunchInputs(
        None,
        candidate,
        policy,
        authority,
        qualification,
        (),
        (),
        (),
        tmp_path / "client",
        "/usr/bin/client",
        tmp_path / "credential",
        stage,
        tmp_path / "prices.json",
    )
    launch = CL.QualifiedComponentLaunch(inputs, source, "1" * 64, "control", "control", object())
    monkeypatch.setattr(CL.ComponentLaunchInputs, "verify", lambda *args, **kwargs: None)
    monkeypatch.setattr(CL.QualifiedComponentLaunch, "verify", lambda *args, **kwargs: None)
    monkeypatch.setattr(CL, "_read_paths", lambda inputs: ())
    monkeypatch.setattr(transcript_audit, "audit_codex_transcript", lambda *args: {"clean": True})

    def native_control(workspace, output, *args, **kwargs):
        # Run an actual isolated local process boundary. No model is invoked.
        # Its emitted stream and authored bytes are read by the actual recorder.
        program = (
            "from pathlib import Path; import json,sys; "
            "Path(sys.argv[1]).write_text('authored control output\\n'); "
            "print(json.dumps({'type':'thread.started','thread_id':'native-control'}))"
        )
        process = subprocess.run(
            [sys.executable, "-I", "-B", "-c", program, str(workspace / "compiler.py")], capture_output=True, check=True
        )
        transcript = output / "native-control.jsonl"
        transcript.write_bytes(process.stdout)
        return process.returncode, transcript

    monkeypatch.setattr(CA, "run_round", native_control)
    rc, transcript, lineage = L.run_component_origin_round(
        launch, inputs, model="control", effort="control", wall_budget_seconds=10, prompt="control only"
    )
    assert rc == 0 and json.loads(transcript.read_text())["thread_id"] == "native-control"
    observed = invocation_record.verify(lineage.author_invocation)
    assert observed["stage"] == "fresh_compiler_phase2_authoring" and observed["outcome"] == "returned"
    assert lineage.author_invocation.parent.parent.name == "invocations"
    assert not (stage / "author_origin" / "invocation.json").exists()
    lineage.verify(candidate=candidate)
    assert (lineage.candidate / "compiler.py").read_text() == "authored control output\n"
    with pytest.raises(StageGateError, match="authority fields changed"):
        replace(lineage, candidate_sha256="2" * 64).verify(candidate=candidate)
