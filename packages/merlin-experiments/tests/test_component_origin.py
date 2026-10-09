"""Fresh-origin refusal controls; no paid author or target hardware is launched."""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import component_origin as O
from merlin_experiments.phase1 import component_qualification as Q
from merlin_experiments.phase1.component_origin import (
    FreshCompilerOrigin,
    FreshToolProbe,
    _tree,
    inert_scaffold,
)
from merlin_experiments.phase2.contracts import StageGateError


def test_complete_initial_scaffold_is_inert_and_contains_no_lowering(tmp_path):
    members = inert_scaffold("synthetic")
    assert set(members) == {"manifest.yaml", "driver.py"}
    manifest = json.loads(members["manifest.yaml"])
    assert manifest["integrity_exempt"] is False
    assert set(manifest["commands"]) == {
        "parse",
        "lower_interface_to_target",
        "emit_command_buffer",
        "lower_target_to_llvm",
    }
    for name, text in members.items():
        (tmp_path / name).write_text(text)
    # Execute the actual initial CLI: even parse cannot emit a prebuilt answer.
    result = subprocess.run(["python3", str(tmp_path / "driver.py"), "parse"], capture_output=True, text=True)
    assert result.returncode != 0 and result.stdout == ""
    assert "no dialect or operation lowering" in result.stderr


@pytest.mark.parametrize("claim", [None, {"status": "qualified", "fresh": True}, object()])
def test_json_and_callers_cannot_mint_origin_for_arbitrary_candidate(tmp_path, claim):
    (tmp_path / "compiler.py").write_text("preexisting implementation\n")
    with pytest.raises(StageGateError, match="issued fresh Phase 1"):
        Q._verify_origin(claim, None, tmp_path)


def test_forged_origin_with_all_apparent_hash_fields_cannot_qualify(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "compiler.py").write_text("preexisting implementation\n")
    receipt = tmp_path / "compiler_origin.json"
    receipt.write_text('{"schema":"merlin.fresh_compiler_origin.v1","status":"fresh_origin_observed"}')
    origin = FreshCompilerOrigin(
        None,
        candidate,
        _tree(candidate)["sha256"],
        candidate,
        _tree(candidate)["sha256"],
        receipt,
        "1" * 64,
        receipt,
        "1" * 64,
        "{}",
        (),
        object(),
    )
    with pytest.raises(StageGateError, match="not issued around actual fresh"):
        origin.verify(candidate=candidate)
    with pytest.raises(StageGateError, match="not issued around actual fresh"):
        Q._verify_origin(origin, None, candidate)


def test_mutating_forged_record_to_qualified_does_not_grant_authority(tmp_path):
    origin = FreshCompilerOrigin(
        None,
        tmp_path,
        "1" * 64,
        tmp_path,
        "1" * 64,
        tmp_path / "r.json",
        "1" * 64,
        tmp_path / "t.json",
        "1" * 64,
        "{}",
        (),
        None,
    )
    with pytest.raises(StageGateError, match="not issued around actual fresh"):
        replace(origin, candidate_sha256="2" * 64, receipt_sha256="2" * 64).verify()


def test_retaining_a_control_issuance_token_cannot_rebind_candidate_bytes(tmp_path, monkeypatch):
    # Private synthetic control isolates registry tamper detection. It never
    # calls authoring or produces an admitted experimental compiler origin.
    token = object()
    origin = FreshCompilerOrigin(
        None,
        tmp_path,
        "1" * 64,
        tmp_path,
        "1" * 64,
        tmp_path / "r.json",
        "1" * 64,
        tmp_path / "t.json",
        "1" * 64,
        "{}",
        (),
        token,
    )
    monkeypatch.setitem(O._ISSUED, token, O._authority_identity(origin))
    with pytest.raises(StageGateError, match="authority fields changed"):
        replace(origin, candidate_sha256="2" * 64).verify()


def test_origin_does_not_hide_untracked_members_and_refuses_symlink_ancestor(tmp_path):
    directory = tmp_path / "source"
    directory.mkdir()
    (directory / "compiler.py").write_text("source\n")
    before = _tree(directory)["sha256"]
    (directory / "ignored.py").write_text("additional source\n")
    assert _tree(directory)["sha256"] != before
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    with pytest.raises(StageGateError, match="indirect"):
        _tree(alias)


def test_phase1_bootstrap_checks_shared_tools_instead_of_candidate_compiler():
    for capability in ("translator", "host_compiler", "linker", "functional_simulator"):
        FreshToolProbe(capability, ("/usr/bin/tool", "--version"), "1" * 64).verify()
    with pytest.raises(StageGateError, match="shared-tool probes"):
        FreshToolProbe("cca", ("/candidate/driver.py", "inspect"), "1" * 64).verify()


def test_missing_independent_runtime_refuses_before_legacy_backend_lookup(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_runner

    old_backend_calls = []
    monkeypatch.setattr(capsule_runner, "qa_checkpoint_adapters", lambda *args: old_backend_calls.append(args))
    # Isolate the next admission gate; this does not issue origin or qualify an
    # experiment. Production callers still need its actual issued origin.
    monkeypatch.setattr(Q, "_verify_origin", lambda *_args: {"phase1_origin_sha256": "1" * 64})
    with pytest.raises(StageGateError, match="independently issued"):
        Q.qualify_component_compiler(
            tmp_path,
            corpus_root=tmp_path,
            target_experiment=SimpleNamespace(path=tmp_path / "target.yaml"),
            contract_root=tmp_path,
            source_root=tmp_path,
            evidence_root=tmp_path / "evidence",
            runtime=(),
            view=None,
        )
    assert old_backend_calls == [] and not (tmp_path / "evidence").exists()


def test_real_shared_probe_reads_stdout_from_recorders_actual_directory(tmp_path):
    from merlin.common import invocation_record

    command = (sys.executable, "-I", "-B", "-c", "print('native readiness control')")
    probe = FreshToolProbe("linker", command, hashlib.sha256(b"native readiness control\n").hexdigest())
    O._probe_shared_tools(
        SimpleNamespace(output=tmp_path, readiness=(probe,), runtime=(), compiler_transport=None), ("/usr/bin/env",)
    )
    records = list(tmp_path.rglob("invocation.json"))
    assert len(records) == 1
    observed = invocation_record.verify(records[0])
    assert observed["stage"] == "fresh_phase1_linker" and observed["returncode"] == 0
    assert not (tmp_path / "readiness" / "linker" / "stdout.bin").exists()
