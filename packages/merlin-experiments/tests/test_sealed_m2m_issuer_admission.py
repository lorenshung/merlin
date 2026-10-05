"""The sealed Model2MLIR runner is the one admitted capture-execution issuer, re-verified from disk.

Admitting it is an operator policy decision: the attestation it issues names the
preselection, the sealed receipt, the model and the materialized receipt by digest,
and every admission re-reads those bytes. Everything else stays refused.
"""

from __future__ import annotations

import copy
import json
import shutil
from pathlib import Path

import pytest
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_static import _file_digest
from merlin_experiments.phase0 import capture_execution_attestation as attestation
from merlin_experiments.phase0 import capture_selection as selected
from merlin_experiments.phase0 import coverage_commitment as CC


def _sealed_run(tmp_path, monkeypatch):
    library = tmp_path / "libfixture.so"
    library.write_bytes(b"selected system library")
    bwrap = tmp_path / "bwrap"
    bwrap.write_bytes(b"selected bubblewrap")
    roots = {name: tmp_path / name for name in ("m2m", "workload", "merlin", "schemas", "venv", "base")}
    for path in roots.values():
        path.mkdir()
    worker = roots["merlin"] / "targetgen/worker.py"
    worker.parent.mkdir()
    worker.write_bytes(b"selected worker")
    plan = {
        "schema": sealed_m2m.SCHEMA,
        "status": "plan_only",
        "m2m_root": str(roots["m2m"]),
        "workload_root": str(roots["workload"]),
        "merlin_root": str(roots["merlin"]),
        "schemas_root": str(roots["schemas"]),
        "venv": str(roots["venv"]),
        "base": str(roots["base"]),
        "worker": str(worker),
        "dtype": "fp32",
        "recipe": None,
        "system_libs": [str(library)],
        "max_snapshot_bytes": 15_000_000_000,
    }
    monkeypatch.setattr(sealed_m2m, "prepare_plan", lambda **_: dict(plan))
    monkeypatch.setattr(selected, "_bwrap_binary", lambda *_: bwrap)
    run = tmp_path / "fresh-run"
    identity = selected.select(
        m2m_root=roots["m2m"],
        workload_root=roots["workload"],
        worker=worker,
        venv=roots["venv"],
        schemas_root=roots["schemas"],
        run_dir=run,
        output_dir=tmp_path / "selection",
    )
    manifest = selected.load(Path(identity["path"]), expected_sha256=identity["sha256"])
    capture = run / "capture"
    capture.mkdir(parents=True)
    run.chmod(0o700)
    (capture / "model.mlir").write_bytes(b"module {}\n")
    (capture / "capture_receipt.json").write_bytes(b"{}\n")
    guest_library = run / "snapshots/guest-root" / library.relative_to("/")
    guest_library.parent.mkdir(parents=True)
    shutil.copy2(library, guest_library)
    pending = run / "sealed_m2m_pending.json"
    pending.write_text(
        json.dumps(
            {
                "schema": sealed_m2m.SCHEMA,
                "status": "pending_replay",
                "capture_selection_sha256": identity["sha256"],
                "plan": plan,
                "policy_sha256": manifest["sandbox_policy_sha256"],
                "bwrap_sha256": manifest["bwrap"]["sha256"],
                "issuer_sha256": manifest["issuer_source_sha256"],
            }
        )
    )
    monkeypatch.setattr(
        sealed_m2m,
        "replay_verify",
        lambda *_args, **_kwargs: {
            "status": "verified_sandbox_replay",
            "sealed_source_closure_replayed": True,
            "receipt_sha256": _file_digest(pending),
        },
    )
    monkeypatch.setattr(attestation, "verify_capture_receipt", lambda _model: {"status": "verified_materialized"})
    selection_path = Path(identity["path"])
    replay = selected.verify(selection_path, expected_sha256=identity["sha256"], model_path=capture / "model.mlir")
    return replay, selection_path, capture / "model.mlir"


def test_preselected_sealed_replay_is_admitted_and_names_its_policy(tmp_path, monkeypatch):
    replay, selection_path, model = _sealed_run(tmp_path, monkeypatch)
    document = attestation.attest_sealed_m2m(replay, selection_path=selection_path, model_path=model)
    assert document["issuer"] == attestation.SEALED_M2M_ISSUER == sealed_m2m.SCHEMA
    assert document["status"] == "verified_sealed_execution"
    assert document["policy"]["decision"].startswith("operator policy decision")
    assert document["policy"]["accepted_residuals"]
    assert document["capture"]["model_sha256"] == _file_digest(model) == replay["model_sha256"]
    attestation.require_verified_execution(document)
    # The replay record itself remains nonadmissible: only the issued attestation passes.
    with pytest.raises(attestation.AttestationNotVerified):
        attestation.require_verified_execution(replay)


def test_admission_rereads_the_bytes_and_refuses_edits_or_other_issuers(tmp_path, monkeypatch):
    replay, selection_path, model = _sealed_run(tmp_path, monkeypatch)
    document = attestation.attest_sealed_m2m(replay, selection_path=selection_path, model_path=model)
    for mutate, match in (
        (lambda d: d.update(issuer="merlin.sealed-static-capture.v1"), "no supported"),
        (lambda d: d.update(issuer="self-reported"), "no supported"),
        (lambda d: d["policy"].update(accepted_residuals=[]), "lacks its selection"),
        (lambda d: d["replay"].update(status="verified_sandbox_replay"), "lacks its selection"),
        (lambda d: d["capture"].update(model_sha256="0" * 64), "model_sha256 differs"),
        (lambda d: d["selection"].update(sha256="0" * 64), "unreadable"),
        (lambda d: d.update(fresh_execution=False), "no verified fresh"),
    ):
        forged = copy.deepcopy(document)
        mutate(forged)
        with pytest.raises(attestation.AttestationNotVerified, match=match):
            attestation.require_verified_execution(forged)
    model.write_bytes(b"module { changed }\n")
    with pytest.raises(attestation.AttestationNotVerified, match="model_sha256 differs"):
        attestation.require_verified_execution(document)


def test_only_a_preselected_replay_record_can_be_attested(tmp_path, monkeypatch):
    replay, selection_path, model = _sealed_run(tmp_path, monkeypatch)
    for forged in (
        {**replay, "schema": "merlin.phase0.sealed_m2m_assessment.v2"},
        {**replay, "status": "replay_verified_nonadmissible"},
        {**replay, "model_sha256": "0" * 64},
    ):
        with pytest.raises(attestation.AttestationNotVerified):
            attestation.attest_sealed_m2m(forged, selection_path=selection_path, model_path=model)
    pending = model.parent.parent / "sealed_m2m_pending.json"
    receipt = json.loads(pending.read_bytes())
    receipt["capture_selection_sha256"] = "0" * 64
    pending.write_text(json.dumps(receipt))
    with pytest.raises(attestation.AttestationNotVerified, match="does not bind"):
        attestation.attest_sealed_m2m(replay, selection_path=selection_path, model_path=model)


def test_coverage_admits_closure_only_for_the_bound_application(tmp_path, monkeypatch):
    replay, selection_path, model = _sealed_run(tmp_path, monkeypatch)
    document = attestation.attest_sealed_m2m(replay, selection_path=selection_path, model_path=model)
    application = {
        "capture_sha256": document["capture"]["model_sha256"],
        "capture_receipt": {"status": "verified_materialized", "receipt_sha256": document["capture"]["receipt_sha256"]},
        "capture_execution_attestation": document,
    }
    assert CC.attestation_failure(application) is None
    other = {**application, "capture_sha256": "f" * 64}
    assert "does not bind" in CC.attestation_failure(other)
    assert "unsupported" in CC.attestation_failure({**application, "capture_execution_attestation": None})


def test_evidence_counts_an_attested_capture_as_verified_only_while_its_bytes_hold(tmp_path, monkeypatch):
    from merlin_experiments.phase0 import evidence_status as ES

    replay, selection_path, model = _sealed_run(tmp_path, monkeypatch)
    document = attestation.attest_sealed_m2m(replay, selection_path=selection_path, model_path=model)
    applications = {
        "app": {
            "capture_sha256": document["capture"]["model_sha256"],
            "capture_receipt": {"receipt_sha256": document["capture"]["receipt_sha256"]},
        }
    }
    assert ES.capture_attestation_diagnostics(applications, {"app": document}) == []
    assert ES.status(ES.capture_attestation_diagnostics(applications, {"app": document})) == "verified"
    model.write_bytes(b"module { changed }\n")
    assert ES.capture_attestation_diagnostics(applications, {"app": document})[0]["component"] == "capture-execution"
