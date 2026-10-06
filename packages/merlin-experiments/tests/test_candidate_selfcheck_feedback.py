"""Candidate whole-model selfcheck feedback never borrows the host-graph result."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from merlin_experiments.phase1.context import InvocationContext
from merlin_experiments.phase1.feedback import qa, selfcheck

SENTINEL = "PRIVATE_OR_LEGACY_PROGRAM_SENTINEL"


def _candidate_result(*, passed=False):
    status = "pass" if passed else "incomplete"
    required = {tier: {"status": "pass" if passed or tier != "L3" else "unverified"} for tier in ("L0", "L1", "L3")}
    return {
        "capsule": "model_case",
        "kind": "model",
        "status": status,
        "numeric": {"status": "pass", "first_mismatch": {"expected": SENTINEL}},
        "tiers": {"L3": {"status": "pass", "cycles": 999}, "L2": {"status": "pass"}},
        "trace_check": {"status": "pass", "summary": SENTINEL},
        "failure": {"plane": "legacy_model", "category": "FUNCTIONAL_MISMATCH", "detail": SENTINEL},
        "candidate_native_execution": {
            "numeric": {"status": "pass", "mismatch_count": 0, "first_mismatch": {"expected": SENTINEL}}
        },
        "candidate_native_model_check": {
            "schema": "merlin_candidate_native_model_check_v1",
            "status": status,
            "violations": [] if passed else ["candidate_required_tiers_unverified"],
            "emitted_host_compute": {"status": "clean"},
            "source_placement": {"status": "clean"},
            "completed_dispatch": {"status": "verified" if passed else "unverified"},
            "candidate_required_tiers": {"status": status, "required_tiers": ["L0", "L1", "L3"], "tiers": required},
            "candidate_source_coverage": (
                {"status": "verified", "n_source_operations": 2, "n_eligible": 1, "n_completed_eligible": 1}
                if passed
                else {"status": "unverified"}
            ),
            "native_status": "numeric_match_diagnostic",
        },
    }


def _write_result(runs_root: Path, result: dict):
    parent = runs_root / "runs" / "fixture-suite" / "model_case"
    parent.mkdir(parents=True, exist_ok=True)
    (parent / "capsule_result.json").write_text(json.dumps(result))
    generated = parent / "generated"
    generated.mkdir()
    (generated / "instruction_trace.json").write_text(json.dumps({"summary": SENTINEL}))
    (generated / "l2_engine_binding.json").write_text(json.dumps({"private": SENTINEL}))
    artifacts = parent / "artifacts"
    artifacts.mkdir()
    (artifacts / "legacy_console.log").write_text(SENTINEL)


def _context(root: Path):
    return InvocationContext(
        root, root / "target.yaml", root, "fixture", root / "runs", root / "reports", root / "bundles", ()
    )


def test_candidate_row_uses_closed_qa_projection_and_never_screen_passes(tmp_path):
    result = _candidate_result()
    _write_result(tmp_path, result)
    closed = qa._per_capsule_from_results(tmp_path)["model_case"]
    row, certified = selfcheck._candidate_selfcheck_row(result, closed, name="model_case", barrier_tier="L3")
    assert not certified and not row["pass"]
    assert row["tiers"] == {"L0": "pass", "L1": "pass", "L3": "unverified"}
    assert row["numeric"] == {"status": "pass", "mismatch_count": 0}
    assert row["execution_digest"] is None
    assert row["failure"]["detail"] == "candidate_required_tiers_unverified"
    assert SENTINEL not in json.dumps(row)
    assert not any(
        key in row
        for key in (
            "trace_summary",
            "trace_check",
            "your_artifacts",
            "sim_console_tail",
            "barrier_cycles",
            "barrier_engine_binding",
        )
    )


def test_candidate_row_only_passes_complete_required_tiers(tmp_path):
    result = _candidate_result(passed=True)
    _write_result(tmp_path, result)
    closed = qa._per_capsule_from_results(tmp_path)["model_case"]
    row, certified = selfcheck._candidate_selfcheck_row(result, closed, name="model_case", barrier_tier="L3")
    assert certified and row["pass"]
    assert row["candidate_native_verification"]["tiers"] == {"L0": "pass", "L1": "pass", "L3": "pass"}
    assert SENTINEL not in json.dumps(row)
    missing, certified = selfcheck._candidate_selfcheck_row(result, None, name="model_case", barrier_tier="L3")
    assert not certified and not missing["pass"]


def test_main_selfcheck_does_not_count_legacy_screen_or_publish_legacy_artifacts(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    submission = tmp_path / "submission"
    submission.mkdir()
    (submission / "manifest.yaml").write_text("{}")
    corpus = tmp_path / "public"
    capsule = corpus / "model_case"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("{}")
    monkeypatch.setattr(selfcheck, "_adapters", lambda *a: ({"L3": object()}, "gsim"))
    monkeypatch.setattr(selfcheck, "_target_sim_via", lambda *a: ("fixture", "chipyard"))
    monkeypatch.setattr(selfcheck.CR, "suite_for", lambda *a: "fixture-suite")
    monkeypatch.setattr(selfcheck, "_log_telemetry", lambda *a: None)

    def grade(_submission, *, runs_root, **_kwargs):
        _write_result(Path(runs_root), _candidate_result())
        return {"n_capsules": 1, "n_passed": 0, "per_capsule": []}

    monkeypatch.setattr(selfcheck.CG, "grade", grade)
    code = selfcheck.main(
        ["--sim", "gsim", "--submission", str(submission)], context=_context(tmp_path), capsules_root=corpus
    )
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["n_passed"] == report["n_screened_only"] == report["n_certified"] == 0
    assert report["per_capsule"][0]["candidate_native_verification"]["status"] == "incomplete"
    assert SENTINEL not in json.dumps(report)


def test_model_layers_uses_candidate_projection_not_legacy_functional_pass(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    corpus = tmp_path / "layers"
    capsule = corpus / "model_case"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text("{}")
    experiment = SimpleNamespace(model_layer_roots=lambda: [corpus], sim_via="fixture")
    monkeypatch.setattr("merlin.targetgen.target_experiment.load_target_experiment", lambda *_: experiment)
    monkeypatch.setattr(selfcheck.CR, "oracle_adapters", lambda *a: {"L3": object()})
    monkeypatch.setattr(selfcheck.CR, "suite_for", lambda *a: "fixture-suite")

    def grade(_submission, *, runs_root, **_kwargs):
        _write_result(Path(runs_root), _candidate_result())
        return {"failure": {"detail": SENTINEL}}

    monkeypatch.setattr(selfcheck.CG, "grade", grade)
    code = selfcheck._model_layers(tmp_path / "submission", "", timeout=1, workers=1, context=_context(tmp_path))
    report = json.loads(capsys.readouterr().out)
    assert code == 1
    assert report["n_passed_functional_tier"] == 0
    assert report["grade_failure"] is None
    assert report["per_capsule"][0]["candidate_native_verification"]["status"] == "incomplete"
    assert SENTINEL not in json.dumps(report)
