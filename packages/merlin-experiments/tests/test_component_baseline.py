"""Fresh baseline authority is distinct from bytes, reports and diagnostics."""
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_baseline as CB
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from test_component_workflow import context


def test_serialized_or_unissued_admission_never_grants_comparison():
    values = {row.name: None for row in fields(CB.ComponentBaselineAdmission)}
    with pytest.raises(StageGateError, match="issued fresh"):
        CB.ComponentBaselineAdmission(**values).verify()
    with pytest.raises(StageGateError, match="fresh Phase 1"):
        CB.verify_baseline_admission({"status": "qualified"}, baseline=None,
                                     corpus=None, target_descriptor=None)


def test_hashed_reference_compiler_and_report_cannot_issue_fresh_baseline(tmp_path):
    inputs = context(tmp_path)
    with pytest.raises(StageGateError, match="evaluated fresh"):
        CB.admit_component_baseline(
            qualification=SimpleNamespace(status="qualified"), baseline=inputs["candidate"],
            corpus=inputs["component_corpus"], target_descriptor=inputs["target_experiment"].path,
        )


def test_unissued_diagnostic_controller_fixture_is_rejected(tmp_path):
    inputs = context(tmp_path)
    with pytest.raises(StageGateError, match="issued fresh"):
        inputs["baseline_admission"].verify()


def test_policy_refuses_comparison_without_fresh_origin(tmp_path):
    from merlin_experiments.phase2 import broker_policy as BP
    from merlin_experiments.phase2 import component_cca as CC

    inputs = context(tmp_path)
    owner = Path(__file__)
    baseline = inputs["baseline_admission"].baseline
    provider = CC.ComponentCCAProvider(lambda **_: {}, owner, sha256_file(owner), baseline,
                                      inputs["baseline_admission"].baseline_sha256)
    with pytest.raises(StageGateError, match="issued fresh"):
        BP.select_workflow(BP.COMPONENT_ONLY_V1, **inputs, component_cca=provider)


def test_independent_cohort_cannot_substitute_withheld_member(tmp_path, monkeypatch):
    inputs = context(tmp_path)
    qualification = SimpleNamespace(corpus_root=tmp_path / "private")
    from merlin_experiments.phase0 import component_coverage

    monkeypatch.setattr(component_coverage, "verify_report", lambda _: {"obligations": [{
        "cohort": "withheld_transfer", "members": [{"name": "member", "state": "generated", "sha256": "a" * 64}],
    }]})
    with pytest.raises(StageGateError, match="independently generated development"):
        CB._development_members(qualification, inputs["component_corpus"])
