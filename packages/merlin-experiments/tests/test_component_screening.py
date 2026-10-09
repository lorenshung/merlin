"""Public ordering replays the exact complete-cost roster, without authority."""

from copy import deepcopy

import pytest
from merlin_experiments.phase2 import component_screening as S
from merlin_experiments.phase2 import component_workflow as W
from merlin_experiments.phase2.contracts import StageGateError, document_sha256

from merlin.perf.component_cost import COMPLETE_STAGES
from merlin.perf.component_screen import ComponentOpportunity, rank_component_opportunities
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


def document():
    def report(interval):
        regime = {"total": interval.to_dict(), "regions": [{
            "id": "whole", "stages": list(COMPLETE_STAGES), "accounting": "inclusive",
            "parent": None, "contained": False, "cycles": interval.to_dict(),
        }]}
        return {"schema": "component_complete_cost_v1", "scope_sha256": "a" * 64,
                "domain_sha256": "b" * 64, "regimes": {"cold": regime, "warm": regime},
                "promotion": "SCREENING_ONLY"}

    members, proposals = [], []
    for family, capsule, cycles, legal in [("f1", "a", 1, True), ("f1", "b", 3, True), ("f2", "c", 6, None)]:
        baseline = CycleInterval.point(10, "synthetic complete cost")
        candidate = (
            CycleInterval.point(cycles, "synthetic complete cost") if legal else CycleInterval.unknown("legality")
        )
        members.append({"family": family, "capsule": capsule, "baseline": baseline.to_dict(),
                        "candidate": candidate.to_dict(), "complete_cost": {
                            "baseline": report(baseline), "candidate": report(candidate), "objective": "warm",
                        }})
        proposals.append(ComponentOpportunity(document_sha256([family, capsule]), family, "warm",
                                              baseline, candidate, 1.0, legal, True))
    screening = rank_component_opportunities(proposals, work_shares={
        proposals[0].id: 0.25, proposals[1].id: 0.25, proposals[2].id: 0.5,
    })
    return {"schema": W.SCHEMA, "workflow_id": "component-only-v1", "tier": "calibrated_component_analytical",
            "candidate_sha256": "a" * 64, "corpus_sha256": "b" * 64, "manifest_sha256": "c" * 64,
            "provider_sha256": "d" * 64, "configuration_sha256": "e" * 64,
            "evidence": members, "screening": S.project_screening(screening, members),
            "promotion": "NO_FINAL_ACCEPTANCE"}


def test_same_complete_cost_order_exposes_unknowns_and_preserves_legacy_feedback():
    original = document()
    assert W.validate_component_feedback(original) == original
    assert len(original["screening"]["order"]) == 2
    assert original["screening"]["evidence"][2]["status"] == "UNRESOLVED"
    assert original["screening"]["evidence"][2]["priority"] is None
    legacy = deepcopy(original)
    legacy.pop("screening")
    assert W.validate_component_feedback(legacy) == legacy


@pytest.mark.parametrize(
    "defect", ["saving", "priority", "order", "gate", "regime", "share", "basis", "roster", "time", "capsule"],
)
def test_tampered_ordering_cannot_override_original_costs_gates_roster_or_independent_weights(defect):
    changed = document()
    screening = changed["screening"]
    row = screening["evidence"][0]
    if defect == "saving":
        row["conservative_saved_cycles"] = 100
    elif defect == "priority":
        row["priority"] *= 2
    elif defect == "order":
        screening["order"].reverse()
    elif defect == "gate":
        row["legal"] = False
    elif defect == "regime":
        row["regime"] = "cold"
    elif defect == "share":
        screening["work_shares_sha256"] = "f" * 64
    elif defect == "basis":
        screening["work_share_basis"] = "protected model frequencies"
    elif defect == "roster":
        screening["evidence"].pop()
    elif defect == "capsule":
        row["capsule"] = "another-source"
    else:
        row["evaluation_seconds"] = True
    with pytest.raises(StageGateError):
        W.validate_component_feedback(changed)


def test_intervals_without_complete_stages_cannot_support_an_experiment_order():
    changed = document()
    changed["evidence"][0].pop("complete_cost")
    with pytest.raises(StageGateError, match="same complete-cost regime"):
        W.validate_component_feedback(changed)


def test_screening_cannot_be_attached_to_another_feedback_tier():
    changed = document()
    changed["tier"] = "certified_component_rtl"
    with pytest.raises(StageGateError, match="complete analytical cost"):
        W.validate_component_feedback(changed)
