"""A form-perf member is decided against its vendor bar; an unscanned candidate is never scored."""

from __future__ import annotations

import copy

import yaml
from merlin_experiments.phase2.claims import dispatch
from merlin_experiments.phase2.claims import vendor_reference as VR

from merlin.common.paths import repo_root

REPLICATES = ("r000", "r001")


def _performance() -> dict:
    template = yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_text())
    performance = copy.deepcopy(next(s for s in template["sweeps"] if s["id"] == "PW")["base"]["performance"])
    performance["acceptance"]["evidence"].update(correctness_simulator="functional", timing_simulator="cycle_model")
    return performance


def _descriptors() -> list[dict]:
    return [{"name": name, "performance": _performance()} for name in ("PW00_stem", "PW01_pool")]


def _results(ours=(100, 102), bar=(150, 151), clean=True) -> list[dict]:
    rows = []
    for capsule in ("PW00_stem", "PW01_pool"):
        for arm, values, digest in (("candidate", ours, "c" * 64), ("vendor_reference", bar, "v" * 64)):
            for replicate, cycles in zip(REPLICATES, values):
                row = {
                    "capsule": capsule,
                    "program_arm": arm,
                    "replicate": replicate,
                    "cycles": cycles,
                    "correct": True,
                    "artifact_sha256": digest,
                    "simulator": "cycle_model",
                    "tier": "L3",
                }
                if arm == "candidate":
                    row["instruction_policy"] = {"status": "clean" if clean else "violations", "violations": []}
                rows.append(row)
    return rows


def test_preflight_authors_both_arms_for_every_member():
    preflight = VR.preflight_vendor_reference_claim(_descriptors(), replicates=list(REPLICATES))
    assert preflight["status"] == "READY"
    arms = {row["program_arm"] for row in preflight["expected_identities"]}
    assert arms == {"candidate", "vendor_reference"} and len(preflight["expected_identities"]) == 2 * 2 * 2 * 2


def test_a_candidate_beating_the_bar_is_established_with_its_ratio():
    verdict = dispatch.analyze(_descriptors(), _results())
    assert verdict["verdict"] == VR.ESTABLISHED and verdict["declared_analyzer"] == VR.ANALYZER
    assert verdict["rows"][0]["ours_over_vendor"] == 100 / 150


def test_a_win_inside_the_band_or_above_the_bar_is_refuted():
    assert VR.analyze_vendor_reference_claim(_descriptors(), _results(ours=(149, 160)))["verdict"] == VR.REFUTED
    assert VR.analyze_vendor_reference_claim(_descriptors(), _results(ours=(200, 201)))["verdict"] == VR.REFUTED


def test_an_unscanned_or_dirty_candidate_is_refused_not_scored():
    rows = _results(clean=False)
    assert VR.analyze_vendor_reference_claim(_descriptors(), rows)["verdict"] == VR.REFUSED
    for row in rows:
        row.pop("instruction_policy", None)
    assert "instruction-policy" in VR.analyze_vendor_reference_claim(_descriptors(), rows)["reason"]


def test_an_unrestricted_candidate_arm_is_refused():
    descriptors = _descriptors()
    for row in descriptors:
        row["performance"]["arms"]["candidate"]["instruction_policy"] = "unrestricted"
    assert VR.preflight_vendor_reference_claim(descriptors, replicates=list(REPLICATES))["status"] == VR.REFUSED
