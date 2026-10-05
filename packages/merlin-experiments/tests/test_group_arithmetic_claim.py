"""A fused member must be decided against the SUM of the parts it replaces, without target knowledge.

Every refusal below is paired with the mutation that triggers it, because the arithmetic in this
claim is otherwise easy to satisfy by accident: three copies of one program, a part that failed to
build, or parts assembled at a different size would all hand the whole a win it did not earn.
"""

from __future__ import annotations

import copy

from merlin_experiments.phase2.claims import dispatch
from merlin_experiments.phase2.claims import group_arithmetic as P

REPLICATES = ("r000", "r001")


def _acceptance() -> dict:
    return {
        "schema_version": 1,
        "analyzer": P.ANALYZER,
        "program_arm": "candidate",
        "group_field": "comparison_group",
        "whole_role": "fused",
        "part_role": "part",
        "parts_per_group": 2,
        "expected_lower": "whole",
        "demand_equal": [
            "performance.lever",
            "performance.emitter.resolved.operand_dtype",
            "performance.emitter.resolved.accum_dtype",
        ],
        "replicates": {"exact_count": 2, "identities": list(REPLICATES)},
        "band": {"kind": "measured_replicate_dispersion", "declared_constant": None},
        "evidence": {
            "correctness_simulator": "functional",
            "correctness_tier": "L2",
            "timing_simulator": "cycle_model",
            "timing_tier": "L3",
        },
    }


def _performance() -> dict:
    return {
        "family": "PF",
        "claim": "DIFFERENTIAL",
        "lever": "epilogue_fusion",
        "acceptance": _acceptance(),
        "emitter": {"resolved": {"operand_dtype": "int8", "accum_dtype": "i32"}},
    }


def _member(name: str, group: str, role: str, op: str, inputs: list[dict]) -> dict:
    return {
        "name": name,
        "inputs": inputs,
        "operation": {"op": op, "attributes": {}},
        "comparison_group": {"name": group, "role": role},
        "performance": _performance(),
    }


def _weight(k: int) -> dict:
    return {"name": "W", "role": "weight", "shape": [k, 16], "dtype": "i8"}


def _activation(k: int) -> dict:
    return {"name": "A0", "role": "input", "shape": [16, k], "dtype": "i8"}


def _bias() -> dict:
    return {"name": "B", "role": "bias", "shape": [16], "dtype": "i32"}


def _intermediate() -> dict:
    return {"name": "X", "role": "input", "shape": [16, 16], "dtype": "i32"}


def _descriptors() -> list[dict]:
    rows = []
    for k in (16, 32):
        group = f"fmb_k{k}"
        rows.append(
            _member(f"PF_fused_k{k}", group, "fused", "fused_matmul_bias", [_weight(k), _activation(k), _bias()])
        )
        rows.append(_member(f"PF_matmul_k{k}", group, "part", "matmul", [_weight(k), _activation(k)]))
        rows.append(_member(f"PF_bias_k{k}", group, "part", "bias_add", [_intermediate(), _bias()]))
    return rows


def _results(*, fused_cycles: int = 150, part_cycles: int = 100, jitter: int = 1) -> list[dict]:
    rows = []
    for k in (16, 32):
        costs = {
            f"PF_fused_k{k}": fused_cycles,
            f"PF_matmul_k{k}": part_cycles,
            f"PF_bias_k{k}": part_cycles,
        }
        for capsule, cycles in costs.items():
            for index, replicate in enumerate(REPLICATES):
                rows.append(
                    {
                        "capsule": capsule,
                        "replicate": replicate,
                        "cycles": cycles + index * jitter,
                        "simulator": "cycle_model",
                        "tier": "L3",
                        "correct": True,
                        "program_arm": "candidate",
                        "artifact_sha256": "a" * 64,
                    }
                )
    return rows


def _refusal(descriptors) -> str:
    preflight = P.preflight_group_arithmetic_claim(descriptors, replicates=list(REPLICATES))
    assert preflight["status"] == P.REFUSED, preflight["status"]
    return "; ".join(preflight["refusal_reasons"])


# ---- the happy path ----------------------------------------------------------------------------
def test_preflight_authors_one_cell_per_member_replicate_and_lane():
    preflight = P.preflight_group_arithmetic_claim(_descriptors(), replicates=list(REPLICATES))
    assert preflight["status"] == "READY"
    assert preflight["family"] == "PF"
    assert preflight["cohort"]["groups"] == ["fmb_k16", "fmb_k32"]
    assert preflight["cohort"]["parts_per_group"] == 2
    # 6 members x 2 replicates x 2 evidence lanes
    assert len(preflight["expected_identities"]) == 24
    assert {row["comparison_role"] for row in preflight["expected_identities"]} == {"fused", "part"}


def test_a_fused_member_cheaper_than_its_summed_parts_is_established():
    verdict = P.analyze_group_arithmetic_claim(_descriptors(), _results())
    assert verdict["verdict"] == P.ESTABLISHED, verdict
    row = next(r for r in verdict["rows"] if r["group"] == "fmb_k16")
    assert row["whole_cycles"] == 150
    assert row["parts_sum_cycles"] == 200
    assert row["delta_cycles"] == 50


# ---- the arithmetic must be able to fail --------------------------------------------------------
def test_a_fused_member_no_cheaper_than_its_parts_is_refuted():
    verdict = P.analyze_group_arithmetic_claim(_descriptors(), _results(fused_cycles=250))
    assert verdict["verdict"] == P.REFUTED
    assert sorted(verdict["groups"]) == ["fmb_k16", "fmb_k32"]


def test_a_win_inside_the_summed_replicate_band_is_refuted():
    """The band of a SUM is the sum of the bands, not the largest of them.

    With a 20-cycle dispersion on each of three members the band is 60; a 40-cycle win is noise.
    """
    verdict = P.analyze_group_arithmetic_claim(_descriptors(), _results(fused_cycles=160, jitter=20))
    row = next(r for r in verdict["rows"] if r["group"] == "fmb_k16")
    assert row["replicate_band"] == 60
    assert row["delta_cycles"] == 40
    assert verdict["verdict"] == P.REFUTED


# ---- the structural guards ----------------------------------------------------------------------
def test_a_group_of_identical_operations_is_refused_not_scored():
    """One whole against copies of itself would 'establish' the claim by counting 1 against 2."""
    descriptors = copy.deepcopy(_descriptors())
    for row in descriptors:
        row["operation"]["op"] = "matmul"
    assert "repeats an operation" in _refusal(descriptors)


def test_a_missing_part_is_refused_rather_than_shrinking_the_sum():
    descriptors = [row for row in _descriptors() if row["name"] != "PF_bias_k16"]
    assert "demands exactly 2" in _refusal(descriptors)


def test_a_group_with_no_whole_is_refused():
    descriptors = [row for row in _descriptors() if row["name"] != "PF_fused_k16"]
    assert "has no 'fused' member" in _refusal(descriptors)


def test_two_wholes_in_one_group_are_refused():
    descriptors = copy.deepcopy(_descriptors())
    descriptors[1]["comparison_group"]["role"] = "fused"
    assert "more than one 'fused' member" in _refusal(descriptors)


def test_a_part_sharing_no_operand_with_the_whole_is_refused():
    descriptors = copy.deepcopy(_descriptors())
    for row in descriptors:
        if row["name"] == "PF_bias_k16":
            row["inputs"] = [{"name": "Q", "role": "input", "shape": [8], "dtype": "i8"}]
    assert "shares no declared operand" in _refusal(descriptors)


def test_a_part_built_at_another_size_is_refused():
    """The shared operand is what holds the parts to the whole's problem."""
    descriptors = copy.deepcopy(_descriptors())
    for row in descriptors:
        if row["name"] == "PF_matmul_k16":
            row["inputs"] = [
                {"name": "W", "role": "weight", "shape": [64, 16], "dtype": "i8"},
                {"name": "A0", "role": "input", "shape": [16, 64], "dtype": "i8"},
            ]
    assert "declares operand" in _refusal(descriptors)


def test_a_demanded_path_that_is_absent_is_refused_never_assumed_equal():
    descriptors = copy.deepcopy(_descriptors())
    for row in descriptors:
        if row["name"] == "PF_bias_k16":
            row["performance"]["emitter"]["resolved"].pop("accum_dtype")
    reason = _refusal(descriptors)
    assert "do not declare demanded path" in reason
    assert "accum_dtype" in reason


def test_members_disagreeing_on_a_demanded_path_are_refused():
    descriptors = copy.deepcopy(_descriptors())
    for row in descriptors:
        if row["name"] == "PF_matmul_k16":
            row["performance"]["emitter"]["resolved"]["operand_dtype"] = "int4"
    assert "disagree on demanded path" in _refusal(descriptors)


def test_an_absent_demanded_path_is_distinct_from_one_declared_null():
    """`None` is a legitimate declared value here, so absence must not collapse into it."""
    descriptors = copy.deepcopy(_descriptors())
    for row in descriptors:
        row["performance"]["emitter"]["resolved"]["accum_dtype"] = None
    preflight = P.preflight_group_arithmetic_claim(descriptors, replicates=list(REPLICATES))
    assert preflight["status"] == "READY"


# ---- the evidence guards -------------------------------------------------------------------------
def test_a_row_without_a_passing_grade_is_not_timing_evidence():
    rows = _results()
    rows[0] = {**rows[0], "correct": False}
    verdict = P.analyze_group_arithmetic_claim(_descriptors(), rows)
    assert verdict["verdict"] == P.REFUSED
    assert "passing correctness" in verdict["reason"]


def test_an_incomplete_timing_cohort_is_refused():
    rows = [row for row in _results() if row["capsule"] != "PF_bias_k16"]
    verdict = P.analyze_group_arithmetic_claim(_descriptors(), rows)
    assert verdict["verdict"] == P.REFUSED
    assert "incomplete" in verdict["reason"]


def test_results_from_two_program_artifacts_are_refused():
    rows = _results()
    rows[0] = {**rows[0], "artifact_sha256": "b" * 64}
    verdict = P.analyze_group_arithmetic_claim(_descriptors(), rows)
    assert verdict["verdict"] == P.REFUSED
    assert "artifact identity" in verdict["reason"]


def test_a_contract_naming_another_analyzer_is_refused():
    descriptors = copy.deepcopy(_descriptors())
    for row in descriptors:
        row["performance"]["acceptance"]["analyzer"] = "merlin.perf.other.analyze/v1"
    assert "not " + repr(P.ANALYZER) in _refusal(descriptors)


def test_a_constant_band_is_refused():
    descriptors = copy.deepcopy(_descriptors())
    for row in descriptors:
        row["performance"]["acceptance"]["band"]["declared_constant"] = 10
    assert "measured replicate-dispersion band" in _refusal(descriptors)


def test_the_run_must_offer_the_frozen_replicate_schedule():
    preflight = P.preflight_group_arithmetic_claim(_descriptors(), replicates=["r000"])
    assert preflight["status"] == P.REFUSED
    assert "the run offers replicates" in "; ".join(preflight["refusal_reasons"])


def test_dispatch_routes_the_frozen_identity_to_this_owner():
    resolved = dispatch.resolve(_descriptors())
    assert resolved.module is P
    assert resolved.analyze is P.analyze_group_arithmetic_claim
    verdict = dispatch.analyze(_descriptors(), _results())
    assert verdict["verdict"] == P.ESTABLISHED and verdict["declared_analyzer"] == P.ANALYZER


def test_the_shared_template_declares_the_pf_contract_this_analyzer_decides():
    import yaml

    from merlin.common.paths import repo_root

    template = yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_text())
    pf = next(s for s in template["sweeps"] if s["id"] == "PF")
    acceptance = pf["base"]["performance"]["acceptance"]
    assert acceptance["analyzer"] == P.ANALYZER
    assert (acceptance["whole_role"], acceptance["part_role"]) == ("fused", "part")
    roles = {v["comparison_group"]["role"] for v in pf["variants"]}
    assert roles == {"fused", "part"} and acceptance["parts_per_group"] == len(pf["variants"]) - 1
