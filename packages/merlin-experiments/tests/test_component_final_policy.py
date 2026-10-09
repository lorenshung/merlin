"""Strict parity boundaries and independent convergence accounting."""

from dataclasses import replace

import pytest
from merlin_experiments.phase2.component_experiment import FinalMemberComparison
from merlin_experiments.phase2.component_final_policy import strict_final_component_campaign_gate as gate
from merlin_experiments.phase2.contracts import StageGateError


def rows():
    return tuple(
        FinalMemberComparison(name, 1000, 1000, "1" * 64, "2" * 64, "3" * 64, True, True, True)
        for name in ("case-a", "case-b", "case-c")
    )


def evaluate(values=None, **kwargs):
    return gate(rows() if values is None else values, expected_members=("case-a", "case-b", "case-c"), **kwargs)


def test_exact_parity_passes_without_historical_time():
    result = evaluate()
    assert result["status"] == "pass"
    assert result["parity_allowance"] == 0
    assert result["convergence"]["status"] == "unknown"
    assert result["convergence"]["gates_performance"] is False


def test_one_cycle_regression_cannot_hide_behind_another_member_gain():
    values = list(rows())
    values[0] = replace(values[0], candidate_cycles=1001)
    values[1] = replace(values[1], candidate_cycles=1)
    result = evaluate(tuple(reversed(values)))
    assert result["status"] == "fail"
    assert [row["member"] for row in result["members"]] == ["case-a", "case-b", "case-c"]
    assert result["members"][0]["delta_cycles"] == 1


def test_integer_parity_does_not_round_large_counters():
    values = list(rows())
    values[0] = replace(values[0], reference_cycles=2**61, candidate_cycles=2**61 + 1)
    assert evaluate(tuple(values))["status"] == "fail"


def test_time_is_reported_without_a_speedup_requirement():
    result = evaluate(phase12_wall_s=100, handwritten_wall_s=10)
    assert result["status"] == "pass"
    assert result["convergence"]["speedup"] == 0.1


@pytest.mark.parametrize("field", ["hardware_verified", "reference_cycles", "candidate_cycles"])
def test_missing_hardware_authority_remains_unknown(field):
    values = list(rows())
    values[0] = replace(values[0], **{field: False if field == "hardware_verified" else None})
    assert evaluate(tuple(values))["status"] == "unknown"


@pytest.mark.parametrize("field", ["accuracy_passed", "final_executable_passed"])
def test_failed_scientific_gate_takes_precedence_over_unknown(field):
    values = list(rows())
    values[0] = replace(values[0], **{field: False, "hardware_verified": False})
    assert evaluate(tuple(values))["status"] == "fail"


@pytest.mark.parametrize("field,value", [("candidate_cycles", True), ("reference_cycles", 0), ("accuracy_passed", 1)])
def test_numeric_evidence_is_not_coerced(field, value):
    values = list(rows())
    values[0] = replace(values[0], **{field: value})
    with pytest.raises(StageGateError):
        evaluate(tuple(values))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), True, 0, -1])
def test_invalid_elapsed_time_refuses_even_if_comparison_is_missing(value):
    with pytest.raises(StageGateError):
        evaluate(phase12_wall_s=value)


def test_missing_and_duplicate_members_are_refused():
    with pytest.raises(StageGateError, match="membership"):
        evaluate(rows()[:2])
    with pytest.raises(StageGateError, match="membership"):
        evaluate((rows()[0], rows()[0], rows()[2]))
