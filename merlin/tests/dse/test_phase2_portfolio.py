from __future__ import annotations

import hashlib

import pytest

from merlin.perf.phase2_portfolio import (
    AnalyticalMetrics,
    FastEvaluationPolicy,
    FourModelQualitySchema,
    PortfolioQualitySchema,
    QualityBudget,
    QualityLimit,
    QualityObservation,
    evaluate_fast_portfolio,
    standard_four_model_quality_schema,
    unavailable_fast_evaluation,
)

MODELS = tuple(hashlib.sha256(f"model-{index}".encode()).hexdigest() for index in range(4))


def _quality_budget() -> QualityBudget:
    return QualityBudget(
        limits=(QualityLimit("relative_error", "at_most", 0.10, maximum_degradation=0.02),),
        reference="independent complete-output reference",
    )


def _metrics(
    cycles: float,
    *,
    movement: float,
    utilization: float,
    overlap: float,
    encoding_count: int,
    encoding_bytes: float,
    encoding_cycles: float,
    risk: float = 0.05,
):
    compute_busy = utilization * cycles
    movement_busy = 0.5 * cycles
    faster = cycles < 100
    return {
        "cycles": {
            "lo": cycles,
            "hi": cycles,
            "provenance": ["calibrated analytical whole-plan model"],
        },
        "movement_bytes": movement,
        "movement_scope": "physical",
        "occupancy": {
            "total_cycles": cycles,
            "busy_cycles": {"compute-engine": compute_busy, "movement-engine": movement_busy},
            "compute_resources": ["compute-engine"],
            "movement_resources": ["movement-engine"],
            "movement_elapsed_cycles": movement_busy,
            "overlap_cycles": overlap * movement_busy,
            "overlap_available_cycles": movement_busy,
            "idle_cycles": max(0.0, cycles - max(compute_busy, movement_busy)),
            "movement_bytes": movement,
            "encoding_transitions": encoding_count,
            "provenance": ["explicit target-adapter activity schedule"],
        },
        "coverage": {
            "supported_work_total": 1000,
            "supported_work_placed": 800 if faster else 700,
            "largest_connected_region_work": 500 if faster else 400,
            "connected_region_work": [500, 300] if faster else [400, 300],
            "host_islands": [
                {"taxonomy": "unsupported-compute", "count": 2 if faster else 3, "work": 50},
            ],
            "boundary_crossings": 8 if faster else 10,
            "boundary_bytes": 80 if faster else 100,
            "work_unit": "exact arithmetic operations",
            "provenance": ["verified source ownership and global-plan connectivity"],
        },
        "roofline": {
            "lower_bound_cycles": cycles * 0.5,
            "resource_floors": {"compute": cycles * 0.5, "movement": cycles * 0.4},
            "limiting_resources": ["compute"],
            "optimization_effects": ["movement", "tiling"],
            "composition": "adapter-declared max for independently overlappable resources",
            "provenance": ["target descriptor and calibrated resource rates"],
        },
        "encoding_conversions": {
            "count": encoding_count,
            "bytes": encoding_bytes,
            "cycles": {
                "lo": encoding_cycles,
                "hi": encoding_cycles,
                "provenance": ["warm reduced conversion witness"],
            },
        },
        "risk_score": risk,
        "provenance": ["host-owned analytical adapter"],
    }


def _row(
    model: str,
    *,
    candidate_cycles: float = 80,
    candidate_movement: float = 80,
    candidate_error: float = 0.06,
):
    return {
        "model_id": model,
        "baseline": _metrics(
            100,
            movement=100,
            utilization=0.50,
            overlap=0.25,
            encoding_count=2,
            encoding_bytes=20,
            encoding_cycles=8,
        ),
        "candidate": _metrics(
            candidate_cycles,
            movement=candidate_movement,
            utilization=0.60,
            overlap=0.50,
            encoding_count=1,
            encoding_bytes=10,
            encoding_cycles=4,
        ),
        "baseline_quality": {
            "values": {"relative_error": 0.05},
            "complete": True,
            "provenance": ["baseline complete-output oracle"],
        },
        "candidate_quality": {
            "values": {"relative_error": candidate_error},
            "complete": True,
            "provenance": ["candidate complete-output oracle"],
        },
    }


def _evaluate(rows, *, surfaces=None, policy=None):
    return evaluate_fast_portfolio(
        rows,
        quality_budgets={model: _quality_budget() for model in MODELS},
        policy=policy or FastEvaluationPolicy(),
        authorized_surfaces=surfaces,
        expected_models=MODELS,
    )


def test_standard_quality_schema_binds_required_thresholds_to_four_corpora():
    corpora = {model: hashlib.sha256(f"corpus-{index}".encode()).hexdigest() for index, model in enumerate(MODELS)}

    schema = standard_four_model_quality_schema(
        MODELS,
        classification_member_sha256=MODELS[0],
        corpus_sha256_by_member=corpora,
    )

    assert schema.mode == "accuracy_bounded"
    classification = schema.budget_map[MODELS[0]]
    assert classification.profile == "classification_top1"
    assert classification.limits[0].to_dict() == {
        "metric": "top1_degradation_percentage_points",
        "direction": "at_most",
        "threshold": 0.7,
        "maximum_degradation": None,
    }
    for model in MODELS[1:]:
        limits = {limit.metric: limit for limit in schema.budget_map[model].limits}
        assert limits["cosine_similarity"].threshold == 0.99
        assert limits["cosine_similarity"].direction == "at_least"
        assert limits["normalized_root_mean_square_error"].threshold == 0.02
        assert limits["normalized_root_mean_square_error"].direction == "at_most"


def test_missing_quality_corpus_falls_back_to_exact_only():
    schema = standard_four_model_quality_schema(
        MODELS,
        classification_member_sha256=MODELS[0],
        corpus_sha256_by_member={MODELS[0]: hashlib.sha256(b"only-one").hexdigest()},
    )
    report = unavailable_fast_evaluation(reason=schema.reason)

    assert schema.mode == "exact_only" and schema.budgets == ()
    assert report["status"] == "exact_only_fallback"
    assert report["approximation_allowed"] is False
    assert report["maximum_parallel_model_evaluations"] == 1


def test_named_quality_profiles_cannot_be_relaxed_while_keeping_their_label():
    with pytest.raises(ValueError, match="0.7 point"):
        QualityBudget(
            (QualityLimit("top1_degradation_percentage_points", "at_most", 7.0),),
            "forged relaxed corpus policy",
            "classification_top1",
        )


def test_four_model_portfolio_retains_only_a_quality_safe_pareto_win():
    report = _evaluate([_row(model) for model in MODELS])

    assert report["status"] == "retain"
    assert report["models_evaluated"] == 4
    assert report["portfolio_conservative_cycle_speedup_geomean"] == pytest.approx(1.25)
    assert all(row["quality_gate"]["status"] == "passed" for row in report["models"])
    assert all(row["status"] == "pareto_admissible" for row in report["models"])
    assert report["models"][0]["candidate"]["roofline"]["headroom_to_lower_bound"] == 2.0
    assert report["maximum_parallel_model_evaluations"] == 1
    assert "never summed" in report["aggregation"]


def test_non_content_addressed_or_incomplete_portfolio_is_refused():
    rows = [_row(model) for model in MODELS]
    with pytest.raises(ValueError, match="nonempty portfolio"):
        evaluate_fast_portfolio(
            rows,
            quality_budgets={model: _quality_budget() for model in MODELS},
            policy=FastEvaluationPolicy(),
            expected_models=(),
        )
    with pytest.raises(ValueError, match="exactly cover"):
        _evaluate(rows[:3])


def test_quality_budget_rejects_a_faster_candidate():
    rows = [_row(model) for model in MODELS]
    rows[2] = _row(MODELS[2], candidate_cycles=50, candidate_error=0.12)

    report = _evaluate(rows)

    assert report["status"] == "reject"
    assert report["models"][2]["quality_gate"]["status"] == "failed"


@pytest.mark.parametrize("count", [1, 3, 5])
def test_any_nonempty_portfolio_keeps_per_member_quality_and_dimensionless_aggregation(count):
    members = tuple(hashlib.sha256(f"independent-member-{index}".encode()).hexdigest() for index in range(count))
    rows = [_row(member) for member in members]
    budgets = {member: _quality_budget() for member in members}
    report = evaluate_fast_portfolio(
        list(reversed(rows)), quality_budgets=budgets, policy=FastEvaluationPolicy(), expected_models=members
    )
    assert report["status"] == "retain" and report["models_evaluated"] == count
    assert tuple(row["model_id"] for row in report["models"]) == members
    assert report["portfolio_conservative_cycle_speedup_geomean"] == pytest.approx(1.25)
    rows[-1]["candidate_quality"]["values"]["relative_error"] = 0.5
    failed = evaluate_fast_portfolio(rows, quality_budgets=budgets, policy=FastEvaluationPolicy())
    assert failed["status"] == "reject"


@pytest.mark.parametrize("members", [(), (MODELS[0], MODELS[0]), ("unbound",)])
def test_generic_portfolio_refuses_empty_duplicate_and_non_content_identities(members):
    with pytest.raises(ValueError):
        PortfolioQualitySchema("exact_only", members, (), "no accuracy evaluator")
    with pytest.raises(ValueError):
        evaluate_fast_portfolio([], quality_budgets={}, policy=FastEvaluationPolicy(), expected_models=members)


def test_generic_quality_schema_freezes_order_and_legacy_wrapper_wire_identity():
    members = MODELS[:3]
    budgets = tuple((member, QualityBudget.exact("complete raw outputs")) for member in members)
    schema = PortfolioQualitySchema("accuracy_bounded", members, budgets, "independent exact observer")
    assert schema.to_dict()["schema"] == "phase2_portfolio_quality_schema_v1"
    assert schema.to_dict()["approximation_allowed"] is False
    with pytest.raises(ValueError, match="portfolio order"):
        PortfolioQualitySchema("accuracy_bounded", members, tuple(reversed(budgets)), "wrong order")
    with pytest.raises(TypeError, match="typed QualityBudget"):
        PortfolioQualitySchema("accuracy_bounded", (MODELS[0],), ((MODELS[0], {}),), "forged budget")
    with pytest.raises(ValueError, match="exactly four"):
        FourModelQualitySchema("exact_only", members, (), "legacy wrapper")
    legacy = FourModelQualitySchema("exact_only", MODELS, (), "corpus absent")
    assert legacy.to_dict() == {
        "schema": "phase2_four_model_quality_schema_v1",
        "mode": "exact_only",
        "ordered_member_sha256s": list(MODELS),
        "budgets": {},
        "reason": "corpus absent",
        "approximation_allowed": False,
    }


def _explicit_quality_report(budget, values, *, parameters=None, complete=True, baseline=None):
    row = _row(MODELS[0])
    row["candidate_quality"] = {
        "values": values,
        "provenance": ["independent full-output quality evaluator"],
        "complete": complete,
    }
    if parameters is not None:
        row["candidate_quality"]["parameters"] = parameters
    if baseline is not None:
        row["baseline_quality"] = baseline
    return evaluate_fast_portfolio([row], quality_budgets={MODELS[0]: budget}, policy=FastEvaluationPolicy())


@pytest.mark.parametrize("count, expected", [(0, "retain"), (1, "reject"), (-1, "needs_evidence")])
def test_exact_quality_uses_complete_bitwise_mismatch_count(count, expected):
    report = _explicit_quality_report(QualityBudget.exact("all raw output words"), {"bitwise_mismatch_count": count})
    assert report["status"] == expected


@pytest.mark.parametrize("values", [{"cosine_similarity": 1.0}, {"maximum_abs_error": 0.0}, {}])
def test_proxies_cannot_supply_an_exact_or_elementwise_quality_metric(values):
    exact = _explicit_quality_report(QualityBudget.exact("all raw output words"), values)
    elementwise = _explicit_quality_report(
        QualityBudget.elementwise("all outputs", atol=0.03, rtol=0.02),
        values,
        parameters={"atol": 0.03, "rtol": 0.02},
    )
    assert exact["status"] == elementwise["status"] == "needs_evidence"


@pytest.mark.parametrize("parameters", [None, {"atol": 0.03}, {"atol": 0.3, "rtol": 0.02}])
def test_elementwise_observer_must_bind_the_exact_selected_tolerances(parameters):
    budget = QualityBudget.elementwise("all outputs", atol=0.03, rtol=0.02)
    report = _explicit_quality_report(budget, {"elementwise_violation_count": 0}, parameters=parameters)
    assert report["status"] == "needs_evidence"
    assert "selected metric parameters" in " ".join(report["models"][0]["quality_gate"]["blockers"])


@pytest.mark.parametrize("count, expected", [(0, "retain"), (3, "reject"), (-1, "needs_evidence")])
def test_elementwise_count_preserves_combined_tolerance_metric(count, expected):
    budget = QualityBudget.elementwise("all outputs", atol=0.03, rtol=0.02)
    report = _explicit_quality_report(
        budget, {"elementwise_violation_count": count}, parameters={"atol": 0.03, "rtol": 0.02}
    )
    assert report["status"] == expected
    assert budget.to_dict()["parameters"] == {"atol": 0.03, "rtol": 0.02}


@pytest.mark.parametrize(
    "profile, metric", [("exact", "bitwise_mismatch_count"), ("elementwise", "elementwise_violation_count")]
)
def test_incomplete_or_fractional_quality_counts_cannot_pass(profile, metric):
    budget = (
        QualityBudget.exact("outputs") if profile == "exact" else QualityBudget.elementwise("outputs", atol=0, rtol=0)
    )
    parameters = None if profile == "exact" else {"atol": 0, "rtol": 0}
    incomplete = _explicit_quality_report(budget, {metric: 0}, parameters=parameters, complete=False)
    fractional = _explicit_quality_report(budget, {metric: -0.5}, parameters=parameters)
    assert incomplete["status"] == fractional["status"] == "needs_evidence"
    with pytest.raises(ValueError, match="zero complete-output"):
        QualityBudget((QualityLimit(metric, "at_most", 10),), "relaxed named policy", profile)


@pytest.mark.parametrize("value", [-1, float("nan"), float("inf"), True])
def test_elementwise_tolerances_require_finite_nonnegative_numbers(value):
    with pytest.raises((TypeError, ValueError)):
        QualityBudget.elementwise("outputs", atol=value, rtol=0.02)


def test_task_quality_has_explicit_metrics_and_degradation_against_complete_baseline():
    budget = QualityBudget.task(
        "held-out task corpus", limits=(QualityLimit("task_success_rate", "at_least", 0.85, maximum_degradation=0.02),)
    )
    baseline = {"values": {"task_success_rate": 0.90}, "provenance": ["independent task evaluator"], "complete": True}
    assert _explicit_quality_report(budget, {"task_success_rate": 0.89}, baseline=baseline)["status"] == "retain"
    assert _explicit_quality_report(budget, {"task_success_rate": 0.86}, baseline=baseline)["status"] == "reject"
    assert _explicit_quality_report(budget, {"cosine_similarity": 1.0}, baseline=baseline)["status"] == "needs_evidence"
    with pytest.raises(ValueError, match="at least one limit"):
        QualityBudget.task("task corpus", limits=())


def test_legacy_quality_serialization_and_direct_observer_completeness_remain_explicit():
    assert "parameters" not in _quality_budget().to_dict()
    assert QualityObservation((("relative_error", 0.1),), ("complete observer",)).to_dict() == {
        "values": {"relative_error": 0.1},
        "provenance": ["complete observer"],
        "complete": True,
    }
    with pytest.raises(TypeError, match="boolean"):
        QualityObservation((("relative_error", 0.1),), ("observer",), complete="true")
    with pytest.raises(ValueError, match="numeric"):
        QualityObservation((("relative_error", "0.1"),), ("observer",))


def test_unknown_occupancy_and_encoding_are_not_treated_as_zero():
    rows = [_row(model) for model in MODELS]
    incomplete = dict(rows[0]["candidate"])
    incomplete["occupancy"] = None
    incomplete["encoding_conversions"] = {}
    rows[0] = {**rows[0], "candidate": incomplete}

    report = _evaluate(rows)

    assert report["status"] == "needs_evidence"
    assert any("compute_utilization is UNKNOWN" in item for item in report["models"][0]["blockers"])
    assert report["models"][0]["candidate"]["encoding_conversions"]["count"] is None


def test_one_model_movement_regression_rejects_instead_of_averaging():
    rows = [_row(model) for model in MODELS]
    rows[3] = _row(MODELS[3], candidate_movement=101)

    report = _evaluate(rows)

    assert report["status"] == "reject"
    assert any("movement_bytes regressed" in item for item in report["failures"])


def test_more_coverage_alone_is_not_a_global_benefit():
    rows = [_row(model, candidate_cycles=100, candidate_movement=100) for model in MODELS]
    for row in rows:
        candidate = row["candidate"]
        candidate["occupancy"] = dict(row["baseline"]["occupancy"])
        candidate["encoding_conversions"] = dict(row["baseline"]["encoding_conversions"])
        candidate["coverage"] = {
            **candidate["coverage"],
            "supported_work_placed": 800,
            "largest_connected_region_work": 500,
            "connected_region_work": [500, 300],
            "host_islands": [{"taxonomy": "unsupported-compute", "count": 2, "work": 50}],
            "boundary_crossings": 10,
            "boundary_bytes": 100,
        }

    report = _evaluate(rows)

    assert report["status"] == "reject"
    assert report["failures"] == ["portfolio: no robust global benefit; increased placement alone is insufficient"]


def test_recommended_levers_are_intersected_with_host_frozen_surfaces():
    rows = [_row(model) for model in MODELS]
    rows[0] = _row(MODELS[0], candidate_movement=101)
    surfaces = {
        MODELS[0]: (
            {
                "id": "movement-pass",
                "path": "compiler/move.py",
                "symbol": "plan",
                "scope": "pass",
                "effects": ["movement", "encoding"],
            },
            {
                "id": "unrelated-pass",
                "path": "compiler/math.py",
                "symbol": "quantize",
                "scope": "pass",
                "effects": ["quantization"],
            },
        )
    }

    report = _evaluate(rows, surfaces=surfaces)
    movement = next(item for item in report["models"][0]["recommended_levers"] if item["lever"] == "data_movement")

    assert [surface["id"] for surface in movement["authorized_surfaces"]] == ["movement-pass"]


def test_metric_consistency_and_complete_connected_regions_are_required():
    value = _metrics(
        100,
        movement=100,
        utilization=0.5,
        overlap=0.2,
        encoding_count=2,
        encoding_bytes=20,
        encoding_cycles=8,
    )
    value["movement_bytes"] = 99
    with pytest.raises(ValueError, match="movement bytes disagree"):
        AnalyticalMetrics.from_mapping(value)

    value["movement_bytes"] = 100
    value["coverage"] = {**value["coverage"], "connected_region_work": [400]}
    with pytest.raises(ValueError, match="exactly partition"):
        AnalyticalMetrics.from_mapping(value)
