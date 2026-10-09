"""Fresh admission reopens actual bounded generation; no runtime is issued.

The ordinary synthetic Phase 0 fixture declares hardware facts explicitly.
These cases establish logical reference bounds, never qualified RTL or engines.
"""

import copy
import json

import pytest
import test_component_coverage as coverage_fixtures
import test_component_execution_budget as budget_fixtures
import test_component_generation as generation_fixtures
import yaml
from merlin_experiments.phase0 import component_coverage as coverage
from merlin_experiments.phase0 import generation
from merlin_experiments.phase1.component_generation_admission import verify_bounded_generation
from merlin_experiments.phase2.contracts import StageGateError

independent = generation_fixtures.independent


def write_report(root, report):
    report = copy.deepcopy(report)
    report.pop("sha256", None)
    report["sha256"] = coverage.digest(report)
    return coverage.write_report(root, report)


def generate_bounded(options):
    budget_fixtures.select(options, [budget_fixtures.contraction(options)])
    generation.generate_target("fixture", **options)
    return verify_bounded_generation(options["output_root"])


def test_missing_actual_coverage_is_not_bounded_generation(tmp_path):
    with pytest.raises(StageGateError, match="bounded generation is unavailable"):
        verify_bounded_generation(tmp_path)


def test_actual_admitted_v2_generation_reopens_original_sources_costs_and_complete_goldens(independent):
    report = generate_bounded(independent)
    assert report["schema"] == coverage.BUDGETED_REPORT_SCHEMA and report["status"] == "complete"
    ledger = report["execution_admission"]
    assert len(ledger["decisions"]) == 5 and all(row["state"] == "admitted" for row in ledger["decisions"])
    identity = report["generation_identity"]
    assert identity["execution_budget_sha256"] == coverage.digest(ledger["policy"])
    assert identity["execution_admission_sha256"] == coverage.digest(ledger)
    member = report["obligations"][0]["members"][0]
    from merlin.targetgen import golden_store

    golden = golden_store.load_golden(independent["output_root"] / member["member"])
    assert sorted(golden["outputs"]) == member["output_roster"] == ["Y"]
    # The production gate reads disk again: an earlier returned dictionary
    # cannot substitute for the current independently replayed source ledger.
    assert verify_bounded_generation(independent["output_root"]) == report


def test_legacy_report_remains_readable_but_cannot_admit_a_fresh_origin(independent):
    coverage_fixtures.plan_for(independent, [coverage_fixtures.movement("legacy", "functional_guard")])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    assert report["schema"] == coverage.REPORT_SCHEMA
    with pytest.raises(StageGateError, match="requires verified budgeted Phase 0 coverage v2"):
        verify_bounded_generation(independent["output_root"])


def test_downgrading_v2_to_a_resigned_legacy_report_cannot_admit_a_fresh_origin(independent):
    report = generate_bounded(independent)
    report["schema"] = coverage.REPORT_SCHEMA
    report["declaration"]["schema"] = coverage_fixtures.coverage_plan.PLAN_SCHEMA
    report["declaration"].pop("execution_budget")
    report.pop("execution_admission")
    report["generation_identity"].pop("execution_budget_sha256")
    report["generation_identity"].pop("execution_admission_sha256")
    write_report(independent["output_root"], report)
    with pytest.raises(StageGateError, match="requires verified budgeted Phase 0 coverage v2"):
        verify_bounded_generation(independent["output_root"])


@pytest.mark.parametrize("defect", ["missing_ledger", "cost", "total", "policy", "missing_member"])
def test_resigned_forged_v2_cannot_replace_source_derived_reference_budget(independent, defect):
    report = generate_bounded(independent)
    ledger = report["execution_admission"]
    if defect == "missing_ledger":
        report.pop("execution_admission")
    elif defect == "cost":
        ledger["decisions"][-1]["cost"]["reference_work"] = 1
    elif defect == "total":
        ledger["totals"]["tensor_payload_bytes"] = 1
    elif defect == "policy":
        ledger["policy"]["max_reference_work"] += 1
        ledger["policy_sha256"] = coverage.digest(ledger["policy"])
    else:
        ledger["decisions"].pop()
    report["generation_identity"]["execution_admission_sha256"] = coverage.digest(ledger)
    path = write_report(independent["output_root"], report)
    # The hash is internally consistent, so the normal replay must establish
    # the refusal rather than treating a signed JSON cost as measured evidence.
    written = json.loads(path.read_bytes())
    stated = written.pop("sha256")
    assert coverage.digest(written) == stated
    with pytest.raises(StageGateError, match="bounded generation is unavailable"):
        verify_bounded_generation(independent["output_root"])


def test_generated_source_drift_and_indirect_roots_refuse_before_fresh_admission(independent, tmp_path):
    report = generate_bounded(independent)
    alias = tmp_path / "coverage-alias"
    alias.symlink_to(independent["output_root"], target_is_directory=True)
    with pytest.raises(StageGateError, match="indirect paths"):
        verify_bounded_generation(alias)
    member = report["obligations"][0]["members"][0]
    directory = independent["output_root"] / member["member"]
    capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
    source = directory / capsule["interface_mlir"]
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(StageGateError, match="bounded generation is unavailable"):
        verify_bounded_generation(independent["output_root"])
