"""Normal generation refuses unbounded references before any data allocation."""

import copy
import json

import pytest
import test_component_coverage as fixtures
import test_component_generation as generation_fixtures
import yaml
from merlin_experiments.phase0 import component_coverage as coverage
from merlin_experiments.phase0 import component_coverage_plan as plans
from merlin_experiments.phase0 import component_execution_budget as budget
from merlin_experiments.phase0 import generation, sealed_generation

from merlin.targetgen import component_program, golden_store, input_palette

independent = generation_fixtures.independent


def policy(**changes):
    return {
        "schema": budget.SCHEMA,
        "max_reference_work": 10000,
        "max_materialized_elements": 10000,
        "max_tensor_payload_bytes": 10000,
        "max_scalar_bits": 64,
        "max_total_reference_work": 100000,
        "max_total_materialized_elements": 100000,
        "max_total_tensor_payload_bytes": 100000,
        **changes,
    }


def program(m, k, n, dtype="operand"):
    return {
        "inputs": [
            {"name": "A", "role": "input", "shape": [m, k], "dtype": dtype},
            {"name": "W", "role": "weight", "shape": [k, n], "dtype": dtype},
        ],
        "nodes": [{"name": "P", "op": "matmul", "inputs": ["A", "W"]}],
        "outputs": [{"name": "Y", "value": "P"}],
    }


def contraction(options, m=2, k=3, n=2):
    contract = yaml.safe_load(options["capability_contract"].read_bytes())
    contract["compute_units"][0]["accumulate"] = [{"in": "int8", "weight": "int8", "acc": "i32"}]
    generation_fixtures.write(options["capability_contract"], contract)
    fixtures.update_hardware(options)
    return {
        "id": "bounded_dag",
        "mandatory": True,
        "cohort": "functional_guard",
        "operations": ["contraction"],
        "effects": [],
        "expectation": "admitted_program",
        "frontend": "mlir",
        "base": {"op": "component_program", "kind": "model_slice", "program": program(m, k, n)},
        "axes": {},
        "interactions": [],
    }


def select(options, obligations, **limits):
    document = fixtures.plan_for(options, obligations)
    document.update(schema=plans.BUDGETED_PLAN_SCHEMA, execution_budget=policy(**limits))
    generation_fixtures.write(options["component_coverage"], document)
    return document


def receipt(options):
    return json.loads((options["output_root"] / "_evidence/coverage/component-coverage.json").read_bytes())


def test_normal_bounded_dag_publishes_complete_outputs_and_rederived_receipt(independent):
    select(independent, [contraction(independent)], max_reference_work=60, max_tensor_payload_bytes=75)
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    row = report["obligations"][0]["members"][0]
    decision = next(item for item in report["execution_admission"]["decisions"] if item["name"] == row["name"])
    assert decision["cost"] == {
        "reference_work": 60,
        "materialized_elements": 24,
        "tensor_payload_bytes": 60,
        "scalar_bits": 32,
    }
    assert report["schema"] == coverage.BUDGETED_REPORT_SCHEMA
    assert report["generation_identity"]["execution_budget_sha256"] == coverage.digest(
        policy(max_reference_work=60, max_tensor_payload_bytes=75)
    )
    from merlin.targetgen.capsule_inputs import materialize_capsule_leaves

    directory = independent["output_root"] / row["member"]
    capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
    leaves = materialize_capsule_leaves(capsule)
    # Independent scalar oracle over all outputs, rather than the cost helper.
    expected = [
        [sum(leaves["A"].data[i * 3 + p] * leaves["W"].data[p * 2 + j] for p in range(3)) for j in range(2)]
        for i in range(2)
    ]
    assert golden_store.load_golden(directory)["outputs"] == {"Y": expected}
    assert len(report["execution_admission"]["decisions"]) == 5  # four ordinary dev sweeps + DAG


@pytest.mark.parametrize("limit", ["max_reference_work", "max_materialized_elements", "max_tensor_payload_bytes"])
def test_huge_required_dag_refuses_before_builder_palette_or_reference(independent, monkeypatch, limit):
    obligation = contraction(independent, m=10**9, k=2, n=2)
    obligation["base"].update(reference_work=1, materialized_elements=1, tensor_payload_bytes=1)
    obligation["base"]["input_palette"] = {
        "schema": input_palette.SCHEMA,
        "inputs": [{"name": "A", "axis": "linear", "values": [0, 1], "offset": 0}],
    }
    limits = {"max_" + key: 10**15 for key in ("reference_work", "materialized_elements", "tensor_payload_bytes")}
    limits.update(
        {"max_total_" + key: 10**16 for key in ("reference_work", "materialized_elements", "tensor_payload_bytes")}
    )
    limits[limit] = 100
    select(independent, [obligation], **limits)
    original = component_program.build

    def guarded_build(entry, binding):
        assert entry["op"] != "component_program", "denied DAG reached ordinary builder"
        return original(entry, binding)

    monkeypatch.setattr(component_program, "build", guarded_build)
    monkeypatch.setattr(input_palette, "realize", lambda *a, **k: pytest.fail("denied palette was allocated"))
    from merlin_experiments.phase0 import component_numerics

    monkeypatch.setattr(component_numerics, "evaluate", lambda *a, **k: pytest.fail("denied golden was evaluated"))
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **independent)
    report = receipt(independent)
    required = report["obligations"][0]
    assert required["mandatory"] and required["state"] == "unavailable" and len(required["members"]) == 1
    row = required["members"][0]
    assert row["reason"].endswith("execution budget exceeded: " + limit.removeprefix("max_"))
    assert not (independent["output_root"] / row["requested_member"]).exists()
    with pytest.raises(ValueError, match="mandatory component coverage"):
        coverage.verify_report(independent["output_root"])


def test_unknown_frontend_does_not_reach_capture_selection_or_writer(independent, monkeypatch):
    obligation = contraction(independent)
    obligation.update(id="unknown_frontend", cohort="withheld_transfer")
    obligation["base"] = {"op": "matmul", "kind": "isa", "M": 2, "K": 3, "N": 2}
    obligation["frontend"] = "pytorch"
    select(independent, [obligation])
    original = sealed_generation.bind_source

    def checked_selection(entries, **kwargs):
        assert all(entry.get("source") != "pytorch" for entry in entries)
        return original(entries, **kwargs)

    monkeypatch.setattr(sealed_generation, "bind_source", checked_selection)
    original_writer = generation._write_capsule

    def checked_writer(entry, *args, **kwargs):
        assert entry.get("source") != "pytorch", "unknown-cost frontend reached the writer"
        return original_writer(entry, *args, **kwargs)

    monkeypatch.setattr(generation, "_write_capsule", checked_writer)
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **independent)
    report = receipt(independent)
    assert report["obligations"][0]["state"] == "unavailable"
    assert len(report["obligations"][0]["members"]) == 1
    assert "cost" not in report["execution_admission"]["decisions"][-1]
    assert "no derived cost" in report["execution_admission"]["decisions"][-1]["reason"]


def test_generation_total_budget_charges_all_development_and_guard_cases(independent):
    select(independent, [contraction(independent)], max_total_reference_work=80)
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **independent)
    report = receipt(independent)
    decisions = report["execution_admission"]["decisions"]
    assert len(decisions) == 5 and all(row["state"] == "admitted" for row in decisions[:4])
    assert decisions[-1]["state"] == "unavailable"
    assert "total_reference_work" in decisions[-1]["reason"]
    assert report["execution_admission"]["totals"]["reference_work"] == 70
    assert report["obligations"][0]["state"] == "unavailable"


@pytest.mark.parametrize("change", ["cost", "total", "policy", "missing", "drop_development"])
def test_resigned_budget_receipt_cannot_substitute_supplied_counts(independent, change):
    select(independent, [contraction(independent)])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    changed = copy.deepcopy(report)
    admission = changed["execution_admission"]
    if change == "cost":
        admission["decisions"][-1]["cost"]["reference_work"] = 1
        changed["generation_identity"]["execution_admission_sha256"] = coverage.digest(admission)
    elif change == "total":
        admission["totals"]["reference_work"] = 1
    elif change == "policy":
        changed["declaration"]["execution_budget"]["max_reference_work"] = 1
    elif change == "missing":
        admission["decisions"].pop()
    else:
        removed = admission["decisions"].pop(0)
        for key in admission["totals"]:
            admission["totals"][key] -= removed["cost"][key]
        changed["generation_identity"]["execution_admission_sha256"] = coverage.digest(admission)
    changed.pop("sha256")
    changed["sha256"] = coverage.digest(changed)
    with pytest.raises(ValueError, match="budget|selected plan"):
        coverage.verify_report(independent["output_root"], changed)


def test_large_reduction_is_not_refused_merely_for_one_large_axis():
    typed = component_program.analyze(program(1, 1000, 1), operand_dtype="i8", accumulator_dtype="i32")
    cost = budget.measure({"kind": "component_program", "program": typed, "input_palette": None})
    assert cost["reference_work"] == 6003 and cost["materialized_elements"] == 2003
    assert cost["tensor_payload_bytes"] == 2012
    assert cost["scalar_bits"] == 32


def test_scalar_width_is_bounded_before_reference_shift_or_materialization(independent, monkeypatch):
    obligation = contraction(independent)
    obligation["base"]["program"] = {
        "inputs": [{"name": "A", "role": "input", "shape": [1, 1], "dtype": "i1000000000"}],
        "nodes": [{"name": "P", "op": "copy", "inputs": ["A"]}],
        "outputs": [{"name": "Y", "value": "P"}],
    }
    select(independent, [obligation], max_tensor_payload_bytes=10**10, max_total_tensor_payload_bytes=10**10)
    from merlin_experiments.phase0 import component_numerics

    monkeypatch.setattr(component_numerics, "evaluate", lambda *a, **k: pytest.fail("wide scalar golden allocated"))
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **independent)
    decision = receipt(independent)["execution_admission"]["decisions"][-1]
    assert decision["state"] == "unavailable"
    assert decision["cost"]["scalar_bits"] == 10**9
    assert decision["reason"] == "execution budget exceeded: scalar_bits"


def test_budget_replay_refuses_symlink_before_reading_another_member_source(independent):
    select(independent, [contraction(independent)])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    first = report["execution_admission"]["decisions"][0]
    capsule = independent["output_root"] / first["requested_member"] / "capsule.yaml"
    outside = independent["recipe"].with_name("not-a-generated-capsule.yaml")
    capsule.rename(outside)
    capsule.symlink_to(outside)
    with pytest.raises(ValueError, match="escaped its actual generation root"):
        coverage.verify_report(independent["output_root"], report)


def test_legacy_v1_generation_preserves_unbudgeted_evidence_version(independent):
    fixtures.plan_for(independent, [fixtures.movement("legacy", "functional_guard")])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    assert report["schema"] == coverage.REPORT_SCHEMA
    assert "execution_admission" not in report and "execution_budget_sha256" not in report["generation_identity"]


@pytest.mark.parametrize(
    "mutation",
    [
        lambda p: p.pop("max_total_reference_work"),
        lambda p: p.update(max_reference_work=True),
        lambda p: p.update(max_scalar_bits=0),
    ],
)
def test_policy_requires_closed_explicit_positive_limits(mutation):
    declaration = policy()
    mutation(declaration)
    with pytest.raises(ValueError, match="closed v1 schema"):
        budget.validate(declaration)
