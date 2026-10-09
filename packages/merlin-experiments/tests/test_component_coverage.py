"""Independent normal-generator coverage, source binding and private transfer tests."""

import hashlib
import itertools
import json

import pytest
import test_component_generation as generation_fixtures
import yaml
from merlin_experiments.phase0 import component_coverage as coverage
from merlin_experiments.phase0 import component_coverage_inputs as coverage_inputs
from merlin_experiments.phase0 import component_coverage_plan as coverage_plan
from merlin_experiments.phase0 import generation
from merlin_experiments.phase0.evidence import select_evidence
from test_component_generation import write

from merlin.targetgen import golden_store

independent = generation_fixtures.independent


def plan_for(options, obligations, *, effects=(), max_members=64):
    evidence = select_evidence(
        "fixture",
        descriptor=options["descriptor"],
        capability_contract_path=options["capability_contract"],
        facts_path=options["rtl_facts"],
        software_spec=options["software_spec"],
    )
    software = [row for row in evidence.source_snapshots if row.role == "software-spec"][0]
    document = {
        "schema": coverage_plan.PLAN_SCHEMA,
        "status": "reviewed",
        "hardware": {key: evidence.derivation_identity[key] for key in ("contract_sha256", "raw_facts_sha256")},
        "software_spec_sha256": software.sha256,
        "numerical_semantics_sha256": coverage.digest(evidence.software_spec["numerical_semantics"]),
        "effects": list(effects),
        "budget": {"max_members": max_members, "max_interaction_cells": 2000},
        "obligations": obligations,
    }
    path = write(options["recipe"].with_name("component-coverage.yaml"), document)
    options["component_coverage"] = path
    return document


def movement(identity, cohort, axes=None):
    return {
        "id": identity,
        "mandatory": True,
        "cohort": cohort,
        "operations": ["movement"],
        "effects": [],
        "expectation": "admitted_program",
        "frontend": "mlir",
        "base": {"op": "movement", "kind": "isa"},
        "axes": axes
        or {
            "M": {"kind": "extent", "values": ["tile", "tile+1"]},
            "N": {"kind": "extent", "values": ["tile+2", "2*tile+3"]},
        },
        "interactions": [],
    }


def update_hardware(options, *, tile=None, extra=None):
    contract = yaml.safe_load(options["capability_contract"].read_bytes())
    raw = json.loads(options["rtl_facts"].read_bytes())
    if tile is not None:
        contract["capabilities"]["mesh"] = {"rows": tile, "cols": tile}
        raw["facts"]["arrays"][0].update(rows=tile, cols=tile)
    if extra:
        raw["facts"].update(extra)
    write(options["capability_contract"], contract)
    write(options["rtl_facts"], raw)
    software = yaml.safe_load(options["software_spec"].read_bytes())
    software["component_performance"]["hardware"] = {
        "contract_sha256": hashlib.sha256((json.dumps(contract, sort_keys=True, indent=2) + "\n").encode()).hexdigest(),
        "raw_facts_sha256": hashlib.sha256(options["rtl_facts"].read_bytes()).hexdigest(),
    }
    write(options["software_spec"], software)


@pytest.mark.parametrize("tile", [2, 3])
def test_normal_generator_binds_functional_and_private_transfer_two_geometries(independent, tile):
    update_hardware(independent, tile=tile)
    functional = movement("guard", "functional_guard")
    hidden = movement(
        "secret_transfer_family",
        "withheld_transfer",
        {
            "M": {"kind": "extent", "values": ["tile+4"]},
            "N": {"kind": "extent", "values": ["tile+7"]},
        },
    )
    plan_for(independent, [functional, hidden])
    written = generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    assert report["status"] == "complete"
    assert {row["state"] for row in report["obligations"]} == {"generated"}
    assert len(written) == 9
    guards = coverage.build_guard_link(report)["guards"]
    assert len(guards) == 4
    assert all(row["member"].startswith("isa/") for row in guards)
    manifest = (independent["output_root"] / "MANIFEST.yaml").read_text()
    assert "secret_transfer_family" not in manifest
    assert "tile+7" not in manifest
    phase1 = yaml.safe_load(manifest)["phase_corpora"]["fixture"]["phase1"]
    assert len(phase1["generated_members"]) == 4
    private = independent["output_root"] / "_evidence/coverage/component-coverage.json"
    assert private.stat().st_mode & 0o077 == 0
    hidden_member = next(row for row in report["obligations"] if row["cohort"] == "withheld_transfer")["members"][0]
    cap = yaml.safe_load((independent["output_root"] / hidden_member["member"] / "capsule.yaml").read_bytes())
    assert cap["inputs"][0]["shape"] == [tile + 4, tile + 7]


def test_covering_points_enumerates_declared_triples_and_remaining_pairs():
    axes = {"a": [0, 1], "b": [0, 1, 2], "c": [0, 1], "d": [0, 1, 2]}
    points, record = coverage_inputs.covering_points(axes, [["a", "b", "c"]], max_members=30, max_cells=1000)
    assert record["status"] == "complete"
    assert len(points) < 36
    for group in [*(itertools.combinations(axes, 2)), ("a", "b", "c")]:
        expected = set(itertools.product(*(axes[name] for name in group)))
        assert {tuple(point[name] for name in group) for point in points} == expected
    assert coverage_inputs.covering_points(axes, [["a", "b", "c"]], max_members=30, max_cells=1000) == (points, record)


@pytest.mark.parametrize("change", ["budget", "fact", "effect"])
def test_missing_mandatory_obligation_fails_after_preserving_private_report(independent, change):
    obligation = movement("required", "functional_guard")
    effects = []
    if change == "fact":
        obligation["axes"] = {
            "M": {
                "kind": "resource",
                "declaration": {
                    "derive": "declared_resource_boundary",
                    "resource": "bank",
                    "capacity_fact": ["bank_bytes"],
                    "reservation_facts": [],
                    "quantum": "tile",
                    "tail_offsets": [-1, 1],
                    "allocations": [{"name": "live", "shape": ["M", 2], "dtype": "operand"}],
                },
            },
            "N": {"kind": "extent", "values": [2]},
        }
    elif change == "effect":
        obligation["effects"] = ["mutation"]
        effects = [
            {"id": "mutation", "status": "reviewed", "kind": "epoch_mutation", "basis": "explicit mutable source epoch"}
        ]
    plan_for(independent, [obligation], effects=effects, max_members=1 if change == "budget" else 64)
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **independent)
    report = json.loads((independent["output_root"] / "_evidence/coverage/component-coverage.json").read_bytes())
    assert report["status"] == "incomplete"
    assert report["obligations"][0]["state"] == "unavailable"
    assert report["obligations"][0]["errors"]
    with pytest.raises(ValueError, match="mandatory component coverage"):
        coverage.verify_report(independent["output_root"])


@pytest.mark.parametrize("resource", ["bank", "accumulator", "transfer_segment", "alignment", "cache"])
def test_declared_byte_frontier_writes_actual_boundary_and_tails(independent, resource):
    update_hardware(independent, extra={"frontier_bytes": 32, "reserved_bytes": 4})
    obligation = movement(
        "frontier",
        "functional_guard",
        {
            "M": {
                "kind": "resource",
                "declaration": {
                    "derive": "declared_resource_boundary",
                    "resource": resource,
                    "capacity_fact": ["frontier_bytes"],
                    "reservation_facts": [["reserved_bytes"]],
                    "quantum": "tile",
                    "tail_offsets": [-1, 1],
                    "allocations": [
                        {"name": "producer", "shape": ["M", 2], "dtype": "operand"},
                        {"name": "consumer", "shape": ["M", 2], "dtype": "operand"},
                    ],
                },
            },
            "N": {"kind": "extent", "values": [2]},
        },
    )
    plan_for(independent, [obligation])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    boundary = report["obligations"][0]["resource_boundaries"]["M"]
    assert boundary["capacity_bytes"] == 32
    points = boundary["points"]
    last = next(point for point in points if "last_fitting" in point["roles"])
    over = next(point for point in points if "first_overflow" in point["roles"])
    assert last["extent"] == 6 and last["total_bytes"] == 28
    assert over["extent"] == 8 and over["total_bytes"] == 36
    assert {point["extent"] for point in points} == {4, 5, 6, 7, 8, 9}


def test_report_rechecks_golden_and_selected_plan_bytes(independent):
    plan_for(independent, [movement("guard", "functional_guard")])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    member = report["obligations"][0]["members"][0]
    directory = independent["output_root"] / member["member"]
    document = golden_store.load_golden(directory)
    document["outputs"]["Y0"][0][0] += 1
    golden_store.write_golden(directory, document)
    with pytest.raises(ValueError, match="member bytes changed"):
        coverage.verify_report(independent["output_root"])


@pytest.mark.parametrize("owner", ["report", "generation_identity"])
@pytest.mark.parametrize("mutation", ["changed", "removed"])
def test_resigned_report_cannot_change_one_independent_software_binding(independent, owner, mutation):
    plan_for(independent, [movement("guard", "functional_guard")])
    generation.generate_target("fixture", **independent)
    root = independent["output_root"]
    report = coverage.verify_report(root)
    # The report verifier checks duplicate consistency, not live intake issuance.
    # The fresh author owner separately requires the actual issued capability.
    report.pop("sha256")
    report["software_intake_sha256"] = "a" * 64
    report["generation_identity"]["software_intake_sha256"] = "a" * 64
    report["sha256"] = coverage.digest(report)
    assert coverage.verify_report(root, report)["software_intake_sha256"] == "a" * 64
    selected = report if owner == "report" else report["generation_identity"]
    if mutation == "removed":
        selected.pop("software_intake_sha256")
    else:
        selected["software_intake_sha256"] = "b" * 64
    report.pop("sha256")
    report["sha256"] = coverage.digest(report)
    with pytest.raises(ValueError, match="independent software binding changed"):
        coverage.verify_report(root, report)


def test_unavailable_sealed_frontend_is_explicit_mandatory_unavailable(independent, monkeypatch):
    from merlin_experiments.phase0 import sealed_generation

    obligation = movement("frontend", "functional_guard")
    obligation["frontend"] = "pytorch"
    obligation["base"]["op"] = "matmul"
    obligation["operations"] = ["contraction"]
    monkeypatch.delenv(sealed_generation.CONFIG_ENV, raising=False)
    plan_for(independent, [obligation])
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **independent)
    report = json.loads((independent["output_root"] / "_evidence/coverage/component-coverage.json").read_bytes())
    assert report["status"] == "incomplete"
    assert all(member["state"] == "unavailable" for member in report["obligations"][0]["members"])
    assert all("sealed Model2MLIR" in member["reason"] for member in report["obligations"][0]["members"])


def test_declared_refusal_has_independent_concrete_program_and_golden(independent, monkeypatch):
    from merlin.common.tree_hash import hash_tree
    from merlin.targetgen import target_experiment

    owner = independent["descriptor"].parent
    host = owner / "host"
    write(
        host / "manifest.yaml",
        {
            "target": "fixture_host",
            "run_id": "empty_support",
            "family": "scalar",
            "authoring": {"mode": "deterministic_generated_from_spec"},
            "outputs": {"knobs": "knobs.yaml"},
        },
    )
    write(
        host / "knobs.yaml",
        {"backend": "scalar", "dtype_strategy": "int8_w8a8", "cflags": ["-march=rv64imafdc", "-mabi=lp64d"]},
    )
    host_spec = write(
        owner / "host-support.yaml",
        {
            "schema": "merlin.host_capabilities.v1",
            "status": "reviewed",
            "compiler": {"package_sha256": hash_tree(host)["sha256"], "dtype_strategy": "int8_w8a8"},
            "operations": [],
            "evidence": {"scope": "independent reviewed synthetic absence of host operation support"},
            "unresolved": [],
        },
    )
    descriptor = yaml.safe_load(independent["descriptor"].read_bytes())
    descriptor["host_lane"] = {
        "description": "Reviewed synthetic empty host domain",
        "repo_canonical": "independent fixture",
        "provenance": "in_tree_minted",
        "package": "host",
        "requires_paths": ["manifest.yaml", "knobs.yaml"],
        "read_only": ["host"],
        "deny_modification": [],
        "dtype_strategy": "int8_w8a8",
        "capability_spec": "host-support.yaml",
        "capability_spec_sha256": hashlib.sha256(host_spec.read_bytes()).hexdigest(),
    }
    write(independent["descriptor"], descriptor)
    load = target_experiment.load_target_experiment

    def selected_root(path, **kwargs):
        return load(path, source_root=owner, **kwargs)

    monkeypatch.setattr(target_experiment, "load_target_experiment", selected_root)
    monkeypatch.setattr(generation, "load_target_experiment", selected_root)
    selected = select_evidence(
        "fixture",
        descriptor=independent["descriptor"],
        capability_contract_path=independent["capability_contract"],
        facts_path=independent["rtl_facts"],
        software_spec=independent["software_spec"],
    )
    assert all(row.get("capability_spec") for row in selected.host_capabilities.values()), selected.host_capabilities
    obligation = movement("unsupported_dtype", "functional_guard")
    obligation["expectation"] = "unsupported_program"
    obligation["operations"] = ["contraction"]
    obligation["base"] = {
        "op": "component_program",
        "kind": "model_slice",
        "program": {
            "inputs": [
                {"name": "A", "role": "input", "shape": [{"axis": "M"}, {"axis": "N"}], "dtype": "i32"},
                {"name": "W", "role": "weight", "shape": [{"axis": "N"}, {"axis": "N"}], "dtype": "i32"},
            ],
            "nodes": [{"name": "Product", "op": "matmul", "inputs": ["A", "W"]}],
            "outputs": [{"name": "product", "value": "Product"}],
        },
    }
    plan_for(independent, [obligation])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    assert report["obligations"][0]["state"] == "verified_refusal"
    assert all(
        member["member"].startswith("_diagnostic/") and member["program_sha256"] and member["output_roster"]
        for member in report["obligations"][0]["members"]
    )
    independent["component_coverage"].write_text("changed")
    with pytest.raises(ValueError, match="generation source changed"):
        coverage.verify_report(independent["output_root"])


def graph_program():
    return {
        "inputs": [
            {"name": "A", "role": "input", "shape": [2, 3], "dtype": "operand"},
            {"name": "W", "role": "weight", "shape": [3, 2], "dtype": "operand"},
        ],
        "nodes": [
            {"name": "P", "op": "matmul", "inputs": ["A", "W"]},
            {"name": "Alias", "op": "alias", "inputs": ["P"]},
            {"name": "Before", "op": "copy", "inputs": ["Alias"]},
            {"name": "Double", "op": "add", "inputs": ["P", "P"]},
            {"name": "Epoch", "op": "update", "inputs": ["Alias", "Double"]},
            {"name": "After", "op": "copy", "inputs": ["P"]},
        ],
        "outputs": [
            {"name": "snapshot", "value": "Before"},
            {"name": "mutated_alias", "value": "Alias"},
            {"name": "after", "value": "After"},
            {"name": "escaped", "value": "Double"},
        ],
    }


def test_general_composition_shared_escaped_alias_and_epoch_source_semantics(independent):
    contract = yaml.safe_load(independent["capability_contract"].read_bytes())
    contract["compute_units"][0]["accumulate"] = [{"in": "int8", "weight": "int8", "acc": "i32"}]
    contract["compute_units"].append(
        {
            "name": "scalar",
            "kind": "simt",
            "dtypes": ["i32"],
            "ops": ["add", "movement"],
            "semantic_capabilities": [
                {"family": "elementwise_map", "dtypes": ["i32"], "ranks": [2]},
                {"family": "movement", "dtypes": ["i32"], "ranks": [2], "forms": ["copy"]},
            ],
        }
    )
    write(independent["capability_contract"], contract)
    update_hardware(independent)
    software = yaml.safe_load(independent["software_spec"].read_bytes())
    software["operations"]["elementwise_map"] = {"placement": "accelerator", "dtypes": ["i32"], "ranks": [2]}
    software["operations"]["movement"]["dtypes"] = ["int8", "i32"]
    write(independent["software_spec"], software)
    kinds = ["shared_producer", "multiple_consumers", "escaped_use", "alias", "epoch_mutation", "output_publication"]
    obligation = {
        "id": "dag",
        "mandatory": True,
        "cohort": "functional_guard",
        "operations": ["movement", "contraction", "elementwise_map"],
        "effects": kinds,
        "expectation": "admitted_program",
        "frontend": "mlir",
        "base": {"op": "component_program", "kind": "model_slice", "program": graph_program()},
        "axes": {},
        "interactions": [],
    }
    effects = [
        {"id": kind, "status": "reviewed", "kind": kind, "basis": "functionalized tensor source semantics"}
        for kind in kinds
    ]
    plan_for(independent, [obligation], effects=effects)
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    member = report["obligations"][0]["members"][0]
    directory = independent["output_root"] / member["member"]
    cap = yaml.safe_load((directory / "capsule.yaml").read_bytes())
    golden = golden_store.load_golden(directory)["outputs"]
    from merlin.targetgen.capsule_golden import golden as recompute

    assert recompute(cap, directory) == golden
    assert golden["after"] == golden["mutated_alias"]
    assert golden["mutated_alias"] == [[value * 3 for value in row] for row in golden["snapshot"]]
    assert golden["escaped"] == [[value * 2 for value in row] for row in golden["snapshot"]]
    assert set(cap["component_program"]["effects"]) == set(kinds)
    assert member["output_roster"] == ["after", "escaped", "mutated_alias", "snapshot"]
