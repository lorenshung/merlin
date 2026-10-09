"""Normal bounded graph generation, original all-output proofs and missing pairs.

Synthetic support declarations exercise generic source semantics. No target
runtime, physical alias/reuse/completion or large-shape compiler claim is made.
"""

import copy
import json

import pytest
import test_component_coverage as fixtures
import test_component_execution_budget as budgets
import test_component_generation as generation_fixtures
import yaml
from merlin_experiments.phase0 import component_coverage as coverage
from merlin_experiments.phase0 import component_graph_variants as graphs
from merlin_experiments.phase0 import component_numerics, generation

from merlin.targetgen import component_program, golden_store, input_palette
from merlin.targetgen.capsule_inputs import materialize_capsule_leaves

independent = generation_fixtures.independent


def support(options, *, overflow="bounded_exact"):
    contract = yaml.safe_load(options["capability_contract"].read_bytes())
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
    generation_fixtures.write(options["capability_contract"], contract)
    fixtures.update_hardware(options)
    software = yaml.safe_load(options["software_spec"].read_bytes())
    software["numerical_semantics"]["overflow"] = overflow
    software["operations"]["elementwise_map"] = {"placement": "accelerator", "dtypes": ["i32"], "ranks": [2]}
    software["operations"]["movement"]["dtypes"] = ["int8", "i32"]
    generation_fixtures.write(options["software_spec"], software)


def family(identity="graph", cohort="functional_guard", *, depth=2, fanout=3, axes=None):
    base = {
        "op": "component_program",
        "kind": "model_slice",
        "M": 2,
        "K": 3,
        "N": 2,
        "depth": depth,
        "fanout": fanout,
        "graph_family": {"schema": graphs.SCHEMA, "representations": list(graphs.VARIANTS)},
    }
    for key in axes or {}:
        base[key] = {"axis": key}
    return {
        "id": identity,
        "mandatory": True,
        "cohort": cohort,
        "operations": ["contraction", "movement", "elementwise_map"],
        "effects": [],
        "expectation": "admitted_program",
        "frontend": "mlir",
        "base": base,
        "axes": axes or {},
        "interactions": [],
    }


def scalar_outputs(capsule, values):
    leaves = materialize_capsule_leaves(capsule)
    m, k, n = (values[key] for key in ("M", "K", "N"))
    current = [
        [sum(leaves["A"].data[i * k + p] * leaves["W"].data[p * n + j] for p in range(k)) for j in range(n)]
        for i in range(m)
    ]
    result = {}
    for epoch in range(values["depth"]):
        prefix = "E" + str(epoch) + "_"
        result[prefix + "old"] = copy.deepcopy(current)
        result[prefix + "escaped"] = copy.deepcopy(current)
        joined = copy.deepcopy(current)
        # Original source order: each distinct fork is added to the prior join.
        for _ in range(1, values["fanout"]):
            joined = [[joined[i][j] + current[i][j] for j in range(n)] for i in range(m)]
        current = [[current[i][j] + joined[i][j] for j in range(n)] for i in range(m)]
        result[prefix + "published"] = copy.deepcopy(current)
    result["Final"] = current
    return result


def test_actual_variable_topologies_publish_all_original_outputs_and_private_pairs(independent):
    support(independent)
    public = family(
        axes={
            "depth": {"kind": "choice", "values": [1, 3]},
            "fanout": {"kind": "choice", "values": [2, 4]},
            "M": {"kind": "extent", "values": ["tile", "tile+1"]},
        }
    )
    private = family("private_graph_transfer", "withheld_transfer", depth=2, fanout=3)
    private["base"].update(M=1, K=2, N=4)
    budgets.select(independent, [public, private])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    assert report["status"] == "complete"
    relations = report["graph_relations"]
    assert relations["schema"] == "merlin.component_graph_relations.v1"
    assert len(relations["rows"]) == 2 and all(row["state"] == "established" for row in relations["rows"])
    sources, shapes, topologies = set(), set(), set()
    for row in report["obligations"]:
        for member in row["members"]:
            directory = independent["output_root"] / member["member"]
            capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
            variant = member["graph_variant"]
            values = variant["parameters"]
            typed = capsule["component_program"]
            outputs = golden_store.load_golden(directory)["outputs"]
            assert outputs == scalar_outputs(capsule, values) == component_numerics.evaluate(capsule)
            assert len(outputs) == 3 * values["depth"] + 1
            mlir = (directory / "capsule.interface.mlir").read_text()
            assert mlir.count("linalg.copy ins(") == values["depth"] * (values["fanout"] + 2)
            assert mlir.count("linalg.generic {") == 1 + values["depth"] * values["fanout"]
            assert {"shared_producer", "multiple_consumers", "escaped_use", "output_publication"} <= set(
                typed["effects"]
            )
            if variant["representation"] == "logical_epochs":
                assert typed["logical_epochs"] == {"P": values["depth"]}
                assert len(typed["logical_aliases"]) == values["depth"]
                assert {"alias", "epoch_mutation"} <= set(typed["effects"])
            else:
                assert not typed["logical_epochs"] and not typed["logical_aliases"]
            assert capsule["integer_partial_sum_bound"]["status"] == "proven_safe"
            sources.add(member["program_sha256"])
            shapes.add((values["M"], values["K"], values["N"]))
            topologies.add((values["depth"], values["fanout"]))
    assert len(sources) == sum(len(row["members"]) for row in report["obligations"])
    assert {(1, 2), (1, 4), (3, 2), (3, 4)} <= topologies and len(shapes) == 3
    for row in relations["rows"]:
        for pair in row["pairs"]:
            assert len(pair["members"]) == 2
            assert len({member["full_outputs_sha256"] for member in pair["members"]}) == 1
    public_text = (
        json.dumps(coverage.public_summary(report)) + (independent["output_root"] / "MANIFEST.yaml").read_text()
    )
    assert "private_graph_transfer" not in public_text and "graph_relations" not in coverage.public_summary(report)
    private_report = independent["output_root"] / "_evidence/coverage/component-coverage.json"
    assert private_report.stat().st_mode & 0o077 == 0


@pytest.mark.parametrize("parameter", ["depth", "fanout", "M"])
def test_huge_graph_stops_before_unrolling_builder_data_and_reference(independent, monkeypatch, parameter):
    support(independent)
    required = family()
    required["base"][parameter] = 10**9
    required["base"].update(reference_work=1, materialized_elements=1, tensor_payload_bytes=1)
    budgets.select(independent, [required])
    original_program = graphs.program

    def checked_program(values, variant):
        assert values["depth"] <= 1 and values["fanout"] <= 3, "unbounded graph topology was constructed"
        return original_program(values, variant)

    monkeypatch.setattr(graphs, "program", checked_program)
    monkeypatch.setattr(component_program, "build", lambda *a, **k: pytest.fail("denied graph reached builder"))
    monkeypatch.setattr(input_palette, "realize", lambda *a, **k: pytest.fail("denied graph allocated palette"))
    monkeypatch.setattr(component_numerics, "evaluate", lambda *a, **k: pytest.fail("denied graph evaluated"))
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **independent)
    report = budgets.receipt(independent)
    row = report["obligations"][0]
    assert row["mandatory"] and row["state"] == "unavailable" and len(row["members"]) == 2
    assert all("before topology construction" in member["reason"] for member in row["members"])
    assert report["graph_relations"]["rows"][0]["pairs"][0]["state"] == "unavailable"
    assert all(not (independent["output_root"] / member["requested_member"]).exists() for member in row["members"])


def test_one_missing_representation_cannot_pass_its_relation(independent, monkeypatch):
    support(independent)
    budgets.select(independent, [family()])
    original = generation._write_capsule

    def fail_one(entry, *args, **kwargs):
        variant = (entry.get("component_coverage") or {}).get("graph_variant")
        if variant and variant["representation"] == "fresh_values":
            raise ValueError("independent injected missing fresh source")
        return original(entry, *args, **kwargs)

    monkeypatch.setattr(generation, "_write_capsule", fail_one)
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **independent)
    report = budgets.receipt(independent)
    row = report["obligations"][0]
    assert row["state"] == "unavailable" and len(row["members"]) == 2
    assert {member["state"] for member in row["members"]} == {"generated", "unavailable"}
    assert report["graph_relations"]["rows"][0]["state"] == "unavailable"
    with pytest.raises(ValueError, match="mandatory component coverage"):
        coverage.verify_report(independent["output_root"])


def test_overflowing_graph_is_not_made_safe_by_metamorphic_equality(independent, monkeypatch):
    support(independent)
    budgets.select(independent, [family(depth=20, fanout=3)])
    monkeypatch.setattr(component_program, "build", lambda *a, **k: pytest.fail("overflowing graph reached builder"))
    monkeypatch.setattr(component_numerics, "evaluate", lambda *a, **k: pytest.fail("overflowing graph evaluated"))
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **independent)
    report = budgets.receipt(independent)
    assert all("may overflow signed i32" in member["reason"] for member in report["obligations"][0]["members"])
    assert report["graph_relations"]["rows"][0]["state"] == "unavailable"


@pytest.mark.parametrize("mutation", ["drop_pair", "changed_witness"])
def test_resigned_relation_cannot_drop_original_members_or_substitute_witness(independent, mutation):
    support(independent)
    budgets.select(independent, [family(depth=1)])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    if mutation == "drop_pair":
        report["obligations"][0]["members"] = []
        report["graph_relations"]["rows"][0]["pairs"] = []
    else:
        report["graph_relations"]["rows"][0]["pairs"][0]["members"][0]["full_outputs_sha256"] = "a" * 64
    report.pop("sha256")
    report["sha256"] = coverage.digest(report)
    with pytest.raises(ValueError, match="graph relation"):
        coverage.verify_report(independent["output_root"], report)


def test_relation_recomputes_every_original_output_even_after_member_resealing(independent):
    from merlin_experiments.phase1.source_inputs import fingerprint

    support(independent)
    budgets.select(independent, [family()])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    for member in report["obligations"][0]["members"]:
        directory = independent["output_root"] / member["member"]
        golden = golden_store.load_golden(directory)
        golden["outputs"]["E0_escaped"][0][0] += 1  # identical corruption in both variants; final remains correct
        golden_store.write_golden(directory, golden)
        member["sha256"] = fingerprint(directory)
    report.pop("sha256")
    report["sha256"] = coverage.digest(report)
    with pytest.raises(ValueError, match="graph relation"):
        coverage.verify_report(independent["output_root"], report)


def test_changed_topology_parameters_refuse_before_expected_graph_allocation(independent, monkeypatch):
    from merlin_experiments.phase0 import component_graph_relations as relations
    from merlin_experiments.phase1.source_inputs import fingerprint

    support(independent)
    budgets.select(independent, [family()])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    for member in report["obligations"][0]["members"]:
        member["graph_variant"]["parameters"]["depth"] = 10**9
        directory = independent["output_root"] / member["member"]
        capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
        capsule["component_coverage"]["graph_variant"] = copy.deepcopy(member["graph_variant"])
        generation_fixtures.write(directory / "capsule.yaml", capsule)
        member["sha256"] = fingerprint(directory)
    report.pop("sha256")
    report["sha256"] = coverage.digest(report)
    monkeypatch.setattr(relations, "program", lambda *a, **k: pytest.fail("hostile replay allocated expected graph"))
    monkeypatch.setattr(component_numerics, "evaluate", lambda *a, **k: pytest.fail("hostile replay allocated leaves"))
    with pytest.raises(ValueError, match="graph relation"):
        coverage.verify_report(independent["output_root"], report)


def test_symbolic_cost_replays_actual_source_including_palette_charge(independent):
    support(independent)
    row = family(depth=3, fanout=4)
    row["base"]["input_palette"] = {
        "schema": input_palette.SCHEMA,
        "inputs": [{"name": "A", "axis": "linear", "values": [-1, 0, 1], "offset": 0}],
    }
    budgets.select(independent, [row])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    members = report["obligations"][0]["members"]
    costs = {decision["name"]: decision["cost"] for decision in report["execution_admission"]["decisions"]}
    assert costs[members[0]["name"]]["reference_work"] > costs[members[1]["name"]]["reference_work"]
    for member in members:
        directory = independent["output_root"] / member["member"]
        capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
        assert golden_store.load_golden(directory)["outputs"] == scalar_outputs(
            capsule, member["graph_variant"]["parameters"]
        )


def test_member_budget_cannot_issue_a_half_pair(independent):
    support(independent)
    selected = budgets.select(independent, [family()])
    selected["budget"]["max_members"] = 1
    generation_fixtures.write(independent["component_coverage"], selected)
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **independent)
    report = budgets.receipt(independent)
    assert report["obligations"][0]["state"] == "unavailable"
    assert not report["obligations"][0]["members"]
    assert report["graph_relations"]["rows"][0]["state"] == "unavailable"


def test_unknown_numeric_semantics_keep_both_required_graphs_unavailable(independent, monkeypatch):
    support(independent, overflow="unknown")
    budgets.select(independent, [family()])
    monkeypatch.setattr(component_program, "build", lambda *a, **k: pytest.fail("unknown graph reached builder"))
    monkeypatch.setattr(component_numerics, "evaluate", lambda *a, **k: pytest.fail("unknown graph evaluated"))
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **independent)
    report = budgets.receipt(independent)
    members = report["obligations"][0]["members"]
    assert len(members) == 2 and all(member["state"] == "unavailable" for member in members)
    assert all("requires explicit bounded_exact or modular_wrap" in member["reason"] for member in members)


def test_generated_operations_need_complete_selected_semantic_owners(independent):
    support(independent)
    row = family()
    row["operations"] = ["contraction", "movement"]
    budgets.select(independent, [row])
    with pytest.raises(ValueError, match="graph family lacks a selected semantic owner for add"):
        generation.generate_target("fixture", **independent)


@pytest.mark.parametrize("change", ["representation", "authored_program", "legacy_budget", "frontend"])
def test_family_schema_refuses_partial_or_unbudgeted_sources(independent, change):
    support(independent)
    row = family()
    if change == "representation":
        row["base"]["graph_family"]["representations"] = ["logical_epochs"]
    elif change == "authored_program":
        row["base"]["program"] = budgets.program(2, 3, 2)
    elif change == "frontend":
        row["frontend"] = "pytorch"
    selected = budgets.select(independent, [row])
    if change == "legacy_budget":
        selected["schema"] = "merlin.component_coverage_plan.v1"
        selected.pop("execution_budget")
        generation_fixtures.write(independent["component_coverage"], selected)
    with pytest.raises(ValueError, match="graph family"):
        generation.generate_target("fixture", **independent)
