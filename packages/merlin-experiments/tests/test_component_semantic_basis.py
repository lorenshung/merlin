"""Example semantics use frozen original graphs, never their shapes or counts."""

import hashlib
import json
import os
import subprocess

import pytest
import test_component_coverage as coverage_fixtures
import test_component_generation as generation_fixtures
import test_component_minimal_spec as minimal_fixtures
import test_phase0_freeze as freeze_fixtures
import yaml
from merlin_experiments.phase0 import component_coverage, generation
from merlin_experiments.phase0.component_semantic_basis import PROVENANCE, SCHEMA, ComponentSemanticBasis

from merlin.targetgen.frontend_trace import original_operation_semantics

independent = generation_fixtures.independent


def select_basis(recipe_path):
    graph = {
        "schema": "m2m.frontend_graph.v1",
        "stage": "original",
        "status": "complete",
        "call_count": 1,
        "by_target": {"aten.clone.default": 1},
        "nodes": [
            {
                "id": "input",
                "op": "placeholder",
                "target": "x",
                "results": [{"id": "x", "shape": [43, 79], "dtype": "int8"}],
            },
            {
                "id": "clone",
                "op": "call_function",
                "target": "aten.clone.default",
                "results": [{"id": "copy", "shape": [43, 79], "dtype": "int8"}],
            },
        ],
        "edges": [
            {
                "producer_node_id": "input",
                "consumer_node_id": "clone",
                "producer_value_id": "x",
                "shape": [43, 79],
                "dtype": "int8",
            }
        ],
    }
    graph["sha256"] = component_coverage.digest(graph)
    trace = {"schema": "m2m.frontend_trace.v1", "graphs": {"original": graph}}
    source = generation_fixtures.write(recipe_path.with_name("example-graph.json"), trace)
    roster = {
        "schema": SCHEMA,
        "status": "reviewed",
        "provenance": dict(PROVENANCE),
        "members": [
            {
                "id": "independent-copy",
                "kind": "model2mlir_frontend_trace",
                "path": source.name,
                "sha256": hashlib.sha256(source.read_bytes()).hexdigest(),
                "schema": "m2m.frontend_trace.v1",
                "operation_semantics": ["aten.clone.default"],
                "effect_semantics": [],
            }
        ],
    }
    path = generation_fixtures.write(recipe_path.with_name("semantic-basis.json"), roster)
    recipe = yaml.safe_load(recipe_path.read_bytes())
    recipe["semantic_basis"] = {"path": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    generation_fixtures.write(recipe_path, recipe)
    return recipe, path, roster, source, trace


def repin(recipe_path, recipe, roster_path, roster):
    generation_fixtures.write(roster_path, roster)
    recipe["semantic_basis"]["sha256"] = hashlib.sha256(roster_path.read_bytes()).hexdigest()
    generation_fixtures.write(recipe_path, recipe)


def linked_plan(options, basis):
    row = coverage_fixtures.movement("copy-guard", "functional_guard")
    row["semantic_basis"] = [
        {
            "member": "independent-copy",
            "operations": [{"source": "aten.clone.default", "owner": "movement"}],
            "effects": [],
        }
    ]
    plan = coverage_fixtures.plan_for(options, [row])
    plan["semantic_basis_sha256"] = basis.source.sha256
    generation_fixtures.write(options["component_coverage"], plan)
    return plan


@pytest.mark.parametrize("tile", [2, 3])
def test_normal_generation_consumes_only_reviewed_semantics_and_binds_all_source_bytes(independent, tile):
    coverage_fixtures.update_hardware(independent, tile=tile)
    minimal_fixtures.recipe_objective(independent)
    _, roster_path, _, source, _ = select_basis(independent["recipe"])
    basis = ComponentSemanticBasis.from_recipe(independent["recipe"])
    linked_plan(independent, basis)
    generation.generate_target("fixture", **independent)
    report = component_coverage.verify_report(independent["output_root"])
    assert report["generation_identity"]["semantic_basis"] == basis.reviewed_semantics()
    assert report["semantic_basis_sources"] == basis.sources()
    public = (independent["output_root"] / "MANIFEST.yaml").read_text()
    assert source.name not in public and str(roster_path) not in public
    assert "43" not in json.dumps(basis.reviewed_semantics())
    assert "79" not in json.dumps(basis.reviewed_semantics())
    assert "call_count" not in public and "by_target" not in public
    actual = basis.semantics()
    actual[0]["operation_semantics"].append("authored.op")
    assert basis.semantics()[0]["operation_semantics"] == ["aten.clone.default"]
    source.write_bytes(source.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="generation source changed"):
        component_coverage.verify_report(independent["output_root"])


@pytest.mark.parametrize(
    "change",
    [
        "unreviewed",
        "validation",
        "private",
        "history",
        "after_authoring",
        "operation",
        "graph_digest",
        "source_drift",
        "graph_provenance",
        "weight",
    ],
)
def test_basis_refuses_unreviewed_or_drifting_source_semantics(independent, change):
    recipe, path, roster, source, trace = select_basis(independent["recipe"])
    reason = "closed reviewed"
    if change == "unreviewed":
        roster["status"] = "unreviewed"
    elif change in {"validation", "history"}:
        roster["provenance"]["role"] = change
    elif change == "private":
        roster["provenance"]["visibility"] = "private"
    elif change == "after_authoring":
        roster["provenance"]["selection"] = "after_authoring"
    elif change == "operation":
        roster["members"][0]["operation_semantics"] = ["aten.matmul.default"]
        reason = "roster disagrees"
    elif change == "graph_digest":
        trace["graphs"]["original"]["nodes"][1]["target"] = "aten.matmul.default"
        generation_fixtures.write(source, trace)
        roster["members"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        reason = "graph content SHA256"
    elif change == "source_drift":
        source.write_bytes(source.read_bytes() + b"\n")
        reason = "source bytes changed"
    elif change == "graph_provenance":
        trace["provenance"] = {**PROVENANCE, "role": "validation"}
        generation_fixtures.write(source, trace)
        roster["members"][0]["sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
        reason = "different source provenance"
    else:
        roster["members"][0]["ranking_weight"] = 7
        reason = "closed source"
    repin(independent["recipe"], recipe, path, roster)
    with pytest.raises(ValueError, match=reason):
        generation.generate_target("fixture", **independent)


@pytest.mark.parametrize("change", ["pin", "member", "source", "owner", "missing", "unmapped_owner"])
def test_obligation_requires_exact_selected_semantic_correspondences(independent, change):
    select_basis(independent["recipe"])
    basis = ComponentSemanticBasis.from_recipe(independent["recipe"])
    plan = linked_plan(independent, basis)
    reason = "mapping differs"
    if change == "pin":
        plan["semantic_basis_sha256"] = "0" * 64
        reason = "basis differs"
    elif change == "missing":
        plan["obligations"][0]["semantic_basis"] = []
        reason = "explicit obligation"
    elif change == "member":
        plan["obligations"][0]["semantic_basis"][0]["member"] = "different"
        reason = "member is unknown"
    elif change == "unmapped_owner":
        plan["obligations"][0]["operations"].append("contraction")
        reason = "without a selected source correspondence"
    else:
        plan["obligations"][0]["semantic_basis"][0]["operations"][0][change] = "unknown"
    generation_fixtures.write(independent["component_coverage"], plan)
    with pytest.raises(ValueError, match=reason):
        generation.generate_target("fixture", **independent)


def test_graph_operation_counts_are_checked_but_not_transferred(independent):
    _, _, _, _, trace = select_basis(independent["recipe"])
    snapshot = trace["graphs"]["original"]
    snapshot["by_target"]["aten.clone.default"] = 2
    snapshot["sha256"] = component_coverage.digest({key: value for key, value in snapshot.items() if key != "sha256"})
    with pytest.raises(ValueError, match="operation roster disagrees"):
        original_operation_semantics(trace)


@pytest.mark.parametrize("change", ["none", "different_kind", "unknown_source", "unavailable"])
def test_reviewed_effect_links_still_need_concrete_generated_witnesses(independent, change):
    recipe, path, roster, _, _ = select_basis(independent["recipe"])
    roster["members"][0]["effect_semantics"] = [
        {
            "id": "published-copy",
            "kind": "output_publication",
            "basis": "complete copied graph result",
            "operation_semantics": ["aten.clone.default"],
        }
    ]
    repin(independent["recipe"], recipe, path, roster)
    basis = ComponentSemanticBasis.from_recipe(independent["recipe"])
    plan = linked_plan(independent, basis)
    plan["effects"] = [
        {
            "id": "publication",
            "status": "reviewed",
            "kind": "output_publication",
            "basis": "the ordinary complete output writer",
        }
    ]
    row = plan["obligations"][0]
    row["effects"] = ["publication"]
    row["semantic_basis"][0]["effects"] = [{"source": "published-copy", "owner": "publication"}]
    if change == "different_kind":
        plan["effects"][0]["kind"] = "physical_alias"
        reason = "effect kind differs"
    elif change == "unknown_source":
        row["semantic_basis"][0]["effects"][0]["source"] = "not-reviewed"
        reason = "mapping differs"
    elif change == "unavailable":
        roster["members"][0]["effect_semantics"][0]["kind"] = "physical_alias"
        repin(independent["recipe"], recipe, path, roster)
        plan["semantic_basis_sha256"] = recipe["semantic_basis"]["sha256"]
        plan["effects"][0]["kind"] = "physical_alias"
    generation_fixtures.write(independent["component_coverage"], plan)
    if change == "none":
        generation.generate_target("fixture", **independent)
        report = component_coverage.verify_report(independent["output_root"])
        assert report["obligations"][0]["state"] == "generated"
    elif change == "unavailable":
        with pytest.raises(RuntimeError, match="component coverage"):
            generation.generate_target("fixture", **independent)
        report = json.loads((independent["output_root"] / "_evidence/coverage/component-coverage.json").read_bytes())
        assert report["status"] == "incomplete" and report["obligations"][0]["state"] == "unavailable"
    else:
        with pytest.raises(ValueError, match=reason):
            generation.generate_target("fixture", **independent)


def test_real_guarded_freeze_preserves_full_graph_bytes_after_original_sources_deleted(tmp_path):
    fixture = freeze_fixtures._fixture(tmp_path, "path")
    recipe_path = fixture["profiles"] / "fixture-device.yaml"
    _, _, _, graph_path, _ = select_basis(recipe_path)
    expected = graph_path.read_bytes()
    entrypoint = fixture["installed"] / "merlin_experiments/phase0/__main__.py"
    probe = (
        "import sys, os\nfrom pathlib import Path\n"
        "from merlin_experiments.phase0.component_semantic_basis import ComponentSemanticBasis\n"
        "basis = ComponentSemanticBasis.from_recipe(sys.argv[sys.argv.index('--recipe')+1])\n"
        "assert basis.semantics()[0]['operation_semantics'] == ['aten.clone.default']\n"
        "root = Path(os.environ['MERLIN_REPO_ROOT'])\n"
        "assert all(Path(row['path']).is_relative_to(root) for row in basis.sources())\n"
    )
    entrypoint.write_text(
        entrypoint.read_text().replace(
            "from __future__ import annotations\n", "from __future__ import annotations\n\n" + probe
        )
    )
    driver = tmp_path / "freeze-and-delete.py"
    driver.write_text(
        "import os, shutil, subprocess\nfrom pathlib import Path\n"
        "from merlin_experiments.runner import resolve_plan, run\n"
        "from merlin_experiments.spec import load_spec\n"
        "original = subprocess.Popen\n"
        "def launch(argv, *args, **kwargs):\n"
        "    if '--execute' in argv: shutil.rmtree(os.environ['ORIGINAL_PROFILES'])\n"
        "    return original(argv, *args, **kwargs)\n"
        "subprocess.Popen = launch\n"
        "raise SystemExit(run(resolve_plan(load_spec(os.environ['DEFINITION']), phase='0', "
        "run_dir=Path(os.environ['RUN_DIR']))))\n"
    )
    env = dict(
        fixture["environment"],
        DEFINITION=str(fixture["definition"]),
        RUN_DIR=str(fixture["run"]),
        ORIGINAL_PROFILES=str(fixture["profiles"]),
    )
    result = subprocess.run([os.sys.executable, str(driver)], env=env, text=True, capture_output=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    sources = list((fixture["run"] / "phase0/private/source").rglob(graph_path.name))
    assert len(sources) == 1 and sources[0].read_bytes() == expected
