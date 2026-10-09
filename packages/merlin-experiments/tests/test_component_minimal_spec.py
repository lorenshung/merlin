"""Campaign objectives have an explicit owner outside the minimal software spec."""

import copy

import pytest
import test_component_coverage as coverage_fixtures
import test_component_generation as generation_fixtures
import yaml
from merlin_experiments.phase0 import component_coverage, generation

from merlin.targetgen.golden_store import load_golden

independent = generation_fixtures.independent


def recipe_objective(options):
    software = yaml.safe_load(options["software_spec"].read_bytes())
    declaration = software.pop("component_performance")
    generation_fixtures.write(options["software_spec"], software)
    recipe = yaml.safe_load(options["recipe"].read_bytes())
    recipe["component_performance"] = declaration
    generation_fixtures.write(options["recipe"], recipe)
    return recipe


@pytest.mark.parametrize("tile", [2, 3])
def test_minimal_semantic_spec_and_recipe_objective_generate_bound_cohorts(independent, tile):
    coverage_fixtures.update_hardware(independent, tile=tile)
    recipe_objective(independent)
    software = yaml.safe_load(independent["software_spec"].read_bytes())
    assert set(software) == {"schema", "target", "status", "numerical_semantics", "operations"}
    coverage_fixtures.plan_for(independent, [coverage_fixtures.movement("guard", "functional_guard")])
    written = generation.generate_target("fixture", **independent)
    report = component_coverage.verify_report(independent["output_root"])
    identity = report["generation_identity"]
    assert identity["declaration_source"] == {"kind": "recipe", "sha256": identity["recipe"]["sha256"]}
    public = yaml.safe_load((independent["output_root"] / "MANIFEST.yaml").read_bytes())
    assert (
        public["performance_generation"]["fixture"]["component_generation"]["software_spec_sha256"]
        == report["software_spec_sha256"]
    )
    perf = [path for path in written if path.parent.name == "_perf"]
    assert len(perf) == 4
    for path in perf:
        capsule = yaml.safe_load((path / "capsule.yaml").read_bytes())
        objective = capsule["performance"]["objective"]
        assert any(item.startswith("component-declaration-source:recipe:") for item in objective["provenance"])
        assert objective["metric"] == "complete_component_cycles" and load_golden(path)["outputs"]


@pytest.mark.parametrize("change", ["duplicate_owner", "unreviewed", "different_hardware"])
def test_recipe_objective_cannot_bypass_exact_review_and_selected_hardware(independent, change):
    recipe = recipe_objective(independent)
    if change == "duplicate_owner":
        software = yaml.safe_load(independent["software_spec"].read_bytes())
        software["component_performance"] = copy.deepcopy(recipe["component_performance"])
        generation_fixtures.write(independent["software_spec"], software)
        reason = "one explicit owner"
    elif change == "unreviewed":
        recipe["component_performance"]["status"] = "unreviewed"
        reason = "explicit reviewed component_performance"
    else:
        recipe["component_performance"]["hardware"]["raw_facts_sha256"] = "0" * 64
        reason = "differs from selected hardware"
    generation_fixtures.write(independent["recipe"], recipe)
    with pytest.raises(ValueError, match=reason):
        generation.generate_target("fixture", **independent)
