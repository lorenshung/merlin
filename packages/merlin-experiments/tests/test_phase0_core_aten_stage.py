"""Receipt-bound retained-call selection and private release assembly."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.corpus.preparation import assemble
from merlin_experiments.phase0 import core_aten_stage as stage
from merlin_experiments.spec import SpecError


def fixture_recipe(tmp_path):
    (tmp_path / "descriptor.yaml").write_text("target: synthetic\n")
    cohorts = {}
    for role in stage.COHORTS:
        cases = tmp_path / f"{role}.json"
        case = {"overload": "aten.add.Tensor", "case_id": role}
        cases.write_text(json.dumps({"cases": [case]}))
        from merlin.targetgen.core_aten_capture import case_capture_name

        captures = tmp_path / f"{role}-captures"
        source = captures / case_capture_name(case)
        source.mkdir(parents=True)
        (source / "capture.json").write_text(
            json.dumps(
                {
                    "status": "captured_exact",
                    "overload": case["overload"],
                    "capture_meta": {"opaque": 0, "result_contract": {"results": []}},
                }
            )
        )
        (source / "capsule.linalg.mlir").write_text(role)
        cohorts[role] = [{"cases": str(cases), "captures": str(captures), "select": [role]}]
    recipe = tmp_path / "recipe.yaml"
    recipe.write_text(
        yaml.safe_dump(
            {
                "capsule_policy": "derived_only",
                "core_aten": {"overlay": str(tmp_path / "public.json"), "cohorts": cohorts},
            }
        )
    )
    return recipe


def fake_writer(corpus, captures, public, private, *, label, tiers):
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    rows = []
    for case in corpus["cases"]:
        name = case["case_id"]
        member, answer = public / name, private / name
        member.mkdir()
        answer.mkdir()
        (answer / "golden.yaml").write_text("host-only answer")
        binding = {"golden.yaml": hashlib.sha256((answer / "golden.yaml").read_bytes()).hexdigest()}
        (member / "capsule.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "label": label,
                    "kind": "model_slice",
                    "required_oracle_tiers": list(tiers),
                    "full_call_golden": binding,
                }
            )
        )
        (member / "capsule.interface.mlir").write_text(case["case_id"])
        (member / "inputs.npz").write_bytes(case["case_id"].encode())
        (member / "capsule.pytorch.py").write_text(case["case_id"])
        (member / "call.json").write_text(case["case_id"])
        rows.append({"id": name})
    manifest = {"capsules": rows}
    (public / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def install_writer(monkeypatch, writer=fake_writer):
    monkeypatch.setitem(sys.modules, "merlin.targetgen.core_aten_capsules", SimpleNamespace(write_capsules=writer))


def test_selection_determinism_and_stale_refusal(tmp_path, monkeypatch):
    recipe = fixture_recipe(tmp_path)
    first = stage.derive(recipe, tmp_path / "derive1", target="synthetic", descriptor=tmp_path / "descriptor.yaml")
    second = stage.derive(recipe, tmp_path / "derive2", target="synthetic", descriptor=tmp_path / "descriptor.yaml")
    assert first["selection_sha256"] == second["selection_sha256"]
    assert (tmp_path / "derive1/selection.json").read_bytes() == (tmp_path / "derive2/selection.json").read_bytes()
    install_writer(monkeypatch)
    source = next((tmp_path / "hidden-captures").glob("*/capsule.linalg.mlir"))
    source.write_text("changed")
    with pytest.raises(ValueError, match="selection changed"):
        stage.generate(Path(first["recipe"]), tmp_path / "run/phase0/capsules", target="synthetic")
    assert not (tmp_path / "run/phase0/capsules").exists()


def test_complete_manifest_and_independent_private_copy(tmp_path, monkeypatch):
    recipe = fixture_recipe(tmp_path)
    result = stage.derive(recipe, tmp_path / "derive", target="synthetic", descriptor=tmp_path / "descriptor.yaml")
    install_writer(monkeypatch)
    generated = tmp_path / "run/phase0/capsules"
    stage.generate(Path(result["recipe"]), generated, target="synthetic")
    manifest = yaml.safe_load((generated / "MANIFEST.yaml").read_bytes())
    assert manifest["generated"] == ["host_guard/host_guard", "public/public"]
    assert manifest["held_out"]["n_generated"] == 1
    assert "hidden/hidden" not in (generated / "MANIFEST.yaml").read_text()
    assert not yaml.safe_load((generated / "host_guard/host_guard/capsule.yaml").read_bytes())["scored"]
    te = SimpleNamespace(
        capsule_corpus=tmp_path / "baseline/public", target="synthetic", descriptor=tmp_path / "descriptor.yaml"
    )
    destination = tmp_path / "release/corpus"
    assemble(te, generated, destination, generated_only=True)
    for role in stage.COHORTS:
        source = generated / "_private" / role / role / "golden.yaml"
        copy = destination / "_private" / role / role / "golden.yaml"
        assert source.read_bytes() == copy.read_bytes()
        assert source.stat().st_ino != copy.stat().st_ino
        assert copy.stat().st_mode & 0o077 == 0
    assert yaml.safe_load((destination / "MANIFEST.yaml").read_bytes())["held_out"]["n_generated"] == 1
    answer = generated / "_private/public/public/golden.yaml"
    answer.write_text("tampered")
    with pytest.raises(SpecError, match="differ"):
        assemble(te, generated, tmp_path / "refused", generated_only=True)


def test_writer_failure_never_promotes_partial_corpus(tmp_path, monkeypatch):
    recipe = fixture_recipe(tmp_path)
    result = stage.derive(recipe, tmp_path / "derive", target="synthetic", descriptor=tmp_path / "descriptor.yaml")

    def fail(*args, **kwargs):
        fake_writer(*args, **kwargs)
        raise ValueError("writer failed")

    install_writer(monkeypatch, fail)
    output = tmp_path / "run/phase0/capsules"
    with pytest.raises(ValueError, match="writer failed"):
        stage.generate(Path(result["recipe"]), output, target="synthetic")
    assert not output.exists()
    assert not list(output.parent.glob(".core-aten-*"))


def test_overlap_and_alias_inputs_refused(tmp_path):
    recipe = fixture_recipe(tmp_path)
    document = yaml.safe_load(recipe.read_bytes())
    document["core_aten"]["cohorts"]["hidden"] = document["core_aten"]["cohorts"]["public"]
    recipe.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="disjoint"):
        stage.selection(recipe)
    document["core_aten"]["overlay"] = str(tmp_path / "alias.json")
    (tmp_path / "alias.json").symlink_to(tmp_path / "public.json")
    recipe.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="aliases"):
        stage.selection(recipe)


def test_identical_executed_call_is_refused_across_cohorts(tmp_path, monkeypatch):
    recipe = fixture_recipe(tmp_path)
    result = stage.derive(recipe, tmp_path / "derive", target="synthetic", descriptor=tmp_path / "descriptor.yaml")

    def duplicate_writer(*args, **kwargs):
        manifest = fake_writer(*args, **kwargs)
        for row in manifest["capsules"]:
            root = args[2] / row["id"]
            for filename in ("capsule.interface.mlir", "inputs.npz", "capsule.pytorch.py", "call.json"):
                (root / filename).write_bytes(b"same executed call")
        return manifest

    install_writer(monkeypatch, duplicate_writer)
    output = tmp_path / "run/phase0/capsules"
    with pytest.raises(ValueError, match="pairs overlap"):
        stage.generate(Path(result["recipe"]), output, target="synthetic")
    assert not output.exists()


def test_output_bytes_and_private_lineage_are_deterministic(tmp_path, monkeypatch):
    from merlin_experiments.corpus.preparation import generation_lineage

    recipe = fixture_recipe(tmp_path)
    result = stage.derive(recipe, tmp_path / "derive", target="synthetic", descriptor=tmp_path / "descriptor.yaml")
    install_writer(monkeypatch)
    outputs = [tmp_path / name / "phase0/capsules" for name in ("one", "two")]
    for output in outputs:
        stage.generate(Path(result["recipe"]), output, target="synthetic")
    trees = [{p.relative_to(root): p.read_bytes() for p in root.rglob("*") if p.is_file()} for root in outputs]
    assert trees[0] == trees[1]
    lineage = generation_lineage({"target": "synthetic"}, outputs[0])
    assert lineage["status"] == "byte_bound_diagnostic"
    assert lineage["selection_sha256"] == result["selection_sha256"]
    (outputs[0] / "_private/stage.json").write_text("{}")
    with pytest.raises(SpecError, match="ledger differs"):
        generation_lineage({"target": "synthetic"}, outputs[0])


def test_unattested_legacy_capture_contract_is_refused(tmp_path):
    recipe = fixture_recipe(tmp_path)
    receipt_path = next((tmp_path / "public-captures").glob("*/capture.json"))
    capture = json.loads(receipt_path.read_bytes())
    capture["capture_meta"].pop("result_contract")
    receipt_path.write_text(json.dumps(capture))
    with pytest.raises(ValueError, match="result contract"):
        stage.derive(recipe, tmp_path / "derive", target="synthetic", descriptor=tmp_path / "descriptor.yaml")
    assert not (tmp_path / "derive").exists()


def test_selected_synthesis_binds_finite_conformance_without_execution_claim(tmp_path):
    from merlin_experiments.phase0.profiles import verify_selected_synthesis

    recipe = fixture_recipe(tmp_path)
    derived = stage.derive(recipe, tmp_path / "derived", target="synthetic", descriptor=tmp_path / "descriptor.yaml")
    status = verify_selected_synthesis(
        derived["synth_profile"],
        conformance_spec=derived["conformance_spec"],
        recipe=derived["recipe"],
        descriptor=tmp_path / "descriptor.yaml",
    )
    assert status["status"] == "verified"
    assert status["execution_coverage_status"] == "unverified"
    requirement = yaml.safe_load(Path(derived["conformance_spec"]).read_bytes())
    assert requirement["cohorts"]["hidden"]["count"] == 1
    assert not requirement["cohorts"]["host_guard"]["scored"]
    assert "hidden-captures" not in Path(derived["conformance_spec"]).read_text()


@pytest.mark.parametrize("changed", ["recipe", "conformance_spec", "descriptor", "receipt", "capture", "synthesis"])
def test_selected_synthesis_refuses_changed_selection(tmp_path, changed):
    from merlin_experiments.phase0.profiles import verify_selected_synthesis

    recipe = fixture_recipe(tmp_path)
    descriptor = tmp_path / "descriptor.yaml"
    derived = stage.derive(recipe, tmp_path / "derived", target="synthetic", descriptor=descriptor)
    if changed in ("recipe", "conformance_spec"):
        path = Path(derived[changed])
        path.write_text(path.read_text() + "\n# changed bytes\n")
    elif changed == "descriptor":
        descriptor.write_text("target: synthetic\nworkload_spec: {certification_floor: L3}\n")
    elif changed == "receipt":
        (tmp_path / "derived/selection.json").write_text("{}")
    elif changed == "capture":
        next((tmp_path / "hidden-captures").glob("*/capsule.linalg.mlir")).write_text("changed")
    else:
        path = Path(derived["synth_profile"])
        document = yaml.safe_load(path.read_bytes())
        document["core_aten_selection"]["selection_sha256"] = "0" * 64
        path.write_text(yaml.safe_dump(document))
    with pytest.raises(ValueError, match="stale selected synthesis|selection changed|synthesis differs"):
        verify_selected_synthesis(
            derived["synth_profile"],
            conformance_spec=derived["conformance_spec"],
            recipe=derived["recipe"],
            descriptor=descriptor,
        )
