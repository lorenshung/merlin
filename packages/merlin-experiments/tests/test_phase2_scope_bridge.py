"""Requirement-selected scope bridge through sweep admission and Phase 2 freeze."""

from __future__ import annotations

import copy
import hashlib
import sys
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import sweeps as SW
from merlin_experiments.phase0.software_screen import diagnostic_entry, screen_entry
from merlin_experiments.phase0.typed_scope import typed_required_instances
from merlin_experiments.phase0.writer import _write_capsule
from merlin_experiments.phase2 import corpus as P
from merlin_experiments.phase2.claims.affine import preflight_affine_claim

from merlin.common.paths import repo_root
from merlin.targetgen import capsule_golden as CG
from merlin.targetgen import conformance
from merlin.targetgen import corpus_spec as CS
from merlin.targetgen.application_graph import application_graph_inventory
from merlin.targetgen.software_spec import load_software_spec


def _binding() -> CS.CorpusBinding:
    return CS.CorpusBinding(
        target="fixture", tile_dim=4, operand_dtype="int8", accum_dtype="i32",
        integer=True, tiers=["L0", "L1", "L2", "L3"], compare="exact_int",
        classes_for=lambda **_: ["CONTRACTION"],
    )


def _facts() -> dict:
    fact = {"satisfied": True, "tier": "fixture", "evidence": "fixture proof", "missing": []}
    return {
        "traits": {"structural_pipeline_depth": fact},
        "execution_capabilities": {
            "whole_program_kernel_abi": fact,
            "warm_single_counter_region_cycles": fact,
        },
    }


def _sweep() -> dict:
    document = yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_text())
    return next(row for row in document["sweeps"] if row["id"] == "PN")


def test_scope_sw_screen_checks_emitted_regions_without_inventing_device_maps() -> None:
    entry = {
        "name": "scope", "kind": "model_slice", "source_role": "derived_sweep",
        "source_reference": "selected scope.required", "op": "scope_chain",
        "M": 4, "K": 8, "N": 4,
        "scope_families": ["movement", "contraction", "elementwise_map", "elementwise_map"],
        "operand_dtype": "int8", "accum_dtype": "i32",
    }
    capsule, _ = CS.build(entry, _binding())
    spec = load_software_spec(repo_root() / "examples/gemmini/target/software-spec.yaml")
    pending = screen_entry(spec, entry, defaults=spec["numerical_semantics"])
    assert pending["status"] == "unknown"  # no emitted region inventory yet
    observed = screen_entry(spec, entry, defaults=spec["numerical_semantics"], capsule=capsule)
    assert [row["op"] for row in observed["decisions"]] == ["transpose", "matmul", "add", "add"]
    assert [row["family"] for row in observed["decisions"]] == entry["scope_families"]
    assert observed["status"] == "unsupported"
    assert all(row["status"] == "unsupported" for row in observed["decisions"][2:])
    assert all("placement" in row["reason"] for row in observed["decisions"][2:])
    assert "scope_chain" not in observed["reason"]
    inconsistent = copy.deepcopy(capsule)
    inconsistent["operation"]["attributes"]["scope_region_ops"][2] = "matmul"
    assert screen_entry(spec, entry, capsule=inconsistent)["status"] == "unsupported"
    first = diagnostic_entry(entry, observed)
    again = diagnostic_entry(first, observed)
    assert again["source_reference"] == first["source_reference"]


def test_required_scope_instances_bind_exact_source_ops_types_and_edges(tmp_path) -> None:
    entry = {
        "name": "scope", "kind": "model_slice", "source_role": "derived_sweep",
        "source_reference": "source fixture", "op": "scope_chain", "M": 4, "K": 8, "N": 4,
        "scope_families": ["movement", "contraction", "elementwise_map"],
    }
    _, mlir = CS.build(entry, _binding())
    source = tmp_path / "model.mlir"
    source.write_text(mlir.replace('prov.op = "add"', 'prov.op = "unsupported_custom"'))
    graph = application_graph_inventory(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    signature = "movement -> contraction -> elementwise_map"
    scope = {"required": [{"signature": signature, "occurrences": 1, "length": 3}]}
    inventory = {"applications": {"fixture": {
        "capture_source_path": str(source), "capture_sha256": digest, "operation_graph": graph,
    }}}
    observed = typed_required_instances(scope, inventory)
    assert observed == typed_required_instances(scope, inventory)
    assert observed["status"] == "typed_source_only"
    assert len(observed["instances"]) == 1
    instance = observed["instances"][0]
    assert instance["capture_sha256"] == digest
    assert [row["source_op"] for row in instance["regions"]] == [
        "linalg.transpose", "matmul", "unsupported_custom"
    ]
    assert [row["semantic_family"] for row in instance["regions"]] == entry["scope_families"]
    assert [row["operation_id"] for row in instance["regions"]] == instance["operation_ids"]
    assert all(row["operation_id"].startswith(f"mlir:{digest}:") for row in instance["regions"])
    assert instance["regions"][1]["results"][0]["dtype"] == "i32"
    assert instance["regions"][2]["operands"][0]["dtype"] == "i32"
    assert len(instance["edges"]) == 2
    assert all(edge["dtype"] in {"i8", "i32"} for edge in instance["edges"])
    source.write_text(source.read_text() + "\n")
    with pytest.raises(ValueError, match="capture bytes changed"):
        typed_required_instances(scope, inventory)


def test_selected_scope_derives_four_priceable_members_and_missing_scope_skips(monkeypatch) -> None:
    monkeypatch.setattr(SW, "target_encodings", lambda *args, **kwargs: [])
    monkeypatch.setattr(SW, "_resolve_target_oracle_evidence", lambda performance, target: performance)
    signature = "movement -> contraction -> elementwise_map -> elementwise_map -> elementwise_map"
    requirement = {"scope": {"required": [{"signature": signature, "length": 5, "occurrences": 13}]}}
    digest = hashlib.sha256(yaml.safe_dump(requirement).encode()).hexdigest()
    profile = {"capsules": [], "sweeps": [_sweep()]}
    skipped = []
    entries = SW.expand_sweeps(
        profile, _binding(), trait_facts=_facts(), skipped=skipped,
        selected_requirement=requirement, requirement_sha256=digest,
    )
    assert not skipped and len(entries) == 4
    assert entries[0]["performance"]["family"].startswith("PN_")
    assert len({entry["performance"]["family"] for entry in entries}) == 1
    assert [entry["K"] for entry in entries] == [4, 8, 12, 16]
    assert all(entry["scope_families"] == signature.split(" -> ") for entry in entries)
    assert all(entry["performance"]["requirement_basis"] == {
        "sha256": digest, "axis": "scope.required", "pattern_family": "PN",
        "signature": signature, "occurrences": 13,
    } for entry in entries)
    assert all(entry["performance"]["member_class"] == "LAW" for entry in entries)
    descriptors = []
    for entry in entries:
        descriptor, _ = CS.build(entry, _binding())
        descriptor["performance"] = entry["performance"]
        descriptors.append(descriptor)
    assert preflight_affine_claim(descriptors)["status"] == "READY"
    altered = copy.deepcopy(descriptors)
    altered[-1]["operation"]["attributes"]["scope_signature"] = "different chain"
    assert preflight_affine_claim(altered)["status"] == "REFUSED"
    absent = SW.expand_sweeps(
        profile, _binding(), trait_facts=_facts(), skipped=skipped,
        selected_requirement={"scope": {"required": []}}, requirement_sha256=digest,
    )
    assert absent == []
    assert skipped[-1]["status"] == "skipped_inapplicable"


def test_each_supported_signature_gets_a_separate_cohort_and_over_cap_is_recorded(monkeypatch) -> None:
    monkeypatch.setattr(SW, "target_encodings", lambda *args, **kwargs: [])
    monkeypatch.setattr(SW, "_resolve_target_oracle_evidence", lambda performance, target: performance)
    signatures = [" -> ".join(["movement", "contraction"] + ["elementwise_map"] * count)
                  for count in (1, 3, 7)]
    requirement = {"scope": {"required": [
        {"signature": signature, "length": len(signature.split(" -> ")), "occurrences": index + 1}
        for index, signature in enumerate(signatures)
    ]}}
    skipped = []
    blocked = []
    entries = SW.expand_sweeps(
        {"capsules": [], "sweeps": [_sweep()]}, _binding(), trait_facts=_facts(),
        skipped=skipped, blocked_unimplemented=blocked,
        selected_requirement=requirement, requirement_sha256="frozen-digest",
    )
    families = {entry["performance"]["family"] for entry in entries}
    assert len(entries) == 8 and len(families) == 2
    assert {entry["performance"]["requirement_basis"]["signature"] for entry in entries} == set(signatures[:2])
    assert any(row["status"] == "blocked_unimplemented" and row["signature"] == signatures[2]
               for row in blocked)
    assert skipped == []


def test_scope_law_metric_follows_selected_l3_oracle(monkeypatch) -> None:
    from merlin.targetgen import oracle_policy, target_experiment

    monkeypatch.setattr(target_experiment, "load_capability_manifest", lambda target: SimpleNamespace(
        contract={"runner": {"tier_sim": {"L2": "reference_sim", "L3": "fallback_rtl"}}},
    ))
    monkeypatch.setitem(sys.modules, "merlin.targetgen.capsule_runner", None)
    monkeypatch.setattr(oracle_policy, "selected_l3_engine_report", lambda target: {
        "available": True, "engine": "selected_rtl",
    })
    performance = _sweep()["base"]["performance"]
    resolved = SW._resolve_target_oracle_evidence(performance, "fixture")
    acceptance = resolved["acceptance"]
    assert acceptance["evidence"]["correctness_simulator"] == "reference_sim"
    assert acceptance["evidence"]["timing_simulator"] == "selected_rtl"
    assert acceptance["fit"]["dependent_metric"] == "selected_rtl_L3_cycles"


def test_scope_law_refuses_generic_l3_label_when_engine_unavailable(monkeypatch) -> None:
    from merlin.targetgen import oracle_policy, target_experiment

    monkeypatch.setattr(target_experiment, "load_capability_manifest", lambda target: SimpleNamespace(
        contract={"runner": {"tier_sim": {"L2": "reference_sim", "L3": "elaborated_rtl"}}},
    ))
    monkeypatch.setitem(sys.modules, "merlin.targetgen.capsule_runner", None)
    monkeypatch.setattr(oracle_policy, "selected_l3_engine_report", lambda target: {
        "available": False, "reason": "no selected engine",
    })
    with pytest.raises(ValueError, match="does not resolve L3 to a concrete simulator"):
        SW._resolve_target_oracle_evidence(_sweep()["base"]["performance"], "fixture")


def test_scope_member_survives_phase2_discovery_and_freeze(tmp_path) -> None:
    root = tmp_path / "live"
    member = root / "_tuning" / "scope"
    member.mkdir(parents=True)
    cap, mlir = CS.build({
        "name": "scope", "kind": "model_slice", "source_role": "derived_sweep",
        "source_reference": "selected requirement fixture", "label": "dev", "op": "scope_chain",
        "M": 4, "K": 8, "N": 4,
        "scope_families": ["movement", "contraction", "elementwise_map"],
        "semantic": {"semantic_family": "contraction", "generalization_axis": "composition", "must_accelerate": True},
    }, _binding())
    cap["performance"] = {
        "family": "PN", "claim": "PREDICTS", "member_class": "LAW",
        "acceptance": _sweep()["base"]["performance"]["acceptance"],
        "requirement_basis": {"sha256": "fixture-digest", "axis": "scope.required",
                              "signature": cap["operation"]["attributes"]["scope_signature"]},
    }
    (member / "capsule.yaml").write_text(yaml.safe_dump(cap))
    (member / "capsule.interface.mlir").write_text(mlir)
    required = [{"signature": cap["operation"]["attributes"]["scope_signature"]}]
    gap = conformance._scope_gap(required, root / "_tuning", labels={"dev"})
    assert gap["n_covered"] == 1 and gap["uncovered"] == []
    public = root / "public"
    public.mkdir()
    (root / "MANIFEST.yaml").write_text(yaml.safe_dump({
        "generated": ["_tuning/scope"], "hand_authored": [],
        "performance_generation": {"fixture": {
            "errors": [], "phase": {"category": "_tuning", "label": "dev", "included_in_functional_grade": False},
        }},
    }))
    generated = SimpleNamespace(target="fixture", capsule_corpus=public, graded_roots=lambda: [public])
    discovered = P.discover_performance_corpus(generated)
    frozen = P.freeze_performance_corpus(discovered, tmp_path / "frozen")
    loaded = P.load_frozen_performance_corpus(
        frozen.root, manifest_sha256=frozen.manifest_sha256,
        capsules_sha256=frozen.capsules_sha256, expected_target="fixture",
    )
    P.verify_frozen_performance_corpus(loaded)
    assert loaded.capsules[0].descriptor["performance"]["requirement_basis"]["axis"] == "scope.required"


def test_phase0_writer_materializes_scope_program_and_independent_golden(tmp_path) -> None:
    entry = {
        "name": "scope", "cat": "_perf", "kind": "model_slice", "source": "direct",
        "source_role": "derived_sweep", "source_reference": "selected requirement fixture",
        "label": "dev", "op": "scope_chain", "M": 4, "K": 8, "N": 4,
        "scope_families": ["movement", "contraction", "elementwise_map"],
        "performance": {"family": "PN_fixture", "claim": "PREDICTS", "member_class": "LAW",
                        "acceptance": _sweep()["base"]["performance"]["acceptance"]},
    }
    written = _write_capsule(entry, _binding(), tmp_path)
    assert written is not None
    directory = tmp_path / "_perf" / "scope"
    descriptor = yaml.safe_load((directory / "capsule.yaml").read_text())
    golden = yaml.safe_load((directory / "golden.yaml").read_text())
    assert descriptor["linalg_mlir"] == "capsule.interface.mlir"
    assert descriptor["performance"]["family"] == "PN_fixture"
    assert golden["outputs"] == CG.golden(descriptor)
