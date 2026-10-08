"""A selected capture can shape a performance law without becoming a model claim."""

from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0.sweeps import expand_sweeps
from merlin_experiments.phase0.writer import _carry_declared_blocks

from merlin.common.paths import repo_root
from merlin.targetgen.corpus_spec import CorpusBinding, build


def _fixture():
    profile = yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_text())
    sweep = next(row for row in profile["sweeps"] if row["id"] == "PK")
    binding = CorpusBinding(
        target="gemmini",
        tile_dim=16,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0", "L1", "L2", "L3"],
        compare="exact_int",
        classes_for=lambda **_: ["CONTRACTION"],
    )
    roster = {"decoder": "1" * 64, "cnn": "2" * 64}
    requirement = {
        "target": "gemmini",
        "application_demands": {
            "full_inventory_sha256": "a" * 64,
            "applications": {name: {"capture_sha256": digest} for name, digest in roster.items()},
        },
    }

    def row(m, k, n, count, *, operation="aten._int_mm.default"):
        return {
            "operation": operation,
            "independent_compute_demand": True,
            "semantic_family": "contraction",
            "operand_format": "int8",
            "contraction_shape": {"M": m, "K": k, "N": n},
            "macs": {"total": m * k * n * count, "per_occurrence": m * k * n},
            "count": count,
        }

    basis = {
        "schema": "merlin.phase0.performance_basis.v1",
        "target": "gemmini",
        "selected_inventory": {"content_sha256": "a" * 64, "declared_roster_matches": True},
        "sources": {"selected_inventory_sha256": "a" * 64},
        "applications": {
            "decoder": {
                "capture_sha256": roster["decoder"],
                "rows": [
                    row(8, 32, 64, 3),
                    row(8, 16, 64, 1),
                    row(8, 16, 16, 1),
                ],
            },
            "cnn": {"capture_sha256": roster["cnn"], "rows": [row(256, 72, 8, 2)]},
        },
    }
    traits = {
        "traits": {
            "structural_pipeline_depth": {
                "satisfied": True,
                "tier": "fixture",
                "evidence": "known",
                "missing": [],
            }
        }
    }
    return sweep, binding, requirement, basis, traits


def test_capture_shape_adds_minimal_complete_law_cohorts_with_provenance() -> None:
    sweep, binding, requirement, basis, traits = _fixture()
    profile = {
        "capsules": [],
        "sweeps": [sweep],
        "_performance_oracles": {"L2": "spike", "L3": "firesim"},
    }
    skipped, errors = [], []
    kwargs = {
        "trait_facts": traits,
        "selected_requirement": requirement,
        "requirement_sha256": "b" * 64,
        "selected_performance_basis": basis,
        "performance_basis_sha256": "c" * 64,
    }
    entries = expand_sweeps(profile, binding, skipped=skipped, errors=errors, **kwargs)
    assert not errors
    selected = [entry for entry in entries if entry["performance"]["family"].startswith("PK_capture_")]
    assert len(selected) == 8
    assert {(entry["M"], entry["K"], entry["N"]) for entry in selected} == {
        (8, k, n) for k in (16, 32, 64, 128) for n in (16, 64)
    }
    for entry in selected:
        provenance = entry["performance"]["requirement_basis"]
        assert {row["capture_sha256"] for row in provenance["source_witnesses"]} == {"1" * 64}
        assert provenance["observed_k"] == ([16, 32] if entry["N"] == 64 else [16])
        assert provenance["performance_basis_sha256"] == "c" * 64
        assert provenance["selected_inventory_sha256"] == "a" * 64
        assert provenance["source_match"] == "capture_shape_candidate"
        descriptor, _ = build(entry, binding)
        assert descriptor["operation"]["op"] == "matmul"
        assert _carry_declared_blocks(entry, descriptor)
        assert descriptor["performance"]["requirement_basis"] == provenance
    assert len(skipped) == 1
    reasons = {row["reason"] for row in skipped[0]["remainder"]}
    assert "observed K is outside the unchanged four-point law grid" in reasons
    assert len(skipped[0]["remainder"]) == 1
    assert entries == expand_sweeps(profile, binding, skipped=[], errors=[], **kwargs)


def test_capture_shape_declared_cap_reports_every_deferred_source() -> None:
    sweep, binding, requirement, basis, traits = _fixture()
    sweep["capture_shape_pattern"]["max_cohorts"] = 1
    skipped = []
    entries = expand_sweeps(
        {"capsules": [], "sweeps": [sweep], "_performance_oracles": {"L2": "spike", "L3": "firesim"}},
        binding,
        trait_facts=traits,
        skipped=skipped,
        selected_requirement=requirement,
        requirement_sha256="b" * 64,
        selected_performance_basis=basis,
        performance_basis_sha256="c" * 64,
    )
    selected = [entry for entry in entries if entry["performance"]["family"].startswith("PK_capture_")]
    assert len(selected) == 4
    assert {entry["N"] for entry in selected} == {64}
    reasons = [row["reason"] for row in skipped[0]["remainder"]]
    assert reasons.count("deferred by shared-template max_cohorts=1") == 1


def test_capture_shape_refuses_mixed_requirement_inventory() -> None:
    sweep, binding, requirement, basis, traits = _fixture()
    requirement = copy.deepcopy(requirement)
    requirement["application_demands"]["full_inventory_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="different application inventories"):
        expand_sweeps(
            {"capsules": [], "sweeps": [sweep]},
            binding,
            trait_facts=traits,
            selected_requirement=requirement,
            requirement_sha256="b" * 64,
            selected_performance_basis=basis,
            performance_basis_sha256="c" * 64,
        )


def test_residency_ladder_anchors_exact_development_tail_and_crosses_selected_store(monkeypatch) -> None:
    from merlin.targetgen import address_space as AS

    sweep, binding, requirement, basis, _ = _fixture()
    profile = yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_text())
    sweep = next(row for row in profile["sweeps"] if row["id"] == "PR")
    basis["sources"]["raw_rtl_facts_sha256"] = "d" * 64

    class Store:
        name = "selected-operand-store"

        def working_set_rows(self, shape, dtype):
            elements = 1
            for extent in shape:
                elements *= extent
            return (elements + 7) // 8

    store = Store()
    monkeypatch.setattr(AS, "derive_address_space", lambda target, *, facts: object())
    monkeypatch.setattr(
        AS, "operand_store", lambda space, *, dtype: SimpleNamespace(store=store, capacity_rows=lambda _: 8192)
    )
    selected = SimpleNamespace(
        target="gemmini",
        refreshed_facts={"selected": "rtl"},
        contract={},
        raw_facts_sha256="d" * 64,
        software_spec={
            "status": "reviewed",
            "numerical_semantics": {
                "accumulator_dtype": "i32",
                "internal_arithmetic": {
                    "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
                    "signed_operand_bits": 8,
                    "mac_result_bits": 20,
                    "mac_result_overflow": "wrap_to_20_bits",
                },
            },
        },
    )
    skipped = []
    entries = expand_sweeps(
        {"sweeps": [sweep], "_performance_oracles": {"L2": "spike", "L3": "firesim"}},
        binding,
        evidence=selected,
        trait_facts={
            "traits": {
                name: {"satisfied": True, "tier": "rtl_facts", "evidence": "selected", "missing": []}
                for name in ("managed_scratchpad", "structural_pipeline_depth")
            }
        },
        selected_requirement=requirement,
        requirement_sha256="b" * 64,
        selected_performance_basis=basis,
        performance_basis_sha256="c" * 64,
        skipped=skipped,
    )
    assert len(entries) >= 6
    assert {(entry["M"], entry["N"]) for entry in entries} == {(256, 8)}
    provenance = entries[0]["performance"]["requirement_basis"]
    assert provenance["selected_inventory_sha256"] == "a" * 64
    assert provenance["raw_rtl_facts_sha256"] == "d" * 64
    assert provenance["selected_store"] == {"name": "selected-operand-store", "capacity_rows": 8192}
    assert provenance["source_match"] == "capture_shape_candidate"
    assert provenance["numeric_obligation"]["status"] == "requires_split_k"
    assert provenance["numeric_obligation"]["max_tile_aligned_primitive_k"] == 16
    assert provenance["source_witnesses"][0]["application"] == "cnn"
    assert provenance["boundary_pairs"]
    for pair in provenance["boundary_pairs"]:
        assert pair["below"]["K"] + binding.tile_dim == pair["above"]["K"]
        threshold = pair["capacity_rows"]
        if pair["predicate"] == "twice_live_rows_exceed_capacity":
            assert 2 * pair["below"]["rows"] <= threshold < 2 * pair["above"]["rows"]
        else:
            assert pair["below"]["rows"] <= threshold < pair["above"]["rows"]
    assert all(entry["performance"]["family"] == "PR" for entry in entries)
    assert entries == expand_sweeps(
        {"sweeps": [sweep], "_performance_oracles": {"L2": "spike", "L3": "firesim"}},
        binding,
        evidence=selected,
        trait_facts={
            "traits": {
                name: {"satisfied": True, "tier": "rtl_facts", "evidence": "selected", "missing": []}
                for name in ("managed_scratchpad", "structural_pipeline_depth")
            }
        },
        selected_requirement=requirement,
        requirement_sha256="b" * 64,
        selected_performance_basis=basis,
        performance_basis_sha256="c" * 64,
        skipped=[],
    )


def test_residency_ladder_refuses_without_selected_store(monkeypatch) -> None:
    from merlin.targetgen import address_space as AS

    _, binding, requirement, basis, traits = _fixture()
    basis["sources"]["raw_rtl_facts_sha256"] = "d" * 64
    profile = yaml.safe_load((repo_root() / "experiments/templates/phase0/performance.yaml").read_text())
    sweep = next(row for row in profile["sweeps"] if row["id"] == "PR")
    monkeypatch.setattr(AS, "derive_address_space", lambda target, *, facts: object())
    monkeypatch.setattr(
        AS, "operand_store", lambda space, *, dtype: SimpleNamespace(store=None, capacity_rows=lambda _: None)
    )
    skipped = []
    entries = expand_sweeps(
        {"sweeps": [sweep]},
        binding,
        evidence=SimpleNamespace(target="gemmini", refreshed_facts={}, contract={}, raw_facts_sha256="d" * 64),
        trait_facts=traits,
        selected_requirement=requirement,
        requirement_sha256="b" * 64,
        selected_performance_basis=basis,
        performance_basis_sha256="c" * 64,
        skipped=skipped,
    )
    assert entries == []
    assert skipped[0]["family"] == "PR"
    assert "sized operand store" in skipped[0]["reason"]
