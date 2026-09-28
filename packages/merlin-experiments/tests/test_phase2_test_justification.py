"""Generated Phase 2 rationale is complete, byte-bound and never a measured claim."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import corpus as P


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(C.canonical_json(document))


@pytest.fixture
def generated(tmp_path):
    root = tmp_path / "generated"
    public = root / "public"
    public.mkdir(parents=True)
    performance = {
        "family": "PAIR",
        "level": "L1_tile",
        "lever": "shape",
        "member_class": "OBJECTIVE",
        "claim": "DIFFERENTIAL",
        "gate": {"traits": ["structural_pipeline_depth"], "instrument": "cycle_count"},
        "comparand": {"kind": "comparison_group", "against": "matched_peer", "demand_equal": ["M", "N"]},
        "falsifier": {
            "observation": "paired_cycle_delta",
            "fires_when": "no_delta",
            "negative_control": "same_program",
        },
        "acceptance": {
            "evidence": {
                "correctness_simulator": "spike",
                "correctness_tier": "L2",
                "timing_simulator": "gsim",
                "timing_tier": "L3",
            },
            "replicates": {"exact_count": 2},
            "band": {"kind": "measured_replicate_dispersion"},
        },
        "shape_geometry": {"geometry_class": "squareish_gemm", "in_census": True, "census_mac_fraction": 0.2},
    }
    for name, role in (("left", "candidate"), ("right", "reference")):
        _write(
            root / "_perf" / name / "capsule.yaml",
            {
                "name": name,
                "label": "dev",
                "source_role": "derived_sweep",
                "required_oracle_tiers": ["L2", "L3"],
                "performance": performance,
                "comparison_group": {"name": "matched", "role": role},
                "software_screen": {"status": "unknown"},
            },
        )
    facts = {
        "target": "fixture",
        "sha256": "a" * 64,
        "raw_facts_sha256": "b" * 64,
        "traits": {
            "structural_pipeline_depth": {"satisfied": True, "tier": "rtl_facts", "evidence": "pipeline observed"}
        },
        "execution_capabilities": {},
    }
    accounting = {
        "schema": "merlin.phase0.operation_accounting.v1",
        "scope": "selected captured applications",
        "selected_inventory": {"content_sha256": "c" * 64},
    }
    evidence_root = root / "_evidence"
    fact_path = evidence_root / "hardware/effective-views/performance-facts.json"
    accounting_path = evidence_root / "coverage/operation-accounting.json"
    _write(fact_path, facts)
    _write(accounting_path, accounting)
    _write(
        evidence_root / "evidence-manifest.json",
        {
            "target": "fixture",
            "status": "diagnostic",
            "performance_facts_sha256": facts["sha256"],
            "raw_facts_sha256": facts["raw_facts_sha256"],
            "artifacts": {
                "hardware/effective-views/performance-facts.json": {
                    "sha256": C.sha256_file(fact_path),
                    "size_bytes": fact_path.stat().st_size,
                },
                "coverage/operation-accounting.json": {
                    "sha256": C.sha256_file(accounting_path),
                    "size_bytes": accounting_path.stat().st_size,
                },
            },
        },
    )
    (root / "MANIFEST.yaml").write_text(
        yaml.safe_dump(
            {
                "generated_by": "merlin/contract/capsules/generate_corpus.py",
                "generated": ["_perf/left", "_perf/right"],
                "hand_authored": [],
                "performance_generation": {
                    "fixture": {
                        "errors": [],
                        "phase": {"category": "_perf", "label": "dev", "included_in_functional_grade": False},
                        "shared_template": {"sha256": "d" * 64},
                        "facts": {
                            "target": "fixture",
                            "sha256": facts["sha256"],
                            "raw_facts_sha256": facts["raw_facts_sha256"],
                        },
                    }
                },
            }
        )
    )
    return SimpleNamespace(target="fixture", capsule_corpus=public, graded_roots=lambda: [public])


def test_current_generated_corpus_has_a_diagnostic_plan_for_every_member(generated, tmp_path):
    discovered = P.discover_performance_corpus(generated)
    receipt = discovered.test_justification
    assert receipt["status"] == "diagnostic_unmeasured"
    assert {row["capsule"] for row in receipt["members"]} == {"left", "right"}
    assert all(row["measurement"]["status"] == "unmeasured" for row in receipt["members"])
    assert all(row["matched_comparator"]["matching_status"] == "planned_unverified" for row in receipt["members"])
    assert all(row["negative_control"]["status"] == "planned_unverified" for row in receipt["members"])
    frozen = P.freeze_performance_corpus(discovered, tmp_path / "frozen")
    assert (frozen.root / "test_justification.json").is_file()
    P.load_frozen_performance_corpus(
        frozen.root,
        manifest_sha256=frozen.manifest_sha256,
        capsules_sha256=frozen.capsules_sha256,
        expected_target="fixture",
    )


@pytest.mark.parametrize("mutation", ["comparator", "negative_control", "source_hash"])
def test_freeze_refuses_changed_claim_or_evidence(generated, tmp_path, mutation):
    discovered = P.discover_performance_corpus(generated)
    root = generated.capsule_corpus.parent
    if mutation in {"comparator", "negative_control"}:
        path = root / "_perf/left/capsule.yaml"
        descriptor = json.loads(path.read_text())
        if mutation == "comparator":
            descriptor["performance"]["comparand"]["against"] = "changed"
        else:
            descriptor["performance"]["falsifier"]["negative_control"] = "changed"
        _write(path, descriptor)
    else:
        path = root / "_evidence/hardware/effective-views/performance-facts.json"
        path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(C.StageGateError, match="changed"):
        P.freeze_performance_corpus(discovered, tmp_path / "frozen")


def test_discovery_refuses_missing_frozen_evidence(generated):
    path = generated.capsule_corpus.parent / "_evidence/evidence-manifest.json"
    path.unlink()
    with pytest.raises(C.StageGateError, match="absent"):
        P.discover_performance_corpus(generated)
