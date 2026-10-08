"""Generated Phase 2 rationale is complete, byte-bound and never a measured claim."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2 import corpus as P


def _write(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(C.canonical_json(document))


def _rebind_artifact(root: Path, relative: str, document: dict) -> None:
    path = root / relative
    _write(path, document)
    manifest_path = root / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["artifacts"][relative] = {"sha256": C.sha256_file(path), "size_bytes": path.stat().st_size}
    _write(manifest_path, manifest)


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
        "shape_geometry": {
            "M": 8,
            "K": 32,
            "N": 96,
            "geometry_class": "wide_skinny",
            "in_census": True,
            "census_mac_fraction": 0.2,
        },
    }
    for name, role in (("left", "candidate"), ("right", "reference")):
        _write(
            root / "_perf" / name / "capsule.yaml",
            {
                "name": name,
                "label": "dev",
                "source_role": "derived_sweep",
                "required_oracle_tiers": ["L2", "L3"],
                "operation": {
                    "op": "matmul",
                    "attributes": {"lhs": "A0", "weight": "W", "out": "Y0", "output_dtype": "i32"},
                },
                "inputs": [
                    {"name": "A0", "role": "input", "shape": [8, 32], "dtype": "i8"},
                    {"name": "W", "role": "weight", "shape": [32, 96], "dtype": "i8"},
                ],
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
    software_spec = {"status": "selected", "operations": []}
    contract = {"compute_units": []}
    receipt = {"status": "verified_materialized", "errors": [], "receipt_sha256": "8" * 64}
    numerical_row = {"finite": True, "max_abs": 0.0, "max_rel": 0.0, "atol": 0.0, "rtol": 0.0}
    integerization = {
        "schema": "merlin.capture_integerization.v1",
        "status": "byte_bound_metadata",
        "capture_receipt_sha256": receipt["receipt_sha256"],
        "source_quantization": "int8_static_act_int8_weight",
        "metadata": {"sha256": "9" * 64, "bytes": 1},
        "capture": {"sha256": "7" * 64, "bytes": 1},
        "integerization_receipt": {
            "schema": "m2m.pt2e-integerize.v1",
            "quantized_contractions_seen": 1,
            "quantized_contractions_integerized": 1,
            "quantized_contractions_remaining": 0,
            "accumulator_bound_checked": True,
            "refusals": [],
            "exported_integer_mm_count": 1,
            "integer_mm_emitted": 1,
            "golden_agreement": {
                **numerical_row,
                "status": "passed",
                "samples": 1,
                "outputs": [{**numerical_row, "within_tolerance": True}],
            },
        },
    }
    selected_application = {
        "capture_sha256": "7" * 64,
        "capture_receipt": receipt,
        "capture_quantization": "int8_static_act_int8_weight",
        "capture_integerization": integerization,
    }
    accounting = {
        "schema": "merlin.phase0.operation_accounting.v1",
        "status": "accounted",
        "scope": "selected captured applications",
        "inventory_sha256": "c" * 64,
        "selected_software_spec_sha256": C.document_sha256(software_spec),
        "selected_capability_contract_sha256": C.document_sha256(contract),
        "selected_inventory": {"content_sha256": "c" * 64},
        "applications": {"selected": selected_application},
    }
    evidence_root = root / "_evidence"
    fact_path = evidence_root / "hardware/effective-views/performance-facts.json"
    accounting_path = evidence_root / "coverage/operation-accounting.json"
    basis_path = evidence_root / "coverage/performance-basis.json"
    hardware_path = evidence_root / "software/hardware-spec.json"
    software_path = evidence_root / "software/software-spec.json"
    contract_path = evidence_root / "software/contract.json"
    _write(fact_path, facts)
    _write(accounting_path, accounting)
    _write(hardware_path, {})
    _write(software_path, software_spec)
    _write(contract_path, contract)
    from merlin.targetgen.application_inventory import _INT_MM_BODY, _INT_MM_ITERATORS, _INT_MM_MAPS

    exact = {
        "application": "selected",
        "capture_sha256": "7" * 64,
        "operation": "aten._int_mm.default",
        "mlir_operation": "linalg.generic",
        "frontend_op": "aten._int_mm.default",
        "provenance_op": "int_matmul",
        "semantic_family": "contraction",
        "operand_format": "int8",
        "accumulator_dtypes": ["i32"],
        "indexing_maps": _INT_MM_MAPS,
        "iterator_types": _INT_MM_ITERATORS,
        "body_operations": _INT_MM_BODY,
        "ordered_operand_types": [
            {"shape": [8, 32], "dtype": "i8"},
            {"shape": [32, 96], "dtype": "i8"},
            {"shape": [8, 96], "dtype": "i32"},
        ],
        "ordered_result_types": [{"shape": [8, 96], "dtype": "i32"}],
        "result_shapes": [[8, 96]],
        "contraction_shape": {"M": 8, "K": 32, "N": 96, "rank": 3},
        "shape_confidence": "observed_iteration_space",
        "quant_evidence": {"prov.quant_inner_1": "verified"},
        "independent_compute_demand": True,
        "count": 2,
        "macs": {"per_occurrence": 24576, "total": 49152, "reason": None},
        "static_tensor_bytes": {
            "operands": 6400,
            "results": 3072,
            "per_occurrence": 9472,
            "total": 18944,
            "reason": None,
        },
    }
    other_shape = {
        **exact,
        "operation": "aten.mm.default",
        "mlir_operation": "linalg.matmul",
        "contraction_shape": {"M": 16, "K": 32, "N": 16, "rank": 3},
        "ordered_operand_types": [
            {"shape": [16, 32], "dtype": "i8"},
            {"shape": [32, 16], "dtype": "i8"},
            {"shape": [16, 16], "dtype": "i32"},
        ],
        "ordered_result_types": [{"shape": [16, 16], "dtype": "i32"}],
        "result_shapes": [[16, 16]],
        "count": 1,
        "macs": {"per_occurrence": 8192, "total": 8192, "reason": None},
        "static_tensor_bytes": {
            "operands": 2048,
            "results": 1024,
            "per_occurrence": 3072,
            "total": 3072,
            "reason": None,
        },
    }
    unknown = {
        **exact,
        "body_operations": ["arith.divsi", "linalg.yield"],
        "count": 1,
        "macs": {"per_occurrence": None, "total": None, "reason": "unrecognized_body"},
        "static_tensor_bytes": {
            "operands": 6400,
            "results": 3072,
            "per_occurrence": 9472,
            "total": 9472,
            "reason": None,
        },
    }
    _write(
        basis_path,
        {
            "schema": "merlin.phase0.performance_basis.v1",
            "target": "fixture",
            "status": "accounted",
            "selected_inventory": accounting["selected_inventory"],
            "sources": {
                "operation_accounting_sha256": C.sha256_file(accounting_path),
                "performance_facts_artifact_sha256": C.sha256_file(fact_path),
                "performance_facts_sha256": facts["sha256"],
                "selected_inventory_sha256": accounting["inventory_sha256"],
                "selected_software_spec_sha256": accounting["selected_software_spec_sha256"],
                "selected_capability_contract_sha256": accounting["selected_capability_contract_sha256"],
                "selected_hardware_spec_sha256": C.document_sha256({}),
                "raw_rtl_facts_sha256": facts["raw_facts_sha256"],
            },
            "applications": {
                "selected": {
                    **selected_application,
                    "n_operations": 4,
                    "n_signatures": 3,
                    "rows": [exact, other_shape, unknown],
                }
            },
            "overall": {
                "n_operations": 4,
                "independent_compute_demands": 4,
                "known_macs": 57344,
                "unknown_macs_demands": 1,
                "known_static_tensor_bytes": 31488,
                "unknown_static_tensor_bytes_demands": 0,
            },
            "qualification": "static selected-capture diagnostic only",
        },
    )
    source_requirement = evidence_root / "software/source-snapshots/0000-requirement.bin"
    source_requirement.parent.mkdir(parents=True, exist_ok=True)
    source_requirement.write_text(
        yaml.safe_dump(
            {
                "target": "fixture",
                "derivation": {"phase0_execution": {"raw_facts_sha256": facts["raw_facts_sha256"]}},
                "scope": {
                    "required": [{"signature": "movement -> contraction", "occurrences": 1}],
                    "typed_required_instances": {
                        "schema": "merlin.phase0.typed_scope_instances.v1",
                        "instances": [
                            {
                                "instance_id": "source-chain-1",
                                "application": "selected",
                                "signature": "movement -> contraction",
                            }
                        ],
                    },
                    "performance": {
                        "schema": "merlin.phase0.performance_scope.v1",
                        "status": "no_eligible_chain",
                        "required": [],
                        "excluded": [
                            {
                                "instance_id": "source-chain-1",
                                "signature": "movement -> contraction",
                                "status": "software_refused",
                                "reason": "mixed-lane precision refused",
                            }
                        ],
                        "unresolved": [],
                    },
                },
                "private_unrelated_requirement_field": "must not enter Phase 2 frozen corpus",
            }
        )
    )
    _write(
        evidence_root / "evidence-manifest.json",
        {
            "target": "fixture",
            "status": "diagnostic",
            "performance_facts_sha256": facts["sha256"],
            "raw_facts_sha256": facts["raw_facts_sha256"],
            "sources": [
                {
                    "role": "conformance-spec",
                    "path": "software/source-snapshots/0000-requirement.bin",
                    "sha256": C.sha256_file(evidence_root / "software/source-snapshots/0000-requirement.bin"),
                    "size_bytes": (evidence_root / "software/source-snapshots/0000-requirement.bin").stat().st_size,
                }
            ],
            "artifacts": {
                "software/source-snapshots/0000-requirement.bin": {
                    "sha256": C.sha256_file(evidence_root / "software/source-snapshots/0000-requirement.bin"),
                    "size_bytes": (evidence_root / "software/source-snapshots/0000-requirement.bin").stat().st_size,
                },
                "hardware/effective-views/performance-facts.json": {
                    "sha256": C.sha256_file(fact_path),
                    "size_bytes": fact_path.stat().st_size,
                },
                "coverage/operation-accounting.json": {
                    "sha256": C.sha256_file(accounting_path),
                    "size_bytes": accounting_path.stat().st_size,
                },
                "coverage/performance-basis.json": {
                    "sha256": C.sha256_file(basis_path),
                    "size_bytes": basis_path.stat().st_size,
                },
                "software/hardware-spec.json": {
                    "sha256": C.sha256_file(hardware_path),
                    "size_bytes": hardware_path.stat().st_size,
                },
                "software/software-spec.json": {
                    "sha256": C.sha256_file(software_path),
                    "size_bytes": software_path.stat().st_size,
                },
                "software/contract.json": {
                    "sha256": C.sha256_file(contract_path),
                    "size_bytes": contract_path.stat().st_size,
                },
            },
            "consumers": {
                "performance_basis": [
                    "coverage/operation-accounting.json",
                    "hardware/effective-views/performance-facts.json",
                    "software/hardware-spec.json",
                    "software/software-spec.json",
                    "software/contract.json",
                    "coverage/performance-basis.json",
                ]
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
    assert all(row["workload_need"]["status"] == "exact_arithmetic_form" for row in receipt["members"])
    assert all(row["workload_need"]["exact_known_macs"] == 49152 for row in receipt["members"])
    accounting = receipt["workload_accounting"]
    assert {key: value for key, value in accounting.items() if key != "uncovered_demands"} == {
        "basis_status": "accounted",
        "known_macs": 57344,
        "exact_arithmetic_form_macs": 49152,
        "candidate_only_known_macs": 0,
        "uncovered_known_macs": 8192,
        "unknown_macs_demands": 1,
        "known_static_tensor_bytes": 31488,
        "unknown_static_tensor_bytes_demands": 0,
        "qualification": "captured static MAC work; no execution frequency, cycle or speedup inference",
    }
    assert accounting["uncovered_demands"] == [
        {
            "application": "selected",
            "capture_sha256": "7" * 64,
            "signature_index": 1,
            "operation": "aten.mm.default",
            "mlir_operation": "linalg.matmul",
            "contraction_shape": {"M": 16, "K": 32, "N": 16, "rank": 3},
            "shape_confidence": "observed_iteration_space",
            "operand_format": "int8",
            "ordered_operand_dtypes": ["i8", "i8", "i32"],
            "accumulator_dtypes": ["i32"],
            "placement": None,
            "count": 1,
            "known_macs": 8192,
            "unknown_macs_reason": None,
            "candidate_capsules": [],
        },
        {
            "application": "selected",
            "capture_sha256": "7" * 64,
            "signature_index": 2,
            "operation": "aten._int_mm.default",
            "mlir_operation": "linalg.generic",
            "contraction_shape": {"M": 8, "K": 32, "N": 96, "rank": 3},
            "shape_confidence": "observed_iteration_space",
            "operand_format": "int8",
            "ordered_operand_dtypes": ["i8", "i8", "i32"],
            "accumulator_dtypes": ["i32"],
            "placement": None,
            "count": 1,
            "known_macs": None,
            "unknown_macs_reason": "unrecognized_body",
            "candidate_capsules": [
                {"capsule": "left", "status": "shape_dtype_candidate"},
                {"capsule": "right", "status": "shape_dtype_candidate"},
            ],
        },
    ]
    frozen = P.freeze_performance_corpus(discovered, tmp_path / "frozen")
    assert (frozen.root / "test_justification.json").is_file()
    assert (frozen.root / "test_justification_inputs/performance-basis.json").is_file()
    projected_path = frozen.root / "test_justification_inputs/performance-scope.json"
    projected = json.loads(projected_path.read_bytes())
    assert projected["performance"]["status"] == "no_eligible_chain"
    assert projected["performance"]["excluded"][0]["status"] == "software_refused"
    assert b"private_unrelated_requirement_field" not in projected_path.read_bytes()
    assert not (frozen.root / "test_justification_inputs/selected-requirement.yaml").exists()
    P.load_frozen_performance_corpus(
        frozen.root,
        manifest_sha256=frozen.manifest_sha256,
        capsules_sha256=frozen.capsules_sha256,
        expected_target="fixture",
    )


def test_representative_discovery_freezes_whole_comparison_family(generated, tmp_path):
    selected = P.discover_performance_corpus(generated, capsules="representative")
    assert {member.capsule for member in selected.capsules} == {"left", "right"}
    assert selected.representative_selection["status"] == "diagnostic_subset_unmeasured"
    assert selected.representative_selection["connected_slice"]["excluded"][0]["status"] == "software_refused"
    frozen = P.freeze_performance_corpus(selected, tmp_path / "representative")
    manifest = json.loads(frozen.manifest_path.read_bytes())
    assert manifest["representative_selection"]["path"] == "representative_selection.json"
    assert (frozen.root / "representative_selection.json").is_file()
    P.verify_frozen_performance_corpus(frozen)


def test_phase0_sibling_evidence_layout_is_frozen_by_exact_manifest(generated, tmp_path):
    root = generated.capsule_corpus.parent
    old = root / "_evidence"
    sibling = root.parent
    for item in list(old.iterdir()):
        shutil.move(str(item), str(sibling / item.name))
    old.rmdir()
    manifest_path = root / "MANIFEST.yaml"
    manifest = yaml.safe_load(manifest_path.read_text())
    manifest["phase0_evidence"] = {"manifest": str(sibling / "evidence-manifest.json")}
    manifest_path.write_text(yaml.safe_dump(manifest))
    selected = P.discover_performance_corpus(generated, capsules="representative")
    frozen = P.freeze_performance_corpus(selected, tmp_path / "sibling_evidence")
    assert (frozen.root / "test_justification_inputs/performance-basis.json").is_file()
    P.verify_frozen_performance_corpus(frozen)


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


@pytest.mark.parametrize("mutation", ["missing", "artifact_bytes", "source_identity"])
def test_discovery_refuses_missing_or_unbound_performance_basis(generated, mutation):
    root = generated.capsule_corpus.parent / "_evidence"
    path = root / "coverage/performance-basis.json"
    if mutation == "missing":
        path.unlink()
    elif mutation == "artifact_bytes":
        path.write_bytes(path.read_bytes() + b" ")
    else:
        basis = json.loads(path.read_text())
        basis["sources"]["operation_accounting_sha256"] = "0" * 64
        _write(path, basis)
        manifest_path = root / "evidence-manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["artifacts"]["coverage/performance-basis.json"] = {
            "sha256": C.sha256_file(path),
            "size_bytes": path.stat().st_size,
        }
        _write(manifest_path, manifest)
    with pytest.raises(C.StageGateError, match="absent|changed|performance basis"):
        P.discover_performance_corpus(generated)


def test_frozen_basis_bytes_are_rechecked(generated, tmp_path):
    frozen = P.freeze_performance_corpus(P.discover_performance_corpus(generated), tmp_path / "frozen")
    path = frozen.root / "test_justification_inputs/performance-basis.json"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(C.StageGateError, match="performance-basis.json"):
        P.verify_frozen_performance_corpus(frozen)


def test_frozen_scope_projection_bytes_are_rechecked(generated, tmp_path):
    frozen = P.freeze_performance_corpus(P.discover_performance_corpus(generated), tmp_path / "frozen")
    path = frozen.root / "test_justification_inputs/performance-scope.json"
    path.chmod(0o644)
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(C.StageGateError, match="performance scope bytes changed"):
        P.verify_frozen_performance_corpus(frozen)


def test_selected_scope_refuses_non_public_capture(generated):
    evidence_root = generated.capsule_corpus.parent / "_evidence"
    relative = "software/source-snapshots/0000-requirement.bin"
    path = evidence_root / relative
    requirement = yaml.safe_load(path.read_text())
    requirement["scope"]["typed_required_instances"]["instances"][0]["application"] = "held_out"
    path.write_text(yaml.safe_dump(requirement))
    manifest_path = evidence_root / "evidence-manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["sources"][0]["sha256"] = C.sha256_file(path)
    manifest["sources"][0]["size_bytes"] = path.stat().st_size
    manifest["artifacts"][relative] = {
        "sha256": C.sha256_file(path),
        "size_bytes": path.stat().st_size,
    }
    _write(manifest_path, manifest)
    with pytest.raises(C.StageGateError, match="outside selected public captures"):
        P.discover_performance_corpus(generated)


def test_class_membership_and_equal_shape_do_not_cover_an_unproven_form(generated):
    root = generated.capsule_corpus.parent / "_evidence"
    path = root / "coverage/performance-basis.json"
    basis = json.loads(path.read_text())
    row = basis["applications"]["selected"]["rows"][0]
    row["body_operations"] = ["arith.divsi", "linalg.yield"]
    row["macs"] = {"per_occurrence": None, "total": None, "reason": "unrecognized_body"}
    basis["overall"]["known_macs"] = 8192
    basis["overall"]["unknown_macs_demands"] = 3
    _rebind_artifact(root, "coverage/performance-basis.json", basis)
    receipt = P.discover_performance_corpus(generated).test_justification
    assert all(row["workload_need"]["class_in_census"] is True for row in receipt["members"])
    assert all(row["workload_need"]["status"] == "shape_dtype_candidate" for row in receipt["members"])
    assert receipt["workload_accounting"]["exact_arithmetic_form_macs"] == 0
    assert receipt["workload_accounting"]["uncovered_known_macs"] == 8192
    assert receipt["workload_accounting"]["unknown_macs_demands"] == 3


def test_structural_arithmetic_without_quant_origin_stays_uncovered(generated):
    root = generated.capsule_corpus.parent / "_evidence"
    path = root / "coverage/performance-basis.json"
    basis = json.loads(path.read_text())
    row = basis["applications"]["selected"]["rows"][0]
    row["quant_evidence"] = {}
    row["macs"] = {"per_occurrence": None, "total": None, "reason": "quant_origin_unknown"}
    basis["overall"]["known_macs"] = 8192
    basis["overall"]["unknown_macs_demands"] = 3
    _rebind_artifact(root, "coverage/performance-basis.json", basis)
    receipt = P.discover_performance_corpus(generated).test_justification
    assert all(row["workload_need"]["status"] == "structural_arithmetic_candidate" for row in receipt["members"])
    assert receipt["workload_accounting"]["exact_arithmetic_form_macs"] == 0
    assert receipt["workload_accounting"]["uncovered_demands"][0]["candidate_capsules"] == [
        {"capsule": "left", "status": "structural_arithmetic_candidate"},
        {"capsule": "right", "status": "structural_arithmetic_candidate"},
    ]


def test_unavailable_inventory_preserves_unknown_mass(generated):
    root = generated.capsule_corpus.parent / "_evidence"
    accounting_path = root / "coverage/operation-accounting.json"
    accounting = json.loads(accounting_path.read_text())
    accounting["status"] = "not_available"
    accounting["inventory_sha256"] = None
    accounting["selected_inventory"] = {"status": "not_available"}
    accounting["applications"] = {}
    _rebind_artifact(root, "coverage/operation-accounting.json", accounting)
    basis_path = root / "coverage/performance-basis.json"
    basis = json.loads(basis_path.read_text())
    basis["status"] = "not_available"
    basis["selected_inventory"] = accounting["selected_inventory"]
    basis["sources"]["operation_accounting_sha256"] = C.sha256_file(accounting_path)
    basis["sources"]["selected_inventory_sha256"] = None
    basis["applications"] = {}
    basis["overall"] = {
        "n_operations": None,
        "independent_compute_demands": None,
        "known_macs": None,
        "unknown_macs_demands": None,
        "known_static_tensor_bytes": None,
        "unknown_static_tensor_bytes_demands": None,
        "unknown_reason": "no_selected_detailed_application_inventory",
    }
    _rebind_artifact(root, "coverage/performance-basis.json", basis)
    receipt = P.discover_performance_corpus(generated).test_justification
    assert all(row["workload_need"]["status"] == "basis_unavailable" for row in receipt["members"])
    assert receipt["workload_accounting"]["known_macs"] is None
    assert receipt["workload_accounting"]["uncovered_known_macs"] is None
    assert receipt["workload_accounting"]["unknown_macs_demands"] is None
