"""Coarse operation-accounting checks: complete partition, honest provenance, frozen inputs."""

import json
from collections import Counter
from copy import deepcopy

import pytest

from merlin.targetgen.operation_accounting import build_operation_accounting


def _inputs():
    rows = []
    for name, disposition, family, fmt, frontend, positions in (
        ("builtin.module", "structural", None, None, None, [0]),
        ("linalg.generic", "hardware_admitted", "contraction", "int8", "aten.mm.default", [1, 2]),
        ("arith.addi", "component", "elementwise_map", "int8", "aten.mm.default", [3, 4]),
        ("tensor.empty", "support_required", "movement", "int8", None, [5]),
        ("linalg.generic", "host_required", "normalization", "fp32", "aten.layer_norm.default", [6]),
        ("func.call", "unclassified", None, None, None, [7]),
    ):
        rows.append(
            {
                "operation": frontend or name,
                "mlir_operation": name,
                "frontend_op": frontend,
                "provenance_op": None,
                "semantic_family": family,
                "operand_format": fmt,
                "disposition": disposition,
                "reason": "old inventory annotation",
                "count": len(positions),
                "ordinals": positions,
                "ordered_operand_types": [{"shape": [4, 8], "dtype": "i8"}],
                "ordered_result_types": [{"shape": [4, 4], "dtype": "i32"}],
                "result_dtypes": ["i32"],
                "accumulator_dtypes": ["i32"] if family == "contraction" else None,
                "contraction_shape": {"M": 4, "K": 8, "N": 4, "rank": 3} if family == "contraction" else None,
            }
        )
    counts = Counter()
    for row in rows:
        counts[row["disposition"]] += row["count"]
    inventory = {
        "schema_version": 1,
        "status": "incomplete",
        "n_operations": 8,
        "applications": {
            "small_model": {
                "capture": "capture/model.mlir",
                "capture_sha256": "a" * 64,
                "n_operations": 8,
                "n_signatures": len(rows),
                "counts": dict(counts),
                "signatures": rows,
            }
        },
    }
    spec = {
        "status": "reviewed",
        "operations": [
            {
                "id": "matrix",
                "families": ["contraction"],
                "placement": "accelerator",
                "signature": {"operand_dtypes": ["int8"], "accumulator_dtype": "i32", "ranks": [2]},
            },
            {
                "id": "norm",
                "families": ["normalization"],
                "placement": "host",
                "signature": {"operand_dtypes": ["fp32"]},
            },
            {
                "id": "absent",
                "ops": ["aten.absent.default"],
                "placement": "accelerator",
                "signature": {"operand_dtypes": ["int8"]},
            },
        ],
    }
    contract = {
        "name": "test_device",
        "compute_units": [
            {
                "name": "matrix_unit",
                "kind": "systolic",
                "dtypes": ["int8"],
                "ops": ["matmul"],
                "semantic_capabilities": [{"family": "contraction", "dtypes": ["int8"], "ranks": [2]}],
            }
        ],
    }
    return inventory, spec, contract


def test_complete_partition_preserves_provenance_and_selected_admission_without_lowering_claims():
    inventory, spec, contract = _inputs()
    originals = deepcopy((inventory, spec, contract))
    catalog = {
        "schema": "merlin.pytorch_opset.v1",
        "status": "observed",
        "torch": "test_version",
        "all_ops": ["aten.mm.default", "aten.other.default"],
        "n_all_aten": 2,
        "ops": ["aten.mm.default"],
        "decomposed": [],
    }
    report = build_operation_accounting(inventory, spec, capability_contract=contract, framework_catalog=catalog)
    assert (inventory, spec, contract) == originals
    overall = report["overall"]
    assert overall["n_mlir_operations"] == sum(overall["classification_counts"].values()) == 8
    assert overall["classification_counts"] == {
        "accelerator_candidate": 2,
        "host_required": 1,
        "nested_component": 2,
        "structural": 1,
        "support_lowering_required": 1,
        "unresolved": 1,
    }
    provenance = overall["pytorch_provenance"]
    assert provenance["original_pytorch_invocation_count"] is None
    mm = next(row for row in provenance["groups"] if row["frontend_op"] == "aten.mm.default")
    assert mm["annotated_mlir_operation_count"] == 4  # includes two nested children, not four Torch calls
    assert provenance["unattributed_mlir_operations"] == 3
    matrix = report["applications"]["small_model"]["signatures"][1]
    assert matrix["observed_admission_signature"]["rank"] == 2  # not the 3-loop contraction rank
    assert matrix["ordinals"] == [1, 2]
    assert matrix["hardware_admission"]["status"] == "admitted"
    assert matrix["hardware_admission"]["units"] == ["matrix_unit"]
    assert overall["hardware_admission_counts"]["admitted"] == 2
    assert matrix["matching_declarations"] == ["matrix"]
    assert all(row["lowering_status"] == "unverified" for row in report["applications"]["small_model"]["signatures"])
    absent = next(row for row in report["declared_support_universe"]["operations"] if row["id"] == "absent")
    assert absent["observation_status"] == "absent_from_selected_workloads"
    assert absent["observed_mlir_operation_count"] == 0
    universe = report["framework_universe"]
    assert universe["observed_registered_aten_operators"] == ["aten.mm.default"]
    assert universe["unobserved_registered_aten_operators"] == ["aten.other.default"]
    assert universe["unlisted_frontend_operators"] == ["aten.layer_norm.default"]
    assert universe["accelerator_candidate_frontend_operators"] == ["aten.mm.default"]
    assert json.dumps(report, sort_keys=True) == json.dumps(
        build_operation_accounting(
            deepcopy(inventory),
            deepcopy(spec),
            capability_contract=deepcopy(contract),
            framework_catalog=deepcopy(catalog),
        ),
        sort_keys=True,
    )
    stale = build_operation_accounting(inventory, spec)
    assert stale["applications"]["small_model"]["signatures"][1]["classification"] == "unresolved"
    spec["status"] = "unreviewed"
    diagnostic = build_operation_accounting(inventory, spec, capability_contract=contract)
    assert diagnostic["overall"]["classification_counts"].get("accelerator_candidate", 0) == 0


def test_missing_inputs_are_distinct_from_an_explicitly_undeclared_workload():
    missing = build_operation_accounting(None)
    assert missing["status"] == "not_available"
    assert missing["overall"]["n_mlir_operations"] is None
    empty = build_operation_accounting(
        {"schema_version": 1, "status": "not_declared", "n_operations": 0, "applications": {}}
    )
    assert empty["status"] == "not_declared"
    assert empty["overall"]["n_mlir_operations"] == 0
    with pytest.raises(ValueError, match="detailed application inventory"):
        build_operation_accounting({})


def test_inconsistent_counts_and_lost_ordinals_cannot_create_a_clean_report():
    inventory, _, _ = _inputs()
    for mutate in (
        lambda app: app["signatures"][1].update(count=-1),
        lambda app: app["signatures"][1].update(ordinals=[1, 1]),
        lambda app: app.update(n_operations=9),
        lambda app: app["counts"].update(hardware_admitted=1),
        lambda app: app.update(capture_sha256="not-a-digest"),
    ):
        changed = deepcopy(inventory)
        mutate(changed["applications"]["small_model"])
        with pytest.raises(ValueError):
            build_operation_accounting(changed)
