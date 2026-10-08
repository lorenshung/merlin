"""Static Phase 0 mass remains capture-bound and refuses invented counts."""

from copy import deepcopy

from merlin.targetgen.performance_basis import build_performance_basis


def _basis(rows):
    entries = [
        {
            "operation": row["operation"],
            "count": row["count"],
            "ordinals": row["ordinals"],
            "observed_signature": row,
            "classification": "accelerator_candidate" if row["disposition"] != "structural" else "structural",
            "support_partition": "accelerator_only" if row["disposition"] != "structural" else "not_applicable",
        }
        for row in rows
    ]
    accounting = {
        "schema": "merlin.phase0.operation_accounting.v1",
        "status": "accounted",
        "inventory_sha256": "a" * 64,
        "selected_software_spec_sha256": "b" * 64,
        "selected_capability_contract_sha256": "c" * 64,
        "applications": {
            "iteration": {
                "capture_sha256": "d" * 64,
                "capture_quantization": {"status": "unknown"},
                "n_mlir_operations": sum(row["count"] for row in rows),
                "signatures": entries,
            }
        },
        "overall": {"n_mlir_operations": sum(row["count"] for row in rows)},
    }
    return build_performance_basis(
        accounting,
        {"target": "fixture", "sha256": "e" * 64},
        target="fixture",
        operation_accounting_sha256="f" * 64,
        performance_facts_artifact_sha256="1" * 64,
        selected_hardware_spec_sha256="2" * 64,
        raw_rtl_facts_sha256="3" * 64,
    )


def test_exact_matmul_mass_and_structural_rows_do_not_double_count():
    matmul = {
        "operation": "linalg.matmul",
        "mlir_operation": "linalg.matmul",
        "disposition": "hardware_admitted",
        "semantic_family": "contraction",
        "contraction_shape": {"M": 2, "K": 3, "N": 4, "rank": 3},
        "shape_confidence": "observed_iteration_space",
        "ordered_operand_types": [
            {"shape": [2, 3], "dtype": "f32"},
            {"shape": [3, 4], "dtype": "f32"},
            {"shape": [2, 4], "dtype": "f32"},
        ],
        "ordered_result_types": [{"shape": [2, 4], "dtype": "f32"}],
        "count": 2,
        "ordinals": [1, 2],
    }
    structural = {
        **deepcopy(matmul),
        "operation": "builtin.module",
        "mlir_operation": "builtin.module",
        "disposition": "structural",
        "count": 1,
        "ordinals": [0],
    }
    original = deepcopy((matmul, structural))
    basis = _basis([structural, matmul])
    assert (matmul, structural) == original
    assert basis["schema"] == "merlin.phase0.performance_basis.v1"
    assert basis["applications"]["iteration"]["capture_sha256"] == "d" * 64
    assert basis["sources"]["operation_accounting_sha256"] == "f" * 64
    rows = basis["applications"]["iteration"]["rows"]
    assert all(row["application"] == "iteration" and row["capture_sha256"] == "d" * 64 for row in rows)
    assert rows[0]["independent_compute_demand"] is False
    assert rows[0]["macs"]["total"] is None
    assert rows[1]["macs"] == {"per_occurrence": 24, "total": 48, "reason": None}
    assert rows[1]["static_tensor_bytes"]["per_occurrence"] == (6 + 12 + 8 + 8) * 4
    assert basis["overall"] == {
        "n_operations": 3,
        "independent_compute_demands": 2,
        "known_macs": 48,
        "unknown_macs_demands": 0,
        "known_static_tensor_bytes": 272,
        "unknown_static_tensor_bytes_demands": 0,
    }


def test_generic_lookalike_and_dynamic_types_remain_unknown():
    row = {
        "operation": "aten.mm.default",
        "mlir_operation": "linalg.generic",
        "disposition": "hardware_admitted",
        "semantic_family": "contraction",
        "contraction_shape": {"M": 2, "K": 3, "N": 4, "rank": 3},
        "shape_confidence": "observed_iteration_space",
        "ordered_operand_types": [{"shape": [None, 3], "dtype": "i8"}],
        "ordered_result_types": [{"shape": [2, 4], "dtype": "i32"}],
        "count": 1,
        "ordinals": [0],
    }
    basis = _basis([row])
    mass = basis["applications"]["iteration"]["rows"][0]
    assert mass["macs"]["total"] is None
    assert mass["macs"]["reason"] == "contraction_body_not_a_recognized_single_mac_form"
    assert mass["static_tensor_bytes"]["total"] is None
    assert mass["static_tensor_bytes"]["reason"] == "tensor_shape_not_static"
    assert basis["overall"]["unknown_macs_demands"] == 1
    assert basis["overall"]["unknown_static_tensor_bytes_demands"] == 1


def test_only_recognized_integer_generic_gets_exact_macs():
    from merlin.targetgen.application_inventory import _INT_MM_BODY, _INT_MM_ITERATORS, _INT_MM_MAPS

    row = {
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
        "quant_evidence": {"prov.quant_inner_1": "torchao"},
        "ordered_operand_types": [
            {"shape": [2, 3], "dtype": "i8"},
            {"shape": [3, 4], "dtype": "i8"},
            {"shape": [2, 4], "dtype": "i32"},
        ],
        "ordered_result_types": [{"shape": [2, 4], "dtype": "i32"}],
        "result_shapes": [[2, 4]],
        "contraction_shape": {"M": 2, "K": 3, "N": 4, "rank": 3},
        "shape_confidence": "observed_iteration_space",
        "disposition": "hardware_admitted",
        "count": 1,
        "ordinals": [0],
    }
    exact = _basis([row])["applications"]["iteration"]["rows"][0]
    assert exact["macs"]["total"] == 24
    assert exact["indexing_maps"] == _INT_MM_MAPS
    unproven = deepcopy(row)
    unproven["quant_evidence"] = None
    assert _basis([unproven])["applications"]["iteration"]["rows"][0]["macs"]["total"] == 24


def test_missing_inventory_is_unknown_not_zero_workload():
    basis = build_performance_basis(
        {
            "schema": "merlin.phase0.operation_accounting.v1",
            "status": "not_available",
            "applications": {},
            "overall": {"n_mlir_operations": None},
        },
        {"target": "fixture", "sha256": "e" * 64},
        target="fixture",
        operation_accounting_sha256="f" * 64,
        performance_facts_artifact_sha256="1" * 64,
        selected_hardware_spec_sha256=None,
        raw_rtl_facts_sha256=None,
    )
    assert basis["overall"]["n_operations"] is None
    assert basis["overall"]["known_macs"] is None
    assert basis["overall"]["unknown_reason"] == "no_selected_detailed_application_inventory"
