"""An observed shape is not, by itself, a performance coverage witness."""

from copy import deepcopy

from merlin.targetgen.performance_match import match_operation_form


def _source():
    return {
        "independent_compute_demand": True,
        "operation": "aten._int_mm.default",
        "mlir_operation": "linalg.generic",
        "frontend_op": "aten._int_mm.default",
        "provenance_op": "int_matmul",
        "semantic_family": "contraction",
        "operand_format": "int8",
        "accumulator_dtypes": ["i32"],
        "ordered_operand_types": [
            {"shape": [2, 32], "dtype": "i8"},
            {"shape": [32, 64], "dtype": "i8"},
            {"shape": [2, 64], "dtype": "i32"},
        ],
        "ordered_result_types": [{"shape": [2, 64], "dtype": "i32"}],
        "result_shapes": [[2, 64]],
        "contraction_shape": {"M": 2, "K": 32, "N": 64, "rank": 3},
        "shape_confidence": "observed_iteration_space",
        "indexing_maps": [
            "affine_map<(d0, d1, d2) -> (d0, d2)>",
            "affine_map<(d0, d1, d2) -> (d2, d1)>",
            "affine_map<(d0, d1, d2) -> (d0, d1)>",
        ],
        "iterator_types": [
            "#linalg.iterator_type<parallel>",
            "#linalg.iterator_type<parallel>",
            "#linalg.iterator_type<reduction>",
        ],
        "body_operations": ["arith.extsi", "arith.extsi", "arith.muli", "arith.addi", "linalg.yield"],
        "quant_evidence": {"prov.quant_inner_1": "weights.int_data"},
        "source_integerization_byte_bound": True,
    }


def _capsule():
    return {
        "inputs": [
            {"name": "A0", "shape": [2, 32], "dtype": "i8"},
            {"name": "W", "shape": [32, 64], "dtype": "i8"},
        ],
        "operation": {
            "op": "matmul",
            "attributes": {"lhs": "A0", "weight": "W", "out": "Y0", "epilogue": [], "output_dtype": "i32"},
        },
    }


def test_exact_arithmetic_form_requires_source_body_and_quant_origin():
    assert match_operation_form(_source(), _capsule())["status"] == "exact_arithmetic_form"
    for field in ("body_operations", "indexing_maps", "shape_confidence"):
        source = _source()
        source[field] = None
        assert match_operation_form(source, _capsule())["status"] == "shape_dtype_candidate"
    source = _source()
    source["quant_evidence"] = None
    assert match_operation_form(source, _capsule())["status"] == "structural_arithmetic_candidate"
    source = _source()
    source["source_integerization_byte_bound"] = False
    assert match_operation_form(source, _capsule())["status"] == "structural_arithmetic_candidate"


def test_same_geometry_does_not_certify_fused_or_mismatched_work():
    fused = _capsule()
    fused["operation"]["op"] = "fused_matmul_bias"
    fused["operation"]["attributes"]["epilogue"] = ["bias_add"]
    assert match_operation_form(_source(), fused)["status"] == "shape_dtype_candidate"
    other = deepcopy(_capsule())
    other["inputs"][0]["shape"][0] = 4
    assert match_operation_form(_source(), other)["status"] == "off_shape"
    source = _source()
    source["independent_compute_demand"] = False
    assert match_operation_form(source, _capsule())["status"] == "not_independent"
    wrong_output = _capsule()
    wrong_output["operation"]["attributes"]["output_dtype"] = "f32"
    assert match_operation_form(_source(), wrong_output)["status"] == "dtype_mismatch"
