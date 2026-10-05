"""Captured quantization parameters must constrain reviewed host placement."""

from copy import deepcopy

import pytest

from merlin.targetgen.application_inventory import application_demand_inventory
from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.operation_accounting import admit_operation_row


_QUANT = "quantized_decomposed.quantize_per_tensor.default"
_DEQUANT = "quantized_decomposed.dequantize_per_tensor.default"
_CONTRACT = {"compute_units": [{
    "name": "vector", "kind": "vector", "dtypes": ["int8"], "ops": ["add"],
    "semantic_capabilities": [{"family": "elementwise_map", "dtypes": ["int8"]}],
}]}


def _declaration(operation, operands, result, parameters):
    return {
        "id": operation, "ops": [operation], "placement": "host", "status": "reviewed",
        "signature": {
            "ordered_operand_dtypes": operands,
            "ordered_result_dtypes": [result],
            "ranks": [3],
            "quantization_parameters": parameters,
        },
    }


_DECLARATIONS = [
    _declaration(_QUANT, ["f32", "f32", "i64"], "i8",
                 {"zero_point": 0, "quant_min": -128, "quant_max": 127}),
    _declaration(_DEQUANT, ["i8", "f32", "i64"], "f32", {"zero_point": 0}),
]


@pytest.mark.parametrize(
    ("zero_point", "quant_max", "expected_quant", "expected_dequant"),
    [
        (0, 127, "admitted", "admitted"),
        (1, 127, "unsupported", "unsupported"),
        (0, 126, "unsupported", "admitted"),
        (None, 127, "unknown", "unknown"),
    ],
)
def test_quant_parameter_admission_uses_captured_ssa_and_attributes(
    tmp_path, zero_point, quant_max, expected_quant, expected_dequant
) -> None:
    capture = tmp_path / "model.mlir"
    zero_source = (
        '%z = "tensor.splat"(%zc) : (i64) -> tensor<i64> '
        if zero_point is not None else ""
    )
    zero_definition = (
        f'%zc = "arith.constant"() <{{value = {zero_point} : i64}}> : () -> i64 '
        if zero_point is not None else ""
    )
    function_args = "%x: tensor<1x2x3xf32>" + (
        "" if zero_point is not None else ", %z: tensor<i64>"
    )
    capture.write_text(
        'builtin.module { func.func @forward(' + function_args +
        ') -> tensor<1x2x3xf32> { '
        '%sc = "arith.constant"() <{value = 1.000000e+00 : f32}> : () -> f32 '
        '%s = "tensor.splat"(%sc) : (f32) -> tensor<f32> '
        + zero_definition + zero_source +
        '%q = "quant_ext.quantize_per_tensor"(%x, %s, %z) '
        f'<{{quant_min = -128 : i64, quant_max = {quant_max} : i64}}> '
        f'{{prov.aten = "{_QUANT}"}} : '
        '(tensor<1x2x3xf32>, tensor<f32>, tensor<i64>) -> tensor<1x2x3xi8> '
        '%y = "quant_ext.dequantize_per_tensor"(%q, %s, %z) '
        f'<{{quant_min = -128 : i64, quant_max = {quant_max} : i64}}> '
        f'{{prov.aten = "{_DEQUANT}"}} : '
        '(tensor<1x2x3xi8>, tensor<f32>, tensor<i64>) -> tensor<1x2x3xf32> '
        'func.return %y : tensor<1x2x3xf32> } }',
        encoding="utf-8",
    )
    software = {"status": "reviewed", "operations": _DECLARATIONS}
    host = {"selected": {
        "package_sha256": "a" * 64, "capability_spec_sha256": "b" * 64,
        "dtype_strategy": "int8",
        "capability_spec": {
            "schema": "merlin.host_capabilities.v1", "status": "reviewed",
            "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8"},
            "operations": _DECLARATIONS, "evidence": {},
        },
    }}
    inventory = application_demand_inventory(
        {"iteration": capture}, "synthetic", detailed=True,
        capability_contract=_CONTRACT, software_spec=software, host_capabilities=host,
    )
    rows = inventory["applications"]["iteration"]["signatures"]
    for mlir_name, expected in (
        ("quant_ext.quantize_per_tensor", expected_quant),
        ("quant_ext.dequantize_per_tensor", expected_dequant),
    ):
        row = next(row for row in rows if row["mlir_operation"] == mlir_name)
        decision = admit_operation_row(
            row, software_spec=software, capability_contract=_CONTRACT, host_capabilities=host,
        )
        assert decision["software_admissions"][0]["status"] == expected, decision["software_admissions"][0]["reason"]
        assert decision["host_admission"]["status"] == expected, decision["host_admission"]["reason"]
        if mlir_name == "quant_ext.dequantize_per_tensor":
            assert row["disposition"] == ("host_required" if expected == "admitted" else "unclassified")


def test_host_spec_rejects_malformed_quant_parameter_constraints() -> None:
    host = {
        "schema": "merlin.host_capabilities.v1", "status": "reviewed",
        "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8"},
        "operations": [deepcopy(_DECLARATIONS[0])], "evidence": {},
    }
    host["operations"][0]["signature"]["quantization_parameters"]["zero_point"] = "0"
    with pytest.raises(ValueError, match="quantization_parameters"):
        validate_host_capabilities(host)
