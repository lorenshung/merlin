"""The adapter contract is projected from authored and derived inputs, never filled from defaults.

An external quantization adapter reads its numerics (finite maximum, flush threshold, rounding,
scale encoding) from these bytes. A value no input states must arrive as ``unknown`` and make the
contract diagnostic; a computed value must follow from the declared exponent domain and reserved
codes, so a domain ending one binade lower or a reserved top code each change the maximum.
"""

from __future__ import annotations

import copy
import json
import math

import pytest

from merlin.targetgen import quant_adapter_contract as QAC
from merlin.targetgen.quantization_spec import build_quantization_contract, validate_quantization_declarations
from merlin.targetgen.software_spec import validate_software_spec

_SOURCE = {"path": "software/software-spec.json", "sha256": "c" * 64}


def _spec(domain=None, *, semantics=None, fmt=None):
    numerical = {
        "model": {"engine": "specir_fp_reduce"},
        "operand_dtype": "fp8_e4m3",
        "accumulator_dtype": "bf16",
        "readout_dtype": "bf16",
        "product_rounding": "accumulator_format",
        "rounding": "rne",
        "reduction_order": "index_sequential",
        "reduction_cadence": "per_step",
        "subnormal_operand_flush": True,
        "operand_rounding": "rne",
        "scale_rule": "power_of_two_amax",
        "internal_arithmetic": {
            "operand_domain": {"exponent_range": [1, 14], "signed_zero": True, "reserved_codes": []}
            if domain is None
            else domain
        },
        **(semantics or {}),
    }
    return validate_software_spec(
        {
            "schema": "merlin.software_spec.v1",
            "target": "synthetic",
            "numerical_semantics": numerical,
            "operations": {
                "contraction": {
                    "placement": "accelerator",
                    "operand_dtypes": ["fp8_e4m3"],
                    "accumulator_dtype": "bf16",
                },
                "host_fallback": {"placement": "host", "families": ["softmax", "normalization"], "dtypes": ["f32"]},
                "elementwise_map": {"placement": "unknown", "dtypes": ["bf16"]},
            },
            "quantization": {
                "formats": [
                    {
                        "operand_dtype": "fp8_e4m3",
                        "accumulator_dtype": "bf16",
                        "eligible_operations": ["contraction"],
                        "scale_encoding": "e8m0",
                        **(fmt or {}),
                    }
                ]
            },
        }
    )


def _hardware(unit="matrix_unit"):
    tensor = {"dtype": "fp8_e4m3", "granularity": "tensor", "mode": "static", "symmetric": True}
    return {
        "quantization_candidates": [
            {
                "format": "fp8_e4m3",
                "unit": unit,
                "status": "derived",
                "recipe": {"status": "derived", "weight": dict(tensor), "activation": dict(tensor)},
            }
        ],
        "readout_facets": [{"unit": unit, "accumulator_dtype": "bf16", "scale": {"dtype": "e8m0"}}],
    }


def _contract(spec, hardware=None):
    quantization = build_quantization_contract(spec, _hardware() if hardware is None else hardware)
    return QAC.build(spec, quantization, format_id=quantization["formats"][0]["id"], spec_source=_SOURCE)


def test_finite_max_is_240_when_the_domain_stops_at_exponent_14():
    operand, unknowns = QAC.operand_domain("fp8_e4m3", _spec()["numerical_semantics"])
    assert operand["bias"] == 7
    assert operand["finite_max"] == 240.0
    assert operand["min_normal"] == math.ldexp(1, -6)
    assert "operand.finite_max" not in unknowns


def test_finite_max_is_448_when_exponent_15_is_finite_and_its_top_code_reserved():
    domain = {"exponent_range": [1, 15], "signed_zero": True, "reserved_codes": [0x7F, 0xFF]}
    operand, _ = QAC.operand_domain("fp8_e4m3", _spec(domain)["numerical_semantics"])
    assert operand["finite_max"] == 448.0
    assert operand["reserved_codes"] == [0x7F, 0xFF]


def test_a_fully_declared_format_is_derived_and_canonical():
    contract = _contract(_spec())
    assert contract["status"] == "derived", contract["unknowns"]
    assert contract["unknowns"] == {}
    assert contract["scale"] == {
        "encoding": "e8m0",
        "weight_granularity": "tensor",
        "activation_granularity": "tensor",
        "activation_mode": "static",
        "block_size": "not_applicable",
        "value_rule": "power_of_two_amax",
    }
    assert contract["accumulate"]["unit"] == "matrix_unit"
    assert [row["operation"] for row in contract["families"]["accelerator"]] == ["contraction"]
    assert contract["families"]["host"][0]["families"] == ["normalization", "softmax"]
    assert contract["families"]["unplaced"][0]["operation"] == "elementwise_map"
    raw = QAC.canonical_bytes(contract)
    assert raw.endswith(b"\n") and raw == QAC.canonical_bytes(copy.deepcopy(contract))
    assert json.loads(raw) == contract
    assert raw == QAC.canonical_bytes(_contract(_spec()))


@pytest.mark.parametrize(
    ("change", "unknown"),
    [
        ({"domain": {"exponent_range": [1, 14], "signed_zero": True}}, "operand.finite_max"),
        ({"domain": {"signed_zero": True, "reserved_codes": []}}, "operand.exponent_range"),
        ({"semantics": {"operand_rounding": None}}, "operand.rounding"),
        ({"semantics": {"scale_rule": None}}, "scale.value_rule"),
        ({"fmt": {"scale_encoding": "unknown"}}, "scale.encoding"),
        ({"hardware": {"quantization_candidates": []}}, "hardware_match"),
    ],
)
def test_an_undeclared_value_stays_unknown_and_makes_the_contract_diagnostic(change, unknown):
    semantics = {key: value for key, value in (change.get("semantics") or {}).items() if value is not None}
    spec = _spec(change.get("domain"), semantics=semantics, fmt=change.get("fmt"))
    for key, value in (change.get("semantics") or {}).items():
        if value is None:
            spec["numerical_semantics"].pop(key)
            for row in spec["quantization"]["formats"]:
                row.pop("numerical_semantics", None)
    contract = _contract(spec, change.get("hardware"))
    assert contract["status"] == "diagnostic"
    assert unknown in contract["unknowns"]


def test_a_quantization_contract_whose_bytes_changed_is_refused():
    spec = _spec()
    quantization = build_quantization_contract(spec, _hardware())
    quantization["formats"][0]["declaration"]["scale_encoding"] = "f32"
    with pytest.raises(ValueError, match="recorded digest"):
        QAC.build(spec, quantization, format_id=quantization["formats"][0]["id"], spec_source=_SOURCE)


def test_unknown_format_id_and_foreign_target_are_refused():
    spec = _spec()
    quantization = build_quantization_contract(spec, _hardware())
    with pytest.raises(ValueError, match="no single format"):
        QAC.build(spec, quantization, format_id="absent", spec_source=_SOURCE)
    other = {**spec, "target": "other"}
    with pytest.raises(ValueError, match="different targets"):
        QAC.build(other, quantization, format_id=quantization["formats"][0]["id"], spec_source=_SOURCE)


def test_scale_encoding_must_be_unknown_or_a_registered_dtype():
    spec = _spec()
    spec["quantization"]["formats"][0]["scale_encoding"] = "e9m9"
    with pytest.raises(ValueError, match="scale_encoding must be unknown or a registered dtype"):
        validate_quantization_declarations(spec)


@pytest.mark.parametrize(
    "domain",
    [
        {"exponent_range": [14, 1]},
        {"exponent_range": [1, 14], "reserved_codes": [127, 127]},
        {"exponent_range": [1, 14], "signed_zero": "yes"},
        {"exponent_range": [1, 14], "saturating": True},
    ],
)
def test_malformed_operand_domain_is_refused_when_the_spec_is_read(domain):
    with pytest.raises(ValueError, match="operand_domain"):
        _spec(domain)


def test_a_domain_wider_than_the_format_is_refused():
    with pytest.raises(ValueError, match="4-bit exponent fields"):
        QAC.operand_domain("fp8_e4m3", _spec({"exponent_range": [1, 16], "reserved_codes": []})["numerical_semantics"])
