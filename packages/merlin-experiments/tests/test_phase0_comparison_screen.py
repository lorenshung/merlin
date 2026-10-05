"""A comparison family whose unfused part the declarations refuse is withdrawn at derivation."""

from __future__ import annotations

from types import SimpleNamespace

from merlin_experiments.phase0 import comparison_screen

from merlin.targetgen.software_spec import validate_software_spec

_SPEC = {
    "schema": "merlin.software_spec.v1",
    "target": "fixture",
    "status": "reviewed",
    "numerical_semantics": {
        "model": {"engine": "integer_reference"},
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "subnormal_operand_flush": False,
        "overflow": "wrap_internal_mac",
    },
    "operations": {
        "contraction": {"placement": "accelerator", "operand_dtypes": ["int8"], "accumulator_dtype": "i32"},
        "elementwise_map": {
            "placement": "fused_accelerator",
            "dtypes": ["int8", "i32"],
            "composed_with": ["contraction"],
            "epilogues": ["relu", "acc_scale", "bias"],
            "scale_granularity": ["tensor"],
        },
    },
}


def _spec(**override):
    return validate_software_spec(dict(_SPEC, **override), "fixture")


def test_a_fused_only_part_withdraws_its_group():
    binding = SimpleNamespace(operand_dtype="int8", accum_dtype="i32")
    variants = [
        {"op": "fused_matmul_bias", "comparison_group": {"name": "g", "role": "fused"}},
        {"op": "matmul", "comparison_group": {"name": "g", "role": "part"}},
        {"op": "bias_add", "comparison_group": {"name": "g", "role": "part"}},
    ]
    refusal = comparison_screen.refused_part(variants, {}, software_spec=_spec(), binding=binding)
    assert refusal["member"] == "bias_add" and "composition" in refusal["reason"]
    standalone = dict(_SPEC["operations"]["elementwise_map"])
    standalone.pop("composed_with")
    permissive = _spec(operations={**_SPEC["operations"], "elementwise_map": standalone})
    assert comparison_screen.refused_part(variants, {}, software_spec=permissive, binding=binding) is None
    # Without selected declarations nothing is withdrawn.
    assert comparison_screen.refused_part(variants, {}, software_spec=None, binding=binding) is None
