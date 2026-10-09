"""Concrete deterministic numeric stresses use the ordinary independent engines."""

import ast
import math

import numpy as np
import pytest
import test_component_coverage as coverage_fixtures
import test_component_generation as generation_fixtures
import yaml
from merlin_experiments.phase0 import component_coverage, generation

from merlin.runtime.fp8_formats import float_to_codes, storage_bits
from merlin.targetgen import capsule_source, golden_store, input_palette

independent = generation_fixtures.independent


def palette(name, values, *, axis=-1, offset=0):
    return {
        "schema": input_palette.SCHEMA,
        "inputs": [{"name": name, "axis": axis, "values": values, "offset": offset}],
    }


@pytest.mark.parametrize("dtype", ["f32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2"])
def test_format_landmarks_and_zero_sign_are_exact(dtype):
    values = [input_palette.scalar(token, dtype) for token in ["zero", "-zero", "min_subnormal", "half_ulp_one"]]
    codes = float_to_codes(values, dtype)
    assert codes[0] == 0 and codes[1] == 1 << (storage_bits(dtype) - 1)
    assert values[2] > 0 and math.copysign(1, values[1]) == -1
    assert input_palette.findings(
        palette("X", ["zero", "-zero", "min_subnormal"]), name="X", shape=[2, 3], dtype=dtype
    ) == ["signed_zero_input", "subnormal_input"]


def test_axis_patterns_are_deterministic_and_refuse_missing_values():
    declaration = palette("arg0", ["cancellation_large", 1, "-cancellation_large"], axis=1)
    values = input_palette.realize(declaration, name="X", shape=[2, 3], dtype="f32", index=0)
    assert values[:3] == values[3:] == [2**24, 1, -(2**24)]
    assert "cancellation_input" in input_palette.findings(declaration, name="X", shape=[2, 3], dtype="f32", index=0)
    with pytest.raises(ValueError, match="too short"):
        input_palette.realize(declaration, name="X", shape=[2, 2], dtype="f32", index=0)
    with pytest.raises(ValueError, match="not exactly representable"):
        input_palette.scalar(0.1, "bf16")
    with pytest.raises(ValueError, match="NaN"):
        input_palette.scalar("nan", "f32")


@pytest.mark.parametrize("tile", [2, 3])
def test_f32_cancellation_ties_signed_zero_subnormals_through_normal_generator(independent, tile):
    coverage_fixtures.update_hardware(independent, tile=tile)
    generation_fixtures.floating_sweep(independent, "f32", "attention_qk")
    patterns = {
        "cancel": (["cancellation_large", 1, "-cancellation_large"], ["cancellation_input"]),
        "ties": (["one", "half_ulp_one", "zero"], ["rounding_tie_input"]),
        "zero_subnormal": (["zero", "-zero", "min_subnormal"], ["signed_zero_input", "subnormal_input"]),
    }
    obligations, effects = [], []
    for identity, (values, kinds) in patterns.items():
        selected = palette("Q", values)
        selected["inputs"].append({"name": "K", "axis": "linear", "values": ["one"], "offset": 0})
        obligations.append(
            {
                "id": identity,
                "mandatory": True,
                "cohort": "functional_guard",
                "operations": ["contraction"],
                "effects": kinds,
                "expectation": "admitted_program",
                "frontend": "mlir",
                "base": {"op": "attention_qk", "kind": "isa", "K": 3 * tile, "input_palette": selected},
                "axes": {
                    "M": {"kind": "extent", "values": ["tile", "tile+1"]},
                    "N": {"kind": "extent", "values": ["tile+2"]},
                },
                "interactions": [],
            }
        )
        effects.extend(
            {"id": kind, "status": "reviewed", "kind": kind, "basis": "explicit independent operand landmarks"}
            for kind in kinds
        )
    coverage_fixtures.plan_for(independent, obligations, effects=effects)
    generation.generate_target("fixture", **independent)
    report = component_coverage.verify_report(independent["output_root"])
    for obligation in report["obligations"]:
        required = set(patterns[obligation["id"]][1])
        for member in obligation["members"]:
            assert required <= set(member["input_palette_witness"]["effects"])
            root = independent["output_root"] / member["member"]
            cap = yaml.safe_load((root / "capsule.yaml").read_bytes())
            golden = golden_store.load_golden(root)
            operands = golden["oracle_provenance"]["inputs"]
            q = np.asarray(operands["Q"]["decoded"], dtype=np.float32).reshape(cap["inputs"][0]["shape"])
            k = np.asarray(operands["K"]["decoded"], dtype=np.float32).reshape(cap["inputs"][1]["shape"])
            np.testing.assert_array_equal(golden["outputs"]["Y0"], q @ k.T)


@pytest.mark.parametrize("operation", ["attention_residual_norm", "mlp_residual", "conv_residual_pool"])
def test_connected_closed_pytorch_sources_preserve_selected_numeric_inputs(operation):
    spec = {
        "op": operation,
        "M": 3,
        "K": 5,
        "N": 7,
        "Dv": 4,
        "Himg": 8,
        "Wimg": 10,
        "P": 3,
        "Pool": 2,
        "causal": True,
        "dtype": "f32",
        "input_palette": palette("arg0", ["one", "half_ulp_one", "-zero", "min_subnormal"]),
    }
    source = capsule_source.build_loader_src(spec)
    module = ast.parse(source)
    functions = {node.name for node in module.body if isinstance(node, ast.FunctionDef)}
    assert "get_model_and_inputs" in functions and "_original_get_model_and_inputs =" in source
    assert operation in capsule_source.supported_ops()
    assert capsule_source.build_loader_src(spec) == source
    assert "torch.tensor(flat" in source


@pytest.mark.parametrize("field", ["N", "Dv", "Himg", "P", "eps", "cap", "causal"])
def test_builtin_source_scalar_selectors_cannot_be_program_text(field):
    with pytest.raises(ValueError, match="must be an explicit"):
        capsule_source.build_loader_src({"op": "attention_residual_norm", field: "__import__('os').getcwd()"})


@pytest.mark.parametrize("scenario", ["nonfinite_input", "overflow_output"])
def test_explicit_selected_finite_domain_generates_concrete_refusal(independent, scenario):
    generation_fixtures.floating_sweep(independent, "f32", "attention_qk")
    software = yaml.safe_load(independent["software_spec"].read_bytes())
    software["numerical_semantics"].update(input_domain={"nonfinite": "forbid"}, output_domain={"nonfinite": "forbid"})
    generation_fixtures.write(independent["software_spec"], software)
    values = ["infinity"] if scenario == "nonfinite_input" else ["max_finite"]
    selected = palette("Q", values)
    selected["inputs"].append(
        {"name": "K", "axis": "linear", "values": [1 if scenario == "nonfinite_input" else 2], "offset": 0}
    )
    obligation = {
        "id": scenario,
        "mandatory": True,
        "cohort": "functional_guard",
        "operations": ["contraction"],
        "effects": [],
        "expectation": "unsupported_program",
        "frontend": "mlir",
        "base": {"op": "attention_qk", "kind": "isa", "M": 2, "K": 3, "N": 4, "input_palette": selected},
        "axes": {},
        "interactions": [],
    }
    coverage_fixtures.plan_for(independent, [obligation])
    with np.errstate(over="ignore", invalid="ignore"):
        generation.generate_target("fixture", **independent)
    report = component_coverage.verify_report(independent["output_root"])
    member = report["obligations"][0]["members"][0]
    assert report["obligations"][0]["state"] == "verified_refusal"
    root = independent["output_root"] / member["member"]
    capsule = yaml.safe_load((root / "capsule.yaml").read_bytes())
    domain = capsule["software_screen"]["numeric_domain"]
    field = "input_domain" if scenario == "nonfinite_input" else "output_domain"
    assert domain["status"] == "unsupported" and domain["observations"][field]["nonfinite_elements"] > 0
