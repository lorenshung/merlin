"""Coarse target-contract screening and actual native-cell qualification checks."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase0.cell_probe import characterize, verify_characterization
from merlin_experiments.phase0.software_screen import diagnostic_entry, intersect_requirement, screen_entry

from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.operation_numerics import integer_partial_sum_bound
from merlin.targetgen.software_spec import (
    admit_operation,
    capability_contract,
    load_software_spec,
    validate_software_spec,
)
from merlin.targetgen.transfer_contracts import screen_transfer_contract, validate_transfer_contracts


def test_named_author_inputs_normalize_without_granting_review_or_mutating_bytes():
    from copy import deepcopy

    root = Path(__file__).resolve().parents[3]
    raw = yaml.safe_load((root / "examples/gemmini/target/software-spec.yaml").read_bytes())
    original = deepcopy(raw)
    selected = validate_software_spec(raw)
    assert raw == original and "status" not in raw
    assert selected["status"] == "unreviewed"
    assert selected["operations"][0]["id"] == "contraction"
    assert selected["operations"][0]["families"] == ["contraction"]
    assert selected["transfer_contracts"]["status"] == "unreviewed"
    assert selected["transfer_contracts"]["declarations"][0]["id"] == "operand_load"
    assert selected["transfer_contracts"]["declarations"][0]["evidence"] == {}
    assert validate_software_spec(selected) == selected
    assert (
        selected["operations"][0]["signature"]["operand_dtypes"] == (raw["operations"]["contraction"]["operand_dtypes"])
    )
    assert selected["transfer_contracts"]["declarations"][0]["signature"] == {
        "operand_dtype": "int8",
        "result_dtype": "int8",
        "operand_layout": "row_major_contiguous",
        "result_layout": "row_major_contiguous",
    }
    # Compact author inputs and expanded declarations share one consumer contract.
    legacy = deepcopy(raw)
    legacy["operations"] = deepcopy(selected["operations"])
    legacy["transfer_contracts"] = deepcopy(selected["transfer_contracts"])
    assert validate_software_spec(legacy) == selected
    for conflicting in ({"signature": {"ranks": [3]}}, {"rank": 3}):
        bad = deepcopy(raw)
        bad["operations"]["contraction"].update(conflicting)
        with pytest.raises(ValueError, match="mixes flat|unknown fields"):
            validate_software_spec(bad)
    for conflicting in ({"signature": {"operand_dtype": "f32"}}, {"copy": {"dtype": "int8"}}):
        bad = deepcopy(raw)
        bad["transfer_contracts"]["operand_load"].update(conflicting)
        with pytest.raises(ValueError, match="mix compact|explicit dtype and layout"):
            validate_software_spec(bad)
    with pytest.raises(ValueError, match="explicit declarations"):
        validate_transfer_contracts({"status": "reviewed"})
    bad = deepcopy(raw)
    bad["transfer_contracts"]["accumulator_readout"]["valid_window"] = "ignored"
    with pytest.raises(ValueError, match="valid_window"):
        validate_software_spec(bad)
    custom = deepcopy(raw)
    custom["operations"] = {"custom": {"placement": "accelerator", "signature": {"dtypes": ["int8"]}}}
    custom["quantization"]["formats"][0]["eligible_operations"] = ["custom"]
    with pytest.raises(ValueError, match="explicit ops or families"):
        validate_software_spec(custom)
    custom["operations"]["custom"]["ops"] = ["matmul"]
    assert validate_software_spec(custom)["operations"][0]["ops"] == ["matmul"]
    reviewed = deepcopy(raw["transfer_contracts"])
    reviewed["status"] = "reviewed"
    with pytest.raises(ValueError, match="review evidence"):
        validate_transfer_contracts(reviewed)
    reviewed["operand_load"]["evidence"] = {"status": "unqualified"}
    reviewed["accumulator_readout"]["evidence"] = {"status": "unqualified"}
    assert (
        screen_transfer_contract(
            {"transfer_contracts": reviewed},
            source_placement="host",
            destination_placement="accelerator",
            operand_dtype="int8",
            result_dtype="int8",
            operand_layout="row_major_contiguous",
            result_layout="row_major_contiguous",
        )["status"]
        == "unknown"
    )


def test_backend_candidates_and_actual_nested_stages_respect_authored_sw_scope():
    root = Path(__file__).resolve().parents[3]
    spec = load_software_spec(root / "examples/gemmini/target/software-spec.yaml")
    contract = yaml.safe_load((root / "examples/gemmini/target/contracts/target_contract.yaml").read_bytes())
    defaults = spec["numerical_semantics"]
    signature = {
        "rank": 2,
        "layout": "row_major_contiguous",
        "tails": "zero_pad_valid_window",
        "broadcasting": "none",
        "aliasing": "disjoint_inputs_outputs",
    }
    entry = {
        "name": "probe",
        "op": "matmul",
        "cat": "isa",
        "operand_dtype": "int8",
        "placement": "accelerator",
        "operation_signature": signature,
    }
    assert screen_entry(spec, entry, defaults=defaults)["constraints_status"] == "matched"
    micro = {"name": "micro", "kind": "model", "micro_model": True}
    assert screen_entry(spec, micro, defaults=defaults)["status"] == "unknown"
    compound = {
        "name": "compound",
        "kind": "model_slice",
        "op": "host_island_seam",
        "performance": {"emitter": {"knobs": {"operation": "host_island_seam", "accelerator_regions": 2}}},
    }
    assert screen_entry(spec, compound, defaults=defaults)["status"] == "unknown"
    observed_compound = {
        "kind": "model_slice",
        "operation": {"op": "host_island_seam", "attributes": {"accelerator_contractions": 2}},
    }
    assert screen_entry(spec, compound, defaults=defaults, capsule=observed_compound)["status"] == "unknown"
    assert screen_entry(spec, {"op": "undeclared", "kind": "model_slice"}, defaults=defaults)["status"] == "unsupported"
    unknown_compound = {
        **compound,
        "op": "undeclared",
        "performance": {"emitter": {"knobs": {"operation": "undeclared", "accelerator_regions": 2}}},
    }
    assert screen_entry(spec, unknown_compound, defaults=defaults)["status"] == "unsupported"
    standalone_bias = {**entry, "kind": "model_slice", "op": "bias_add", "epilogue": ["bias_add"]}
    assert screen_entry(spec, standalone_bias, defaults=defaults)["status"] == "unsupported"
    refused = screen_entry(spec, {**entry, "epilogue": ["maxpool"]}, defaults=defaults)
    assert refused["status"] == "unsupported" and refused["decisions"][1]["constraints_status"] == "refused"
    assert (
        screen_entry(spec, {**entry, "kind": "model_slice", "epilogue": ["maxpool"]}, defaults=defaults)["status"]
        == "unsupported"
    )
    assert diagnostic_entry(entry, refused)["cat"] == "_diagnostic"
    actual = {
        "inputs": [{"role": "input", "dtype": "i8"}],
        "operation": {"op": "matmul", "attributes": {"epilogue": ["maxpool"], "output_dtype": "i8"}},
    }
    assert screen_entry(spec, entry, defaults=defaults, capsule=actual)["status"] == "unsupported"
    actual["operation"]["attributes"]["epilogue"] = ["bias_add"]
    bias = screen_entry(spec, entry, defaults=defaults, capsule=actual)
    assert bias["status"] == "unknown" and bias["decisions"][1]["constraints_status"] != "refused"
    actual["inputs"][0]["dtype"] = "f32"
    assert screen_entry(spec, entry, defaults=defaults, capsule=actual)["status"] == "unsupported"
    host = {**entry, "operand_dtype": "f32", "generalization": {"must_accelerate": False, "eligible": False}}
    assert screen_entry(spec, host, defaults=defaults)["status"] == "unknown"
    assert screen_entry(spec, host, defaults=defaults)["scope"].startswith("explicit host-only")
    requirement = {
        "cells": [
            {"cell": "contraction/i8/aligned", "family": "contraction", "dtype": "i8"},
            {"cell": "reduction/i8/aligned", "family": "reduction", "dtype": "i8"},
        ],
        "epilogue": {"required": [{"stage": "maxpool"}, {"stage": "bias_add"}]},
        "application_demands": {"n_operations": 751},
    }
    selected = intersect_requirement(requirement, spec, contract)
    assert [row["family"] for row in selected["cells"]] == ["contraction"]
    assert selected["epilogue"]["required"] == [{"stage": "bias_add"}]
    assert len(selected["software_intersection"]["rejected_backend_capabilities"]) == 2
    assert selected["application_demands"] == requirement["application_demands"]
    assert len(requirement["cells"]) == 2 and spec["status"] == "unreviewed"


def test_example_contracts_are_typed_candidates_not_blanket_admission():
    root = Path(__file__).resolve().parents[3]
    for target, operand, accumulator in (("gemmini", "int8", "i32"), ("atlas", "fp8_e4m3", "bf16")):
        selected = root / "examples" / target / "target"
        spec = load_software_spec(selected / "software-spec.yaml", target=target)
        assert "capability_contract" not in spec and "source_observations" not in spec
        with pytest.raises(ValueError, match="explicit same-target"):
            capability_contract(spec)
        with pytest.raises(ValueError, match="explicit same-target"):
            capability_contract(spec, base_contract={"name": "different"})
        backend = {"name": target, "opaque_provider_policy": {"protocol": ["selected"]}}
        projected = capability_contract(spec, base_contract=backend)
        projected["opaque_provider_policy"]["protocol"].append("mutated")
        assert backend["opaque_provider_policy"]["protocol"] == ["selected"]
        legacy = {**spec, "capability_contract": {"name": target, "legacy": True}}
        assert capability_contract(legacy, base_contract=backend) == legacy["capability_contract"]
        host = validate_host_capabilities(yaml.safe_load((selected / "host-capabilities.yaml").read_bytes()))
        assert host["operations"] and host["status"] == "unreviewed"
        signature = {
            "family": "contraction",
            "operand_dtype": operand,
            "accum_dtype": accumulator,
            "rank": 2,
            "layout": "row_major_contiguous",
            "tails": "zero_pad_valid_window",
            "broadcasting": "none",
            "aliasing": "disjoint_inputs_outputs",
        }
        decision = admit_operation(spec, "matmul", signature, "accelerator")
        assert decision["constraints_status"] == "matched" and decision["status"] == "unknown"
        assert (
            admit_operation(spec, "matmul", {**signature, "operand_dtype": "f32"}, "accelerator")["status"]
            == "unsupported"
        )
        assert admit_operation(spec, "lstm", {**signature, "family": None}, "accelerator")["status"] == "unsupported"
        transfer = screen_transfer_contract(
            spec,
            source_placement="host",
            destination_placement="accelerator",
            operand_dtype=operand,
            result_dtype=operand,
            operand_layout="row_major_contiguous",
            result_layout="row_major_contiguous",
        )
        assert transfer["matching_declarations"] and transfer["status"] == "unknown"
    integer = load_software_spec(root / "examples/gemmini/target/software-spec.yaml")["numerical_semantics"]
    # i32 storage and a small final sum cannot rescue an overflowing i20 prefix.
    assert (
        integer_partial_sum_bound(integer, reduction_extent=31, lhs_values=[-128], rhs_values=[-128])["status"]
        == "proven_safe"
    )
    assert (
        integer_partial_sum_bound(integer, reduction_extent=32, lhs_values=[-128], rhs_values=[-128])["status"]
        == "may_overflow"
    )
    assert (
        integer_partial_sum_bound(integer, reduction_extent=1, lhs_values=[1], rhs_values=[1], initial_values=[524287])[
            "status"
        ]
        == "may_overflow"
    )


def test_native_cell_receipt_checks_differential_execution_and_artifact_integrity(tmp_path):
    circt, verilator = os.environ.get("MERLIN_TEST_CIRCT_OPT"), os.environ.get("MERLIN_TEST_VERILATOR")
    if not circt or not verilator:
        pytest.skip("explicit CIRCT/Verilator tool selection required for native characterization")
    source = tmp_path / "cell.mlir"
    source.write_text(
        "module {\n  hw.module @Cell(in %a: i8, in %b: i8, out d: i8) {\n"
        "    %sum = comb.add %a, %b : i8\n    hw.output %sum : i8\n  }\n}\n"
    )
    cases = [
        {"inputs": {"a": a, "b": b}, "expected": {"d": (a + b) & 255}}
        for a in (0, 127, 128, 255)
        for b in (0, 1, 128, 255)
    ]
    output = tmp_path / "good"
    result = characterize(
        hw_source=source,
        module="Cell",
        inputs={"a": 8, "b": 8},
        outputs={"d": 8},
        cases=cases,
        circt_opt=circt,
        verilator=verilator,
        output=output,
        reference={"engine": "independent Python modular sum"},
        domain={"vectors": "explicit signed-wrap boundaries"},
    )
    assert result["status"] == "passed", json.dumps(result)
    assert verify_characterization(output / "characterization.json")["whole_operation_support"] == "not_qualified"
    (output / "vectors.txt").write_text("0 0 1\n")
    with pytest.raises(ValueError, match="artifact changed"):
        verify_characterization(output / "characterization.json")
    wrong = [{"inputs": {"a": 0, "b": 0}, "expected": {"d": 1}}]
    mismatch = characterize(
        hw_source=source,
        module="Cell",
        inputs={"a": 8, "b": 8},
        outputs={"d": 8},
        cases=wrong,
        circt_opt=circt,
        verilator=verilator,
        output=tmp_path / "wrong",
        reference={"engine": "deliberately wrong counterfactual"},
        domain={"vectors": "one incorrect output"},
    )
    assert mismatch["status"] == "mismatch"
    with pytest.raises(ValueError, match="successful cell execution"):
        verify_characterization(tmp_path / "wrong/characterization.json")
