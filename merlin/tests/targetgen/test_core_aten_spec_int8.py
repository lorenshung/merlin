"""Exact per-site admission and static i32/f32 chain qualification."""

import copy

import numpy as np
import pytest
from xdsl.parser import Parser

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.frontends.linalg_mlir import make_context
from merlin.targetgen.core_aten_spec_int8 import (
    DEQUANTIZE,
    QUANTIZE,
    admission_plan,
    host_signature,
    pipeline_refusals,
)
from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.software_spec import admit_operation, load_software_spec
from merlin.targetgen.spec_fact_drift import resolve_spec


@pytest.fixture
def selection():
    base = repo_root() / "examples/gemmini/target"
    spec = load_software_spec(base / "software-spec.yaml")
    resolved, record = resolve_spec(spec, {})
    return dict(
        spec=resolved,
        host=validate_host_capabilities(load_yaml(base / "host-capabilities.yaml")),
        resolution=record,
        identities={
            name: {"path": str(base / filename)}
            for name, filename in (
                ("software_spec", "software-spec.yaml"),
                ("host_capabilities", "host-capabilities.yaml"),
            )
        },
    )


@pytest.mark.parametrize("rank,expected", [(2, "unsupported"), (3, "admitted"), (4, "admitted"), (5, "unsupported")])
def test_exact_dequantize_rank_and_dtype(selection, rank, expected):
    signature = host_signature(DEQUANTIZE, rank, dict(zero_point=0))
    for authority in ("spec", "host"):
        assert admit_operation(selection[authority], DEQUANTIZE, signature, "host")["status"] == expected
        for operand in (["i32", "f32", "i64"], ["i8", "f64", "i64"], ["i8", "f32", "i32"]):
            wrong = {**signature, "ordered_operand_dtypes": operand}
            assert admit_operation(selection[authority], DEQUANTIZE, wrong, "host")["status"] != "admitted"


def test_exact_quantize_parameters_and_typed_zero_point(selection):
    signature = host_signature(QUANTIZE, 3, dict(zero_point=0, quant_min=-128, quant_max=127))
    for authority in ("spec", "host"):
        assert admit_operation(selection[authority], QUANTIZE, signature, "host")["status"] == "admitted"
        for parameters in (
            dict(zero_point=1, quant_min=-128, quant_max=127),
            dict(zero_point=0, quant_min=-127, quant_max=127),
        ):
            wrong = {**signature, "quantization_parameters": parameters}
            assert admit_operation(selection[authority], QUANTIZE, wrong, "host")["status"] != "admitted"
        wrong = {**signature, "ordered_operand_dtypes": ["f32", "f32", "i32"]}
        assert admit_operation(selection[authority], QUANTIZE, wrong, "host")["status"] != "admitted"


def test_raw_i32_chain_does_not_require_narrowing_product_semantics(selection):
    before = copy.deepcopy(selection)
    assert pipeline_refusals(selection) == []
    assert selection == before
    proposed = copy.deepcopy(selection)
    proposed["host"]["status"] = "unreviewed"
    assert pipeline_refusals(proposed)


def test_rank3_refusal_retains_float_source_and_numerical_bound(selection):
    source = """builtin.module {
      func.func @forward(%a: tensor<2x3x40xf32>, %b: tensor<2x40x4xf32>) -> tensor<2x3x4xf32> {
        %c = arith.constant 0.0 : f32
        %e = tensor.empty() : tensor<2x3x4xf32>
        %z = linalg.fill ins(%c : f32) outs(%e : tensor<2x3x4xf32>) -> tensor<2x3x4xf32>
        %r = linalg.generic {
          indexing_maps = [affine_map<(b,m,n,k) -> (b,m,k)>,
                           affine_map<(b,m,n,k) -> (b,k,n)>, affine_map<(b,m,n,k) -> (b,m,n)>],
          iterator_types = ["parallel", "parallel", "parallel", "reduction"]
        } ins(%a, %b : tensor<2x3x40xf32>, tensor<2x40x4xf32>) outs(%z : tensor<2x3x4xf32>) {
          ^bb0(%x: f32, %y: f32, %z0: f32):
            %v = arith.mulf %x, %y : f32
            %w = arith.addf %z0, %v : f32
            linalg.yield %w : f32
        } -> tensor<2x3x4xf32>
        func.return %r : tensor<2x3x4xf32>
      }
    }"""
    module = Parser(make_context(), source).parse_module()
    before = str(module)
    plan = admission_plan(module, selection)
    assert str(module) == before
    assert plan["quantized_contractions"] == 0
    assert plan["calibration"]["performed"] is False
    assert plan["calibration"]["source"] == "calibration-on-evaluation-input"
    (site,) = plan["sites"]
    contraction = next(row for row in site["steps"] if row["step"] == 3)
    assert contraction["software"]["status"] == "admitted"
    assert contraction["signature"]["ordered_operand_dtypes"] == ["i8", "i8", "i32"]
    assert site["source_dtypes"] == ["f32", "f32", "f32"]
    assert site["status"] == "refused"
    assert site["partial_sum_obligation"]["max_true_k"] == 31
    assert len(site["reasons"]) == 1
    assert "rank 3 not in [2]" in site["reasons"][0]
    assert "host-capabilities.yaml:62" in site["reasons"][0]


def test_dynamic_verdict_is_explicitly_diagnostic():
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    report = dict(provenance={}, cases=[dict(overload="aten.mm.default", status="bundled", numeric_policy="int8_w8a8")])
    verdict = grade_core_aten_batch(report, execution_error="not run")["cases"]["aten.mm.default"]
    assert verdict["numeric_qualification"] == "diagnostic, not spec-conformant"


def test_spec_int8_host_pass_is_counted_only_as_original_host():
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    case = dict(expected=dict(kind="tensor", dtype="float32", shape=[1], values=[2.0]), comparison="torch_close")
    report = dict(
        provenance={},
        cases=[
            dict(
                overload="aten.mm.default",
                status="bundled",
                case=case,
                numeric_policy="spec_int8",
                numeric_admission=dict(status="refused", quantized_contractions=0),
                output_indices=[0],
                output_abi=[dict(dtype="f32", shape=[1])],
            )
        ],
    )
    result = grade_core_aten_batch(report, [np.array([2.0], np.float32).tobytes()])
    verdict = result["cases"]["aten.mm.default"]
    assert verdict["status"] == verdict["numeric_status"] == "pass"
    assert verdict["numeric_execution_policy"] == "original_host"
    assert verdict["numeric_qualification"] == "original host fallback; no spec-int8 result"
    assert result["numeric_passed_by_lane_and_policy"]["host"]["original_host"] == 1
    assert result["numeric_passed_by_lane_and_policy"]["host"]["spec_int8"] == 0
    assert result["numeric_passed_by_lane_and_policy"]["device"]["spec_int8"] == 0
    assert result["spec_int8_passed_count"] == 0
    assert result["spec_int8_host_fallback_passed_count"] == 1


@pytest.mark.parametrize("evidence_valid", [False, True])
def test_capture_spec_grade_keeps_dynamic_gate_and_host_counts_separate(monkeypatch, evidence_valid):
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    monkeypatch.setattr("merlin.targetgen.core_aten_device.verify_execution_evidence", lambda _: evidence_valid)
    case = dict(expected=dict(kind="tensor", dtype="float32", shape=[1], values=[2.0]), comparison="torch_close")
    raw = np.array([2.0], np.float32).tobytes()
    record = dict(
        overload="aten.mm.default",
        status="bundled",
        case=case,
        fp32_case=copy.deepcopy(case),
        numeric_policy="spec_int8",
        output_indices=[0],
        output_abi=[dict(dtype="f32", shape=[1])],
        numeric_admission=dict(
            status="admitted",
            quantized_contractions=1,
            integer_reference=dict(status="prepared", engine="independent", expected_output_bytes={"0": raw.hex()}),
        ),
        routing=dict(
            lane="device",
            routed=True,
            target="gemmini",
            routed_operations=[{}],
            execution_evidence=dict(
                case_id="aten.mm.default",
                status="measured",
                attribution="single_case_shard",
                executed_instructions=1,
                target="gemmini",
            ),
        ),
    )
    result = grade_core_aten_batch(dict(provenance={}, cases=[record]), [raw])
    row = result["cases"][record["overload"]]
    assert row["integer_reference"]["status"] == "pass"
    assert row["fp32_error"][0]["max_abs"] == 0
    assert "w8a8_policy" not in row
    assert row["status"] == ("pass" if evidence_valid else "execution_failed")
    assert result["spec_int8_passed_count"] == int(evidence_valid)
    assert result["spec_int8_host_fallback_passed_count"] == 0


def test_fp32_metrics_distinguish_zero_golden_and_zero_error():
    from merlin.targetgen.core_aten_numeric import fp32_error

    record = dict(
        output_indices=[0],
        output_abi=[dict(dtype="f32")],
        fp32_case=dict(expected=dict(kind="tensor", dtype="float32", shape=[2], values=[0.0, 1.0])),
    )
    (observed,) = fp32_error(record, [np.array([1.0, 1.0], np.float32).tobytes()])
    assert observed["max_rel"] == float("inf")
    assert observed["relative_error_at_zero"] == 1
    assert observed["rmse"] == pytest.approx(2**-0.5)
    assert observed["normalized_l2"] == 1.0
    assert observed["sqnr_db"] == 0.0


def rank2_module():
    return Parser(
        make_context(),
        """builtin.module {
      func.func @forward(%a: tensor<3x40xf32>, %b: tensor<40x4xf32>) -> tensor<3x4xf32> {
        %c = arith.constant 0.0 : f32
        %e = tensor.empty() : tensor<3x4xf32>
        %z = linalg.fill ins(%c : f32) outs(%e : tensor<3x4xf32>) -> tensor<3x4xf32>
        %r = linalg.matmul ins(%a, %b : tensor<3x40xf32>, tensor<40x4xf32>)
          outs(%z : tensor<3x4xf32>) -> tensor<3x4xf32>
        func.return %r : tensor<3x4xf32>
      }
    }""",
    ).parse_module()


def test_rank2_chain_and_each_independent_host_rank_restriction(selection):
    module = rank2_module()
    plan = admission_plan(module, selection)
    (site,) = plan["sites"]
    assert site["status"] == "admitted", site["reasons"]
    assert {row["step"] for row in site["steps"]} == {1, 2, 3, 4, 5, 6}
    for identity, step in (
        ("host_quantize_per_tensor_int8", 1),
        ("host_i32_to_f32_cast", 5),
        ("host_f32_elementwise_exact", 6),
    ):
        restricted = copy.deepcopy(selection)
        row = next(row for row in restricted["host"]["operations"] if row["id"] == identity)
        row["signature"]["ranks"] = [3]
        (refused,) = admission_plan(module, restricted)["sites"]
        assert refused["status"] == "refused"
        assert all(f"step {step}" in r for r in refused["reasons"])


def test_integer_reference_compares_signed_zero_bits():
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    record = dict(
        case_id="test",
        overload="test",
        status="bundled",
        numeric_policy="spec_int8",
        output_indices=[],
        output_abi=[],
        case=dict(expected=[]),
        numeric_admission=dict(
            integer_reference=dict(
                status="prepared",
                engine="independent",
                expected_output_bytes={"0": np.array([-0.0], np.float32).tobytes().hex()},
            )
        ),
    )
    result = grade_core_aten_batch(dict(provenance={}, cases=[record]), [np.array([0.0], np.float32).tobytes()])
    assert result["cases"]["test"]["integer_reference"]["status"] == "mismatch"
    assert result["cases"]["test"]["numeric_status"] == "mismatch"


def test_static_capture_freezes_original_float_intermediate_scales(tmp_path, selection):
    import os
    import subprocess

    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("TorchAO capture interpreter is not configured")
    script = r"""
import sys
import torch
import m2m
from merlin.targetgen import _recipe_quantizer as quant
from merlin.targetgen.core_aten_w8a8_reference import run_pt2e_reference
from merlin.capture.integerization import combine_static_readout_scales
from m2m.capture.pt2e_integerize import integerize_pt2e
recipe = {
    "schema": "quant_recipe_v1", "status": "derived", "families": ["contraction"],
    "weight": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
               "quant_min": -127, "quant_max": 127},
    "activation": {"dtype": "int8", "granularity": "tensor", "symmetric": True,
                   "quant_min": -128, "quant_max": 127, "mode": "static"},
}
class Model(torch.nn.Module):
    def forward(self, a, b, c):
        return (a @ b) @ c
inputs = (
    torch.tensor([[1.25, -.91, .3], [.67, .15, -.79]]),
    torch.tensor([[.93, -1.1], [-.76, .23], [.41, .87]]),
    torch.tensor([[.7, -.1], [-.5, 1.1]]),
)
converted = quant.apply_recipe(Model().eval(), recipe, example_inputs=inputs)
reference = run_pt2e_reference(converted, inputs)
scale, observer = quant.observe_static_tensor(inputs[0] @ inputs[1], recipe["activation"])
assert reference.sites[1]["lhs"] == scale.item()
assert observer == "histogram"
integer, receipt = integerize_pt2e(converted, inputs)
products = combine_static_readout_scales(integer)
assert products == [site["product"] for site in reference.sites]
assert torch.equal(integer(*inputs).view(torch.int32), reference.output.view(torch.int32))
text = m2m.convert(integer, inputs, backend="fx_importer").mlir_text
open(sys.argv[1], "w").write(text)
"""
    path = tmp_path / "captured.mlir"
    result = subprocess.run([python, "-c", script, str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr[-5000:]
    module = Parser(make_context(), path.read_text()).parse_module()
    plan = admission_plan(module, selection, captured=True)
    assert len(plan["sites"]) == 2
    assert len(plan["quantization_sites"]) == 4
    assert all(
        row[authority]["status"] == "admitted"
        for row in plan["quantization_sites"]
        for authority in ("software", "host")
    )
    assert all(site["status"] == "admitted" for site in plan["sites"]), plan
    from merlin.kernels.shapes import observe_contractions

    for contraction, _ in observe_contractions(module):
        casts = [use.operation for use in contraction.results[0].uses]
        assert len(casts) == 1
        assert [op.name for op in casts[0].body.block.ops] == ["arith.sitofp", "linalg.yield"]
        readout = next(iter(casts[0].results[0].uses)).operation
        assert [op.name for op in readout.body.block.ops] == ["arith.mulf", "linalg.yield"]
