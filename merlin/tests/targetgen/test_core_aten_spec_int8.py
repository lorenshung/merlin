"""The current reviewed spec must fail closed without widening its declarations."""

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
        spec=resolved, host=validate_host_capabilities(load_yaml(base / "host-capabilities.yaml")), resolution=record
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


def test_product_type_and_transfer_are_independent_blockers(selection):
    before = copy.deepcopy(selection)
    reasons = pipeline_refusals(selection)
    assert any("acc_scale_product_dtype" in reason for reason in reasons)
    assert any("transfer" in reason for reason in reasons)
    assert selection == before
    # Fixture-only hypothetical declaration, never an edit to the reviewed spec.
    proposed = copy.deepcopy(selection)
    proposed["spec"]["numerical_semantics"]["readout"]["acc_scale_product_dtype"] = "f32"
    assert len(pipeline_refusals(proposed)) == 1


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
    assert plan["calibration"]["source"] == "calibration-on-input"
    (site,) = plan["sites"]
    assert site["contraction_admission"]["status"] == "admitted"
    assert site["proposed_signature"]["ordered_operand_dtypes"] == ["i8", "i8", "i32"]
    assert site["source_dtypes"] == ["f32", "f32", "f32"]
    assert site["status"] == "refused"
    assert site["partial_sum_obligation"]["max_true_k"] == 31
    assert site["actual_dispatch"].startswith("none")


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
