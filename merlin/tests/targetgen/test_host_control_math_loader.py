"""Neutral source-fixture matrix tests; no host placement or native claim."""

from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import repo_root

torch = pytest.importorskip("torch")
_SOURCE = repo_root() / "examples/workloads/host_control_math/loader.py"
_SPEC = importlib.util.spec_from_file_location("host_control_math_probe", _SOURCE)
assert _SPEC is not None and _SPEC.loader is not None
_PROBE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_PROBE)


@pytest.mark.parametrize("rank", (1, 2, 3, 4))
def test_matrix_cases_keep_independent_raw_outputs_and_tail(monkeypatch, rank):
    shape = (2, 2, 3, 7)[-rank:]
    expected = {
        "f32_add": torch.float32,
        "f32_le": torch.bool,
        "i64_add": torch.int64,
        "i64_le": torch.bool,
        "i64_to_f32": torch.float32,
        "i1_xor": torch.bool,
        "i1_to_f32": torch.float32,
        "i1_mul_lhs_projected": torch.bool,
        "i1_mul_rhs_projected": torch.bool,
    }
    for operation, dtype in expected.items():
        monkeypatch.setenv("M2M_HOST_PROBE_CASE", f"matrix_{operation}_r{rank}")
        model, inputs = _PROBE.get_model_and_inputs()
        result = model(*inputs)
        assert result.dtype == dtype
        assert tuple(result.shape) == shape
        assert result.numel() == torch.Size(shape).numel()
        if operation in {"f32_add", "i64_add", "f32_le", "i64_le", "i1_xor"}:
            assert inputs[0].data_ptr() != inputs[1].data_ptr()
    monkeypatch.setenv("M2M_HOST_PROBE_CASE", f"matrix_i64_add_r{rank}")
    _, inputs = _PROBE.get_model_and_inputs()
    assert inputs[0].flatten()[0].item() == -(2**63)
    assert inputs[1].flatten()[0].item() == 1


@pytest.mark.parametrize("case", ("matrix_i64_add_r0", "matrix_i64_add_r5", "matrix_unknown_r1"))
def test_matrix_refuses_unknown_selection(monkeypatch, case):
    monkeypatch.setenv("M2M_HOST_PROBE_CASE", case)
    with pytest.raises(ValueError, match="unknown host-control/math case"):
        _PROBE.get_model_and_inputs()


@pytest.mark.parametrize(
    ("case", "dtype", "shape"),
    [
        ("sine_rank4", torch.float32, (1, 1, 2, 4)),
        ("cosine_rank4", torch.float32, (1, 1, 2, 4)),
        ("arange_f32_exact", torch.float32, (8,)),
        ("arange_f32_fractional", torch.float32, (20,)),
        ("boolean_not", torch.float32, (3, 5)),
        ("boolean_and", torch.float32, (3, 5)),
        ("boolean_select", torch.float32, (3, 5)),
        ("boolean_mul_lhs_singleton", torch.bool, (3, 7)),
        ("boolean_mul_rhs_singleton", torch.bool, (3, 7)),
        ("tensor_not_equal", torch.bool, (2, 5)),
        ("tensor_bitwise_xor", torch.bool, (9,)),
        ("min_values", torch.float32, (3,)),
        ("f32_cumsum_rows", torch.float32, (2, 5)),
        ("f32_cumsum_columns", torch.float32, (3, 2)),
        ("f32_cumsum_signed_zero", torch.float32, (6,)),
        ("bucketize_left_nan", torch.float32, (2, 5)),
        ("bucketize_right_nan", torch.float32, (2, 5)),
    ],
)
def test_extended_probe_cases_retain_source_signatures(monkeypatch, case, dtype, shape):
    monkeypatch.setenv("M2M_HOST_PROBE_CASE", case)
    model, inputs = _PROBE.get_model_and_inputs()
    result = model(*inputs)
    assert result.dtype == dtype
    assert tuple(result.shape) == shape
    if len(inputs) == 2:
        assert inputs[0].data_ptr() != inputs[1].data_ptr()
    if case.startswith("boolean_mul_"):
        assert tuple(inputs[0].shape) != tuple(inputs[1].shape)


def test_ordered_scan_inputs_expose_intermediate_rounding_and_zero_sign(monkeypatch):
    monkeypatch.setenv("M2M_HOST_PROBE_CASE", "f32_cumsum_rows")
    model, inputs = _PROBE.get_model_and_inputs()
    assert torch.equal(model(*inputs)[:, -1], torch.tensor([1.0, 2.0]))
    # An FP32 running accumulator loses the small addend before cancellation.
    assert ((inputs[0][:, 0] + inputs[0][:, 1]) + inputs[0][:, 2]).tolist() == [0.0, 0.0]
    monkeypatch.setenv("M2M_HOST_PROBE_CASE", "f32_cumsum_signed_zero")
    model, inputs = _PROBE.get_model_and_inputs()
    assert torch.signbit(inputs[0]).tolist()[:4] == [True, False, True, False]
    result = model(*inputs)
    assert result.tolist() == [0.0, 0.0, 0.0, 0.0, 1.0, 0.0]
    assert not torch.signbit(result).any()


def test_nonfinite_bucketization_distinguishes_bound_side(monkeypatch):
    outputs = []
    for case in ("bucketize_left_nan", "bucketize_right_nan"):
        monkeypatch.setenv("M2M_HOST_PROBE_CASE", case)
        model, inputs = _PROBE.get_model_and_inputs()
        assert torch.isnan(inputs[0][0, 0])
        assert torch.isneginf(inputs[0][0, 1]) and torch.isposinf(inputs[0][1, 0])
        outputs.append(model(*inputs).tolist())
    assert outputs[0] == [[4.0, 0.0, 1.0, 2.0, 2.0], [4.0, 2.0, 2.0, 3.0, 4.0]]
    assert outputs[1] == [[4.0, 0.0, 1.0, 2.0, 2.0], [4.0, 2.0, 3.0, 3.0, 4.0]]


def test_tensor_comparison_does_not_round_i64_inputs_to_float(monkeypatch):
    monkeypatch.setenv("M2M_HOST_PROBE_CASE", "tensor_not_equal")
    model, inputs = _PROBE.get_model_and_inputs()
    result = model(*inputs)
    assert result.tolist() == [[False, True, True, False, True], [False, True, True, False, True]]
    rounded_equal = inputs[0].to(torch.float32) == inputs[1].to(torch.float32)
    assert (rounded_equal & result).any()
    assert result[0, -1] and inputs[0][0, -1].item() == 2**63 - 1
