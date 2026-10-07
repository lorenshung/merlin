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
