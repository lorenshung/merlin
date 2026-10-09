"""The capture worker's receipts live in a sibling module the worker can import in both layouts.

The worker runs as a script under the capture interpreter and is imported as a package module by
Merlin. Its receipts (output agreement, FP32 staging observation, the exported integer contraction
count and the realized scheme) must behave the same either way, and the capture cache must bind
their source bytes, or an edit to a receipt would reuse captures recorded under the old rule.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from merlin.common.paths import module_source_path
from merlin.targetgen import _capture_receipts as CR
from merlin.targetgen import _m2m_capture_worker as worker
from merlin.targetgen.capture_cache import implementation_identity

_INT8_STATIC = "int8_static_act_int8_weight"
_INT8_DYNAMIC = "int8_dyn_act_int8_weight"


def _realized(**overrides):
    selection = {
        "requested": None,
        "recipe_selected": True,
        "already_quantized": False,
        "applied_scheme": None,
        "quant_stats": None,
        "integerization_ok": False,
    }
    selection.update(overrides)
    return CR.realized_scheme(**selection)


def test_worker_uses_the_sibling_receipts_module():
    assert worker.CR is CR


def test_worker_imports_receipts_as_a_script_sibling():
    path = module_source_path("merlin.targetgen._m2m_capture_worker")
    probe = (
        "import runpy, sys\n"
        f"namespace = runpy.run_path({str(path)!r}, run_name='_probe')\n"
        "print(namespace['CR'].__name__)\n"
    )
    done = subprocess.run([sys.executable, "-I", "-c", probe], capture_output=True, text=True, check=False)
    assert done.returncode == 0, done.stderr
    assert done.stdout.strip() == "_capture_receipts"


def test_capture_cache_binds_the_receipt_source():
    identity = implementation_identity()
    assert identity["merlin.targetgen._capture_receipts"]["path"] == str(
        module_source_path("merlin.targetgen._capture_receipts")
    )


def test_scheme_named_capture_records_its_request_and_materialized_records_none():
    assert _realized(recipe_selected=False, requested=_INT8_DYNAMIC) == _INT8_DYNAMIC
    assert _realized(recipe_selected=False, requested=None) is None
    assert _realized(recipe_selected=False, requested=_INT8_DYNAMIC, already_quantized=True) is None


def test_static_int8_recipe_requires_a_passing_integer_receipt():
    stats = {"api": "pt2e"}
    assert _realized(applied_scheme=_INT8_STATIC, quant_stats=stats, integerization_ok=True) == _INT8_STATIC
    assert _realized(applied_scheme=_INT8_STATIC, quant_stats=stats, integerization_ok=False) is None
    assert _realized(applied_scheme=_INT8_STATIC, quant_stats={"api": "quantize_"}, integerization_ok=True) is None


@pytest.mark.parametrize(
    ("scheme", "stats", "expected"),
    [
        (_INT8_DYNAMIC, {"api": "quantize_", "layers_quantized": 2}, True),
        (_INT8_DYNAMIC, {"api": "quantize_", "layers_quantized": 0}, False),
        ("fp8_e4m3_static_act_weight", {"api": "pt2e", "annotated_contractions": 1}, True),
        ("fp8_e4m3_static_act_weight", {"api": "pt2e", "annotated_contractions": 0}, False),
        ("fp8_e4m3_dynamic_act_weight", {"api": "quantize_", "layers_quantized": 3}, True),
        ("fp8_e4m3_static_act_weight", {"api": "quantize_", "layers_quantized": 3}, False),
        ("fp8_e4m3_dynamic_act_weight", {"api": "quantize_", "layers_quantized": 0}, False),
        ("unregistered_scheme", {"api": "quantize_", "layers_quantized": 3}, False),
    ],
)
def test_recipe_scheme_is_recorded_only_when_the_transform_realized_it(scheme, stats, expected):
    assert _realized(applied_scheme=scheme, quant_stats=stats) == (scheme if expected else None)


def test_recipe_without_stats_or_scheme_records_none():
    assert _realized(applied_scheme=_INT8_DYNAMIC, quant_stats=None) is None
    assert _realized(applied_scheme=None, quant_stats={"api": "quantize_", "layers_quantized": 1}) is None


def test_exported_integer_mm_count_of_no_module_is_zero():
    assert CR.exported_integer_mm_count(None) == 0


def test_integerized_agreement_reports_shape_and_tolerance():
    torch = pytest.importorskip("torch")
    same = CR.integerized_agreement(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.0005]), atol=1e-3, rtol=0.0)
    assert same["status"] == "passed" and same["finite"] is True
    far = CR.integerized_agreement(torch.tensor([1.0, 2.0]), torch.tensor([1.0, 2.5]), atol=1e-3, rtol=0.0)
    assert far["status"] == "failed" and far["max_abs"] == pytest.approx(0.5)
    shape = CR.integerized_agreement(torch.zeros(2), torch.zeros(3), atol=1.0, rtol=1.0)
    assert shape["status"] == "failed"


def test_fp32_stage_observation_requires_fp32_floating_outputs():
    torch = pytest.importorskip("torch")
    original = torch.tensor([1.0, 2.0], dtype=torch.bfloat16)
    leaves, abi = CR.output_abi(original)
    record = {"leaves": leaves, "output_abi": abi}
    observed = CR.fp32_stage_observation(record, [torch.tensor([1.0, 2.0])], [], [])
    assert observed["output_metrics"][0]["original_dtype"] == "bf16"
    assert observed["output_metrics"][0]["staged_dtype"] == "f32"
    with pytest.raises(RuntimeError, match="non-FP32"):
        CR.fp32_stage_observation(record, [original], [], [])
