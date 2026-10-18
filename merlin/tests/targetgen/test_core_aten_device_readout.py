# ruff: noqa: E501 -- the captured module is printed one op per line
"""A capture in the integer_readout numerics is admitted with its readout ON THE DEVICE.

WHY THIS EXISTS. The core-aten int8 pipeline read every contraction's int32 accumulator back to the
host, cast it to f32 and multiplied twice. That host cast is exact only for |acc| <= 2^24, which bounded K
to 1024 at int8 (TinyLlama's K = 2048/5632 and addmm's K = 3072 were refused), and the f32 bias stayed a
host add. The frontend's integer_readout numerics instead seed the accumulator with an int32 bias and
read it out once, fused, as int8 (``prov.role = "readout"`` with ``prov.quant.*``). This plan admits that
form: an accumulator-seed load, the contraction, the fused acc_scale/ReLU stage and an int8 readout
crossing -- and no cast bound.
"""

from __future__ import annotations

import pytest
from phase0_fixture_selection import fixture_resolved

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.kernels.shapes import accumulator_init, observe_contractions, zero_initialised
from merlin.targetgen.core_aten_spec_int8 import admission_plan, device_readout
from merlin.targetgen.host_capabilities import validate_host_capabilities

pytestmark = pytest.mark.target("gemmini")
_BASE = repo_root() / "examples/gemmini/target"


@pytest.fixture(scope="module")
def source_text(tmp_path_factory):
    import os
    import subprocess

    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON is required for real model2mlir readout capture")
    path = tmp_path_factory.mktemp("device-readout") / "source.mlir"
    script = "import sys,torch,m2m\nfrom m2m.capture import integer_readout\nclass Model(torch.nn.Module):\n def forward(self,a,b,bias):\n  acc=torch.ops.m2m_q.int_mm_seeded.default(a,b,bias)\n  return torch.ops.m2m_q.requantize.default(acc,[0.5],0,-128,127,True)\ninputs=(torch.ones(4,8,dtype=torch.int8),torch.ones(8,16,dtype=torch.int8),torch.arange(16,dtype=torch.int32))\nresult=m2m.convert(Model(),inputs,backend='fx_importer')\nassert result.ok,result\nopen(sys.argv[1],'w').write(result.mlir_text)\n"
    result = subprocess.run([python, "-c", script, str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    return path.read_text()


@pytest.fixture(scope="module")
def selection():
    return {
        "spec": fixture_resolved(_BASE, "gemmini"),
        "host": validate_host_capabilities(load_yaml(_BASE / "host-capabilities.yaml")),
    }


def _plan(selection, source, proof=None):
    with IR_LOCK:
        module = mq.parse(source)
        ((op, _shape),) = observe_contractions(module)
        return op, admission_plan(module, selection, captured=True, provider_proof=proof)


def test_the_seed_and_the_fused_readout_are_read_structurally(source_text):
    with IR_LOCK:
        ((op, _shape),) = observe_contractions(mq.parse(source_text))
        assert accumulator_init(op) == "int32_seed" and not zero_initialised(op)
        readout = device_readout(op)
    assert readout["relu"] and readout["quant_min"] == -128 and readout["granularity"] == "tensor"
    assert readout["scales"] == [0.5]


def test_the_device_readout_plan_replaces_the_host_cast_and_its_k_bound(selection, source_text):
    _op, plan = _plan(selection, source_text)
    (site,) = plan["sites"]
    assert site["readout"] == "device_int8" and site["accumulator_init"] == "int32_seed"
    steps = {(row["step"], row.get("declaration") or row.get("op")) for row in site["steps"]}
    assert (2, "transfer_contracts.accumulator_seed_load") in steps
    assert (4, "transfer_contracts.int8_readout") in steps and (4, "elementwise_map") in steps
    assert not any(step for step, _ in steps if step in (5, 6)) and site["cast_exact_range_obligation"] is None
    fused = next(row for row in site["steps"] if row.get("op") == "elementwise_map")
    assert fused["software"]["status"] == "admitted"
    assert fused["device_readout"]["relu"] is True


def test_seeded_admission_requires_selected_provider_coverage(selection, source_text):
    import hashlib

    with IR_LOCK:
        module = mq.parse(source_text)
        ((op, _shape),) = observe_contractions(module)
        ordinal = list(module.walk()).index(op)
        proof = dict(
            status="proven",
            source_sha256=hashlib.sha256(str(module).encode()).hexdigest(),
            kernels=[dict(symbol="seeded", status="proven", exact_arithmetic=True)],
            bindings=[
                dict(
                    source_operation_ordinal=ordinal,
                    symbol="seeded",
                    replaced_ordinals=[ordinal, list(module.walk()).index(device_readout(op)["operation"])],
                )
            ],
        )
    _, absent = _plan(selection, source_text)
    assert absent["sites"][0]["status"] == "refused"
    assert "coverage proof unavailable" in " ".join(absent["sites"][0]["reasons"])
    _, admitted = _plan(selection, source_text, proof)
    assert admitted["sites"][0]["status"] == "admitted", admitted["sites"][0]["reasons"]
    for damaged in (
        {**proof, "bindings": []},
        {**proof, "source_sha256": "stale"},
        {**proof, "kernels": []},
        {**proof, "kernels": [dict(symbol="seeded", status="proven")]},
        {**proof, "bindings": [{**proof["bindings"][0], "replaced_ordinals": [ordinal]}]},
    ):
        _, refused = _plan(selection, source_text, damaged)
        assert refused["sites"][0]["status"] == "refused"


@pytest.mark.parametrize("key", ["qmin", "qmax", "mystery"])
def test_unknown_quantization_attributes_are_rejected(source_text, key):
    from xdsl.dialects.builtin import IntegerAttr, i64

    with IR_LOCK:
        ((op, _shape),) = observe_contractions(mq.parse(source_text))
        reader = device_readout(op)["operation"]
        reader.attributes[f"prov.quant.{key}"] = IntegerAttr(0, i64)
        with pytest.raises(ValueError, match="unknown readout quantization"):
            device_readout(op)


def test_a_readout_relu_with_a_nonpositive_scale_is_refused(selection, source_text):
    negative = source_text.replace("prov.quant.scale = ", "prov.quant.scale = -")
    _op, plan = _plan(selection, negative)
    assert any("ReLU" in reason for reason in plan["sites"][0]["reasons"])


def test_unknown_readout_numerics_are_refused_before_any_frontend_import():
    from merlin.capture.integerization import integerize_static_pt2e

    with pytest.raises(ValueError, match="numerics"):
        integerize_static_pt2e(None, (), numerics="fixed_point")


def test_exact_ratio_metadata_comes_from_real_model2mlir(tmp_path):
    import json
    import os
    import subprocess
    from fractions import Fraction

    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON is required")
    path = tmp_path / "rational-readout.mlir"
    script = r"""
import sys,torch,m2m
from m2m.capture import integer_readout
class Model(torch.nn.Module):
 def forward(self,a,b,bias):
  acc=torch.ops.m2m_q.int_mm_seeded.default(a,b,bias)
  return torch.ops.m2m_q.exact_readout.default(acc,.03,[.02],1.,None,1.,.01,0,-128,127,False,False)
inputs=(torch.ones(4,8,dtype=torch.int8),torch.ones(8,16,dtype=torch.int8),torch.zeros(16,dtype=torch.int32))
result=m2m.convert(Model(),inputs,backend='fx_importer')
assert result.ok,result
open(sys.argv[1],'w').write(result.mlir_text)
"""
    result = subprocess.run([python, "-c", script, str(path)], capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    with IR_LOCK:
        ((op, _),) = observe_contractions(mq.parse(path.read_text()))
        readout = device_readout(op)
        expected = Fraction(0.03) * Fraction(0.02) / Fraction(0.01)
        assert readout["scales"] == []
        assert readout["scale_ratios"] == [
            dict(numerator=str(expected.numerator), denominator=str(expected.denominator))
        ]
        json.dumps({key: value for key, value in readout.items() if key != "operation"})
        from xdsl.dialects.builtin import ArrayAttr, StringAttr

        readout["operation"].attributes["prov.quant.scale_denominator"] = ArrayAttr([StringAttr("0")])
        with pytest.raises(ValueError, match="positive denominators"):
            device_readout(op)
