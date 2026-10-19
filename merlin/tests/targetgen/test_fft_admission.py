"""An admissible contraction cannot grant an undeclared FFT boundary op."""

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.targetgen.fft_admission import screen_fft_candidates, screen_fft_component_dtype
from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.software_spec import load_software_spec


def _selection():
    root = repo_root() / "examples" / "gemmini" / "target"
    spec, host = root / "software-spec.yaml", root / "host-capabilities.yaml"
    return {
        "spec": load_software_spec(spec),
        "host": validate_host_capabilities(load_yaml(host)),
        "identities": {"software_spec": {"path": str(spec)}, "host_capabilities": {"path": str(host)}},
    }


def _candidate(rank=2):
    return [
        {
            "source_node": "fft",
            "source_op": "aten._fft_r2c.default",
            "operations": [
                {
                    "op": "aten.matmul.default",
                    "rank": rank,
                    "operand_ranks": [rank, 2],
                    "operand_dtypes": ["torch.float32", "torch.float32"],
                    "result_dtype": "torch.float32",
                },
                {
                    "op": "aten.complex.default",
                    "rank": rank,
                    "operand_ranks": [rank, rank],
                    "operand_dtypes": ["torch.float32", "torch.float32"],
                    "result_dtype": "torch.complex64",
                },
            ],
        }
    ]


def test_complete_candidate_refuses_undeclared_assembly():
    (site,) = screen_fft_candidates(_candidate(), _selection())
    assert site["status"] == "refused"
    assert all("aten.complex.default" in reason for reason in site["reasons"])
    assert site["steps"][-1]["host"]["source"]["line"] == 17
    assert site["steps"][-1]["software"]["source"]["line"] == 35
    assert all(step["software"]["status"] == "admitted" for step in site["steps"][:-1])


def test_rank3_cast_refusal_keeps_exact_restriction_line():
    (site,) = screen_fft_candidates(_candidate(3), _selection())
    assert any("host-capabilities.yaml:62: rank 3 not in [2]" in reason for reason in site["reasons"])


def test_component_extraction_has_no_declaration():
    candidate = _candidate()
    candidate[0]["operations"] = [
        {
            "op": "aten.real.default",
            "rank": 2,
            "operand_ranks": [2],
            "operand_dtypes": ["torch.complex64"],
            "result_dtype": "torch.float32",
        }
    ]
    (site,) = screen_fft_candidates(candidate, _selection())
    assert site["status"] == "refused"
    assert len(site["reasons"]) == 2


def test_no_fft_is_no_candidate():
    assert screen_fft_candidates([], _selection()) == []


def test_f64_component_refusal_names_exact_precision_lines():
    reasons = screen_fft_component_dtype("torch.complex128", 2, _selection())
    assert any("software-spec.yaml:111:" in reason for reason in reasons)
    assert any("host-capabilities.yaml:71:" in reason for reason in reasons)
    assert all("ordered_operand_dtypes" in reason for reason in reasons)
    assert screen_fft_component_dtype("torch.complex64", 2, _selection()) == []


def test_fft_refusal_preserves_original_host_execution(tmp_path, monkeypatch):
    from merlin.runtime.backends import spike_model
    from merlin.targetgen import core_aten_device, core_aten_numeric, core_aten_provenance

    reason = "host-capabilities.yaml:17: no SW operation declaration admits 'aten.complex.default'"
    monkeypatch.setattr(core_aten_device, "selected_facts", lambda *args: nullcontext({}))
    monkeypatch.setattr(
        core_aten_device,
        "load_execution_provider",
        lambda *args: SimpleNamespace(__file__=str(tmp_path / "provider.py"), runner_options=lambda: {}),
    )
    plan = {"fft_refusals": [reason], "refusal_code": "fft_admission_refused", "quantized_contractions": 0, "sites": []}
    monkeypatch.setattr(core_aten_numeric, "prepare_spec_int8", lambda *args, **kwargs: plan)
    monkeypatch.setattr(core_aten_provenance, "batch_provenance", lambda *args, **kwargs: {"artifact_digests": {}})
    calls = []

    def build(*args, **kwargs):
        calls.append(kwargs)
        assert kwargs["device"] is None
        return {"elf": tmp_path / "host.elf", "mem_bytes": 1024}

    monkeypatch.setattr(spike_model, "build", build)
    monkeypatch.setattr(spike_model, "run", lambda *args, **kwargs: {"output_bytes": [bytes(4)], "console": "DONE\n"})

    def route(*args, **kwargs):
        raise AssertionError("an undeclared FFT cannot enter device routing")

    monkeypatch.setattr(core_aten_device, "routing_for_bundle", route)
    (tmp_path / "spec.yaml").write_text("{}\n")
    original = "original unquantized FFT source"
    (tmp_path / "model.mlir").write_text(original)
    report = {
        "output_count": 1,
        "cases": [
            {
                "case_id": "fft",
                "overload": "aten._fft_r2c.default",
                "status": "bundled",
                "output_indices": [0],
                "output_abi": [{"dtype": "f32", "shape": [1]}],
                "case": {"expected": {"kind": "tensor", "dtype": "float32", "shape": [1], "values": [0.0]}},
            }
        ],
    }
    execution, verdict = core_aten_device.run_spike_bundle(
        tmp_path,
        report,
        target="selected",
        device_package=tmp_path,
        numeric_policy="spec_int8",
        software_spec=tmp_path / "spec.yaml",
        host_capabilities=tmp_path / "host.yaml",
        arena_mb=1,
        timeout=1,
    )
    assert len(calls) == 1
    assert execution["status"] == "ran"
    assert verdict["passed_by_lane"] == {"host": 1, "device": 0}
    assert report["numeric_preparation"]["refusal_code"] == "fft_admission_refused"
    assert report["cases"][0]["routing"]["lane"] == "host"
    assert not report["cases"][0]["routing"]["routed"]
    assert reason in report["cases"][0]["routing"]["reason"]
    assert (tmp_path / "model.mlir").read_text() == original


def test_fft_refusal_receipt_preserves_reason_and_code(tmp_path):
    from merlin.targetgen.capture_refusals import NumericCaptureRefused, read_refusal, write_refusal

    reason = "aten.imag.default has no operation declaration"
    write_refusal(tmp_path, code="fft_admission_refused", reason=reason)
    refusal = read_refusal(tmp_path / "refusal.json")
    assert isinstance(refusal, NumericCaptureRefused)
    assert refusal.code == "fft_admission_refused"
    assert str(refusal) == reason


def test_all_fft_boundaries_refuse_before_pt2e():
    import os
    import subprocess

    script = r"""
import torch
import torchao.quantization.pt2e.quantize_pt2e as pt2e
from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml
from merlin.targetgen.host_capabilities import validate_host_capabilities
from merlin.targetgen.software_spec import load_software_spec
from merlin.targetgen._recipe_quantizer import apply_recipe, FFTAdmissionRefused
root = repo_root() / "examples" / "gemmini" / "target"
spec, host = root / "software-spec.yaml", root / "host-capabilities.yaml"
recipe = {
    "schema":"quant_recipe_v1", "status":"derived", "families":["contraction"],
    "weight":dict(dtype="int8",granularity="tensor",symmetric=True,quant_min=-127,quant_max=127),
    "activation":dict(dtype="int8",granularity="tensor",symmetric=True,quant_min=-128,quant_max=127,mode="static"),
    "fft_admission_selection":dict(spec=load_software_spec(spec),
        host=validate_host_capabilities(load_yaml(host)),
        identities=dict(software_spec=dict(path=str(spec)),host_capabilities=dict(path=str(host)))),
}
def forbidden(*args, **kwargs):
    raise AssertionError("PT2E must not run for undeclared FFT boundary")
pt2e.prepare_pt2e = forbidden
class Model(torch.nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
    def forward(self, x):
        if self.mode == "r2c":
            return torch.ops.aten._fft_r2c.default(x, [1], 0, True)
        if self.mode == "c2r":
            return torch.ops.aten._fft_c2r.default(x, [1], 0, 4)
        return torch.ops.aten._fft_c2c.default(x, [1], 0, True)
for mode, x, required in [
    ("r2c",torch.ones(2,4),"aten.complex.default"),
    ("c2r",torch.ones(2,3,dtype=torch.complex64),"aten.real.default"),
    ("c2c",torch.ones(2,4,dtype=torch.complex64),"aten.imag.default"),
    ("r2c",torch.ones(2,4,dtype=torch.float64),"ordered_operand_dtypes"),
]:
    try:
        apply_recipe(Model(mode),recipe,example_inputs=(x,))
    except FFTAdmissionRefused as exc:
        assert exc.code == "fft_admission_refused"
        assert required in str(exc), str(exc)
    else:
        raise AssertionError("FFT needs complete operation declarations")
"""
    result = subprocess.run(
        [os.environ["MERLIN_M2M_PYTHON"], "-c", script], capture_output=True, text=True, timeout=120
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("recipe_count", [0, 2])
def test_fft_without_unique_recipe_retains_host_source(tmp_path, monkeypatch, recipe_count):
    from merlin.targetgen import core_aten_numeric, core_aten_spec_int8
    from merlin.targetgen.capsule_source import PytorchRefSource

    selection = _selection()
    selection["recipes"] = [
        {
            "activation": dict(
                dtype="int8", mode="static", granularity="tensor", symmetric=True, quant_min=-128, quant_max=127
            )
        }
        for _ in range(recipe_count)
    ]
    monkeypatch.setattr(core_aten_spec_int8, "select_spec", lambda *args: selection)

    def capture(*args, **kwargs):
        raise AssertionError("ambiguous or absent recipes cannot recapture the source")

    monkeypatch.setattr(PytorchRefSource, "capture_loader", capture)
    (tmp_path / "spec.yaml").write_text(Path(selection["identities"]["software_spec"]["path"]).read_text())
    original = "builtin.module {}\n"
    (tmp_path / "model.mlir").write_text(original)
    report = {
        "cases": [
            {
                "overload": "aten._fft_r2c.default",
                "status": "bundled",
                "input_abi": [{"dtype": "f32", "shape": [2, 4]}],
                "output_abi": [{"dtype": "f32", "shape": [2, 3, 2]}],
            }
        ]
    }
    plan = core_aten_numeric.prepare_spec_int8(
        tmp_path,
        report,
        target="selected",
        facts={},
        software_spec=tmp_path / "spec.yaml",
        host_capabilities=tmp_path / "host.yaml",
        provider=None,
        package=tmp_path,
    )
    assert plan["quantized_contractions"] == 0
    assert plan["refusal_code"] == "fft_admission_refused"
    assert plan["refusal_details"]["matching_static_recipes"] == recipe_count
    assert plan["fft_refusals"]
    assert (tmp_path / "model.mlir").read_text() == original
    assert plan["source_sha256_before"] == plan["source_sha256_after"]
    assert report["cases"][0]["numeric_admission"]["refusal_code"] == "fft_admission_refused"
