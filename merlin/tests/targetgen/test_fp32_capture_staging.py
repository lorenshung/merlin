"""Opt-in FP32 capture staging over real BF16 frontend programs."""

import json
import math
import os
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower.session_bundle import load
from merlin.targetgen import _m2m_capture_worker as worker
from merlin.targetgen._recipe_quantizer import _source_weight_geometry_matches
from merlin.targetgen.application_inventory import verify_capture_receipt
from merlin.targetgen.quant_recipe import digest as recipe_digest

SINGLE_LOADER = """
import torch
from torch import nn

class Neutral(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.tensor([[1., 2.], [2., 1.]], dtype=torch.bfloat16))

    def forward(self, x):
        return x @ self.weight + torch.zeros((1, 2), dtype=torch.bfloat16)

def get_model_and_inputs():
    return Neutral().eval(), (torch.tensor([[2., 4.]], dtype=torch.bfloat16),)
"""

MIXED_OUTPUT_LOADER = SINGLE_LOADER.replace(
    "return x @ self.weight + torch.zeros((1, 2), dtype=torch.bfloat16)",
    "return (x @ self.weight, x > 0, x.to(torch.int64))",
)

LOADER_OWNED_CALIBRATION = """
import torch
from torch import nn

class Neutral(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(2, 2, bias=False).bfloat16()
        with torch.no_grad():
            self.projection.weight.copy_(torch.eye(2, dtype=torch.bfloat16))
        self.calibration_images = torch.tensor(
            [[[1., 2.]], [[3., 4.]]], dtype=torch.bfloat16
        )

    def forward(self, x):
        return self.projection(x)

def get_model_and_inputs():
    return Neutral().eval(), (torch.tensor([[1., 2.]], dtype=torch.bfloat16),)

def get_calibration_inputs(model, inputs):
    expected = torch.tensor([[[1., 2.]], [[3., 4.]]], dtype=torch.bfloat16)
    assert torch.equal(model.calibration_images, expected)
    assert model.calibration_images.dtype == torch.bfloat16
    assert model.projection.weight.dtype == torch.bfloat16
    assert torch.equal(model.projection.weight, torch.eye(2, dtype=torch.bfloat16))
    assert len(inputs) == 1 and inputs[0].dtype == torch.bfloat16
    return [(model.calibration_images[i],) for i in range(2)]
"""

MULTI_LOADER = """
import torch
from torch import nn
from m2m.capture.external_runtime import ExternalRuntimeProgram, make_external_runtime_session

class Core(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(4, 4, bias=False).bfloat16()
        with torch.no_grad():
            self.layer.weight.copy_(torch.eye(4, dtype=torch.bfloat16))

    def forward(self, x):
        return self.layer(x) + torch.zeros_like(x, dtype=torch.bfloat16)

class Last(nn.Module):
    def forward(self, x):
        return x[:, :2]

class Capture:
    def external_runtime_session(self):
        core = Core().eval()
        x = torch.tensor([[1., 2., 3., 4.]], dtype=torch.bfloat16)
        with torch.no_grad():
            context = core(x)
        programs = (ExternalRuntimeProgram("first", core, (x,), 1),
                    ExternalRuntimeProgram("second", core, (context,), 1),
                    ExternalRuntimeProgram("last", Last().eval(), (context,), 1))
        metadata = dict(kind="generic_recurrent", paper_ready=False,
                        stages=[p.name for p in programs], quality_program="second", states=[],
                        stage_schedule=[dict(name=p.name, steps=1, execution="compiled", timed=True)
                                        for p in programs],
                        provenance=dict(synthetic_inputs=True, full_checkpoint=False),
                        bindings=[dict(name="context", **{"from":dict(program="first", output_index=0),
                                                         "to":dict(program="second", input_index=0)}),
                                  dict(name="result", **{"from":dict(program="second", output_index=0),
                                                        "to":dict(program="last", input_index=0)})])
        return make_external_runtime_session(version=2, programs=programs, metadata=metadata)

def get_model_and_inputs():
    return Capture(), ()
"""


def _static_recipe():
    tensor = dict(dtype="int8", granularity="tensor", symmetric=True, block=None, mode="static")
    recipe = dict(
        schema="quant_recipe_v1",
        target="t",
        unit="matrix",
        status="derived",
        weight={**tensor, "quant_min": -127, "quant_max": 127},
        activation={**tensor, "quant_min": -128, "quant_max": 127},
        families=["contraction"],
        bias_domain="accumulator",
        software_numerical_engine="integer_reference",
    )
    recipe["recipe_sha256"] = recipe_digest(recipe)
    return recipe


def test_staged_quantizer_requires_unique_source_owner_and_unchanged_weight_geometry():
    source = [{"fqn": "projection", "kind": "Linear", "weight_shape": [4, 4]}]
    assert _source_weight_geometry_matches(source, "projection", [4, 4])
    assert not _source_weight_geometry_matches(source, "missing", [4, 4])
    assert not _source_weight_geometry_matches(source, "projection", [5, 4])
    assert not _source_weight_geometry_matches(source * 2, "projection", [4, 4])


@pytest.mark.slow
def test_explicit_fp32_stage_keeps_original_outputs_and_materializes_one_typed_program(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    if not python or not root:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    loader.write_text(SINGLE_LOADER)
    out = tmp_path / "capture"
    result = subprocess.run(
        [
            python,
            str(Path(worker.__file__)),
            "--m2m-dir",
            root,
            "--loader",
            str(loader),
            "--dtype",
            "fp32",
            "--stage-fp32",
            "--materialize-bundle",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr[-5000:]
    meta = json.loads((out / "meta.json").read_text())
    staged = meta["fp32_staging"]
    trace = json.loads((out / "frontend-trace.json").read_text())
    assert trace["graphs"]["original"]["sha256"] == staged["original_graph_sha256"]
    assert staged["status"] == "observed"
    assert staged["original_input_abi"] == [{"shape": [1, 2], "dtype": "bf16"}]
    assert staged["original_output_abi"] == [{"shape": [1, 2], "dtype": "bf16"}]
    assert meta["input_abi"] == meta["output_abi"] == [{"shape": [1, 2], "dtype": "f32"}]
    assert staged["output_cardinality"] == 1
    assert staged["output_metrics"][0]["max_abs"] == 0.0
    assert json.loads((out / "float_reference.json").read_text()) == json.loads((out / "golden.json").read_text())
    assert meta["precision_conversion"]["staged_precision_audit"]["status"] == "complete"
    assert verify_capture_receipt(out / "model.mlir")["status"] == "verified_materialized"


@pytest.mark.slow
def test_explicit_fp32_stage_keeps_a_source_when_eval_returns_none(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    if not python or not root:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    loader.write_text(
        SINGLE_LOADER.replace("def get_model_and_inputs():", "def source_model_and_inputs():")
        + """
class NullableEval(nn.Module):
    def __init__(self, inner):
        super().__init__()
        self.inner = inner

    def forward(self, x):
        return self.inner(x)

    def eval(self):
        super().eval()
        return None

def get_model_and_inputs():
    model, inputs = source_model_and_inputs()
    return NullableEval(model), inputs
"""
    )
    out = tmp_path / "capture"
    result = subprocess.run(
        [
            python,
            str(Path(worker.__file__)),
            "--m2m-dir",
            root,
            "--loader",
            str(loader),
            "--dtype",
            "fp32",
            "--stage-fp32",
            "--materialize-bundle",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr[-5000:]
    assert json.loads((out / "meta.json").read_text())["fp32_staging"]["status"] == "observed"


@pytest.mark.slow
def test_fp32_stage_preserves_exact_boolean_and_integer_results(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    if not python or not root:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    loader.write_text(MIXED_OUTPUT_LOADER)
    out = tmp_path / "capture"
    result = subprocess.run(
        [
            python,
            str(Path(worker.__file__)),
            "--m2m-dir",
            root,
            "--loader",
            str(loader),
            "--dtype",
            "fp32",
            "--stage-fp32",
            "--materialize-bundle",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=120,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr[-5000:]
    stage = json.loads((out / "meta.json").read_text())["fp32_staging"]
    assert [row["comparison"] for row in stage["output_metrics"]] == [
        "observed_floating",
        "exact_nonfloating",
        "exact_nonfloating",
    ]
    assert [row["dtype"] for row in stage["original_output_abi"]] == ["bf16", "i1", "i64"]
    assert [row["dtype"] for row in stage["staged_output_abi"]] == ["f32", "i1", "i64"]


@pytest.mark.slow
@pytest.mark.parametrize("quantized", [False, True])
def test_explicit_fp32_stage_preserves_shared_source_and_resolves_multi_program_bindings(tmp_path, quantized):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    if not python or not root:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    loader.write_text(MULTI_LOADER)
    out = tmp_path / "capture"
    recipe_args = []
    if quantized:
        recipe_path = tmp_path / "recipe.json"
        recipe_path.write_text(json.dumps(_static_recipe()))
        recipe_args = ["--recipe", str(recipe_path)]
    result = subprocess.run(
        [
            python,
            str(Path(worker.__file__)),
            "--m2m-dir",
            root,
            "--loader",
            str(loader),
            "--dtype",
            "int8" if quantized else "fp32",
            *recipe_args,
            "--stage-fp32",
            "--materialize-bundle",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr[-5000:]
    session = load(out)
    assert session.program_names == ("first", "second", "last")
    assert len(session.bindings) == 2
    by_name = {program.name: program for program in session.programs}
    for binding in session.bindings:
        assert by_name[binding.source_program].outputs[binding.source_output][1] == "f32"
        assert by_name[binding.target_program].inputs[binding.target_input][1] == "f32"
    report = json.loads((out / "session-receipt.json").read_text())
    assert [row["precision_selection"] for row in report["programs"]] == (
        ["recipe", "recipe", "no_recipe_work"] if quantized else ["fp32_staged"] * 3
    )
    assert report["source_state_unchanged"] is True
    for program in session.programs:
        stage = program.bundle
        meta = json.loads((stage / "meta.json").read_text())
        assert meta["fp32_staging"]["status"] == "observed"
        trace = json.loads((stage / "frontend-trace.json").read_text())
        assert trace["graphs"]["original"]["sha256"] == meta["fp32_staging"]["original_graph_sha256"]
        assert all(row["dtype"] == "bf16" for row in meta["fp32_staging"]["original_input_abi"])
        assert all(row["dtype"] == "f32" for row in meta["input_abi"])
        assert all(row["dtype"] == "f32" for row in meta["fp32_staging"]["staged_output_abi"])
        if quantized and program.name != "last":
            assert meta["fp32_staging"]["source_layer_plan_sha256"] == meta["quantization_stats"]["plan_sha256"]
        assert verify_capture_receipt(stage / "model.mlir")["status"] == "verified_materialized"


@pytest.mark.slow
def test_explicit_bf16_to_fp32_to_w8a8_keeps_three_distinct_numeric_references(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    if not python or not root:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    loader.write_text(SINGLE_LOADER)
    recipe = _static_recipe()
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(recipe))
    out = tmp_path / "capture"
    result = subprocess.run(
        [
            python,
            str(Path(worker.__file__)),
            "--m2m-dir",
            root,
            "--loader",
            str(loader),
            "--dtype",
            "int8",
            "--recipe",
            str(recipe_path),
            "--stage-fp32",
            "--materialize-bundle",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr[-5000:]
    meta = json.loads((out / "meta.json").read_text())
    staged = meta["fp32_staging"]
    trace = json.loads((out / "frontend-trace.json").read_text())
    assert trace["graphs"]["original"]["sha256"] == staged["original_graph_sha256"]
    assert staged["original_output_abi"][0]["dtype"] == "bf16"
    assert staged["staged_output_abi"][0]["dtype"] == "f32"
    assert staged["output_metrics"][0]["comparison"] == "observed_floating"
    assert meta["recipe_agreement"]["samples"] >= 1
    assert staged["source_layer_plan_sha256"] == meta["quantization_stats"]["plan_sha256"]
    assert meta["integerization_receipt"]["golden_agreement"]["reference"] == "pt2e_integer"
    assert (out / "float_reference.json").is_file()
    assert (out / "integer-reference.json").is_file()
    assert verify_capture_receipt(out / "model.mlir")["status"] == "verified_materialized"


@pytest.mark.slow
@pytest.mark.parametrize("reused_generator", [False, True])
def test_fp32_stage_uses_original_loader_owned_bf16_calibration_for_static_w8a8(tmp_path, reused_generator):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    selected_worker = os.environ.get("MERLIN_CAPTURE_WORKER", worker.__file__)
    if not python or not root:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    loader = tmp_path / "loader.py"
    source = LOADER_OWNED_CALIBRATION
    if reused_generator:
        value_check = subprocess.run(
            [
                python,
                "-c",
                f"""
import runpy
import torch
freeze = runpy.run_path({str(Path(selected_worker))!r})['_freeze_calibration']
scratch = torch.empty((1, 2), dtype=torch.bfloat16)
flag, index = torch.tensor([True]), torch.tensor([7], dtype=torch.int64)
def samples():
    for i in range(2):
        scratch.copy_(torch.tensor([[1. + 2*i, 2. + 2*i]], dtype=torch.bfloat16))
        yield (scratch, flag, index)
    raise RuntimeError('requested a third sample')
owned = freeze(samples(), limit=2, normalize=tuple, torch=torch)
assert torch.equal(owned[0][0], torch.tensor([[1., 2.]], dtype=torch.bfloat16))
assert torch.equal(owned[1][0], torch.tensor([[3., 4.]], dtype=torch.bfloat16))
assert owned[0][0].data_ptr() != owned[1][0].data_ptr()
flag.fill_(False)
index.fill_(9)
assert all(row[1].dtype == torch.bool and bool(row[1][0]) for row in owned)
assert all(row[2].dtype == torch.int64 and int(row[2][0]) == 7 for row in owned)
""",
            ],
            cwd=tmp_path,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert value_check.returncode == 0, value_check.stderr
        source = source.replace(
            "return [(model.calibration_images[i],) for i in range(2)]",
            """def samples():
        scratch = torch.empty_like(model.calibration_images[0])
        for i in range(100):
            scratch.copy_(model.calibration_images[i % 2])
            yield (scratch,)
        raise RuntimeError("calibration requested beyond its selected budget")
    return samples()""",
        )
    loader.write_text(source)
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(_static_recipe()))
    out = tmp_path / "capture"
    result = subprocess.run(
        [
            python,
            str(Path(selected_worker)),
            "--m2m-dir",
            root,
            "--loader",
            str(loader),
            "--dtype",
            "int8",
            "--recipe",
            str(recipe_path),
            "--stage-fp32",
            "--materialize-bundle",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr[-5000:]
    meta = json.loads((out / "meta.json").read_text())
    stage = meta["fp32_staging"]
    assert stage["original_input_abi"] == [{"shape": [1, 2], "dtype": "bf16"}]
    assert stage["staged_input_abi"] == [{"shape": [1, 2], "dtype": "f32"}]
    assert stage["original_output_abi"] == [{"shape": [1, 2], "dtype": "bf16"}]
    assert stage["staged_output_abi"] == [{"shape": [1, 2], "dtype": "f32"}]
    assert stage["output_metrics"][0]["max_abs"] == 0.0
    assert stage["output_metrics"][0]["max_rel"] == 0.0
    assert stage["calibration_source"] == "loader_declared_pre_transform"
    assert stage["calibration_sample_limit"] == 100
    count = 100 if reused_generator else 2
    assert (
        stage["calibration_input_abis"]
        == [
            {
                "original": [{"shape": [1, 2], "dtype": "bf16"}],
                "staged": [{"shape": [1, 2], "dtype": "f32"}],
            },
        ]
        * count
    )
    assert meta["precision_conversion"]["staged_precision_audit"]["non_target_floating_values"] == 0
    stats = meta["quantization_stats"]
    assert stats["framework_capture_policy"]["calibration_source"] == "explicit_calibration_stream"
    assert stats["calibration_samples"] == count
    assert meta["recipe_agreement"]["samples"] == min(count + 1, 64)
    assert math.isfinite(meta["recipe_agreement"]["max_abs_error"])
    assert verify_capture_receipt(out / "model.mlir")["status"] == "verified_materialized"


@pytest.mark.slow
def test_real_tiny_loader_session_keeps_complete_source_snapshot_before_static_recipe(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    cache = os.environ.get("MERLIN_TINY_HF_CACHE")
    if not python or not root or not cache:
        pytest.skip("explicit M2M runtime, source and offline TinyLlama cache are required")
    loader = Path(root) / "workloads/tiny_llama/loader.py"
    if not loader.is_file() or not Path(cache).is_dir():
        pytest.skip("selected TinyLlama loader or offline cache is absent")
    recipe_path = tmp_path / "recipe.json"
    recipe_path.write_text(json.dumps(_static_recipe()))
    out = tmp_path / "capture"
    env = {
        **os.environ,
        "HF_HUB_CACHE": cache,
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "M2M_LLAMA_ATTENTION": "eager",
        "M2M_LLAMA_SESSION": "e2e",
        "M2M_LLAMA_LAYERS": "1",
        "M2M_PREFILL_TOKENS": "16",
        "M2M_DECODE_TOKENS": "2",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
    }
    for name in ("M2M_LLAMA_TOKEN_IDS", "M2M_LLAMA_TOKEN_SOURCE", "M2M_LLAMA_PAPER_READY", "M2M_SEQ"):
        env.pop(name, None)
    result = subprocess.run(
        [
            python,
            str(Path(worker.__file__)),
            "--m2m-dir",
            root,
            "--loader",
            str(loader),
            "--dtype",
            "int8",
            "--recipe",
            str(recipe_path),
            "--stage-fp32",
            "--materialize-bundle",
            "--out",
            str(out),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )
    assert result.returncode == 0, result.stderr[-5000:]
    session = load(out)
    assert session.program_names == ("prefill", "decode")
    for program in session.programs:
        stage = program.bundle
        meta = json.loads((stage / "meta.json").read_text())
        trace = json.loads((stage / "frontend-trace.json").read_text())
        relation = trace["transformations"][0]
        source_nodes = {row["id"]: row["target"] for row in trace["graphs"]["original"]["nodes"]}
        dest_nodes = {row["id"]: row["target"] for row in trace["graphs"]["quantized"]["nodes"]}
        unresolved = {
            "source": [(key, source_nodes[key]) for key in relation["unresolved_source_ids"]],
            "destination": [(key, dest_nodes[key]) for key in relation["unresolved_destination_ids"]],
        }
        assert trace["status"] == "complete", {"blockers": trace.get("blockers"), "unresolved": unresolved}
        assert trace["graphs"]["original"]["status"] == "complete"
        assert trace["graphs"]["original"]["sha256"] == meta["fp32_staging"]["original_graph_sha256"]
        assert meta["precision_conversion"]["staged_precision_audit"]["non_target_floating_values"] == 0
        assert verify_capture_receipt(stage / "model.mlir")["status"] == "verified_materialized"
