"""Current fused source, unchanged native outputs and explicit stage refusals."""

import hashlib
import importlib.util
import json
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from merlin.common.paths import merlin_dir
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.codegen import mlir_runtime_c
from merlin.llvmlower.lower import lower_model, lower_model_file
from merlin.llvmlower.pipeline import PipelineError
from merlin.llvmlower.source_expression_interval import IntervalEffectContract
from merlin.llvmlower.source_observation_stage import (
    CHECKPOINT,
    FEATURE,
    MARKER,
    REPORT,
    _edit_pipeline,
    bind_runner,
    load_source_observations,
    require_report,
    validate_pipeline,
)
from merlin.llvmlower.toolchain import clang

EFFECTS = IntervalEffectContract(True, True, True, True, True)


def source(length=7, *, finishing="arith.mulf", name="forward"):
    fixture = merlin_dir() / "tests/ir/test_source_expression_interval.py"
    spec = importlib.util.spec_from_file_location("interval_fixture", fixture)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    quantizer = "%prod=" + module.source().split("%prod=", 1)[1].split("return %r:i8", 1)[0]
    quantizer = quantizer.replace("%prod=arith.mulf", "%prod=" + finishing)
    return f"""module {{func.func @{name}(%input:tensor<{length}xf32>,%upstream:tensor<{length}xf32>)
       -> tensor<{length}xi8> {{
      %e=tensor.empty():tensor<{length}xf32>
      %activated_tensor=linalg.generic {{
        indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%input:tensor<{length}xf32>) outs(%e:tensor<{length}xf32>) {{
        ^bb0(%x:f32,%unused_source:f32):
          %two=arith.constant 2.0:f32
          %xx=arith.mulf %x,%x:f32
          %den=arith.addf %xx,%two:f32
          %ratio=arith.divf %two,%den:f32
          %activation=arith.mulf %x,%ratio:f32
          linalg.yield %activation:f32
      }} -> tensor<{length}xf32>
      %out=tensor.empty():tensor<{length}xi8>
      %result=linalg.generic {{indexing_maps=[
        affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%activated_tensor,%upstream:tensor<{length}xf32>,tensor<{length}xf32>) outs(%out:tensor<{length}xi8>) {{
        ^bb0(%act:f32,%up:f32,%unused:i8):
          {quantizer}
          linalg.yield %r:i8
      }} -> tensor<{length}xi8>
      return %result:tensor<{length}xi8>
    }} }}"""


def native(result):
    digest = hashlib.sha256(result.ll_path.read_bytes()).hexdigest()[:16]
    shared = result.workdir / ("model_" + digest + ".so")
    subprocess.run(
        [
            str(clang()),
            "-O2",
            "-shared",
            "-fPIC",
            "-ffp-contract=off",
            str(result.ll_path),
            str(mlir_runtime_c()),
            "-lm",
            "-o",
            str(shared),
        ],
        check=True,
        capture_output=True,
    )
    return HostModel.load(str(shared))


@pytest.mark.parametrize("length,file_api", [(7, False), (31, True), (129, False)])
def test_current_fused_source_and_complete_native_output(tmp_path, length, file_api):
    text = source(length)
    normal = lower_model(text, tmp_path / "normal", targets=())
    work = tmp_path / "selected"
    options = dict(targets=(), features=frozenset({FEATURE}), source_observation_effects=EFFECTS)
    if file_api:
        path = tmp_path / "source.mlir"
        path.write_text(text)
        selected = lower_model_file(path, work, **options)
    else:
        selected = lower_model(text, work, **options)
    # Ordinary fusion creates a closed observer unavailable in the unfused
    # source. The checkpoint must retain this actual source and its declaration.
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.source_expression_interval import find_closed_scalar_i8_observers

    before, _ = find_closed_scalar_i8_observers(parse_mlir_text(text), effects=EFFECTS)
    observed = load_source_observations(work, effects=EFFECTS)
    assert not before and len(observed.proofs) == 1 and not observed.refusals
    assert normal.ll_path.read_bytes() == selected.ll_path.read_bytes()
    report = selected.stats["source_observation"]
    recipe = json.loads((work / "lowering_recipe.json").read_text())
    assert recipe["status"] == "returned" and recipe["sources"]["source_observation"] == report["source"]
    assert report["effect_permissions"] == vars(EFFECTS) and len(report["observers"]) == 1
    rng = np.random.default_rng(839 + length)
    values = rng.uniform(-7, 7, length).astype(np.float32)
    up = rng.uniform(-150, 150, length).astype(np.float32)
    original = values.copy(), up.copy()
    act = (values * (np.float32(2) / (values * values + np.float32(2)))).astype(np.float32)
    expected = np.clip(np.rint((act * up).astype(np.float32) * np.float32(3)), -128, 127).astype(np.int8)
    for result in (normal, selected):
        output = np.full(length + 32, 109, dtype=np.int8)
        native(result)(
            [(values.ctypes.data, values.shape), (up.ctypes.data, up.shape), (output.ctypes.data + 16, values.shape)]
        )
        np.testing.assert_array_equal(output[16:-16], expected)
        assert np.all(output[:16] == 109) and np.all(output[-16:] == 109)
    np.testing.assert_array_equal(values, original[0])
    np.testing.assert_array_equal(up, original[1])


def test_unknown_consumer_refuses_observer_without_changing_source_code(tmp_path):
    text = source(finishing="arith.addf")
    normal = lower_model(text, tmp_path / "normal", targets=())
    selected = lower_model(
        text, tmp_path / "selected", targets=(), features={FEATURE}, source_observation_effects=EFFECTS
    )
    observed = load_source_observations(selected.workdir, effects=EFFECTS)
    assert not observed.proofs and len(observed.refusals) == 1
    assert "first unchanged rounded" in observed.refusals[0][1]
    assert normal.ll_path.read_bytes() == selected.ll_path.read_bytes()


@pytest.mark.parametrize(
    "permission", ["rne", "gradual_underflow", "nontrapping", "flags_unobserved", "signed_zero_unobserved"]
)
def test_each_effect_permission_remains_explicit(tmp_path, permission):
    with pytest.raises(ValueError, match="explicit"):
        lower_model(
            source(),
            tmp_path,
            targets=(),
            features={FEATURE},
            source_observation_effects=replace(EFFECTS, **{permission: False}),
        )
    assert not (tmp_path / REPORT).exists() and not (tmp_path / "lowering_recipe.json").exists()


def test_missing_or_unused_contract_and_reused_workdir_invalidation(tmp_path):
    lower_model(source(), tmp_path, targets=(), features={FEATURE}, source_observation_effects=EFFECTS)
    with pytest.raises(PipelineError, match="explicit IntervalEffectContract"):
        lower_model(source(), tmp_path, targets=(), features={FEATURE})
    assert not (tmp_path / REPORT).exists() and not (tmp_path / CHECKPOINT).exists()
    with pytest.raises(PipelineError, match="without selected"):
        lower_model(source(), tmp_path, targets=(), source_observation_effects=EFFECTS)


def test_checkpoint_and_native_report_are_bound_by_bytes(tmp_path):
    lower_model(source(), tmp_path, targets=(), features={FEATURE}, source_observation_effects=EFFECTS)
    with pytest.raises(ValueError, match="absent, ambiguous or stale"):
        require_report("OK source_observation_stage " + "0" * 64, tmp_path, effects=EFFECTS)
    digest = hashlib.sha256((tmp_path / CHECKPOINT).read_bytes()).hexdigest()
    line = "OK source_observation_stage " + digest
    with pytest.raises(ValueError, match="ambiguous"):
        require_report(line + "\n" + line, tmp_path, effects=EFFECTS)
    (tmp_path / CHECKPOINT).write_text((tmp_path / CHECKPOINT).read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        load_source_observations(tmp_path, effects=EFFECTS)


@pytest.mark.parametrize(
    "mutation",
    ["missing_fusion", "duplicate_fusion", "buffer_before", "pointwise_before", "marker_late", "duplicate_marker"],
)
def test_pipeline_refuses_missing_ambiguous_or_late_source_stage(mutation):
    passes = [
        "func.func(linalg-fuse-elementwise-ops)",
        "func.func(linalg-generalize-named-ops)",
        "one-shot-bufferize{bufferize-function-boundaries}",
    ]
    if mutation == "missing_fusion":
        passes.pop(0)
    elif mutation == "duplicate_fusion":
        passes.insert(0, passes[0])
    elif mutation == "buffer_before":
        passes.insert(0, passes.pop())
    elif mutation == "pointwise_before":
        passes.insert(0, "__merlin_pointwise_packet_2__")
    if mutation in ("marker_late", "duplicate_marker"):
        selected = _edit_pipeline(passes)
        if mutation == "marker_late":
            selected.remove(MARKER)
            selected.append(MARKER)
        else:
            selected.insert(0, MARKER)
        with pytest.raises(ValueError):
            validate_pipeline(",".join(selected))
    else:
        with pytest.raises(ValueError):
            _edit_pipeline(passes)


def test_runner_hook_is_explicit_and_default_source_is_exact():
    baseline = "def _run_stages(*args): pass\n_run_stages(None)\n"
    assert bind_runner(baseline, selected=False) == baseline
    with pytest.raises(ValueError, match="one staged"):
        bind_runner("print('no staged runner')", selected=True)
    with pytest.raises(ValueError, match="one staged"):
        bind_runner(baseline + "_run_stages(None)\n", selected=True)
