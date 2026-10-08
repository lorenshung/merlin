"""Actual normal public model build preserves the original scalar output."""

import json

import numpy as np


def test_normal_model_source_observation_stage_and_actual_spike(tmp_path):
    from merlin.llvmlower.source_expression_interval import IntervalEffectContract
    from merlin.llvmlower.source_observation_stage import FEATURE, REPORT, load_source_observations
    from merlin.runtime.backends import spike_model

    source = """module {func.func @forward(%a:tensor<5xf32>)->tensor<5xf32> {
      %e=tensor.empty():tensor<5xf32>
      %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
        ins(%a:tensor<5xf32>) outs(%e:tensor<5xf32>) {
        ^bb0(%x:f32,%unused:f32):
          %two=arith.constant 2.0:f32
          %r=arith.addf %x,%two:f32
          linalg.yield %r:f32
      } -> tensor<5xf32>
      return %r:tensor<5xf32>
    }}"""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.mlir").write_text(source)
    (bundle / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"values"}}')
    values = np.array([-5, -0.75, 0, 1.25, 7], dtype=np.float32)
    np.savez(bundle / "inputs.npz", in0=values)
    effects = IntervalEffectContract(True, True, True, True, True)
    work = tmp_path / "build"
    built = spike_model.build(
        bundle,
        work,
        arena_mb=1,
        backend="scalar",
        features={FEATURE},
        source_observation_effects=effects,
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
    )
    observation = load_source_observations(work / "lower", effects=effects)
    assert not observation.proofs and not observation.refusals
    report = json.loads((work / "lower" / REPORT).read_text())
    assert report["status"] == "OBSERVED_CURRENT_TYPED_SOURCE"
    recipe = json.loads((work / "compilation_recipe.json").read_text())
    assert recipe["status"] == "completed"
    assert recipe["preparation"]["source_observation"]["path"] == str((work / "lower" / REPORT).resolve())
    result = spike_model.run(built["elf"], mem_bytes=built["mem_bytes"], isa="rv64gc", timeout=60)
    np.testing.assert_array_equal(result["outputs"], values + np.float32(2))
