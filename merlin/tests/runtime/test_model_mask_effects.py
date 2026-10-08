"""Actual normal bare-metal contract delivery through preparation and lowering."""

import json

import numpy as np
import pytest


def test_normal_baremetal_mask_effects_reach_lowering_and_spike(tmp_path):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.masked_contraction import FEATURE, MaskEffectContract
    from merlin.llvmlower.scalar_contraction import RECTANGULAR_FEATURE
    from merlin.runtime.backends import spike, spike_model

    if not spike.available() or not toolchain.m2m_python().is_file() or not toolchain.clang().is_file():
        pytest.skip("bare-metal and upstream toolchain unavailable")

    # The source is an independently observed matrix result, rather than a
    # contraction whose discarded values justify masking.
    source = """module { func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> {
      %zero = arith.constant 0.0 : f32
      %empty = tensor.empty() : tensor<2x2xf32>
      %init = linalg.fill ins(%zero : f32) outs(%empty : tensor<2x2xf32>) -> tensor<2x2xf32>
      %r = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x2xf32>) outs(%init : tensor<2x2xf32>) -> tensor<2x2xf32>
      return %r : tensor<2x2xf32>
    } }"""
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "model.mlir").write_text(source)
    (bundle / "weights.safetensors.manifest.json").write_text(
        '{"0":{"kind":"input","name":"left"},"1":{"kind":"input","name":"right"}}'
    )
    a = np.array([[1, 2, 3], [-4, 5, -6]], dtype=np.float32)
    b = np.array([[1, -2], [3, 4], [-5, 6]], dtype=np.float32)
    np.savez(bundle / "inputs.npz", in0=a, in1=b)
    work = tmp_path / "build"
    built = spike_model.build(
        bundle,
        work,
        arena_mb=1,
        backend="scalar",
        features=frozenset({FEATURE, RECTANGULAR_FEATURE}),
        masked_contraction_effects=MaskEffectContract(True, True),
        cflags_override=[
            "-march=rv64gc",
            "-mabi=lp64d",
            "-mcmodel=medany",
            "-O2",
            "-ffreestanding",
            "-fno-builtin",
            "-ffp-contract=off",
        ],
    )
    assert json.loads((work / "lower/masked_contraction_report.json").read_text())["rewritten_contractions"] == 0
    assert json.loads((work / "compilation_recipe.json").read_text())["status"] == "completed"
    result = spike_model.run(built["elf"], mem_bytes=built["mem_bytes"], isa="rv64gc", timeout=60)
    np.testing.assert_array_equal(result["outputs"].reshape(2, 2), a @ b)
