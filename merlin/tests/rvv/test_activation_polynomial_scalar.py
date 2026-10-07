"""Activation approximation can run without a contraction or vector schedule."""

import subprocess
from textwrap import dedent

import numpy as np

from merlin.llvmlower import impr_features as features
from merlin.llvmlower.abi import HostModel
from merlin.llvmlower.act_poly import rewrite_source
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.toolchain import clang, compiler_python

FEATURE = "approximate_transcendental_activation"
SOURCE = """module {
  func.func @forward(%x: memref<64xf32>, %act: memref<64xf32>, %norm: memref<64xf32>) attributes {llvm.emit_c_interface} {
    linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
      ins(%x: memref<64xf32>) outs(%act: memref<64xf32>) attrs = {prov.op = "silu", prov.family = "elementwise"} {
    ^bb0(%value: f32, %unused: f32):
      %e = math.exp %value : f32
      linalg.yield %e : f32
    }
    linalg.generic {indexing_maps = [affine_map<(d0)->(d0)>, affine_map<(d0)->(d0)>], iterator_types = ["parallel"]}
      ins(%x: memref<64xf32>) outs(%norm: memref<64xf32>) attrs = {prov.op = "softmax", prov.family = "normalization"} {
    ^bb0(%value: f32, %unused: f32):
      %e = math.exp %value : f32
      linalg.yield %e : f32
    }
    return
  }
}"""


def test_explicit_approximation_does_not_select_a_vector_schedule():
    selected = features.get(FEATURE)
    assert selected.edit_schedule is None and selected.edit_pipeline is None
    assert features.apply_schedule("unchanged", {FEATURE}) == "unchanged"
    assert FEATURE in features.normalize({"vectorized_transcendental_activation"})


def test_executed_scalar_path_preserves_normalization_and_original_default(tmp_path):
    inputs = np.linspace(-9, 9, 64, dtype=np.float32)
    outputs = []
    for label, selected in [
        ("baseline", set()),
        ("approximate", {FEATURE}),
        ("fused", {"fuse_activation_polynomial_fma"}),
    ]:
        work = tmp_path / label
        lowered = lower_to_llvm_ir(SOURCE, workdir=work, vectorize=False, features=selected)
        assert not (work / "rvv_schedule.mlir").exists()
        expected_calls = 2 if not selected else 1
        assert lowered.count("call float @expf(") == expected_calls
        if label == "fused":
            assert "@llvm.fma.f32" in lowered and "@fmaf" not in lowered
        library = work / "model.so"
        subprocess.run(
            [str(clang()), "-O0", "-fPIC", "-shared", str(work / "model.ll"), "-lm", "-o", str(library)], check=True
        )
        act, norm = np.empty_like(inputs), np.empty_like(inputs)
        HostModel.load(str(library))([(x.ctypes.data, x.shape) for x in (inputs, act, norm)])
        outputs.append((act, norm))
    baseline, *selected_variants = outputs
    assert np.array_equal(baseline[0].view(np.uint32), baseline[1].view(np.uint32))
    for selected in selected_variants:
        assert np.array_equal(selected[1].view(np.uint32), baseline[1].view(np.uint32))
        assert np.allclose(selected[0], baseline[0], rtol=2e-6, atol=1e-8)


def test_unsupported_float_dtypes_are_left_to_existing_lowering(tmp_path):
    # The compiler's MLIR bindings belong to its selected Python environment,
    # independently of the interpreter running pytest or capturing Torch models.
    script = tmp_path / "dtype_probe.py"
    script.write_text(
        "from torch_mlir import ir\n"
        f"SOURCE = {SOURCE!r}\n"
        f"REWRITE = {rewrite_source()!r}\n"
        + dedent("""
    namespace = {"_ir": ir}
    exec(REWRITE, namespace)
    for dtype, expected in [("f32", 1), ("f64", 0), ("bf16", 0)]:
        with ir.Context() as context, ir.Location.unknown():
            source = SOURCE.replace("f32", dtype)
            module = ir.Module.parse(source)
            count = namespace["apply_activation_polynomial"](module, context)
            assert count == expected
            module.operation.verify()
""")
    )
    result = subprocess.run([str(compiler_python()), str(script)], capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
