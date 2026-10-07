"""Execute the optional source cascade at vector and cascade boundaries."""

import os
import subprocess

import numpy as np
import pytest
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from merlin.llvmlower import toolchain
from merlin.llvmlower.cpu_sum import CPU_SUM_F32_AVX2_POLICY, build_cpu_sum_lastdim
from merlin.xdsl_dialects._common import text


def _module(shape):
    typ = builtin.TensorType(builtin.f32, shape)
    body = Block(arg_types=[typ])
    ops, result = build_cpu_sum_lastdim(
        body.args[0], backend_policy=CPU_SUM_F32_AVX2_POLICY, allow_reassociation=True, assume_finite_intermediates=True
    )
    body.add_ops([*ops, func.ReturnOp(result)])
    module = builtin.ModuleOp([func.FuncOp("forward", ([typ], [result.type]), Region(body))])
    module.verify()
    return module


@pytest.mark.parametrize("n", [8, 9, 31, 32, 33, 511, 512, 513, 720, 960, 2048, 8191])
@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_native_source_cascade_with_tails(tmp_path, n, optimization):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    rng = np.random.default_rng(7)
    shape = (n,) if n == 960 else (2, 3, n)
    x = np.asarray(rng.normal(size=shape) * np.exp2(rng.integers(-16, 16, size=shape)), dtype=np.float32)
    x.reshape(-1, n)[0] = np.resize(np.array([1e8, 1.0, -1e8, 3.0, -0.0, 0.0, 1e-20, -1e-20], dtype=np.float32), n)
    np.save(tmp_path / "input.npy", x)
    reference = "import sys,numpy as np,torch;from pathlib import Path;torch.set_num_threads(8);p=Path(sys.argv[1]);x=torch.from_numpy(np.load(p/'input.npy'));np.save(p/'golden.npy',x.sum(-1).numpy())"
    subprocess.run(
        [str(toolchain.compiler_python()), "-c", reference, str(tmp_path)],
        check=True,
        env={**os.environ, "ATEN_CPU_CAPABILITY": "avx2"},
        capture_output=True,
    )
    source = tmp_path / "input.mlir"
    source.write_text(text(parse_mlir_text(text(_module(x.shape)))))
    lowered = lower_model_file(source, tmp_path / "lower", targets=(), textual=True)
    obj = tmp_path / f"model_{n}_{optimization}.o"
    lib = tmp_path / f"model_{n}_{optimization}.so"
    subprocess.run(
        [str(toolchain.clang()), optimization, "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(lib)],
        check=True,
        capture_output=True,
    )
    expected = np.load(tmp_path / "golden.npy")
    actual = np.zeros_like(expected)
    HostModel.load(str(lib))([(v.ctypes.data, v.shape) for v in (x, actual)])
    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))


@pytest.mark.parametrize("shape", [[], [7], [8192], [2, 0], [2, -1]])
def test_refuse_uncovered_axis(shape):
    with pytest.raises(ValueError):
        _module(shape)


def test_require_policy_finite_and_dtype_contracts():
    v = Block(arg_types=[builtin.TensorType(builtin.f32, [960])]).args[0]
    with pytest.raises(ValueError, match="pinned"):
        build_cpu_sum_lastdim(v, backend_policy="math")
    with pytest.raises(ValueError, match="finite"):
        build_cpu_sum_lastdim(v, backend_policy=CPU_SUM_F32_AVX2_POLICY)
    v = Block(arg_types=[builtin.TensorType(builtin.bf16, [960])]).args[0]
    with pytest.raises(TypeError):
        build_cpu_sum_lastdim(
            v, backend_policy=CPU_SUM_F32_AVX2_POLICY, allow_reassociation=True, assume_finite_intermediates=True
        )
