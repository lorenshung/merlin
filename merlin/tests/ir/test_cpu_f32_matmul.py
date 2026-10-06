"""Execute source-selected ordered FMA without granting global fast math."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest
from xdsl.dialects import builtin, func, math, scf
from xdsl.ir import Block, Region

from merlin.llvmlower.cpu_f32_matmul import CPU_F32_ORDERED_FMA_POLICY, build_cpu_f32_matmul_ordered_fma
from merlin.xdsl_dialects._common import text


def _module(ls, rs):
    types = [builtin.TensorType(builtin.f32, s) for s in (ls, rs)]
    block = Block(arg_types=types)
    ops, result = build_cpu_f32_matmul_ordered_fma(
        *block.args,
        backend_policy=CPU_F32_ORDERED_FMA_POLICY,
        allow_contraction=True,
        assume_finite_intermediates=True,
    )
    block.add_ops([*ops, func.ReturnOp(result)])
    module = builtin.ModuleOp([func.FuncOp("forward", (types, [result.type]), Region(block))])
    module.verify()
    assert sum(isinstance(op, math.FmaOp) for op in module.walk()) == 1
    assert sum(isinstance(op, scf.ForOp) for op in module.walk()) == 1
    return module


def _reference(a, b):
    fma = ctypes.CDLL("libm.so.6").fmaf
    fma.argtypes = [ctypes.c_float] * 3
    fma.restype = ctypes.c_float
    out = np.zeros((*a.shape[:-2], a.shape[-2], b.shape[-1]), dtype=np.float32)
    for idx in np.ndindex(out.shape):
        batch, row, col = idx[:-2], idx[-2], idx[-1]
        value = 0.0
        for k in range(a.shape[-1]):
            value = fma(float(a[(*batch, row, k)]), float(b[(*batch, k, col)]), value)
        out[idx] = value
    return out


@pytest.mark.parametrize("k", [1, 2, 3, 7, 64, 113])
@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_native_ordered_fma(tmp_path, k, optimization):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    batch = [(), (2,), (1, 2)][k % 3]
    shapes = ((*batch, 3, k), (*batch, k, 5))
    rng = np.random.default_rng(219)
    a, b = [np.asarray(rng.normal(size=s) * np.exp2(rng.integers(-10, 11, size=s)), dtype="f4") for s in shapes]
    if k >= 2:
        a[..., 0, :] = 0
        b[..., :, 0] = 0
        a[..., 0, 0], b[..., 0, 0] = -1, 1
        a[..., 0, 1], b[..., 1, 0] = np.float32(1 + 2**-23), np.float32(1 - 2**-23)
    expected = _reference(a, b)
    if k >= 2:
        assert np.all(expected[..., 0, 0] == np.float32(-(2**-46)))
    source = tmp_path / "input.mlir"
    source.write_text(text(parse_mlir_text(text(_module(*shapes)))))
    lower = lower_model_file(
        source, tmp_path / "lower", targets=(), textual=True, features=frozenset({"lower_fma_to_intrinsic"})
    )
    obj, lib = tmp_path / "model.o", tmp_path / "model.so"
    subprocess.run(
        [str(toolchain.clang()), optimization, "-fPIC", "-c", str(lower.ll_path), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(lib)],
        check=True,
        capture_output=True,
    )
    actual = np.zeros_like(expected)
    HostModel.load(str(lib))([(v.ctypes.data, v.shape) for v in (a, b, actual)])
    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))


def test_requires_source_contract():
    args = Block(arg_types=[builtin.TensorType(builtin.f32, [2, 3]), builtin.TensorType(builtin.f32, [3, 4])]).args
    with pytest.raises(ValueError, match="explicit"):
        build_cpu_f32_matmul_ordered_fma(*args, backend_policy="generic")
    with pytest.raises(ValueError, match="finite"):
        build_cpu_f32_matmul_ordered_fma(*args, backend_policy=CPU_F32_ORDERED_FMA_POLICY)


@pytest.mark.parametrize(
    "ls,rs,dtype",
    [
        ([2, 3], [4, 5], builtin.f32),
        ([1, 2, 3], [2, 3, 4], builtin.f32),
        ([2, 0], [0, 3], builtin.f32),
        ([3], [3], builtin.f32),
        ([2, 3], [3, 4], builtin.bf16),
    ],
)
def test_refuse_unsupported(ls, rs, dtype):
    args = Block(arg_types=[builtin.TensorType(dtype, s) for s in (ls, rs)]).args
    with pytest.raises((ValueError, TypeError)):
        build_cpu_f32_matmul_ordered_fma(
            *args, backend_policy=CPU_F32_ORDERED_FMA_POLICY, allow_contraction=True, assume_finite_intermediates=True
        )
