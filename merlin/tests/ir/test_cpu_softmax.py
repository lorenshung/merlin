"""Execute the selected source softmax schedule, including incomplete vector tails."""

from __future__ import annotations

import os
import subprocess

import numpy as np
import pytest
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from merlin.llvmlower import toolchain
from merlin.llvmlower.cpu_softmax import CPU_SOFTMAX_F32_AVX2_POLICY, build_cpu_softmax_lastdim


def _module(shape):
    typ = builtin.TensorType(builtin.f32, shape)
    b = Block(arg_types=[typ])
    ops, result = build_cpu_softmax_lastdim(
        b.args[0], backend_policy=CPU_SOFTMAX_F32_AVX2_POLICY, assume_finite_inputs=True
    )
    b.add_ops([*ops, func.ReturnOp(result)])
    m = builtin.ModuleOp([func.FuncOp("forward", ([typ], [typ]), Region(b))])
    m.verify()
    return m


@pytest.mark.parametrize("shape", [[7], [2, 0], [2, -1]])
def test_refuse_unsupported_shape(shape):
    with pytest.raises(ValueError):
        _module(shape)


def test_require_source_and_finite_contracts():
    v = Block(arg_types=[builtin.TensorType(builtin.f32, [2, 113])]).args[0]
    with pytest.raises(ValueError, match="finite-input"):
        build_cpu_softmax_lastdim(v, backend_policy=CPU_SOFTMAX_F32_AVX2_POLICY)
    with pytest.raises(ValueError, match="pinned"):
        build_cpu_softmax_lastdim(v, backend_policy="math", assume_finite_inputs=True)


@pytest.mark.parametrize("dtype", [builtin.bf16, builtin.f64])
def test_refuse_other_dtype(dtype):
    v = Block(arg_types=[builtin.TensorType(dtype, [2, 113])]).args[0]
    with pytest.raises(TypeError):
        build_cpu_softmax_lastdim(v, backend_policy=CPU_SOFTMAX_F32_AVX2_POLICY, assume_finite_inputs=True)


@pytest.mark.parametrize("n", [8, 9, 15, 113, 163, 257])
@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_native_matches_original_torch_all_bits(tmp_path, n, optimization):
    if not toolchain.available():
        pytest.skip("native toolchain unavailable")
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    reference = """
import torch,numpy as np,sys
from pathlib import Path
if not torch.__version__.startswith('2.10.') or torch.backends.cpu.get_cpu_capability()!='AVX2':sys.exit(77)
n=int(sys.argv[2]);root=Path(sys.argv[1]);torch.manual_seed(21);x=torch.randn(2,5,n)*4
x[0,0]=torch.finfo(torch.float32).min;x[0,1]=torch.linspace(-120,0,n);x[0,2]=1
x[1,0]=torch.finfo(torch.float32).min;x[1,0,0]=torch.finfo(torch.float32).max
x[1,1]=(-((torch.arange(n,dtype=torch.float32)%150)+.5))/1.4426950408889634;x[1,1,-1]=0
edge=torch.tensor([-104.,np.nextafter(np.float32(-104),np.float32(-np.inf)),-103.9,-87.33655,-0.,0.,-120.,-1e30]);x[1,2]=edge.repeat((n+7)//8)[:n]
np.save(root/'input.npy',x.numpy());np.save(root/'golden.npy',torch.softmax(x,-1).numpy())
"""
    run = subprocess.run(
        [str(toolchain.compiler_python()), "-c", reference, str(tmp_path), str(n)],
        env={**os.environ, "ATEN_CPU_CAPABILITY": "avx2"},
        capture_output=True,
        text=True,
    )
    if run.returncode == 77:
        pytest.skip("requires pinned Torch2.10 AVX2")
    assert run.returncode == 0, run.stderr
    source = tmp_path / "model.mlir"
    source.write_text(str(_module([2, 5, n])))
    lowered = lower_model_file(source, tmp_path / "lower", targets=(), textual=True)
    obj = tmp_path / f"model_{n}_{optimization}.o"
    shared = tmp_path / f"model_{n}_{optimization}.so"
    subprocess.run(
        [str(toolchain.clang()), optimization, "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(shared)],
        check=True,
        capture_output=True,
    )
    x = np.load(tmp_path / "input.npy")
    g = np.load(tmp_path / "golden.npy")
    a = np.zeros_like(x)
    HostModel.load(str(shared))([(z.ctypes.data, z.shape) for z in [x, a]])
    np.testing.assert_array_equal(a.view("u4"), g.view("u4"))
