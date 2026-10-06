"""Execute the explicit four-partial dot schedule and its source tail rule."""

from __future__ import annotations

import subprocess

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg import ops as L
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from merlin.llvmlower.cpu_bf16_matmul import CPU_BF16_BLAS_ILP4_POLICY, build_cpu_bf16_matmul_ilp4
from merlin.xdsl_dialects._common import text


def _module(ls, rs):
    types = [builtin.TensorType(builtin.f32, s) for s in (ls, rs)]
    b = Block(arg_types=types)

    def cast(v, dtype):
        shape = v.type.get_shape()
        typ = builtin.TensorType(dtype, shape)
        empty = tensor.EmptyOp((), typ)
        body = Block(arg_types=[v.type.element_type, dtype])
        op = arith.TruncFOp(body.args[0], dtype) if dtype == builtin.bf16 else arith.ExtFOp(body.args[0], dtype)
        body.add_ops([op, L.YieldOp(op.result)])
        generic = L.GenericOp(
            inputs=(v,),
            outputs=(empty.tensor,),
            body=Region(body),
            indexing_maps=builtin.ArrayAttr([builtin.AffineMapAttr(AffineMap.identity(len(shape)))] * 2),
            iterator_types=builtin.ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(shape)),
            result_types=(typ,),
        )
        b.add_ops([empty, generic])
        return generic.results[0]

    lhs, rhs = [cast(v, builtin.bf16) for v in b.args]
    ops, result = build_cpu_bf16_matmul_ilp4(
        lhs, rhs, backend_policy=CPU_BF16_BLAS_ILP4_POLICY, allow_reassociation=True, assume_finite_intermediates=True
    )
    b.add_ops(ops)
    output = cast(result, builtin.f32)
    b.add_op(func.ReturnOp(output))
    module = builtin.ModuleOp([func.FuncOp("forward", (types, [output.type]), Region(b))])
    module.verify()
    return module


def _bf16(x):
    x = np.asarray(x, dtype=np.float32)
    bits = x.view(np.uint32)
    return ((bits + np.uint32(0x7FFF) + ((bits >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def _reference(a, b):
    k = a.shape[-1]
    partials = [np.zeros((*a.shape[:-2], a.shape[-2], b.shape[-1]), dtype=np.float32) for _ in range(4)]
    for p in range(k - k % 4):
        partials[p % 4] += a[..., :, p, None] * b[..., None, p, :]
    for p in range(k - k % 4, k):
        partials[0] += a[..., :, p, None] * b[..., None, p, :]
    return _bf16(((partials[0] + partials[1]) + partials[2]) + partials[3])


@pytest.mark.parametrize("k", [1, 2, 3, 4, 5, 7, 113, 163])
@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_native_source_order_with_tails(tmp_path, k, optimization):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    batch = [(), (2,), (1, 2)][k % 3]
    ls, rs = (*batch, 3, k), (*batch, k, 5)
    rng = np.random.default_rng(937)
    a, b = [_bf16(rng.normal(size=s) * np.exp2(rng.integers(-6, 7, size=s))) for s in (ls, rs)]
    expected = _reference(a, b)
    source = tmp_path / "input.mlir"
    # Both compiler parse seams are exercised, not just direct upstream parsing.
    from merlin.frontends.linalg_mlir import parse_mlir_text

    source.write_text(text(parse_mlir_text(text(_module(ls, rs)))))
    lower = lower_model_file(source, tmp_path / "lower", targets=(), textual=True)
    obj = tmp_path / f"model_{k}_{optimization}.o"
    lib = tmp_path / f"model_{k}_{optimization}.so"
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


def test_refuse_unproven_policy():
    a, b = Block(arg_types=[builtin.TensorType(builtin.bf16, [2, 3]), builtin.TensorType(builtin.bf16, [3, 4])]).args
    with pytest.raises(ValueError, match="pinned"):
        build_cpu_bf16_matmul_ilp4(a, b, backend_policy="math")
    with pytest.raises(ValueError, match="finite"):
        build_cpu_bf16_matmul_ilp4(a, b, backend_policy=CPU_BF16_BLAS_ILP4_POLICY)


@pytest.mark.parametrize(
    "ls,rs,dtype",
    [
        ([2, 3], [4, 5], builtin.bf16),
        ([1, 2, 3], [2, 3, 4], builtin.bf16),
        ([2, 0], [0, 3], builtin.bf16),
        ([3], [3], builtin.bf16),
        ([2, 3], [3, 4], builtin.f32),
    ],
)
def test_refuse_unsupported_operands(ls, rs, dtype):
    a, b = Block(arg_types=[builtin.TensorType(dtype, ls), builtin.TensorType(dtype, rs)]).args
    with pytest.raises((ValueError, TypeError)):
        build_cpu_bf16_matmul_ilp4(
            a, b, backend_policy=CPU_BF16_BLAS_ILP4_POLICY, allow_reassociation=True, assume_finite_intermediates=True
        )
