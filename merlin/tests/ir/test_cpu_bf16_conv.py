"""Pinned four-partial convolution: refusal, native order and recorded replay."""

import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg import ops as linalg
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from merlin.llvmlower.cpu_bf16_conv import CPU_BF16_BLAS_ILP4_POLICY, rewrite_cpu_bf16_conv_ilp4


def _module(m=2, n=3, k=16):
    shapes = [(m, k), (k, n), (m,)]
    types = [builtin.TensorType(builtin.f32, s) for s in shapes]
    block = Block(arg_types=types)
    attrs = {
        key: builtin.StringAttr(value)
        for key, value in {
            "prov.aten": "aten.convolution.default",
            "prov.orig_dtype": "bfloat16",
            "prov.conv_path": "im2col_matmul",
        }.items()
    }
    attrs["prov.source_node_ids"] = builtin.ArrayAttr([builtin.StringAttr("test:conv")])

    def generic(inputs, shape, dtype, scalar, maps=None):
        typ = builtin.TensorType(dtype, shape)
        empty = tensor.EmptyOp((), typ)
        b = Block(arg_types=[x.type.element_type for x in inputs] + [dtype])
        inst = scalar(b.args)
        b.add_ops([inst, linalg.YieldOp(inst.results[0])])
        rank = len(shape)
        identity = builtin.AffineMapAttr(AffineMap.identity(rank))
        op = linalg.GenericOp(
            inputs=inputs,
            outputs=[empty.tensor],
            body=Region(b),
            indexing_maps=builtin.ArrayAttr([*(maps or [identity] * len(inputs)), identity]),
            iterator_types=builtin.ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * rank),
            result_types=[typ],
        )
        op.attributes.update(attrs)
        block.add_ops([empty, op])
        return op.results[0]

    inputs = []
    for arg, shape in zip(block.args, shapes):
        narrow = generic([arg], shape, builtin.bf16, lambda a: arith.TruncFOp(a[0], builtin.bf16))
        inputs.append(generic([narrow], shape, builtin.f32, lambda a: arith.ExtFOp(a[0], builtin.f32)))
    zero = arith.ConstantOp(builtin.FloatAttr(0, builtin.f32))
    init = tensor.SplatOp(zero.result, [], builtin.TensorType(builtin.f32, [m, n]))
    contraction = linalg.MatmulOp(inputs[:2], [init.results[0]], res=[init.results[0].type])
    contraction.attributes.update(attrs)
    block.add_ops([zero, init, contraction])
    identity = builtin.AffineMapAttr(AffineMap.identity(2))
    biasmap = builtin.AffineMapAttr(AffineMap.from_callable(lambda i, j: (i,)))
    biased = generic(
        [contraction.results[0], inputs[2]],
        [m, n],
        builtin.f32,
        lambda a: arith.AddfOp(a[0], a[1]),
        [identity, biasmap],
    )
    result = generic([biased], [m, n], builtin.bf16, lambda a: arith.TruncFOp(a[0], builtin.bf16))
    result = generic([result], [m, n], builtin.f32, lambda a: arith.ExtFOp(a[0], builtin.f32))
    block.add_op(func.ReturnOp(result))
    module = builtin.ModuleOp([func.FuncOp("forward", (types, [result.type]), Region(block))])
    return module, contraction


def _rewrite(op, **kw):
    options = dict(
        backend_policy=CPU_BF16_BLAS_ILP4_POLICY,
        source_node_ids=("test:conv",),
        allow_reassociation=True,
        assume_finite_intermediates=True,
    )
    options.update(kw)
    return rewrite_cpu_bf16_conv_ilp4(op, **options)


@pytest.mark.parametrize(
    "options",
    [
        dict(backend_policy="math"),
        dict(source_node_ids=()),
        dict(source_node_ids=("other",)),
        dict(allow_reassociation=False),
        dict(assume_finite_intermediates=False),
    ],
)
def test_refuse_obligations_without_mutation(options):
    module, op = _module()
    before = str(module)
    with pytest.raises(ValueError):
        _rewrite(op, **options)
    assert str(module) == before


@pytest.mark.parametrize("bad", ["tail", "dtype", "provenance", "bias", "widening", "init", "extra_use"])
def test_refuse_structural_mismatch(bad):
    module, op = _module(k=15 if bad == "tail" else 16)
    if bad == "dtype":
        op.attributes["prov.orig_dtype"] = builtin.StringAttr("float32")
    elif bad == "provenance":
        del op.operands[0].owner.attributes["prov.aten"]
    elif bad == "bias":
        bias = next(iter(op.results[0].uses)).operation
        b = bias.body.block
        old = b.first_op
        from xdsl.rewriter import Rewriter

        Rewriter.replace_op(old, arith.MulfOp(*old.operands))
    elif bad == "widening":
        op.operands = (op.parent.args[0], *op.operands[1:])
    elif bad == "init":
        op.operands[2].owner.operands[0].owner.properties["value"] = builtin.FloatAttr(1, builtin.f32)
    elif bad == "extra_use":
        op.parent.insert_op_before(func.ReturnOp(op.results[0]), op.parent.last_op)
    before = str(module)
    with pytest.raises(ValueError):
        _rewrite(op)
    assert str(module) == before


def _bf16(x):
    bits = np.asarray(x, dtype=np.float32).view(np.uint32)
    return ((bits + 0x7FFF + ((bits >> 16) & 1)) & 0xFFFF0000).view(np.float32)


def _reference(a, b, bias):
    s = np.zeros((a.shape[0], b.shape[1], 4), np.float32)
    for k in range(a.shape[1]):
        s[:, :, k % 4] += a[:, k, None] * b[None, k, :]
    return _bf16(((s[:, :, 0] + s[:, :, 1]) + s[:, :, 2]) + s[:, :, 3] + bias[:, None])


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
@pytest.mark.parametrize("case", ["small", "recorded"])
def test_native_exact_four_partial_order(tmp_path, case, optimization):
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    if not toolchain.available():
        pytest.skip("native compiler unavailable")
    if case == "recorded":
        directory = os.environ.get("MERLIN_BF16_CONV_FIXTURE")
        if directory is None:
            pytest.skip("set MERLIN_BF16_CONV_FIXTURE to original BF16 conv audit directory")
        root = Path(directory)
        a = np.load(root / "weight.npy").reshape(768, 768).astype(np.float32)
        b = np.load(root / "columns.npy").reshape(768, 1024).astype(np.float32)
        bias = np.load(root / "bias.npy").astype(np.float32)
        expected = np.load(root / "output.npy").reshape(768, 1024).astype(np.float32)
        np.testing.assert_array_equal(_reference(a, b, bias), expected)
    else:
        rng = np.random.default_rng(18)
        a = _bf16(rng.normal(size=(5, 768)).astype(np.float32))
        b = _bf16(rng.normal(size=(768, 9)).astype(np.float32))
        bias = _bf16(rng.normal(size=5).astype(np.float32))
        # A cancellation witness distinguishes four lanes from serial reduction:
        # lane0 cancels its large terms before lane1 contributes one.
        a[0, :] = 1
        b[:, 0] = 0
        b[0, 0], b[1, 0], b[4, 0] = 2**24, 1, -(2**24)
        bias[0] = 0
        expected = _reference(a, b, bias)
        assert expected[0, 0] == 1
        assert np.float32(np.float32(2**24) + np.float32(1)) - np.float32(2**24) == 0
    module, op = _module(a.shape[0], b.shape[1], a.shape[1])
    _rewrite(op)
    module.verify()
    tmp_path = tmp_path / (case + optimization)
    tmp_path.mkdir()
    source = tmp_path / (case + ".mlir")
    source.write_text(str(module))
    lowered = lower_model_file(source, tmp_path / case, targets=(), textual=True)
    shared = tmp_path / (case + ".so")
    subprocess.run(
        [
            str(toolchain.clang()),
            optimization,
            "-fPIC",
            "-shared",
            str(lowered.ll_path),
            str(mlir_runtime_c()),
            "-lm",
            "-o",
            str(shared),
        ],
        check=True,
        capture_output=True,
    )
    output = np.full_like(expected, np.nan)
    arrays = [a, b, bias, output]
    HostModel.load(str(shared))([(x.ctypes.data, x.shape) for x in arrays])
    np.testing.assert_array_equal(output, expected)


def test_generic_contraction_and_explicit_left_fold():
    from xdsl.rewriter import Rewriter

    module, named = _module()
    maps = [
        AffineMap.from_callable(f) for f in (lambda m, n, k: (m, k), lambda m, n, k: (k, n), lambda m, n, k: (m, n))
    ]
    generic = linalg.GenericOp(
        inputs=named.inputs,
        outputs=named.outputs,
        body=named.body.clone(),
        indexing_maps=builtin.ArrayAttr([builtin.AffineMapAttr(x) for x in maps]),
        iterator_types=builtin.ArrayAttr(
            [
                linalg.IteratorTypeAttr(x)
                for x in (linalg.IteratorType.PARALLEL, linalg.IteratorType.PARALLEL, linalg.IteratorType.REDUCTION)
            ]
        ),
        result_types=[named.results[0].type],
    )
    generic.attributes.update(named.attributes)
    Rewriter.replace_op(named, generic)
    result = _rewrite(generic)
    module.verify()
    body = result.owner.body.block
    adds = list(body.ops)[:-1]
    assert len(adds) == 3
    assert tuple(adds[0].operands) == (body.args[0], body.args[1])
    assert tuple(adds[1].operands) == (adds[0].result, body.args[2])
    assert tuple(adds[2].operands) == (adds[1].result, body.args[3])
    before = str(module)
    with pytest.raises(ValueError, match="already scheduled"):
        _rewrite(result.owner)
    assert str(module) == before
