"""The tiled schedule must preserve every output's original scalar FMA chain."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from merlin.llvmlower.ordered_fma_matmul import build_ordered_fma_matmul
from merlin.xdsl_dialects._common import text


def module(ls, rs, dtype, tile, transposed=False, pre_widen="none"):
    types = [builtin.TensorType(dtype, s) for s in (ls, rs)]
    block = Block(arg_types=types)
    ops, result = build_ordered_fma_matmul(
        *block.args,
        output_tile=tile,
        rhs_transposed=transposed,
        pre_widen_operands=pre_widen,
        accumulation_order="zero_seeded_increasing_k_fma",
        assume_rne=True,
        assume_finite_intermediates=True,
    )
    block.add_ops([*ops, func.ReturnOp(result)])
    result = builtin.ModuleOp([func.FuncOp("forward", (types, [result.type]), Region(block))])
    result.verify()
    return result


def execute_native_case(tmp_path, dtype, shape, tile, transposed, pre_widen="none", rewrite=False):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    m, k, n = shape
    shapes = ((2, m, k), (2, k, n))
    rng = np.random.default_rng(315)
    values = [np.asarray(rng.normal(size=s) * np.exp2(rng.integers(-12, 13, size=s)), dtype="f4") for s in shapes]
    if dtype == builtin.bf16:
        values = [((v.view("u4") >> 16) << 16).view("f4") for v in values]
        inputs = [(v.view("u4") >> 16).astype("u2") for v in values]
    else:
        inputs = values
    fma = ctypes.CDLL("libm.so.6").fmaf
    fma.argtypes = [ctypes.c_float] * 3
    fma.restype = ctypes.c_float
    expected = np.zeros((2, m, n), dtype="f4")
    for batch, i, j in np.ndindex(expected.shape):
        v = 0.0
        for q in range(k):
            v = fma(float(values[0][batch, i, q]), float(values[1][batch, q, j]), v)
        expected[batch, i, j] = v
    if transposed:
        inputs[1] = np.ascontiguousarray(inputs[1].swapaxes(-1, -2))
        shapes = (shapes[0], inputs[1].shape)
    source = tmp_path / "model.mlir"
    if rewrite:
        from merlin.llvmlower.ordered_fma_rewrite import rewrite_ordered_fma_contractions

        ir = generic_module(*shapes, dtype, transposed)
        report = rewrite_ordered_fma_contractions(
            ir, output_tile=tile, pre_widen_operands=pre_widen, assume_rne=True, assume_finite_intermediates=True
        )
        assert report["rewritten"] == 1
    else:
        ir = module(*shapes, dtype, tile, transposed, pre_widen)
    source.write_text(text(parse_mlir_text(text(ir))))
    lower = lower_model_file(
        source, tmp_path / "lower", targets=(), textual=True, features=frozenset({"lower_fma_to_intrinsic"})
    )
    obj, lib = tmp_path / "model.o", tmp_path / f"model_{dtype}_{m}_{k}_{n}_{tile}_{transposed}_{pre_widen}.so"
    subprocess.run(
        [str(toolchain.clang()), "-O2", "-fPIC", "-c", str(lower.ll_path), "-o", str(obj)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["cc", "-shared", "-Wl,-Bsymbolic", "-fPIC", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(lib)],
        check=True,
        capture_output=True,
    )
    actual = np.zeros_like(expected)
    HostModel.load(str(lib))([(v.ctypes.data, v.shape) for v in (*inputs, actual)])
    np.testing.assert_array_equal(actual.view("u4"), expected.view("u4"))


@pytest.mark.parametrize("dtype", [builtin.f32, builtin.bf16])
@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize(
    "shape,tile", [((2, 1, 1), 4), ((2, 7, 5), 4), ((2, 64, 8), 4), ((2, 13, 3), 1), ((2, 19, 7), 3)]
)
def test_native_preserves_order_tail_and_batch(tmp_path, dtype, shape, tile, transposed):
    execute_native_case(tmp_path, dtype, shape, tile, transposed)


@pytest.mark.parametrize("transposed", [False, True])
@pytest.mark.parametrize("pre_widen", ["lhs", "rhs", "both"])
@pytest.mark.parametrize("shape,tile", [((2, 1, 1), 4), ((2, 7, 5), 4), ((2, 64, 8), 4)])
def test_pre_widening_preserves_original_bf16_fma_chain(tmp_path, shape, tile, transposed, pre_widen):
    execute_native_case(tmp_path, builtin.bf16, shape, tile, transposed, pre_widen=pre_widen)


def test_pre_widening_refuses_wrong_type():
    with pytest.raises(ValueError, match="BF16 operands"):
        module([2, 3], [3, 5], builtin.f32, 4, pre_widen="both")


@pytest.mark.parametrize("mode", [True, "automatic", 1, None])
def test_pre_widening_requires_explicit_operand_selection(mode):
    with pytest.raises(ValueError, match="none, lhs, rhs or both"):
        module([2, 3], [3, 5], builtin.bf16, 4, pre_widen=mode)


@pytest.mark.parametrize("tile", [0, -1, True, 1.5])
def test_refuses_bad_tiles(tile):
    with pytest.raises(ValueError, match="positive integer"):
        module([2, 3], [3, 5], builtin.f32, tile)


def test_requires_numeric_contract():
    args = Block(arg_types=[builtin.TensorType(builtin.bf16, [2, 3]), builtin.TensorType(builtin.bf16, [3, 5])]).args
    with pytest.raises(ValueError, match="contracts"):
        build_ordered_fma_matmul(*args, output_tile=4, accumulation_order="zero_seeded_increasing_k_fma")


@pytest.mark.parametrize(
    "ls,rs,dtype",
    [
        ([2, 3], [4, 5], builtin.f32),
        ([2, 3], [3, 5], builtin.f16),
        ([1, 2, 3], [2, 3, 5], builtin.bf16),
        ([2, -1], [-1, 5], builtin.f32),
    ],
)
def test_refuses_unproven_shape_and_type(ls, rs, dtype):
    with pytest.raises((ValueError, TypeError)):
        module(ls, rs, dtype, 4)


def generic_module(ls, rs, dtype, transposed=False, seed_value=0.0, fill=False):
    from xdsl.dialects import arith, math, tensor
    from xdsl.dialects.builtin import AffineMapAttr, FloatAttr
    from xdsl.dialects.linalg import ops as linalg
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    types = [builtin.TensorType(dtype, s) for s in (ls, rs)]
    block = Block(arg_types=types)
    shape = (*ls[:-2], ls[-2], rs[-2] if transposed else rs[-1])
    typ = builtin.TensorType(builtin.f32, shape)
    zero = arith.ConstantOp(FloatAttr(seed_value, builtin.f32))
    block.add_op(zero)
    if fill:
        empty = tensor.EmptyOp((), typ)
        seed = linalg.FillOp(inputs=[zero.result], outputs=[empty.tensor], res=[typ])
        block.add_ops([empty, seed])
    else:
        seed = tensor.SplatOp(zero.result, [], typ)
        block.add_op(seed)
    r = len(shape)
    prefix = tuple(AffineDimExpr(i) for i in range(r - 2))
    m, n, k = (AffineDimExpr(i) for i in (r - 2, r - 1, r))
    maps = [
        AffineMapAttr(AffineMap(r + 1, 0, x))
        for x in ((*prefix, m, k), (*prefix, n, k) if transposed else (*prefix, k, n), (*prefix, m, n))
    ]
    body = Block(arg_types=[dtype, dtype, builtin.f32])
    a, b = body.args[:2]
    if dtype == builtin.bf16:
        ext = [arith.ExtFOp(v, builtin.f32) for v in (a, b)]
        body.add_ops(ext)
        a, b = (x.result for x in ext)
    fused = math.FmaOp(a, b, body.args[2])
    body.add_ops([fused, linalg.YieldOp(fused.result)])
    generic = linalg.GenericOp(
        inputs=block.args,
        outputs=[seed.results[0]],
        body=Region(body),
        indexing_maps=builtin.ArrayAttr(maps),
        iterator_types=builtin.ArrayAttr(
            [linalg.IteratorTypeAttr(x) for x in (*([linalg.IteratorType.PARALLEL] * r), linalg.IteratorType.REDUCTION)]
        ),
        result_types=[typ],
    )
    block.add_ops([generic, func.ReturnOp(generic.results[0])])
    result = builtin.ModuleOp([func.FuncOp("forward", (types, [typ]), Region(block))])
    result.verify()
    return result


@pytest.mark.parametrize("dtype", [builtin.bf16, builtin.f32])
@pytest.mark.parametrize("transposed", [False, True])
def test_source_rewrite_preserves_batch_and_tail_fma_bits(tmp_path, dtype, transposed):
    execute_native_case(tmp_path, dtype, (2, 7, 5), 4, transposed, pre_widen="both", rewrite=True)


def test_source_fill_and_provenance_preservation():
    from xdsl.dialects.linalg import ops as linalg

    from merlin.llvmlower.ordered_fma_rewrite import rewrite_ordered_fma_contractions

    ir = generic_module([2, 7], [7, 5], builtin.f32, fill=True)
    op = next(x for x in ir.walk() if isinstance(x, linalg.GenericOp))
    op.attributes["arbitrary.source"] = builtin.StringAttr("unrelated identity")
    report = rewrite_ordered_fma_contractions(ir, output_tile=4, assume_rne=True, assume_finite_intermediates=True)
    assert report == dict(rewritten=1, bf16=0, f32=1, source_macs=70)
    assert any(x.attributes.get("arbitrary.source") == builtin.StringAttr("unrelated identity") for x in ir.walk())


@pytest.mark.parametrize("seed", [-0.0, 1.0, -2.0])
def test_source_rewrite_preserves_unsupported_initializers(seed):
    from merlin.llvmlower.ordered_fma_rewrite import rewrite_ordered_fma_contractions

    ir = generic_module([2, 7], [7, 5], builtin.bf16, seed_value=seed)
    before = text(ir)
    assert (
        rewrite_ordered_fma_contractions(ir, output_tile=4, assume_rne=True, assume_finite_intermediates=True)[
            "rewritten"
        ]
        == 0
    )
    assert text(ir) == before


def test_source_rewrite_refuses_changed_fma_wiring_and_reduction_order():
    from xdsl.dialects import math
    from xdsl.dialects.linalg import ops as linalg

    from merlin.llvmlower.ordered_fma_rewrite import rewrite_ordered_fma_contractions

    for changed in ("wiring", "order"):
        ir = generic_module([2, 7], [7, 5], builtin.bf16)
        op = next(x for x in ir.walk() if isinstance(x, linalg.GenericOp))
        if changed == "wiring":
            fused = next(x for x in op.body.block.ops if isinstance(x, math.FmaOp))
            fused.operands = (fused.operands[1], fused.operands[0], fused.operands[2])
        else:
            op.properties["iterator_types"] = builtin.ArrayAttr(
                [
                    linalg.IteratorTypeAttr(x)
                    for x in (linalg.IteratorType.REDUCTION, linalg.IteratorType.PARALLEL, linalg.IteratorType.PARALLEL)
                ]
            )
        before = text(ir)
        assert (
            rewrite_ordered_fma_contractions(ir, output_tile=4, assume_rne=True, assume_finite_intermediates=True)[
                "rewritten"
            ]
            == 0
        )
        assert text(ir) == before
