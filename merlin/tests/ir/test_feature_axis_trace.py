"""A rank-4 activation's feature axis, derived from the im2col contractions its data flows through.

A capture whose convolutions were rewritten as slices, transposes and reshapes feeding an integer
matmul records no framework operator that states a rank-4 axis order. Its dataflow does: the matmul's
reduction is the input's features and its output column is the output's. These modules are that
rewrite in miniature -- one-by-one convolutions, so every reshape and transpose the real rewrite
emits is here without its window taps.
"""

from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.xdsl_dialects.lowering import feature_axis_trace as T
from merlin.xdsl_dialects.lowering import group_demand as GD


class _Module:
    """Builds a function of im2col convolutions in the generic syntax a capture is printed in."""

    def __init__(self, arguments: list[str]):
        self.arguments = arguments
        self.lines: list[str] = []
        self.count = len(arguments)

    def name(self) -> str:
        self.count += 1
        return f"%v{self.count}"

    def empty(self, shape: str) -> str:
        out = self.name()
        self.lines.append(f'{out} = "tensor.empty"() : () -> tensor<{shape}>')
        return out

    def transpose(self, value: str, src: str, dst: str, perm: str, element: str = "i8") -> str:
        init, out = self.empty(f"{dst}x{element}"), self.name()
        self.lines.append(
            f'{out} = "linalg.transpose"({value}, {init}) <{{permutation = array<i64: {perm}>}}> ({{\n'
            f'^bb0(%a: {element}, %b: {element}):\n  "linalg.yield"(%a) : ({element}) -> ()\n'
            f"}}) : (tensor<{src}x{element}>, tensor<{dst}x{element}>) -> tensor<{dst}x{element}>"
        )
        return out

    def collapse(self, value: str, src: str, flat: int, element: str = "i8") -> str:
        rank = src.count("x") + 1
        out = self.name()
        group = ", ".join(f"{i} : i64" for i in range(rank))
        self.lines.append(
            f'{out} = "tensor.collapse_shape"({value}) <{{reassociation = [[{group}]]}}> : '
            f"(tensor<{src}x{element}>) -> tensor<{flat}x{element}>"
        )
        return out

    def expand(self, value: str, flat: int, dst: str, element: str = "i8") -> str:
        rank = dst.count("x") + 1
        out = self.name()
        group = ", ".join(f"{i} : i64" for i in range(rank))
        sizes = ", ".join(dst.split("x"))
        self.lines.append(
            f'{out} = "tensor.expand_shape"({value}) <{{reassociation = [[{group}]], '
            f"static_output_shape = array<i64: {sizes}>}}> : (tensor<{flat}x{element}>) -> tensor<{dst}x{element}>"
        )
        return out

    def matmul(self, lhs: str, rhs: str, m: int, k: int, n: int) -> str:
        zero, init, out = self.name(), self.name(), self.name()
        self.lines += [
            f'{zero} = "arith.constant"() <{{value = 0 : i32}}> : () -> i32',
            f'{init} = "tensor.splat"({zero}) : (i32) -> tensor<{m}x{n}xi32>',
            f'{out} = "linalg.generic"({lhs}, {rhs}, {init}) <{{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, '
            "affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = "
            "[#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], "
            "operandSegmentSizes = array<i32: 2, 1>}> ({\n"
            "^bb0(%x: i8, %y: i8, %acc: i32):\n"
            '  %xe = "arith.extsi"(%x) : (i8) -> i32\n'
            '  %ye = "arith.extsi"(%y) : (i8) -> i32\n'
            '  %p = "arith.muli"(%xe, %ye) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32\n'
            '  %s = "arith.addi"(%acc, %p) <{overflowFlags = #arith.overflow<none>}> : (i32, i32) -> i32\n'
            '  "linalg.yield"(%s) : (i32) -> ()\n'
            f"}}) : (tensor<{m}x{k}xi8>, tensor<{k}x{n}xi8>, tensor<{m}x{n}xi32>) -> tensor<{m}x{n}xi32>",
        ]
        return out

    def narrow(self, value: str, shape: str) -> str:
        """The accumulator back to the element type, elementwise (the requantize in miniature)."""
        init, out = self.empty(f"{shape}xi8"), self.name()
        rank = shape.count("x") + 1
        dims = ", ".join(f"d{i}" for i in range(rank))
        identity = f"affine_map<({dims}) -> ({dims})>"
        parallel = ", ".join(["#linalg.iterator_type<parallel>"] * rank)
        self.lines.append(
            f'{out} = "linalg.generic"({value}, {init}) <{{indexing_maps = [{identity}, {identity}], '
            f"iterator_types = [{parallel}], operandSegmentSizes = array<i32: 1, 1>}}> ({{\n"
            '^bb0(%a: i32, %b: i8):\n  %t = "arith.trunci"(%a) : (i32) -> i8\n  "linalg.yield"(%t) : (i8) -> ()\n'
            f"}}) : (tensor<{shape}xi32>, tensor<{shape}xi8>) -> tensor<{shape}xi8>"
        )
        return out

    def add(self, lhs: str, rhs: str, shape: str) -> str:
        init, out = self.empty(f"{shape}xi8"), self.name()
        identity = "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>"
        parallel = ", ".join(["#linalg.iterator_type<parallel>"] * 4)
        self.lines.append(
            f'{out} = "linalg.generic"({lhs}, {rhs}, {init}) <{{indexing_maps = [{identity}, {identity}, {identity}], '
            f"iterator_types = [{parallel}], operandSegmentSizes = array<i32: 2, 1>}}> ({{\n"
            '^bb0(%a: i8, %b: i8, %c: i8):\n  %s = "arith.addi"(%a, %b) <{overflowFlags = #arith.overflow<none>}> '
            ': (i8, i8) -> i8\n  "linalg.yield"(%s) : (i8) -> ()\n'
            f"}}) : (tensor<{shape}xi8>, tensor<{shape}xi8>, tensor<{shape}xi8>) -> tensor<{shape}xi8>"
        )
        return out

    def conv1x1(self, x: str, weight: str, c: int, hw: int, n: int, *, back_to_nchw: bool = True) -> str:
        """``x[1, c, hw, hw]`` through a one-by-one convolution as the int8 rewrite states it: NHWC
        patches by transpose and reshape, ``patches @ weight^T``, and the result reshaped (and, unless
        told otherwise, transposed) back to rank 4."""
        positions = hw * hw
        nhwc = self.transpose(x, f"1x{c}x{hw}x{hw}", f"1x{hw}x{hw}x{c}", "0, 2, 3, 1")
        patches = self.expand(self.collapse(nhwc, f"1x{hw}x{hw}x{c}", positions * c), positions * c, f"{positions}x{c}")
        rhs = self.transpose(weight, f"{n}x{c}", f"{c}x{n}", "1, 0")
        product = self.narrow(self.matmul(patches, rhs, positions, c, n), f"{positions}x{n}")
        rows = self.expand(self.collapse(product, f"{positions}x{n}", positions * n), positions * n, f"1x{hw}x{hw}x{n}")
        if not back_to_nchw:
            return rows
        return self.transpose(rows, f"1x{hw}x{hw}x{n}", f"1x{n}x{hw}x{hw}", "0, 3, 1, 2")

    def text(self, result: str, result_type: str) -> str:
        args = ", ".join(f"%a{i}: {t}" for i, t in enumerate(self.arguments))
        types = ", ".join(self.arguments)
        body = "\n    ".join(self.lines)
        return (
            '"builtin.module"() ({\n'
            f'  "func.func"() <{{sym_name = "forward", function_type = ({types}) -> {result_type}}}> ({{\n'
            f"  ^bb0({args}):\n    {body}\n"
            f'    "func.return"({result}) : ({result_type}) -> ()\n'
            "  }) : () -> ()\n"
            "}) : () -> ()\n"
        )


def _ops(text: str) -> dict[str, list]:
    module = mq.parse(text)
    found: dict[str, list] = {}
    for op in module.walk():
        found.setdefault(mq.op_name(op), []).append(op)
    return found


def _residual(*, conv2_back_to_nchw: bool = True, hw: int = 4, c: int = 3, n: int = 5) -> str:
    """Two im2col convolutions and the residual sum of their outputs: ``y1 + conv(y1)``."""
    b = _Module([f"tensor<1x{c}x{hw}x{hw}xi8>", f"tensor<{n}x{c}xi8>", f"tensor<{n}x{n}xi8>"])
    y1 = b.conv1x1("%a0", "%a1", c, hw, n)
    y2 = b.conv1x1(y1, "%a2", n, hw, n, back_to_nchw=conv2_back_to_nchw)
    total = b.add(y1, y2, f"1x{n}x{hw}x{hw}")
    return b.text(total, f"tensor<1x{n}x{hw}x{hw}xi8>")


def test_an_im2col_convolutions_output_and_input_feature_axes_are_derived():
    """The output's channels are the matmul's output column, put back at axis 1 by the reshape and the
    transpose; the input's channels are its reduction, the one axis of the image whose whole extent
    lands there."""
    with IR_LOCK:
        ops = _ops(_residual())
        (add,) = [
            op
            for op in ops["linalg.generic"]
            if len(op.operands) == 3 and op.results and len(mq.type_shape_dtype(op.results[0].type)[0]) == 4
        ]
        image = add.parent_block().args[0]
        derived = T.derive_feature_axis([image])
        assert derived.axis == 1 and any("reduction" in w for w in derived.witnesses[1])
        conv_output = add.operands[0]
        derived = T.derive_feature_axis([conv_output])
        assert derived.axis == 1 and any("output column" in w for w in derived.witnesses[1])


def test_a_residual_sum_of_two_such_outputs_is_stated_with_one_axis():
    """The sum's two operands, followed to their contractions, agree: features at axis 1, so the sum is
    stated in the [positions, features] plane its neighbours commit -- here [16, 5]."""
    with IR_LOCK:
        ops = _ops(_residual())
        add = next(
            op
            for op in ops["linalg.generic"]
            if len(op.operands) == 3 and len(mq.type_shape_dtype(op.results[0].type)[0]) == 4
        )
        assert GD.feature_axis(add, 4) == 1
        assert GD._activation_plane([1, 5, 4, 4], add) == (16, 5)


def test_operands_whose_contractions_disagree_are_refused():
    """The same-shaped sum of an output put back in NCHW and one left in NHWC: one path says features
    are at axis 1, the other at axis 3. Neither is guessed."""
    with IR_LOCK:
        ops = _ops(_residual(conv2_back_to_nchw=False, hw=4, c=4, n=4))
        add = next(
            op
            for op in ops["linalg.generic"]
            if len(op.operands) == 3 and len(mq.type_shape_dtype(op.results[0].type)[0]) == 4
        )
        with pytest.raises(GD.NoCapsuleForm, match="disagree"):
            GD.feature_axis(add, 4)


def test_an_ambiguous_chain_is_refused():
    """The output column reshaped so that features share a dim with positions ([16, 5] read back as
    [1, 2, 4, 10]): no axis of the sum lands wholly in the column, so nothing states the features."""
    b = _Module(["tensor<1x3x4x4xi8>", "tensor<5x3xi8>"])
    rows = b.conv1x1("%a0", "%a1", 3, 4, 5, back_to_nchw=False)
    flat = b.collapse(rows, "1x4x4x5", 80)
    folded = b.expand(flat, 80, "1x2x4x10")
    total = b.add(folded, folded, "1x2x4x10")
    with IR_LOCK:
        ops = _ops(b.text(total, "tensor<1x2x4x10xi8>"))
        add = next(
            op
            for op in ops["linalg.generic"]
            if len(op.operands) == 3 and len(mq.type_shape_dtype(op.results[0].type)[0]) == 4
        )
        with pytest.raises(GD.NoCapsuleForm, match="dataflow does not state it"):
            GD.feature_axis(add, 4)


def test_a_capture_whose_operators_state_the_axis_is_answered_by_them_alone():
    """The tagged answer keeps precedence, unchanged: a module recording a convolution's provenance is
    read off it, and the dataflow -- which would refuse this module -- is not consulted."""
    from xdsl.dialects.builtin import StringAttr

    with IR_LOCK:
        module = mq.parse(_residual(conv2_back_to_nchw=False, hw=4, c=4, n=4))
        add = next(
            op
            for op in module.walk()
            if mq.op_name(op) == "linalg.generic"
            and len(op.operands) == 3
            and len(mq.type_shape_dtype(op.results[0].type)[0]) == 4
        )
        first = next(op for op in module.walk() if mq.op_name(op) == "linalg.transpose")
        first.attributes["prov.aten"] = StringAttr("aten.conv2d.default")
        assert GD.feature_axis(add, 4) == 1
