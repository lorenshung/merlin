"""Common layouts are structural, do not consume fanout, and admit arbitrary tails."""

import itertools

import numpy as np
import pytest
from xdsl.dialects import tensor
from xdsl.dialects.builtin import DenseArrayBase, i64
from xdsl.dialects.linalg import Linalg
from xdsl.parser import Parser

from merlin.llvmlower.shared_permutation import prove_shared_transpose
from merlin.xdsl_dialects._common import make_context, text


def fixture(shape, permutation, dtype="i8", count=2):
    logical = tuple(shape[i] for i in permutation)
    ty = lambda s: "tensor<" + "x".join(map(str, s)) + "x" + dtype + ">"
    a, b = ty(shape), ty(logical)
    args = ", ".join(f"%a{i}: {a}" for i in range(count))
    lines = [
        f"module {{ func.func @f({args}) -> ({a}, " + ", ".join([b] * (count + 1)) + ") {",
        f"%init = tensor.empty() : {b}",
    ]
    for i in range(count):
        lines.append(f"%t{i} = linalg.transpose ins(%a{i} : {a}) outs(%init : {b}) permutation = {list(permutation)}")
    # Preserve both the original input and transpose results as live fanout.
    lines.append(
        "return %a0, %t0, " + ", ".join(f"%t{i}" for i in range(count)) + f" : {a}, " + ", ".join([b] * (count + 1))
    )
    lines.append("}}")
    module = Parser(make_context(tensor.Tensor, Linalg), "\n".join(lines)).parse_module()
    module.verify()
    values = [op.results[0] for op in module.walk() if op.name == "linalg.transpose"]
    return module, values, logical


@pytest.mark.parametrize(
    "shape,permutation,dtype,count",
    [
        ((3, 5), (1, 0), "i8", 2),
        ((2, 3, 7), (2, 0, 1), "f32", 3),
        ((1, 5, 3, 7), (0, 3, 1, 2), "i32", 2),
        ((7,), (0,), "f32", 1),
    ],
)
def test_tail_shapes_multiple_types_and_fanout_are_proved_without_mutation(shape, permutation, dtype, count):
    module, values, logical = fixture(shape, permutation, dtype, count)
    before = text(module, generic=True)
    uses = [tuple(v.uses) for v in values]
    assert len(uses[0]) == 2
    proof = prove_shared_transpose(values, logical)
    assert proof == dict(
        kind="shared_explicit_transpose",
        permutation=list(permutation),
        physical_shape=list(shape),
        logical_shape=list(logical),
    )
    assert text(module, generic=True) == before and [tuple(v.uses) for v in values] == uses
    # Independent element indexing validates moving a uniform binary operation
    # through the selected permutation, including dimensions smaller than tiles.
    a = np.arange(np.prod(shape)).reshape(shape)
    b = (a * 3 + 5) % 17
    np.testing.assert_array_equal(
        (a + 2 * b).transpose(permutation), a.transpose(permutation) + 2 * b.transpose(permutation)
    )


def test_every_rank_three_permutation_with_unequal_tail_dimensions():
    for permutation in itertools.permutations(range(3)):
        _, values, logical = fixture((2, 3, 5), permutation)
        assert prove_shared_transpose(values, logical)["permutation"] == list(permutation)


def test_equal_dimensions_do_not_prove_different_axis_identity():
    module, values, logical = fixture((3, 3), (1, 0))
    values[1].owner.properties["permutation"] = DenseArrayBase.from_list(i64, [0, 1])
    module.verify()
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="disagree"):
        prove_shared_transpose(values, logical)
    assert text(module, generic=True) == before


def test_unknown_producer_shape_change_and_empty_selection_refuse():
    module, values, logical = fixture((3, 5), (1, 0))
    before = text(module, generic=True)
    with pytest.raises(ValueError, match="explicit transpose"):
        prove_shared_transpose([values[0].owner.operands[0]], logical)
    with pytest.raises(ValueError, match="shape proof"):
        prove_shared_transpose(values, (3, 5))
    with pytest.raises(ValueError, match="requires operands"):
        prove_shared_transpose([], logical)
    assert text(module, generic=True) == before


def test_different_element_types_and_invalid_permutation_refuse():
    _, a, shape = fixture((3, 5), (1, 0), "i8")
    _, b, _ = fixture((3, 5), (1, 0), "f32")
    with pytest.raises(ValueError, match="element types"):
        prove_shared_transpose([a[0], b[0]], shape)
    a[0].owner.properties["permutation"] = DenseArrayBase.from_list(i64, [1, 1])
    with pytest.raises(ValueError, match="positive static axes"):
        prove_shared_transpose(a, shape)


@pytest.mark.parametrize("shape", [(0, 5), (-1, 5)])
def test_empty_or_dynamic_source_axes_refuse(shape):
    from xdsl.dialects.builtin import TensorType, i8

    _, values, logical = fixture((3, 5), (1, 0))
    values[0].owner.operands[0]._type = TensorType(i8, shape)
    with pytest.raises(ValueError, match="positive static axes"):
        prove_shared_transpose(values, logical)
