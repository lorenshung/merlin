"""Neutral prepared-source checks for one-axis f32 maximum reductions."""

from hashlib import sha256

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_f32_maximum_patterns import (
    recognize_static_f32_maximum,
    screen_static_f32_maximum_source,
)
from merlin.frontends.linalg_patterns import InvalidLinalgPattern


def _program() -> str:
    return """builtin.module {
  func.func @max(%x: tensor<2x3x7xf32>) -> tensor<2x3xf32> {
    %neg_inf = arith.constant 0xff800000 : f32
    %init = tensor.splat %neg_inf : tensor<2x3xf32>
    %result = "linalg.reduce"(%x, %init) <{dimensions = array<i64: 2>}> ({
      ^bb0(%value: f32, %acc: f32):
        %maximum = "arith.maximumf"(%value, %acc) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "linalg.yield"(%maximum) : (f32) -> ()
    }) : (tensor<2x3x7xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    func.return %result : tensor<2x3xf32>
  }
}"""


def _reduce(text: str):
    module = mq.parse(text)
    return next(mq.walk(module, "linalg.reduce"))


def test_maximum_source_binds_exact_body_axis_seed_and_bytes(tmp_path):
    text = _program()
    mq.parse(text).verify()
    op = _reduce(text)
    pattern = recognize_static_f32_maximum(op, index_bits=64)
    assert (pattern.axis, pattern.input_shape, pattern.output_shape) == (2, (2, 3, 7), (2, 3))
    assert pattern.ordered_types == ("f32", "f32", "f32")
    assert pattern.index_bits_premise == 64
    path = tmp_path / "neutral.mlir"
    path.write_text(text)
    module = mq.parse(text)
    ordinal = tuple(mq.walk(module)).index(next(mq.walk(module, "linalg.reduce")))
    witness = screen_static_f32_maximum_source(path, (ordinal,), index_bits=64)
    assert witness.raw_sha256 == sha256(text.encode()).hexdigest()
    assert witness.ordinals == ((ordinal, pattern),)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("0xff800000", "0x7f800000"),
        ("arith.maximumf", "arith.minimumf"),
        ("#arith.fastmath<none>", "#arith.fastmath<fast>"),
        ("%value, %acc", "%acc, %acc"),
        ("linalg.yield\"(%maximum)", "linalg.yield\"(%acc)"),
        ("array<i64: 2>", "array<i64: 1>"),
        ("tensor<2x3xf32>", "tensor<2x7xf32>"),
    ],
)
def test_maximum_refuses_changed_body_seed_axis_or_geometry(old, new):
    text = _program().replace(old, new)
    with pytest.raises((InvalidLinalgPattern, ValueError)):
        recognize_static_f32_maximum(_reduce(text), index_bits=64)


def test_maximum_refuses_unselected_or_too_narrow_index_premise():
    op = _reduce(_program())
    for bits in (None, True, 1, 3, 129):
        with pytest.raises(InvalidLinalgPattern):
            recognize_static_f32_maximum(op, index_bits=bits)


def test_maximum_refuses_flattened_span_overflow_even_when_each_extent_fits():
    text = _program().replace("2x3x7", "16x16x7").replace("2x3xf32", "16x16xf32")
    with pytest.raises(InvalidLinalgPattern, match="element span"):
        recognize_static_f32_maximum(_reduce(text), index_bits=8)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("tensor<2x3x7xf32>", "tensor<2x3x?xf32>"),
        ("tensor<2x3x7xf32>", 'tensor<2x3x7xf32, "encoded">'),
    ],
)
def test_maximum_refuses_dynamic_or_encoded_source(old, new):
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_f32_maximum(_reduce(_program().replace(old, new)), index_bits=64)


def test_maximum_refuses_foreign_root_attribute_on_registered_reduction():
    from xdsl.dialects.builtin import BoolAttr

    op = _reduce(_program())
    op.attributes["unproved"] = BoolAttr.from_bool(True)
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_f32_maximum(op, index_bits=64)


def test_maximum_accepts_a_different_axis_and_scalar_result_without_abi_claim():
    middle = _program().replace("2x3x7", "2x7x3").replace("array<i64: 2>", "array<i64: 1>")
    assert recognize_static_f32_maximum(_reduce(middle), index_bits=64).output_shape == (2, 3)
    scalar = _program().replace("2x3x7", "7").replace("2x3xf32", "f32")
    scalar = scalar.replace("array<i64: 2>", "array<i64: 0>")
    assert recognize_static_f32_maximum(_reduce(scalar), index_bits=64).output_shape == ()


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ('"linalg.yield"(%maximum)', '"linalg.yield"(%maximum) {unsafe = true}'),
        ("%maximum =", "%extra = arith.addf %value, %acc : f32\n        %maximum ="),
    ],
)
def test_maximum_refuses_unproved_owner_or_body_mutations(old, new):
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_f32_maximum(_reduce(_program().replace(old, new)), index_bits=64)
