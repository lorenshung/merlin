"""Neutral two-result integer-minimum source contracts, not generated kernels."""

from hashlib import sha256
from itertools import permutations

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_extremum_patterns import (
    recognize_static_i64_argmin,
    screen_static_i64_argmin_source,
)
from merlin.frontends.linalg_patterns import InvalidLinalgPattern


def _program():
    return """builtin.module {
  func.func @pair(%x: tensor<2x5xi64>) -> (tensor<2xi64>, tensor<2xi64>) {
    %seed = arith.constant 9223372036854775807 : i64
    %vinit = tensor.splat %seed : tensor<2xi64>
    %iinit = tensor.splat %seed : tensor<2xi64>
    %value, %index = "linalg.generic"(%x, %vinit, %iinit) <{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      operandSegmentSizes = array<i32: 1, 2>}> ({
      ^bb0(%input: i64, %old: i64, %old_index: i64):
        %coordinate = "linalg.index"() <{dim = 1 : i64}> : () -> index
        %position = arith.index_cast %coordinate : index to i64
        %better = arith.cmpi slt, %input, %old : i64
        %same = arith.cmpi eq, %input, %old : i64
        %earlier = arith.cmpi ult, %position, %old_index : i64
        %tie = arith.andi %same, %earlier : i1
        %choose = arith.ori %better, %tie : i1
        %new_value = arith.select %choose, %input, %old : i64
        %new_index = arith.select %choose, %position, %old_index : i64
        "linalg.yield"(%new_value, %new_index) : (i64, i64) -> ()
    }) : (tensor<2x5xi64>, tensor<2xi64>, tensor<2xi64>) -> (tensor<2xi64>, tensor<2xi64>)
    func.return %value, %index : tensor<2xi64>, tensor<2xi64>
  }
}"""


def _generic(text):
    return next(mq.walk(mq.parse(text), "linalg.generic"))


def test_minimum_first_index_source_and_exact_module_binding(tmp_path):
    text = _program()
    module = mq.parse(text)
    module.verify()
    generic = next(mq.walk(module, "linalg.generic"))
    pattern = recognize_static_i64_argmin(generic, index_bits=64)
    assert (pattern.operation, pattern.axis) == ("i64_min_first_index", 1)
    assert pattern.input_shape == (2, 5) and pattern.output_shape == (2,)
    assert pattern.index_bits == 64 and pattern.ordered_types == ("i64",) * 5
    path = tmp_path / "neutral.mlir"
    path.write_text(text)
    ordinal = tuple(mq.walk(module)).index(generic)
    witness = screen_static_i64_argmin_source(path, (ordinal,), index_bits=64)
    assert witness.raw_sha256 == sha256(text.encode()).hexdigest()
    assert witness.ordinals == ((ordinal, pattern),)


@pytest.mark.parametrize(
    ("old", "new"),
    [
        ("9223372036854775807", "9223372036854775806"),
        ("cmpi slt", "cmpi sgt"),
        ("cmpi ult", "cmpi slt"),
        ("cmpi eq", "cmpi ne"),
        ("andi %same, %earlier", "andi %same, %same"),
        ("select %choose, %position, %old_index", "select %choose, %old_index, %position"),
        ('linalg.yield"(%new_value, %new_index)', 'linalg.yield"(%new_index, %new_value)'),
        ("dim = 1", "dim = 0"),
        ("tensor<2x5xi64>", "tensor<2x0xi64>"),
        ("#linalg.iterator_type<reduction>", "#linalg.iterator_type<parallel>"),
    ],
)
def test_minimum_rejects_wrong_seeds_signedness_ties_results_and_axis(old, new):
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_i64_argmin(_generic(_program().replace(old, new)), index_bits=64)


def test_minimum_requires_representable_loop_extent_and_explicit_index_premise():
    generic = _generic(_program())
    for bits in (None, True, 1, 2, 3, 129):
        with pytest.raises(InvalidLinalgPattern):
            recognize_static_i64_argmin(generic, index_bits=bits)
    # A reduced tensor may be scalar; that must not turn into an unqualified ABI claim.
    scalar_text = _program().replace("2x5", "5").replace("2xi64", "i64")
    scalar_text = scalar_text.replace("(d0, d1) -> (d0, d1)", "(d0) -> (d0)")
    scalar_text = scalar_text.replace("(d0, d1) -> (d0)", "(d0) -> ()")
    scalar_text = scalar_text.replace("#linalg.iterator_type<parallel>, ", "").replace("dim = 1", "dim = 0")
    assert recognize_static_i64_argmin(_generic(scalar_text), index_bits=64).output_shape == ()


def test_integer_pair_minimum_ties_are_independent_of_visit_order():
    # Finite algebra regression; this is not a theorem about compiler output.
    limit = (1 << 63) - 1
    for values in ((limit, limit, limit), (-5, 7, -5), (-(1 << 63), 0, -(1 << 63))):
        for order in permutations(range(len(values))):
            best = (limit, limit)
            for index in order:
                candidate = (values[index], index)
                if candidate[0] < best[0] or (candidate[0] == best[0] and candidate[1] < best[1]):
                    best = candidate
            assert best == min((value, index) for index, value in enumerate(values))
