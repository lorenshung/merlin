"""Access observations are read from the graph, in the declarations' vocabulary, never assumed."""

from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.targetgen import access_observations as AO

_MAPS_MM = (
    "[affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d2, d1)>, affine_map<(d0, d1, d2) -> (d0, d1)>]"
)
_MAPS_BCAST = (
    "[affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d3, d2)>, "
    "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]"
)
_MAPS_BMM = (
    "[affine_map<(d0, d1, d2, d3) -> (d0, d1, d3)>, affine_map<(d0, d1, d2, d3) -> (d0, d3, d2)>, "
    "affine_map<(d0, d1, d2, d3) -> (d0, d1, d2)>]"
)


def _program(tmp_path, *, maps, iters, a, b, c, body, init="%e", acc="i32"):
    text = f""""builtin.module"() ({{
  "func.func"() <{{function_type = ({a}, {b}) -> {c}, sym_name = "forward"}}> ({{
  ^bb0(%x: {a}, %w: {b}):
    %e = "tensor.empty"() : () -> {c}
    %r = "linalg.generic"(%x, %w, {init}) <{{indexing_maps = {maps}, iterator_types = [{iters}], operandSegmentSizes = array<i32: 2, 1>}}> ({{
    ^bb1(%p: i8, %q: i8, %s: {acc}):
{body}
    }}) : ({a}, {b}, {c}) -> {c}
    "func.return"(%r) : ({c}) -> ()
  }}) : () -> ()
}}) : () -> ()
"""
    path = tmp_path / "program.mlir"
    path.write_text(text)
    module = mq.parse(str(path))
    return next(op for op in mq.walk(module) if mq.op_name(op) == "linalg.generic")


_MAC = """      %0 = "arith.extsi"(%p) : (i8) -> i32
      %1 = "arith.extsi"(%q) : (i8) -> i32
      %2 = "arith.muli"(%0, %1) : (i32, i32) -> i32
      %3 = "arith.addi"(%s, %2) : (i32, i32) -> i32
      "linalg.yield"(%3) : (i32) -> ()"""
_P, _R = "#linalg.iterator_type<parallel>", "#linalg.iterator_type<reduction>"


def test_an_integer_matmul_observes_every_constraint(tmp_path):
    op = _program(
        tmp_path,
        maps=_MAPS_MM,
        iters=f"{_P}, {_P}, {_R}",
        a="tensor<4x8xi8>",
        b="tensor<8x4xi8>",
        c="tensor<4x4xi32>",
        body=_MAC,
    )
    assert AO.observe(op, "contraction") == {
        "layout": "row_major_contiguous",
        "tails": "zero_pad_valid_window",
        "broadcasting": "none",
        "aliasing": "disjoint_inputs_outputs",
    }


def test_batches_and_broadcast_operands_are_told_apart(tmp_path):
    iters = f"{_P}, {_P}, {_P}, {_R}"
    batched = _program(
        tmp_path,
        maps=_MAPS_BMM,
        iters=iters,
        a="tensor<2x4x8xi8>",
        b="tensor<2x8x4xi8>",
        c="tensor<2x4x4xi32>",
        body=_MAC,
    )
    assert AO.observe(batched, "contraction")["broadcasting"] == "independent_batches"
    broadcast = _program(
        tmp_path,
        maps=_MAPS_BCAST,
        iters=iters,
        a="tensor<2x4x8xi8>",
        b="tensor<8x4xi8>",
        c="tensor<2x4x4xi32>",
        body=_MAC,
    )
    assert AO.observe(broadcast, "contraction")["broadcasting"] == "operand_broadcast"


def test_a_body_where_zero_padding_is_not_exact_has_no_tail_observation(tmp_path):
    body = _MAC.replace('"arith.addi"(%s, %2)', '"arith.maxsi"(%s, %2)')
    op = _program(
        tmp_path,
        maps=_MAPS_MM,
        iters=f"{_P}, {_P}, {_R}",
        a="tensor<4x8xi8>",
        b="tensor<8x4xi8>",
        c="tensor<4x4xi32>",
        body=body,
    )
    assert AO.observe(op, "contraction")["tails"] is None


def test_unknowns_stay_unknown(tmp_path):
    op = _program(
        tmp_path,
        maps=_MAPS_MM,
        iters=f"{_P}, {_P}, {_R}",
        a="tensor<4x8xi8>",
        b="tensor<8x4xi8>",
        c="tensor<4x4xi32>",
        body=_MAC,
    )
    assert (
        AO.observe(op, "elementwise_map")["tails"] is None and AO.observe(op, "elementwise_map")["broadcasting"] is None
    )
    empty = next(o for o in mq.walk(op.parent_op().parent_op()) if mq.op_name(o) == "tensor.empty")
    assert AO.observe(empty, "movement") == {"layout": None, "tails": None, "broadcasting": None, "aliasing": None}


def test_the_screen_admits_only_from_observed_values():
    from merlin.targetgen.operation_accounting import _observed_signature

    row = {"semantic_family": "contraction", "layout": "row_major_contiguous", "tails": None}
    observed = _observed_signature(row)
    assert observed["layout"] == "row_major_contiguous" and "tails" not in observed
