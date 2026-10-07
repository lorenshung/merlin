"""Shared iteration geometry is checked independently of scalar algorithms."""

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_patterns import InvalidLinalgPattern, _checked_static_indexed_linalg_shell


def _program(*, middle_axis=False):
    if middle_axis:
        shapes = ("2x5x7", "2x5x7")
        dims = "d0, d1, d2, d3"
        maps = ("d0, d3, d2", "d0, d1, d2")
        iterators = ("parallel", "parallel", "parallel", "reduction")
        tensor_types = tuple(f"tensor<{shape}xi64>" for shape in shapes)
        return f"""builtin.module {{
  func.func @scan(%x: {tensor_types[0]}, %init: {tensor_types[1]}) -> {tensor_types[1]} {{
    %out = "linalg.generic"(%x, %init) <{{
      indexing_maps = [{", ".join(f"affine_map<({dims}) -> ({mapping})>" for mapping in maps)}],
      iterator_types = [{", ".join(f"#linalg.iterator_type<{kind}>" for kind in iterators)}],
      operandSegmentSizes = array<i32: 1, 1>}}> ({{
      ^bb0(%value: i64, %acc: i64):
        %sum = arith.addi %value, %acc : i64
        "linalg.yield"(%sum) : (i64) -> ()
    }}) : ({tensor_types[0]}, {tensor_types[1]}) -> {tensor_types[1]}
    func.return %out : {tensor_types[1]}
  }}
}}"""
    return """builtin.module {
  func.func @pair(%x: tensor<3x5xi64>, %vinit: tensor<3xi64>,
                  %iinit: tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
    %value, %index = "linalg.generic"(%x, %vinit, %iinit) <{
      indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
                       affine_map<(d0, d1) -> (d0)>, affine_map<(d0, d1) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>],
      operandSegmentSizes = array<i32: 1, 2>}> ({
      ^bb0(%input: i64, %old: i64, %position: i64):
        "linalg.yield"(%input, %position) : (i64, i64) -> ()
    }) : (tensor<3x5xi64>, tensor<3xi64>, tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>)
    func.return %value, %index : tensor<3xi64>, tensor<3xi64>
  }
}"""


def _generic(text):
    return next(mq.walk(mq.parse(text), "linalg.generic"))


def test_geometry_handles_multiple_results_and_middle_axis_without_proving_body():
    shell = _checked_static_indexed_linalg_shell(_generic(_program()))
    assert shell.shape == (3, 5)
    assert shell.input_shapes == ((3, 5),)
    assert shell.output_shapes == ((3,), (3,))
    assert shell.ordered_types == ("i64",) * 5
    # This fixture is deliberately not an extremum algorithm: geometry is not admission.
    assert tuple(op.name for op in shell.body) == ("linalg.yield",)
    scan = _checked_static_indexed_linalg_shell(_generic(_program(middle_axis=True)))
    assert scan.shape == (2, 5, 7, 5)
    assert scan.output_shapes == ((2, 5, 7),)


@pytest.mark.parametrize(
    ("old", "new", "reason"),
    [
        ("tensor<3x5xi64>", "tensor<?x5xi64>", "static"),
        ("tensor<3x5xi64>", "tensor<4x5xi64>", "bounds disagree"),
        ("-> (d0, d1)>", "-> (d0, d0)>", "distinct-dimension"),
        ("-> (d0, d1)>", "-> (d0, d1 + 1)>", "distinct-dimension"),
        ("(d0, d1) -> (d0, d1)", "(d0, d1)[s0] -> (d0, d1)", "symbol"),
        ("array<i32: 1, 2>", "array<i64: 1, 2>", "segments"),
        ("array<i32: 1, 2>", "array<i32: 2, 1>", "segments"),
        ("operandSegmentSizes =", "extra = true, operandSegmentSizes =", "properties"),
    ],
)
def test_geometry_refuses_ambiguous_or_unreviewed_shells(old, new, reason):
    with pytest.raises(InvalidLinalgPattern, match=reason):
        _checked_static_indexed_linalg_shell(_generic(_program().replace(old, new)))


def test_geometry_requires_every_loop_bound_and_exact_result_types():
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, TensorType, i64
    from xdsl.dialects.linalg.attrs import IteratorTypeAttr
    from xdsl.ir.affine import AffineMap

    generic = _generic(_program())
    generic.properties["iterator_types"] = ArrayAttr([*generic.iterator_types, IteratorTypeAttr.parallel()])
    generic.properties["indexing_maps"] = ArrayAttr(
        AffineMapAttr(AffineMap(3, 0, mapping.data.results)) for mapping in generic.indexing_maps
    )
    with pytest.raises(InvalidLinalgPattern, match="unbounded"):
        _checked_static_indexed_linalg_shell(generic)
    generic = _generic(_program())
    generic.results[1]._type = TensorType(i64, (4,))
    with pytest.raises(InvalidLinalgPattern, match="initialized output"):
        _checked_static_indexed_linalg_shell(generic)
