"""Neutral, target-independent source-body checks for static pointwise Linalg."""

from __future__ import annotations

from hashlib import sha256

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_patterns import (
    InvalidLinalgPattern,
    recognize_static_pointwise,
    screen_static_pointwise_source,
)


def _program(
    *,
    shape: str = "3x5",
    lhs_type: str = "f32",
    rhs_type: str = "f32",
    result_type: str = "f32",
    scalar_op: str = "arith.addf",
    scalar_properties: str = " <{fastmath = #arith.fastmath<none>}>",
    map_result: str = "(d0, d1)",
    iterator: str = "#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>",
    scalar_args: str = "%a, %b",
    extra_body: str = "",
) -> str:
    rank = len(shape.split("x"))
    dims = ", ".join(f"d{i}" for i in range(rank))
    identity = f"({dims})"
    maps = ", ".join(
        f"affine_map<({dims}) -> {value}>" for value in (identity, identity, map_result if rank == 2 else identity)
    )
    iters = iterator if rank == 2 else ", ".join("#linalg.iterator_type<parallel>" for _ in range(rank))
    return f'''builtin.module {{
  func.func @forward(%x: tensor<{shape}x{lhs_type}>, %y: tensor<{shape}x{rhs_type}>,
                     %init: tensor<{shape}x{result_type}>) -> tensor<{shape}x{result_type}> {{
    %out = "linalg.generic"(%x, %y, %init) <{{indexing_maps = [{maps}],
      iterator_types = [{iters}], operandSegmentSizes = array<i32: 2, 1>}}> ({{
      ^bb0(%a: {lhs_type}, %b: {rhs_type}, %acc: {result_type}):
        %v = "{scalar_op}"({scalar_args}){scalar_properties} : ({lhs_type}, {rhs_type}) -> {result_type}
        {extra_body}
        "linalg.yield"(%v) : ({result_type}) -> ()
    }}) : (tensor<{shape}x{lhs_type}>, tensor<{shape}x{rhs_type}>,
           tensor<{shape}x{result_type}>) -> tensor<{shape}x{result_type}>
    func.return %out : tensor<{shape}x{result_type}>
  }}
}}'''


def _generic(text: str):
    return next(mq.walk(mq.parse(text), "linalg.generic"))


def _typed_program(
    operation: str,
    inputs: tuple[str, ...],
    result: str,
    *,
    shape: str = "3x5",
    properties: str = "",
    operands: tuple[int, ...] | None = None,
    extra_body: str = "",
) -> str:
    rank = len(shape.split("x"))
    dims = ", ".join(f"d{i}" for i in range(rank))
    args = tuple(f"%v{i}" for i in range(len(inputs)))
    refs = tuple(args[i] for i in (operands if operands is not None else tuple(range(len(inputs)))))
    signatures = [f"%x{i}: tensor<{shape}x{dtype}>" for i, dtype in enumerate(inputs)]
    signatures.append(f"%init: tensor<{shape}x{result}>")
    input_tensors = ", ".join(f"tensor<{shape}x{dtype}>" for dtype in inputs)
    all_tensors = f"{input_tensors}, tensor<{shape}x{result}>"
    block_args = ", ".join(f"{arg}: {dtype}" for arg, dtype in zip(args, inputs))
    block_args += f", %acc: {result}"
    maps = ", ".join(f"affine_map<({dims}) -> ({dims})>" for _ in range(len(inputs) + 1))
    iterators = ", ".join("#linalg.iterator_type<parallel>" for _ in range(rank))
    return f'''builtin.module {{
  func.func @forward({", ".join(signatures)}) -> tensor<{shape}x{result}> {{
    %out = "linalg.generic"({", ".join(f"%x{i}" for i in range(len(inputs)))}, %init)
      <{{indexing_maps = [{maps}], iterator_types = [{iterators}],
        operandSegmentSizes = array<i32: {len(inputs)}, 1>}}> ({{
      ^bb0({block_args}):
        %v = "{operation}"({", ".join(refs)}){properties} : ({", ".join(inputs)}) -> {result}
        {extra_body}
        "linalg.yield"(%v) : ({result}) -> ()
      }}) : ({all_tensors}) -> tensor<{shape}x{result}>
    func.return %out : tensor<{shape}x{result}>
  }}
}}'''


def test_f32_add_identity_iteration_is_recognized():
    proof = recognize_static_pointwise(_generic(_program()))
    assert proof.operation == "arith.addf"
    assert proof.shape == (3, 5)
    assert proof.ordered_types == ("f32", "f32", "f32", "f32")


def test_nonidentity_output_map_is_refused():
    with pytest.raises(InvalidLinalgPattern, match="identity"):
        recognize_static_pointwise(_generic(_program(map_result="(d1, d0)")))


@pytest.mark.parametrize(
    ("operation", "inputs", "result", "properties", "predicate"),
    [
        ("arith.addf", ("f32", "f32"), "f32", "", None),
        ("arith.subf", ("f32", "f32"), "f32", "", None),
        ("arith.mulf", ("f32", "f32"), "f32", "", None),
        ("arith.negf", ("f32",), "f32", "", None),
        ("arith.select", ("i1", "f32", "f32"), "f32", "", None),
        ("arith.cmpf", ("f32", "f32"), "i1", " <{predicate = 4 : i64}>", "olt"),
        ("arith.addi", ("i64", "i64"), "i64", "", None),
        ("arith.subi", ("i64", "i64"), "i64", "", None),
        ("arith.muli", ("i64", "i64"), "i64", "", None),
        ("arith.cmpi", ("i64", "i64"), "i1", " <{predicate = 2 : i64}>", "slt"),
        ("arith.andi", ("i1", "i1"), "i1", "", None),
        ("arith.xori", ("i1", "i1"), "i1", "", None),
        ("arith.extui", ("i1",), "i64", "", None),
        ("arith.uitofp", ("i1",), "f32", "", None),
        ("arith.sitofp", ("i64",), "f32", "", None),
    ],
)
def test_exact_scalar_forms_across_neutral_ranks(operation, inputs, result, properties, predicate):
    for shape in ("7", "3x5x2"):
        proof = recognize_static_pointwise(
            _generic(_typed_program(operation, inputs, result, shape=shape, properties=properties))
        )
        assert proof.operation == operation
        assert proof.shape == tuple(int(dim) for dim in shape.split("x"))
        assert proof.ordered_types == (*inputs, result, result)
        assert proof.predicate == predicate


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (_program(iterator="#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>"), "parallel"),
        (_program(scalar_args="%a, %acc"), "input block arguments"),
        (_program(scalar_properties=" <{fastmath = #arith.fastmath<reassoc>}>"), "fast-math"),
        (_program(extra_body='%other = "arith.negf"(%v) : (f32) -> f32'), "one scalar"),
        (_typed_program("arith.cmpf", ("f32", "f32"), "i1", properties=" <{predicate = 10 : i64}>"), "ordered/signed"),
        (_typed_program("arith.cmpi", ("i64", "i64"), "i1", properties=" <{predicate = 6 : i64}>"), "ordered/signed"),
        (
            _typed_program("arith.addi", ("i64", "i64"), "i64", properties=" <{overflowFlags = #arith.overflow<nsw>}>"),
            "overflow",
        ),
        (_typed_program("arith.addf", ("f64", "f64"), "f64"), "signature"),
        (_typed_program("arith.addf", ("f32", "f32"), "f32", shape="?x5"), "static"),
        (_typed_program("arith.select", ("i1", "f32", "f32"), "f32", operands=(0, 2, 1)), "input block arguments"),
        (_typed_program("arith.ori", ("i1", "i1"), "i1"), "registered pure arith pointwise"),
        (
            _program().replace(
                " <{fastmath = #arith.fastmath<none>}>", " <{fastmath = #arith.fastmath<none>}> {unreviewed = 1 : i64}"
            ),
            "unrecognized attributes",
        ),
    ],
)
def test_structural_mutations_are_refused(text, reason):
    with pytest.raises(InvalidLinalgPattern, match=reason):
        recognize_static_pointwise(_generic(text))


def test_source_wrapper_binds_raw_normalized_hashes_and_exact_ordinals(tmp_path):
    path = tmp_path / "neutral.mlir"
    path.write_text(_program())
    parsed = mq.parse(_program())
    ordinal = next(i for i, op in enumerate(mq.walk(parsed)) if mq.op_name(op) == "linalg.generic")
    proof = screen_static_pointwise_source(path, (ordinal,))
    assert proof.raw_sha256 == sha256(path.read_bytes()).hexdigest()
    assert proof.normalized_sha256 != ""
    assert proof.ordinals == ((ordinal, recognize_static_pointwise(_generic(_program()))),)
    path.write_text(_program(shape="2x7"))
    assert screen_static_pointwise_source(path, (ordinal,)).raw_sha256 != proof.raw_sha256
    for bad in ((True,), (-1,), (ordinal, ordinal), (10000,)):
        with pytest.raises(InvalidLinalgPattern):
            screen_static_pointwise_source(path, bad)
    path.write_text(_program(scalar_args="%a, %acc"))
    with pytest.raises(InvalidLinalgPattern, match="input block arguments"):
        screen_static_pointwise_source(path, (ordinal,))


def test_provenance_annotations_do_not_act_as_semantic_permissions():
    text = _program().replace(
        " <{fastmath = #arith.fastmath<none>}>",
        ' <{fastmath = #arith.fastmath<none>}> {prov.source_node_ids = ["neutral"]}',
    )
    assert recognize_static_pointwise(_generic(text)).operation == "arith.addf"


def test_registered_operations_and_closed_region_are_required():
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.ir import Block, Region

    class SameNameProxy:
        def __init__(self, actual):
            self.actual = actual

        def __getattr__(self, key):
            return getattr(self.actual, key)

    op = _generic(_program())
    with pytest.raises(InvalidLinalgPattern, match="registered"):
        recognize_static_pointwise(SameNameProxy(op))

    op = _generic(_program())
    scalar, yld = tuple(op.regions[0].block.ops)
    scalar.__class__ = type("SameNameAddf", (arith.AddfOp,), {})
    with pytest.raises(InvalidLinalgPattern, match="registered"):
        recognize_static_pointwise(op)

    op = _generic(_program())
    scalar, yld = tuple(op.regions[0].block.ops)
    yld.__class__ = type("SameNameYield", (YieldOp,), {})
    with pytest.raises(InvalidLinalgPattern, match="registered"):
        recognize_static_pointwise(op)

    op = _generic(_program())
    scalar, yld = tuple(op.regions[0].block.ops)
    scalar.add_region(Region())
    with pytest.raises(InvalidLinalgPattern, match="region"):
        recognize_static_pointwise(op)

    op = _generic(_program())
    scalar, yld = tuple(op.regions[0].block.ops)
    scalar._successors = (Block(),)
    with pytest.raises(InvalidLinalgPattern, match="successor"):
        recognize_static_pointwise(op)

    op = _generic(_program())
    scalar, yld = tuple(op.regions[0].block.ops)
    yld.attributes["unexpected"] = IntegerAttr(1, 64)
    with pytest.raises(InvalidLinalgPattern, match="yield.*attribute"):
        recognize_static_pointwise(op)

    op = _generic(_program())
    scalar, yld = tuple(op.regions[0].block.ops)
    yld.properties["unexpected"] = IntegerAttr(1, 64)
    with pytest.raises(InvalidLinalgPattern, match="yield.*propert"):
        recognize_static_pointwise(op)

    op = _generic(_program())
    scalar, yld = tuple(op.regions[0].block.ops)
    scalar.properties["fastmath"] = IntegerAttr(1, 64)
    with pytest.raises(InvalidLinalgPattern, match="fast-math"):
        recognize_static_pointwise(op)

    op = _generic(_typed_program("arith.cmpf", ("f32", "f32"), "i1", properties=" <{predicate = 4 : i64}>"))
    scalar, yld = tuple(op.regions[0].block.ops)
    scalar.properties.pop("predicate")
    with pytest.raises(InvalidLinalgPattern, match="predicate"):
        recognize_static_pointwise(op)


def test_source_wrapper_rehashes_normalized_text_and_verifies_complete_module(tmp_path, monkeypatch):
    from merlin.frontends import capture_normalization as cn

    path = tmp_path / "neutral.mlir"
    path.write_text(_program())
    normalized, receipt = cn.normalize_capture_mlir(path.read_text())
    ordinal = next(i for i, op in enumerate(mq.walk(mq.parse(normalized))) if op.name == "linalg.generic")
    assert mq.parse(normalized).verify() is None

    monkeypatch.setattr(cn, "normalize_capture_mlir", lambda _: (normalized, {**receipt, "output_sha256": "0" * 64}))
    with pytest.raises(InvalidLinalgPattern, match="output hash"):
        screen_static_pointwise_source(path, (ordinal,))

    invalid_module = normalized.replace("-> tensor<3x5xf32>}>", "-> tensor<2x7xf32>}>", 1)
    with pytest.raises(Exception):
        mq.parse(invalid_module).verify()
    monkeypatch.setattr(
        cn,
        "normalize_capture_mlir",
        lambda _: (
            invalid_module,
            {
                **receipt,
                "output_sha256": sha256(invalid_module.encode()).hexdigest(),
            },
        ),
    )
    with pytest.raises(InvalidLinalgPattern, match="module.*verif"):
        screen_static_pointwise_source(path, (ordinal,))
