"""Neutral source-only checks for closed Boolean Linalg bodies."""

from __future__ import annotations

from hashlib import sha256

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_boolean_patterns import (
    DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA,
    recognize_dynamic_boolean_cast_body,
    recognize_static_boolean_body,
    screen_static_boolean_source,
    static_boolean_ordered_types,
    validate_dynamic_boolean_cast_source_body,
    validate_serialized_dynamic_boolean_cast_pattern,
    validate_static_boolean_source_body,
)
from merlin.frontends.linalg_patterns import InvalidLinalgPattern, recognize_static_pointwise


def _program(
    form: str,
    *,
    shape: str = "4x7",
    input_shapes: tuple[str, ...] | None = None,
    maps: tuple[str, ...] | None = None,
    body: str | None = None,
    iterators: str | None = None,
    result_type: str = "i1",
    input_dtypes: tuple[str, ...] | None = None,
) -> str:
    input_types = input_dtypes or {"not": ("i1",), "cast": ("f32",), "mul": ("i1", "i1")}[form]
    if input_shapes is None:
        input_shapes = (shape,) * len(input_types)
    rank = len(shape.split("x"))
    dims = ", ".join(f"d{i}" for i in range(rank))
    if maps is None:
        maps = (f"({dims})",) * len(input_types)
    if body is None:
        body = {
            "not": """%constant = "arith.constant"() <{value = true}> : () -> i1
        %value = "arith.xori"(%a0, %constant) : (i1, i1) -> i1""",
            "cast": """%constant = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
        %value = "arith.cmpf"(%a0, %constant)
          <{predicate = 13 : i64, fastmath = #arith.fastmath<none>}> : (f32, f32) -> i1""",
            "mul": """%value = "arith.muli"(%a0, %a1)
          <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1""",
        }[form]
    signature = [f"%x{i}: tensor<{extent}x{dtype}>" for i, (extent, dtype) in enumerate(zip(input_shapes, input_types))]
    signature.append(f"%init: tensor<{shape}x{result_type}>")
    tensors = [f"tensor<{extent}x{dtype}>" for extent, dtype in zip(input_shapes, input_types)]
    tensors.append(f"tensor<{shape}x{result_type}>")
    args = [f"%a{i}: {dtype}" for i, dtype in enumerate(input_types)]
    args.append(f"%acc: {result_type}")
    mapping = ", ".join(f"affine_map<({dims}) -> {value}>" for value in (*maps, f"({dims})"))
    iters = iterators or ", ".join("#linalg.iterator_type<parallel>" for _ in range(rank))
    return f"""builtin.module {{
  func.func @forward({", ".join(signature)}) -> tensor<{shape}x{result_type}> {{
    %out = "linalg.generic"({", ".join(f"%x{i}" for i in range(len(input_types)))}, %init)
      <{{indexing_maps = [{mapping}], iterator_types = [{iters}],
        operandSegmentSizes = array<i32: {len(input_types)}, 1>}}> ({{
      ^bb0({", ".join(args)}):
        {body}
        "linalg.yield"(%value) : ({result_type}) -> ()
      }}) : ({", ".join(tensors)}) -> tensor<{shape}x{result_type}>
    func.return %out : tensor<{shape}x{result_type}>
  }}
}}"""


def _generic(text: str):
    return next(mq.walk(mq.parse(text), "linalg.generic"))


def _dynamic_cast() -> str:
    return """builtin.module {
  func.func @forward(%x: tensor<?xi1>) -> tensor<?xi64> {
    %axis = "arith.constant"() <{value = 0 : index}> : () -> index
    %extent = "tensor.dim"(%x, %axis) : (tensor<?xi1>, index) -> index
    %init = "tensor.empty"(%extent) : (index) -> tensor<?xi64>
    %out = "linalg.generic"(%x, %init) <{indexing_maps = [
      affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = [#linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: 1, 1>}> ({
      ^bb0(%bit: i1, %old: i64):
        %wide = "arith.extui"(%bit) : (i1) -> i64
        "linalg.yield"(%wide) : (i64) -> ()
    }) : (tensor<?xi1>, tensor<?xi64>) -> tensor<?xi64>
    func.return %out : tensor<?xi64>
  }
}"""


def test_dynamic_boolean_extension_has_separate_closed_source_contract():
    import json
    from dataclasses import asdict

    from xdsl.dialects.builtin import DYNAMIC_INDEX

    declaration = {"schema": DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA, "operation": "i1_to_i64_extui"}
    assert validate_dynamic_boolean_cast_source_body(declaration) == declaration
    pattern = recognize_dynamic_boolean_cast_body(_generic(_dynamic_cast()))
    assert pattern.shape == (DYNAMIC_INDEX,)
    assert pattern.ordered_types == ("i1", "i64", "i64")
    validate_serialized_dynamic_boolean_cast_pattern(asdict(pattern))
    validate_serialized_dynamic_boolean_cast_pattern(json.loads(json.dumps(asdict(pattern))))
    for changed in (
        {**declaration, "predicate": "eq"},
        {**declaration, "schema": "merlin.static_boolean_body.v1"},
        {**declaration, "operation": "arith.extui"},
    ):
        with pytest.raises(InvalidLinalgPattern, match="source_body"):
            validate_dynamic_boolean_cast_source_body(changed)
    for key, value in (
        ("shape", [True]),
        ("ordered_types", ["i1", "i1", "i64"]),
        ("input_maps", ["(d0) -> (0)"]),
        ("shape_source", "ambient_length"),
    ):
        with pytest.raises(InvalidLinalgPattern, match="serialized"):
            validate_serialized_dynamic_boolean_cast_pattern({**asdict(pattern), key: value})


@pytest.mark.parametrize(
    "changed",
    [
        lambda s: s.replace("%x: tensor<?xi1>)", "%x: tensor<?xi1>, %y: tensor<?xi1>)").replace(
            '"tensor.dim"(%x, %axis)', '"tensor.dim"(%y, %axis)'
        ),
        lambda s: s.replace('"tensor.empty"(%extent)', '"tensor.empty"(%axis)'),
        lambda s: s.replace('"arith.extui"(%bit)', '"arith.extsi"(%bit)'),
        lambda s: s.replace('"arith.extui"(%bit) : (i1) -> i64', '"arith.addi"(%old, %old) : (i64, i64) -> i64'),
        lambda s: s.replace('"linalg.yield"(%wide)', '"linalg.yield"(%old)'),
        lambda s: s.replace("affine_map<(d0) -> (d0)>],", "affine_map<(d0) -> (0)>],"),
        lambda s: s.replace("#linalg.iterator_type<parallel>", "#linalg.iterator_type<reduction>"),
        lambda s: s.replace("tensor<?xi64>", "tensor<4xi64>"),
    ],
)
def test_dynamic_boolean_cast_near_misses_refuse(changed):
    with pytest.raises(InvalidLinalgPattern):
        recognize_dynamic_boolean_cast_body(_generic(changed(_dynamic_cast())))


def test_dynamic_boolean_host_declaration_is_source_bound_and_not_a_review_grant():
    from copy import deepcopy

    from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities

    declaration = {
        "id": "neutral_dynamic_cast",
        "ops": ["linalg.generic"],
        "placement": "host",
        "signature": {
            "ordered_operand_dtypes": ["i1", "i64"],
            "ordered_result_dtypes": ["i64"],
            "ranks": [1],
        },
        "source_body": {"schema": DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA, "operation": "i1_to_i64_extui"},
    }
    document = {
        "schema": "merlin.host_capabilities.v1",
        "status": "reviewed",
        "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "neutral"},
        "operations": [declaration],
        "evidence": {"scope": "synthetic declaration only"},
    }
    selected = {
        "host": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "neutral",
            "capability_spec": document,
        }
    }
    row = {"mlir_operation": "linalg.generic", "count": 1}
    signature = {
        "family": "cast",
        "ordered_operand_dtypes": ["i1", "i64"],
        "ordered_result_dtypes": ["i64"],
        "rank": 1,
    }
    validate_host_capabilities(document)
    assert admit_host_operation(selected, row, signature)["status"] == "unknown"
    assert (
        admit_host_operation(selected, row, signature, source_operations=(_generic(_dynamic_cast()),))["status"]
        == "admitted"
    )
    unreviewed = deepcopy(selected)
    unreviewed["host"]["capability_spec"]["operations"][0]["status"] = "unreviewed"
    assert (
        admit_host_operation(unreviewed, row, signature, source_operations=(_generic(_dynamic_cast()),))["status"]
        == "unknown"
    )
    declaration["source_body"] = {**declaration["source_body"], "allow_other_shapes": True}
    with pytest.raises(InvalidLinalgPattern, match="source_body"):
        validate_host_capabilities(document)


@pytest.mark.parametrize(
    ("operation", "types"),
    [
        ("i1_not", ("i1", "i1", "i1")),
        ("f32_nonzero_to_i1", ("f32", "i1", "i1")),
        ("i1_mul_singleton_projected", ("i1", "i1", "i1", "i1")),
    ],
)
def test_static_boolean_declaration_is_separate_and_closed(operation, types):
    declaration = {"schema": "merlin.static_boolean_body.v1", "operation": operation}
    assert validate_static_boolean_source_body(declaration) == declaration
    assert static_boolean_ordered_types(declaration) == types
    for invalid in (
        {**declaration, "predicate": "une"},
        {**declaration, "operation": "arith.xori"},
        {**declaration, "schema": "merlin.static_pointwise_source_body.v1"},
    ):
        with pytest.raises(InvalidLinalgPattern, match="source_body"):
            validate_static_boolean_source_body(invalid)


@pytest.mark.parametrize(
    ("form", "operation", "types"),
    [
        ("not", "i1_not", ("i1", "i1", "i1")),
        ("cast", "f32_nonzero_to_i1", ("f32", "i1", "i1")),
        ("mul", "i1_mul_singleton_projected", ("i1", "i1", "i1", "i1")),
    ],
)
def test_closed_boolean_forms_are_source_only(form, operation, types):
    proof = recognize_static_boolean_body(_generic(_program(form)))
    assert proof.operation == operation
    assert proof.shape == (4, 7)
    assert proof.ordered_types == types
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_pointwise(_generic(_program(form)))


def test_singleton_projection_on_each_boolean_input_is_proved():
    text = _program(
        "mul",
        shape="2x3x5",
        input_shapes=("2x1x5", "2x3x1"),
        maps=("(d0, 0, d2)", "(d0, d1, 0)"),
    )
    proof = recognize_static_boolean_body(_generic(text))
    assert proof.operation == "i1_mul_singleton_projected"
    assert proof.shape == (2, 3, 5)
    assert proof.input_maps == ("(d0, d1, d2) -> (d0, 0, d2)", "(d0, d1, d2) -> (d0, d1, 0)")


@pytest.mark.parametrize(
    ("text", "reason"),
    [
        (_program("not").replace("value = true", "value = false"), "true i1 constant"),
        (_program("not").replace('"arith.xori"(%a0, %constant)', '"arith.xori"(%acc, %constant)'), "input with true"),
        (_program("not").replace('"arith.xori"(%a0, %constant)', '"arith.ori"(%a0, %constant)'), "closed forms"),
        (_program("cast").replace("0.000000e+00", "1.000000e+00"), "positive zero"),
        (_program("cast").replace("0.000000e+00", "-0.000000e+00"), "positive zero"),
        (_program("cast").replace("predicate = 13", "predicate = 6"), "UNE"),
        (_program("cast").replace("#arith.fastmath<none>", "#arith.fastmath<nnan>"), "UNE"),
        (_program("cast").replace('"arith.cmpf"(%a0, %constant)', '"arith.cmpf"(%constant, %a0)'), "UNE"),
        (_program("mul").replace("#arith.overflow<none>", "#arith.overflow<nsw>"), "unflagged"),
        (_program("mul").replace('"arith.muli"(%a0, %a1)', '"arith.muli"(%a0, %acc)'), "unflagged"),
        (
            _program(
                "mul",
                result_type="i64",
                input_dtypes=("i64", "i64"),
                body="""%value = "arith.muli"(%a0, %a1)
          <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64""",
            ),
            "incorrect tensor types",
        ),
        (
            _program("mul", shape="2x3x5", input_shapes=("2x2x5", "2x3x1"), maps=("(d0, 0, d2)", "(d0, d1, 0)")),
            "singleton projection",
        ),
        (
            _program("mul", shape="2x3x5", input_shapes=("2x1x5", "2x3x1"), maps=("(d0, 1, d2)", "(d0, d1, 0)")),
            "singleton projection",
        ),
        (
            _program("mul", shape="2x3x5", input_shapes=("2x1x5", "2x3x1"), maps=("(d0, 0, d2 + 1)", "(d0, d1, 0)")),
            "singleton projection",
        ),
        (
            _program("mul", shape="2x2", input_shapes=("2x2", "2x2"), maps=("(d1, d0)", "(d0, d1)")),
            "singleton projection",
        ),
        (_program("not").replace("tensor<4x7xi1>", 'tensor<4x7xi1, "encoded">'), "encoding"),
        (_program("cast", shape="?x7"), "static"),
        (_program("mul", iterators="#linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>"), "parallel"),
    ],
)
def test_near_misses_are_refused(text, reason):
    with pytest.raises(InvalidLinalgPattern, match=reason):
        recognize_static_boolean_body(_generic(text))


def test_extra_region_operation_and_unregistered_same_name_refuse():
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr

    op = _generic(_program("not"))
    scalar = tuple(op.regions[0].blocks[0].ops)[1]
    scalar.__class__ = type("SameNameXor", (arith.XOrIOp,), {})
    with pytest.raises(InvalidLinalgPattern, match="unregistered"):
        recognize_static_boolean_body(op)

    extra = _program("not").replace(
        '"linalg.yield"(%value)',
        '%extra = "arith.xori"(%value, %constant) : (i1, i1) -> i1\n        "linalg.yield"(%value)',
    )
    with pytest.raises(InvalidLinalgPattern, match="closed forms"):
        recognize_static_boolean_body(_generic(extra))

    op = _generic(_program("cast"))
    op.attributes["unknown"] = IntegerAttr(1, 64)
    with pytest.raises(InvalidLinalgPattern, match="unrecognized linalg.generic attribute"):
        recognize_static_boolean_body(op)

    op = _generic(_program("mul"))
    scalar = tuple(op.regions[0].blocks[0].ops)[0]
    scalar.attributes["unknown"] = IntegerAttr(1, 64)
    with pytest.raises(InvalidLinalgPattern, match="unknown attribute"):
        recognize_static_boolean_body(op)

    op = _generic(_program("not"))
    yld = tuple(op.regions[0].blocks[0].ops)[-1]
    yld.properties["unknown"] = IntegerAttr(1, 64)
    with pytest.raises(InvalidLinalgPattern, match="yield only"):
        recognize_static_boolean_body(op)


def test_source_screen_binds_every_ordinal_and_both_hashes(tmp_path):
    path = tmp_path / "neutral.mlir"
    text = _program("not")
    path.write_text(text)
    ordinal = next(i for i, op in enumerate(mq.walk(mq.parse(text))) if mq.op_name(op) == "linalg.generic")
    proof = screen_static_boolean_source(path, (ordinal,))
    assert proof.raw_sha256 == sha256(text.encode()).hexdigest()
    assert len(proof.normalized_sha256) == 64
    assert proof.ordinals == ((ordinal, recognize_static_boolean_body(_generic(text))),)
    for invalid in ((ordinal, ordinal), (10000,), (True,)):
        with pytest.raises(InvalidLinalgPattern):
            screen_static_boolean_source(path, invalid)
    path.write_text(_program("cast"))
    assert screen_static_boolean_source(path, (ordinal,)).raw_sha256 != proof.raw_sha256
    path.write_text(_program("not").replace("value = true", "value = false"))
    with pytest.raises(InvalidLinalgPattern, match="true i1 constant"):
        screen_static_boolean_source(path, (ordinal,))


def test_source_screen_rejects_a_mutated_second_occurrence(tmp_path):
    first = _program("not")
    second = _program("not").replace("@forward", "@other").replace("value = true", "value = false")
    path = tmp_path / "paired.mlir"
    path.write_text(first[:-1] + second.removeprefix("builtin.module {")[1:])
    ordinals = tuple(i for i, op in enumerate(mq.walk(mq.parse(path))) if mq.op_name(op) == "linalg.generic")
    assert len(ordinals) == 2
    assert len(screen_static_boolean_source(path, ordinals[:1]).ordinals) == 1
    with pytest.raises(InvalidLinalgPattern, match="true i1 constant"):
        screen_static_boolean_source(path, ordinals)
