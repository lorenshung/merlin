"""Boolean source declarations require complete typed source, not a name match."""

from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities


def _source(form: str, *, false_constant: bool = False):
    body = {
        "not": """%one = "arith.constant"() <{value = true}> : () -> i1
        %v = "arith.xori"(%a, %one) : (i1, i1) -> i1""",
        "cast": """%zero = "arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32
        %v = "arith.cmpf"(%a, %zero)
          <{predicate = 13 : i64, fastmath = #arith.fastmath<none>}> : (f32, f32) -> i1""",
        "mul": """%v = "arith.muli"(%a, %b)
          <{overflowFlags = #arith.overflow<none>}> : (i1, i1) -> i1""",
    }[form]
    if false_constant:
        body = body.replace("value = true", "value = false")
    inputs = {
        "not": ("%x: tensor<2x3xi1>",),
        "cast": ("%x: tensor<2x3xf32>",),
        "mul": ("%x: tensor<2x1xi1>", "%y: tensor<2x3xi1>"),
    }[form]
    args = {
        "not": "%a: i1, %init_value: i1",
        "cast": "%a: f32, %init_value: i1",
        "mul": "%a: i1, %b: i1, %init_value: i1",
    }[form]
    operands = "%x, %y, %init" if form == "mul" else "%x, %init"
    tensor_types = {
        "not": "tensor<2x3xi1>, tensor<2x3xi1>",
        "cast": "tensor<2x3xf32>, tensor<2x3xi1>",
        "mul": "tensor<2x1xi1>, tensor<2x3xi1>, tensor<2x3xi1>",
    }[form]
    maps = (
        "affine_map<(d0, d1) -> (d0, 0)>, affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>"
        if form == "mul"
        else "affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>"
    )
    text = f"""builtin.module {{
  func.func @forward({", ".join((*inputs, "%init: tensor<2x3xi1>"))}) -> tensor<2x3xi1> {{
    %out = "linalg.generic"({operands}) <{{indexing_maps = [{maps}],
      iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: {len(inputs)}, 1>}}> ({{
      ^bb0({args}):
        {body}
        "linalg.yield"(%v) : (i1) -> ()
    }}) : ({tensor_types}) -> tensor<2x3xi1>
    func.return %out : tensor<2x3xi1>
  }}
}}"""
    return next(mq.walk(mq.parse(text), "linalg.generic"))


def _selection(operation: str, operand_types: list[str]):
    return {
        "host": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [
                    {
                        "id": "neutral_boolean_body",
                        "ops": ["linalg.generic"],
                        "placement": "host",
                        "signature": {
                            "ordered_operand_dtypes": operand_types,
                            "ordered_result_dtypes": ["i1"],
                            "ranks": [2],
                        },
                        "source_body": {"schema": "merlin.static_boolean_body.v1", "operation": operation},
                    }
                ],
                "evidence": {"scope": "synthetic declaration screen only"},
            },
        }
    }


@pytest.mark.parametrize(
    ("form", "operation", "types"),
    [
        ("not", "i1_not", ["i1", "i1"]),
        ("cast", "f32_nonzero_to_i1", ["f32", "i1"]),
        ("mul", "i1_mul_singleton_projected", ["i1", "i1", "i1"]),
    ],
)
def test_boolean_host_screen_needs_every_typed_source_occurrence(form, operation, types):
    selected = _selection(operation, types)
    observed = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": types,
        "ordered_result_dtypes": ["i1"],
        "rank": 2,
    }
    row = {"mlir_operation": "linalg.generic", "count": 2}
    assert admit_host_operation(selected, row, observed)["status"] == "unknown"
    good = (_source(form), _source(form))
    passed = admit_host_operation(selected, row, observed, source_operations=good)
    assert passed["status"] == "admitted" and passed["reviewed"] is True
    assert passed["source_body_proof"]["schema"] == "merlin.static_boolean_body.v1"
    assert len(passed["source_body_proof"]["patterns"]) == 2
    assert admit_host_operation(selected, row, observed, source_operations=good[:1])["status"] == "unsupported"
    assert (
        admit_host_operation(selected, row, observed, source_operations=(good[0], good[0]))["status"] == "unsupported"
    )
    assert (
        admit_host_operation(selected, row, {**observed, "rank": 3}, source_operations=good)["status"] == "unsupported"
    )
    selected["host"]["capability_spec"]["operations"][0]["source_body"]["operation"] = "i1_not"
    if operation != "i1_not":
        assert admit_host_operation(selected, row, observed, source_operations=good)["status"] == "unsupported"


def test_boolean_host_screen_refuses_a_bad_second_body_and_source_free_alias():
    selected = _selection("i1_not", ["i1", "i1"])
    row = {"mlir_operation": "linalg.generic", "count": 2}
    observed = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": ["i1", "i1"],
        "ordered_result_dtypes": ["i1"],
        "rank": 2,
    }
    mixed = (_source("not"), _source("not", false_constant=True))
    assert admit_host_operation(selected, row, observed, source_operations=mixed)["status"] == "unsupported"
    assert admit_host_operation(selected, row, observed, source_operations=None)["status"] == "unknown"
    selected["host"]["capability_spec"]["operations"][0]["source_body"] = {
        "schema": "merlin.static_pointwise_source_body.v1",
        "operation": "arith.xori",
    }
    assert admit_host_operation(selected, row, observed, source_operations=mixed)["status"] == "unsupported"


def test_boolean_source_declaration_has_closed_schema_and_no_nested_alias():
    selected = _selection("i1_not", ["i1", "i1"])
    row = selected["host"]["capability_spec"]["operations"][0]
    validate_host_capabilities(selected["host"]["capability_spec"])
    for invalid in (
        {**row["source_body"], "predicate": "une"},
        {**row["source_body"], "operation": "arith.xori"},
        {**row["source_body"], "schema": "merlin.unreviewed_body.v1"},
    ):
        row["source_body"] = invalid
        with pytest.raises(ValueError, match="source_body"):
            validate_host_capabilities(selected["host"]["capability_spec"])
    row.pop("source_body")
    row["signature"]["source_body"] = {"schema": "merlin.static_boolean_body.v1"}
    with pytest.raises(ValueError, match="source_body"):
        validate_host_capabilities(selected["host"]["capability_spec"])
