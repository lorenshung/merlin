"""Exact host selectors may not turn a schedule into family-wide support."""

import pytest

from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities


def _pointwise_source(scalar: str = "arith.addf", *, flags: str = "none", shape: str = "3x5"):
    from merlin.common import mlir_query as mq

    text = f'''builtin.module {{
  func.func @f(%x: tensor<{shape}xf32>, %y: tensor<{shape}xf32>,
               %init: tensor<{shape}xf32>) -> tensor<{shape}xf32> {{
    %out = "linalg.generic"(%x, %y, %init) <{{indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [
      #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: 2, 1>}}> ({{
      ^bb0(%a: f32, %b: f32, %old: f32):
        %v = "{scalar}"(%a, %b) <{{fastmath = #arith.fastmath<{flags}>}}> : (f32, f32) -> f32
        "linalg.yield"(%v) : (f32) -> ()
    }}) : (tensor<{shape}xf32>, tensor<{shape}xf32>, tensor<{shape}xf32>) -> tensor<{shape}xf32>
    func.return %out : tensor<{shape}xf32>
  }}
}}'''
    return next(mq.walk(mq.parse(text), "linalg.generic"))


def _pointwise_selection():
    selected = _selection()
    selected["host"]["capability_spec"]["operations"] = [
        {
            "id": "typed_add",
            "ops": ["linalg.generic"],
            "families": ["elementwise_map"],
            "placement": "host",
            "signature": {
                "ordered_operand_dtypes": ["f32", "f32", "f32"],
                "ordered_result_dtypes": ["f32"],
                "ranks": [2],
            },
            "source_body": {"schema": "merlin.static_pointwise_source_body.v1", "operation": "arith.addf"},
        }
    ]
    return selected


def _selection():
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
                        "id": "named_matmul",
                        "ops": ["linalg.matmul"],
                        "families": ["contraction"],
                        "placement": "host",
                        "signature": {"operand_dtypes": ["int8"]},
                    }
                ],
                "evidence": {"scope": "declaration screen only"},
            },
        }
    }


def test_exact_host_operation_selector_does_not_expand_to_generic_contraction():
    selected = _selection()
    signature = {"family": "contraction", "operand_dtype": "int8"}
    generic = admit_host_operation(selected, {"mlir_operation": "linalg.generic"}, signature)
    assert generic["status"] == "unsupported"
    assert generic["profiles"][0]["decisions"] == []
    named = admit_host_operation(selected, {"mlir_operation": "linalg.matmul"}, signature)
    assert named["status"] == "admitted"
    assert named["qualification"].startswith("selected declaration screen")


def test_family_only_host_selector_remains_available_when_explicitly_declared():
    selected = _selection()
    selected["host"]["capability_spec"]["operations"][0].pop("ops")
    generic = admit_host_operation(
        selected, {"mlir_operation": "linalg.generic"}, {"family": "contraction", "operand_dtype": "int8"}
    )
    assert generic["status"] == "admitted"


def test_unreviewed_operation_inside_reviewed_host_document_remains_unknown():
    selected = _selection()
    declaration = selected["host"]["capability_spec"]["operations"][0]
    declaration["status"] = "unreviewed"
    named = admit_host_operation(
        selected, {"mlir_operation": "linalg.matmul"}, {"family": "contraction", "operand_dtype": "int8"}
    )
    assert named["status"] == "unknown"
    assert named["review_status"] == "unreviewed"
    assert named["reviewed"] is False
    declaration["status"] = "accepted"
    with pytest.raises(ValueError, match="invalid review status"):
        validate_host_capabilities(selected["host"]["capability_spec"])


def test_pointwise_declaration_needs_every_exact_parsed_source_occurrence():
    from merlin.targetgen.operation_accounting import admit_operation_row

    selected = _pointwise_selection()
    row = {"mlir_operation": "linalg.generic", "count": 2}
    observed = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": ["f32", "f32", "f32"],
        "ordered_result_dtypes": ["f32"],
        "rank": 2,
    }
    assert admit_host_operation(selected, row, observed)["status"] == "unknown"
    assert (
        admit_operation_row(
            {**row, "operation": "linalg.generic", "disposition": "unclassified"},
            software_spec=None,
            capability_contract=None,
            host_capabilities=selected,
            observed=observed,
        )["host_admission"]["status"]
        == "unknown"
    )
    good = (_pointwise_source(), _pointwise_source())
    result = admit_host_operation(selected, row, observed, source_operations=good)
    assert result["status"] == "admitted"
    assert result["source_body_proof"]["declaration"] == "typed_add"
    assert len(result["source_body_proof"]["patterns"]) == 2
    assert (
        admit_host_operation(selected, row, observed, source_operations=(good[0], _pointwise_source("arith.subf")))[
            "status"
        ]
        == "unsupported"
    )
    assert (
        admit_host_operation(selected, row, observed, source_operations=(good[0], _pointwise_source(flags="reassoc")))[
            "status"
        ]
        == "unsupported"
    )
    assert (
        admit_host_operation(selected, row, {**observed, "rank": 3}, source_operations=good)["status"] == "unsupported"
    )
    assert admit_host_operation(selected, row, observed, source_operations=(good[0],))["status"] == "unsupported"
    assert (
        admit_host_operation(selected, row, observed, source_operations=(good[0], good[0]))["status"] == "unsupported"
    )


def test_pointwise_source_body_schema_is_closed_and_never_nested_silently():
    selected = _pointwise_selection()
    declaration = selected["host"]["capability_spec"]["operations"][0]
    validate_host_capabilities(selected["host"]["capability_spec"])
    for invalid in (
        {**declaration["source_body"], "unsupported": True},
        {"schema": "merlin.static_pointwise_source_body.v1", "operation": "arith.cmpf"},
        {"schema": "merlin.static_pointwise_source_body.v1", "operation": "arith.addf", "predicate": "oeq"},
    ):
        declaration["source_body"] = invalid
        with pytest.raises(ValueError, match="source_body"):
            validate_host_capabilities(selected["host"]["capability_spec"])
    declaration.pop("source_body")
    declaration["signature"]["source_body"] = {"schema": "merlin.static_pointwise_source_body.v1"}
    with pytest.raises(ValueError, match="source_body"):
        validate_host_capabilities(selected["host"]["capability_spec"])
