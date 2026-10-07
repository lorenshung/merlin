"""Exact host selectors may not turn a schedule into family-wide support."""

import pytest

from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities


def _pointwise_source(scalar: str = "arith.addf", *, flags: str = "none", shape: str = "3x5", projected: bool = False):
    from merlin.common import mlir_query as mq

    first_shape = "3x1" if projected else shape
    first_map = "(d0, 0)" if projected else "(d0, d1)"
    text = f'''builtin.module {{
  func.func @f(%x: tensor<{first_shape}xf32>, %y: tensor<{shape}xf32>,
               %init: tensor<{shape}xf32>) -> tensor<{shape}xf32> {{
    %out = "linalg.generic"(%x, %y, %init) <{{indexing_maps = [
      affine_map<(d0, d1) -> {first_map}>, affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [
      #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: 2, 1>}}> ({{
      ^bb0(%a: f32, %b: f32, %old: f32):
        %v = "{scalar}"(%a, %b) <{{fastmath = #arith.fastmath<{flags}>}}> : (f32, f32) -> f32
        "linalg.yield"(%v) : (f32) -> ()
    }}) : (tensor<{first_shape}xf32>, tensor<{shape}xf32>, tensor<{shape}xf32>) -> tensor<{shape}xf32>
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


@pytest.mark.parametrize(
    "frontend,kind,operands,results",
    [
        ("aten.sum.dim_IntList", "sum", ["i64", "i64"], ["i64"]),
        ("aten.cumsum.default", "cumsum", ["i1", "i64"], ["i64"]),
        ("aten.cumsum.default", "cumsum", ["i64", "i64"], ["i64"]),
        ("aten.min.dim", "i64_min_first_index", ["i64", "i64", "i64"], ["i64", "i64"]),
    ],
)
def test_integer_reduction_draft_schema_is_closed_without_admission(frontend, kind, operands, results):
    selected = _selection()
    document = selected["host"]["capability_spec"]
    document["status"] = "unreviewed"
    declaration = {
        "id": "draft_integer_reduction",
        "status": "unreviewed",
        "ops": [frontend],
        "families": ["reduction"],
        "placement": "host",
        "signature": {
            "family": "reduction",
            "ordered_operand_dtypes": operands,
            "ordered_result_dtypes": results,
        },
        "source_body": {"schema": "merlin.static_integer_reduction_source_body.v1", "operation": kind},
    }
    document["operations"] = [declaration]
    validate_host_capabilities(document)
    assert (
        admit_host_operation(
            selected,
            {"mlir_operation": "linalg.reduce" if kind == "sum" else "linalg.generic", "frontend_op": frontend},
            {"family": "reduction", "ordered_operand_dtypes": operands, "ordered_result_dtypes": results},
        )["status"]
        != "admitted"
    )
    for changed in (
        {**declaration, "ops": ["aten.other.default"]},
        {**declaration, "families": ["elementwise_map"]},
        {**declaration, "signature": {**declaration["signature"], "ordered_result_dtypes": ["f32"]}},
        {**declaration, "source_body": {**declaration["source_body"], "other": True}},
    ):
        document["operations"] = [changed]
        with pytest.raises(ValueError, match="source_body|integer-reduction"):
            validate_host_capabilities(document)


def test_projected_pointwise_needs_explicit_separate_schema_and_every_source_occurrence():
    selected = _pointwise_selection()
    declaration = selected["host"]["capability_spec"]["operations"][0]
    row = {"mlir_operation": "linalg.generic", "count": 2}
    observed = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": ["f32", "f32", "f32"],
        "ordered_result_dtypes": ["f32"],
        "rank": 2,
    }
    projected = (_pointwise_source(projected=True), _pointwise_source(projected=True))
    assert admit_host_operation(selected, row, observed, source_operations=projected)["status"] == "unsupported"
    declaration["source_body"] = {"schema": "merlin.static_projected_pointwise_body.v1", "operation": "arith.addf"}
    validate_host_capabilities(selected["host"]["capability_spec"])
    admitted = admit_host_operation(selected, row, observed, source_operations=projected)
    assert admitted["status"] == "admitted"  # synthetic reviewed declaration, never a policy edit
    assert admitted["source_body_proof"]["patterns"][0]["input_shapes"] == ((3, 1), (3, 5))
    assert admit_host_operation(selected, row, observed)["status"] == "unknown"
    assert admit_host_operation(selected, row, observed, source_operations=(projected[0],))["status"] == "unsupported"
    declaration["source_body"] = {"schema": "merlin.static_projected_pointwise_body.v1", "operation": "arith.cmpf"}
    with pytest.raises(ValueError, match="source_body"):
        validate_host_capabilities(selected["host"]["capability_spec"])
