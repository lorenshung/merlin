"""Neutral prepared-source witnesses; no host placement or linked-value claim."""

# ruff: noqa: E501 - MLIR text fixtures retain the producer's single-line operation form.

from pathlib import Path

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_integer_reductions import (
    InvalidLinalgPattern,
    recognize_static_integer_reduction,
    screen_static_integer_reduction_source,
)


def _sum_source(*, boolean: bool) -> str:
    source_type = "i1" if boolean else "i64"
    cast = (
        ""
        if not boolean
        else """
    %empty = \"tensor.empty\"() : () -> tensor<2x3xi64>
    %cast = \"linalg.generic\"(%arg0, %empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%b: i1, %unused: i64):
      %w = \"arith.extui\"(%b) : (i1) -> i64
      \"linalg.yield\"(%w) : (i64) -> ()
    }) : (tensor<2x3xi1>, tensor<2x3xi64>) -> tensor<2x3xi64>
"""
    )
    source = "%cast" if boolean else "%arg0"
    return f"""builtin.module {{
  func.func @forward(%arg0: tensor<2x3x{source_type}>) -> tensor<2xi64> {{
{cast}
    %zero = \"arith.constant\"() <{{value = 0 : i64}}> : () -> i64
    %init = \"tensor.splat\"(%zero) : (i64) -> tensor<2xi64>
    %sum = \"linalg.reduce\"({source}, %init) <{{dimensions = array<i64: 1>}}> ({{
    ^bb1(%elem: i64, %acc: i64):
      %added = \"arith.addi\"(%elem, %acc) <{{overflowFlags = #arith.overflow<none>}}> : (i64, i64) -> i64
      \"linalg.yield\"(%added) : (i64) -> ()
    }}) : (tensor<2x3xi64>, tensor<2xi64>) -> tensor<2xi64>
    func.return %sum : tensor<2xi64>
  }}
}}
"""


def _scan_source(*, boolean: bool) -> str:
    source_type = "i1" if boolean else "i64"
    cast = '      %cast = "arith.extui"(%value) : (i1) -> i64\n' if boolean else ""
    operand = "%cast" if boolean else "%value"
    return f"""builtin.module {{
  func.func @forward(%arg0: tensor<2x3x{source_type}>) -> tensor<2x3xi64> {{
    %zero = \"arith.constant\"() <{{value = 0 : i64}}> : () -> i64
    %init = \"tensor.splat\"(%zero) : (i64) -> tensor<2x3xi64>
    %scan = \"linalg.generic\"(%arg0, %init) <{{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d2)>, affine_map<(d0, d1, d2) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>, #linalg.iterator_type<reduction>], operandSegmentSizes = array<i32: 1, 1>}}> ({{
    ^bb0(%value: {source_type}, %acc: i64):
      %i = \"linalg.index\"() <{{dim = 1 : i64}}> : () -> index
      %j = \"linalg.index\"() <{{dim = 2 : i64}}> : () -> index
      %mask = \"arith.cmpi\"(%j, %i) <{{predicate = 7 : i64}}> : (index, index) -> i1
{cast}      %selected = \"arith.select\"(%mask, {operand}, %zero) : (i1, i64, i64) -> i64
      %added = \"arith.addi\"(%acc, %selected) <{{overflowFlags = #arith.overflow<none>}}> : (i64, i64) -> i64
      \"linalg.yield\"(%added) : (i64) -> ()
    }}) : (tensor<2x3x{source_type}>, tensor<2x3xi64>) -> tensor<2x3xi64>
    func.return %scan : tensor<2x3xi64>
  }}
}}
"""


def _float_scan_source() -> str:
    """The structurally valid floating analogue must not enter the i64 witness."""
    text = _scan_source(boolean=False).replace("tensor<2x3xi64>", "tensor<2x3xf32>")
    text = text.replace("value = 0 : i64", "value = 0.000000e+00 : f32")
    text = text.replace(
        '"arith.constant"() <{value = 0.000000e+00 : f32}> : () -> i64',
        '"arith.constant"() <{value = 0.000000e+00 : f32}> : () -> f32',
    )
    text = text.replace('"tensor.splat"(%zero) : (i64)', '"tensor.splat"(%zero) : (f32)')
    text = text.replace("%value: i64, %acc: i64", "%value: f32, %acc: f32")
    text = text.replace(
        '"arith.select"(%mask, %value, %zero) : (i1, i64, i64) -> i64',
        '"arith.select"(%mask, %value, %zero) : (i1, f32, f32) -> f32',
    )
    text = text.replace(
        '"arith.addi"(%acc, %selected) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64',
        '"arith.addf"(%acc, %selected) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32',
    )
    return text.replace('"linalg.yield"(%added) : (i64) -> ()', '"linalg.yield"(%added) : (f32) -> ()')


def _selected(text: str, name: str):
    module = mq.parse(text)
    module.verify()
    return next(op for op in mq.walk(module) if op.name == name)


def _host_case(op, frontend: str, kind: str):
    from xdsl.dialects.builtin import StringAttr

    from merlin.targetgen.application_inventory import operation_structure

    op.attributes["prov.aten"] = StringAttr(frontend)
    structure = operation_structure(op)
    inputs, outputs = structure["ordered_operand_types"], structure["ordered_result_types"]
    row = {"mlir_operation": op.name, "frontend_op": frontend, "count": 1, **structure}
    observed = {
        "family": "reduction",
        "ordered_operand_dtypes": [item["dtype"] for item in inputs],
        "ordered_result_dtypes": [item["dtype"] for item in outputs],
        "rank": len(outputs[0]["shape"]),
    }
    body = {"schema": "merlin.static_integer_reduction_source_body.v1", "operation": kind}
    declaration = {
        "id": "neutral_integer_reduction",
        "ops": [frontend],
        "families": ["reduction"],
        "placement": "host",
        "signature": {
            "family": "reduction",
            "ordered_operand_dtypes": observed["ordered_operand_dtypes"],
            "ordered_result_dtypes": observed["ordered_result_dtypes"],
        },
        "source_body": body,
    }
    selected = {
        "neutral": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [declaration],
                "evidence": {"scope": "neutral structural test, not numerical qualification"},
            },
        }
    }
    context = {
        "selected_index_observation": {
            "schema": "merlin.selected-index-lowering.v1",
            "compiler_requested": "neutral-clang",
            "compiler_resolved": "/neutral/clang",
            "compiler_sha256": "c" * 64,
            "cross_flags": ["--target=neutral"],
            "data_layout": "e-p:64:64",
            "index_bits": 64,
            "scope": "neutral selected-width premise only",
        }
    }
    return selected, row, observed, context


@pytest.mark.parametrize("kind,boolean", [("sum", False), ("sum", True), ("cumsum", False), ("cumsum", True)])
def test_integer_reduction_host_screen_needs_exact_body_and_selected_index(kind: str, boolean: bool):
    from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities
    from merlin.targetgen.operation_accounting import admit_operation_row

    source = _sum_source(boolean=boolean) if kind == "sum" else _scan_source(boolean=boolean)
    op = _selected(source, "linalg.reduce" if kind == "sum" else "linalg.generic")
    frontend = "aten.sum.dim_IntList" if kind == "sum" else "aten.cumsum.default"
    selected, row, observed, context = _host_case(op, frontend, kind)
    validate_host_capabilities(selected["neutral"]["capability_spec"])
    assert admit_host_operation(selected, row, observed, source_operations=(op,))["status"] == "unknown"
    admitted = admit_operation_row(
        {**row, "operation": op.name, "disposition": "unclassified"},
        software_spec=None,
        capability_contract=None,
        host_capabilities=selected,
        observed=observed,
        source_operations=(op,),
        source_context=context,
    )["host_admission"]
    assert admitted["status"] == "admitted"
    assert admitted["source_body_proof"]["patterns"][0]["operation"] == kind
    assert admitted["source_body_proof"]["patterns"][0]["input_type"] == ("i1" if boolean else "i64")
    assert admitted["source_body_proof"]["selected_index_observation"] == context["selected_index_observation"]
    wrong = {"selected_index_observation": {**context["selected_index_observation"], "index_bits": 32}}
    assert (
        admit_host_operation(selected, row, observed, source_operations=(op,), source_context=wrong)["status"]
        == "unknown"
    )
    assert (
        admit_host_operation(selected, {**row, "count": 2}, observed, source_operations=(op,), source_context=context)[
            "status"
        ]
        == "unsupported"
    )
    assert (
        admit_host_operation(
            selected, row, {**observed, "rank": observed["rank"] + 1}, source_operations=(op,), source_context=context
        )["status"]
        == "unsupported"
    )
    assert (
        admit_host_operation(
            selected, {**row, "ordered_operand_types": []}, observed, source_operations=(op,), source_context=context
        )["status"]
        == "unsupported"
    )


def test_integer_reduction_host_screen_rejects_mutated_ssa_and_draft_schema():
    from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities

    source = _sum_source(boolean=True).replace("(%elem, %acc)", "(%acc, %acc)")
    op = _selected(source, "linalg.reduce")
    selected, row, observed, context = _host_case(op, "aten.sum.dim_IntList", "sum")
    assert (
        admit_host_operation(selected, row, observed, source_operations=(op,), source_context=context)["status"]
        == "unsupported"
    )
    declaration = selected["neutral"]["capability_spec"]["operations"][0]
    declaration["status"] = "unreviewed"
    validate_host_capabilities(selected["neutral"]["capability_spec"])
    assert (
        admit_host_operation(selected, row, observed, source_operations=(op,), source_context=context)["status"]
        != "admitted"
    )
    for body in (
        {**declaration["source_body"], "extra": "unsafe"},
        {"schema": declaration["source_body"]["schema"], "operation": "bucketize"},
    ):
        declaration["source_body"] = body
        with pytest.raises(ValueError, match="source_body"):
            validate_host_capabilities(selected["neutral"]["capability_spec"])


@pytest.mark.parametrize(
    "kind,old,new",
    [
        ("sum", "array<i64: 1>", "array<i64: 0>"),
        ("sum", "#arith.overflow<none>", "#arith.overflow<nsw>"),
        ("cumsum", "predicate = 7 : i64", "predicate = 6 : i64"),
        ("cumsum", "#arith.overflow<none>", "#arith.overflow<nuw>"),
    ],
)
def test_integer_reduction_host_screen_refuses_changed_axes_or_arithmetic(kind, old, new):
    from merlin.targetgen.host_capabilities import admit_host_operation

    source = _sum_source(boolean=True) if kind == "sum" else _scan_source(boolean=True)
    name = "linalg.reduce" if kind == "sum" else "linalg.generic"
    op = next(mq.walk(mq.parse(source.replace(old, new)), name))
    frontend = "aten.sum.dim_IntList" if kind == "sum" else "aten.cumsum.default"
    selected, row, observed, context = _host_case(op, frontend, kind)
    assert (
        admit_host_operation(selected, row, observed, source_operations=(op,), source_context=context)["status"]
        == "unsupported"
    )


def test_integer_sum_host_screen_accepts_independent_scalar_reduction_shape():
    from merlin.targetgen.host_capabilities import admit_host_operation

    source = (
        _sum_source(boolean=True).replace("tensor<2xi64>", "tensor<i64>").replace("array<i64: 1>", "array<i64: 0, 1>")
    )
    op = _selected(source, "linalg.reduce")
    selected, row, observed, context = _host_case(op, "aten.sum.dim_IntList", "sum")
    assert observed["rank"] == 0
    assert (
        admit_host_operation(selected, row, observed, source_operations=(op,), source_context=context)["status"]
        == "admitted"
    )


@pytest.mark.parametrize("boolean", [False, True])
def test_static_sum_checks_zero_seed_cast_axes_and_add(boolean: bool):
    pattern = recognize_static_integer_reduction(
        _selected(_sum_source(boolean=boolean), "linalg.reduce"), index_bits=64
    )
    assert (pattern.operation, pattern.input_shape, pattern.output_shape, pattern.axis, pattern.input_type) == (
        "sum",
        (2, 3),
        (2,),
        (1,),
        "i1" if boolean else "i64",
    )


@pytest.mark.parametrize("boolean", [False, True])
def test_static_scan_checks_masked_prefix_source(boolean: bool):
    pattern = recognize_static_integer_reduction(
        _selected(_scan_source(boolean=boolean), "linalg.generic"), index_bits=64
    )
    assert (pattern.operation, pattern.input_shape, pattern.output_shape, pattern.axis, pattern.input_type) == (
        "cumsum",
        (2, 3),
        (2, 3),
        (1,),
        "i1" if boolean else "i64",
    )


@pytest.mark.parametrize(
    "old,new",
    [
        ("value = 0 : i64", "value = 1 : i64"),
        ("arith.addi", "arith.subi"),
        ("#arith.overflow<none>", "#arith.overflow<nsw>"),
        ("(%elem, %acc)", "(%acc, %acc)"),
    ],
)
def test_sum_mutations_refuse(old: str, new: str):
    text = _sum_source(boolean=True).replace(old, new)
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_integer_reduction(_selected(text, "linalg.reduce"), index_bits=64)


@pytest.mark.parametrize(
    "old,new",
    [
        ("value = 0 : i64", "value = 1 : i64"),
        ("predicate = 7 : i64", "predicate = 6 : i64"),
        ("#linalg.iterator_type<reduction>", "#linalg.iterator_type<parallel>"),
        ("#arith.overflow<none>", "#arith.overflow<nsw>"),
        ("arith.extui", "arith.extsi"),
        ("(%acc, %selected)", "(%acc, %acc)"),
        ("(%mask, %cast, %zero)", "(%mask, %zero, %zero)"),
    ],
)
def test_scan_mutations_refuse(old: str, new: str):
    text = _scan_source(boolean=True).replace(old, new)
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_integer_reduction(_selected(text, "linalg.generic"), index_bits=64)


def test_source_binds_exact_bytes_and_normalized_ordinals(tmp_path: Path):
    source = tmp_path / "neutral.mlir"
    source.write_text(_sum_source(boolean=True), encoding="utf-8")
    module = mq.parse(source)
    ordinal = next(index for index, op in enumerate(mq.walk(module)) if op.name == "linalg.reduce")
    proof = screen_static_integer_reduction_source(source, (ordinal,), index_bits=64)
    assert len(proof.raw_sha256) == len(proof.normalized_sha256) == 64
    assert proof.ordinals[0][0] == ordinal
    with pytest.raises(InvalidLinalgPattern):
        screen_static_integer_reduction_source(source, (ordinal, ordinal), index_bits=64)


@pytest.mark.parametrize("index_bits", [None, True, 2, 129])
def test_index_width_is_an_explicit_bounded_premise(index_bits):
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_integer_reduction(
            _selected(_scan_source(boolean=True), "linalg.generic"), index_bits=index_bits
        )


@pytest.mark.parametrize("old,new", [("(d0, d2)", "(d0, d1)"), ("arith.addi", "arith.addf")])
def test_malformed_module_refuses_at_full_source_boundary(tmp_path: Path, old: str, new: str):
    source = tmp_path / "invalid.mlir"
    source.write_text(_scan_source(boolean=True).replace(old, new), encoding="utf-8")
    module = mq.parse(source)
    ordinal = next(index for index, op in enumerate(mq.walk(module)) if op.name == "linalg.generic")
    with pytest.raises(InvalidLinalgPattern):
        screen_static_integer_reduction_source(source, (ordinal,), index_bits=64)


def test_neutral_tail_shape_is_structural_not_model_selected():
    sum_text = _sum_source(boolean=True).replace("2x3", "1x7").replace("tensor<2xi64>", "tensor<1xi64>")
    scan_text = _scan_source(boolean=True).replace("2x3", "1x7")
    assert recognize_static_integer_reduction(_selected(sum_text, "linalg.reduce"), index_bits=64).input_shape == (1, 7)
    assert recognize_static_integer_reduction(_selected(scan_text, "linalg.generic"), index_bits=64).input_shape == (
        1,
        7,
    )


def test_neutral_first_axis_uses_its_own_static_map_and_index():
    sum_text = (
        _sum_source(boolean=True).replace("tensor<2xi64>", "tensor<3xi64>").replace("array<i64: 1>", "array<i64: 0>")
    )
    scan_text = _scan_source(boolean=True).replace("(d0, d2)", "(d2, d1)").replace("dim = 1 : i64", "dim = 0 : i64")
    assert recognize_static_integer_reduction(_selected(sum_text, "linalg.reduce"), index_bits=64).axis == (0,)
    assert recognize_static_integer_reduction(_selected(scan_text, "linalg.generic"), index_bits=64).axis == (0,)


def test_valid_f32_cumsum_remains_outside_integer_source_scope():
    op = _selected(_float_scan_source(), "linalg.generic")
    with pytest.raises(InvalidLinalgPattern):
        recognize_static_integer_reduction(op, index_bits=64)


def test_all_axes_sum_has_a_real_scalar_tensor_zero_init():
    text = (
        _sum_source(boolean=True).replace("tensor<2xi64>", "tensor<i64>").replace("array<i64: 1>", "array<i64: 0, 1>")
    )
    pattern = recognize_static_integer_reduction(_selected(text, "linalg.reduce"), index_bits=64)
    assert (pattern.input_shape, pattern.output_shape, pattern.axis, pattern.input_type) == (
        (2, 3),
        (),
        (0, 1),
        "i1",
    )


def test_i64_pointwise_input_remains_an_i64_sum_obligation():
    text = _sum_source(boolean=False)
    producer = """
    %empty = "tensor.empty"() : () -> tensor<2x3xi64>
    %computed = "linalg.generic"(%arg0, %empty) <{indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>], operandSegmentSizes = array<i32: 1, 1>}> ({
    ^bb0(%x: i64, %unused: i64):
      %twice = "arith.addi"(%x, %x) <{overflowFlags = #arith.overflow<none>}> : (i64, i64) -> i64
      "linalg.yield"(%twice) : (i64) -> ()
    }) : (tensor<2x3xi64>, tensor<2x3xi64>) -> tensor<2x3xi64>
"""
    text = text.replace('    %zero = "arith.constant"', producer + '    %zero = "arith.constant"')
    text = text.replace('"linalg.reduce"(%arg0, %init)', '"linalg.reduce"(%computed, %init)')
    pattern = recognize_static_integer_reduction(_selected(text, "linalg.reduce"), index_bits=64)
    assert pattern.input_type == "i64"


def test_one_boolean_input_with_a_different_i64_body_does_not_claim_extui():
    text = _sum_source(boolean=True).replace(
        '%w = "arith.extui"(%b) : (i1) -> i64',
        '%one = "arith.constant"() <{value = 1 : i64}> : () -> i64\n'
        '      %none = "arith.constant"() <{value = 0 : i64}> : () -> i64\n'
        '      %w = "arith.select"(%b, %one, %none) : (i1, i64, i64) -> i64',
    )
    pattern = recognize_static_integer_reduction(_selected(text, "linalg.reduce"), index_bits=64)
    assert pattern.input_type == "i64"


def test_sign_extension_never_claims_boolean_count():
    text = _sum_source(boolean=True).replace("arith.extui", "arith.extsi")
    pattern = recognize_static_integer_reduction(_selected(text, "linalg.reduce"), index_bits=64)
    assert pattern.input_type == "i64"
