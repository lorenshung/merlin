"""Prepared maximum placement needs every typed source root and selected index width."""

from __future__ import annotations

import copy

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.linalg_f32_maximum_patterns import STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA
from merlin.targetgen.application_inventory import operation_structure
from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities


def _source(*, body: str = "arith.maximumf", seed: str = "0xff800000", flags: str = "none"):
    text = f'''builtin.module {{
  func.func @neutral(%x: tensor<2x3x7xf32>) -> tensor<2x3xf32> {{
    %seed = arith.constant {seed} : f32
    %init = tensor.splat %seed : tensor<2x3xf32>
    %out = "linalg.reduce"(%x, %init) <{{dimensions = array<i64: 2>}}> ({{
      ^bb0(%value: f32, %acc: f32):
        %v = "{body}"(%value, %acc) <{{fastmath = #arith.fastmath<{flags}>}}> : (f32, f32) -> f32
        "linalg.yield"(%v) : (f32) -> ()
    }}) {{prov.aten = "aten.amax.default"}}
      : (tensor<2x3x7xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    func.return %out : tensor<2x3xf32>
  }}
}}'''
    return next(mq.walk(mq.parse(text), "linalg.reduce"))


def _selected():
    return {
        "neutral": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [
                    {
                        "id": "neutral_prepared_maximum",
                        "ops": ["aten.amax.default"],
                        "placement": "host",
                        "signature": {
                            "family": "reduction",
                            "ordered_operand_dtypes": ["f32", "f32"],
                            "ordered_result_dtypes": ["f32"],
                        },
                        "source_body": {
                            "schema": STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA,
                            "operation": "arith.maximumf",
                        },
                        "numerical_contract": {"status": "unreviewed"},
                    }
                ],
                "evidence": {"scope": "neutral declaration screen, not numerical qualification"},
            },
        }
    }


def _screen(
    ops: tuple | None,
    *,
    count: int = 1,
    rank: int = 2,
    width: int | None = 64,
    index_mutation: dict | None = None,
    context_mutation: dict | None = None,
):
    reference = _source()
    structure = operation_structure(reference)
    observed = {
        "family": "reduction",
        "ordered_operand_dtypes": ["f32", "f32"],
        "ordered_result_dtypes": ["f32"],
        "rank": rank,
    }
    context = (
        {
            "selected_index_observation": {
                "schema": "merlin.selected-index-lowering.v1",
                "compiler_requested": "neutral-clang",
                "compiler_resolved": "/neutral/clang",
                "compiler_sha256": "c" * 64,
                "cross_flags": ["--target=neutral"],
                "data_layout": f"e-p:{width}:{width}",
                "index_bits": width,
                "scope": "neutral selected compiler observation",
            }
        }
        if width is not None
        else None
    )
    if context is not None:
        context["selected_index_observation"].update(index_mutation or {})
        context.update(context_mutation or {})
    return admit_host_operation(
        _selected(),
        {
            "frontend_op": "aten.amax.default",
            "mlir_operation": "linalg.reduce",
            "count": count,
            **structure,
        },
        observed,
        source_operations=ops,
        source_context=context,
    )


def test_closed_maximum_declaration_requires_all_parsed_roots_and_selected_width():
    validate_host_capabilities(_selected()["neutral"]["capability_spec"])
    one, two = _source(), _source()
    result = _screen((one, two), count=2)
    assert result["status"] == "admitted"
    proof = result["source_body_proof"]
    assert proof["schema"] == STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA
    assert proof["declaration"] == "neutral_prepared_maximum"
    assert proof["operation"] == "arith.maximumf"
    assert len(proof["patterns"]) == 2
    assert proof["patterns"][0] == {
        "axis": 2,
        "input_shape": [2, 3, 7],
        "output_shape": [2, 3],
        "ordered_types": ["f32", "f32", "f32"],
        "index_bits_premise": 64,
    }
    assert proof["profile"] == "neutral" and proof["capability_spec_sha256"] == "b" * 64
    assert _screen(None)["status"] == "unknown"
    assert _screen((one,), width=None)["status"] == "unknown"
    assert _screen((one,), rank=3)["status"] == "unsupported"
    assert _screen((one,), count=2)["status"] == "unsupported"
    assert _screen((one, one), count=2)["status"] == "unsupported"


@pytest.mark.parametrize(
    ("index_mutation", "context_mutation"),
    [
        ({"index_bits": 32}, None),
        ({"compiler_sha256": "not-a-digest"}, None),
        ({"data_layout": "e-p:64:64:64:32"}, None),
        (None, {"unowned_context": True}),
    ],
)
def test_maximum_rejects_malformed_selected_index_observation(index_mutation, context_mutation):
    assert (
        _screen((_source(),), index_mutation=index_mutation, context_mutation=context_mutation)["status"] == "unknown"
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"source_body": {"schema": STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA, "operation": "arith.minimumf"}},
        {"source_body": {"schema": STATIC_F32_MAXIMUM_SOURCE_BODY_SCHEMA, "operation": "arith.maximumf", "axis": 2}},
        {"ops": ["linalg.reduce"]},
        {"families": ["reduction"]},
        {"signature": {"family": "reduction", "ordered_operand_dtypes": ["f32"], "ordered_result_dtypes": ["f32"]}},
        {"numerical_contract": {"status": "reviewed"}},
    ],
)
def test_maximum_declaration_cannot_widen_or_claim_numerics(changes):
    document = copy.deepcopy(_selected()["neutral"]["capability_spec"])
    document["operations"][0].update(changes)
    with pytest.raises(ValueError, match="maximum|source_body"):
        validate_host_capabilities(document)


@pytest.mark.parametrize(
    "changed",
    [
        {"body": "arith.minimumf"},
        {"seed": "0x7f800000"},
        {"flags": "fast"},
    ],
)
def test_maximum_source_mutation_refuses_placement(changed):
    assert _screen((_source(**changed),))["status"] == "unsupported"
