"""Neutral host declaration screen for a closed prepared bucketize source body."""

from __future__ import annotations

import pytest

from merlin.common import mlir_query as mq
from merlin.frontends.bucketize_source import STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA
from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities


def _source(*, right: bool = False, dynamic: bool = False, boundaries: str = "0.0, 0.25, 0.25, 1.0"):
    boundary_arg = ", %boundaries: tensor<4xf32>" if dynamic else ""
    dense = f'dense<"{boundaries}">' if boundaries.startswith("0x") else f"dense<[{boundaries}]>"
    boundary_literal = (
        "" if dynamic else f'%boundaries = "arith.constant"() <{{value = {dense} : tensor<4xf32>}}> : () -> tensor<4xf32>'
    )
    text = f'''builtin.module {{
  func.func @neutral(%input: tensor<2x3xf32>{boundary_arg}) -> tensor<2x3xi64> {{
    {boundary_literal}
    %zero = "arith.constant"() <{{value = 0 : i64}}> : () -> i64
    %one = "arith.constant"() <{{value = 1 : i64}}> : () -> i64
    %init = "tensor.splat"(%zero) : (i64) -> tensor<2x3xi64>
    %out = "linalg.generic"(%input, %boundaries, %init)
      <{{indexing_maps = [affine_map<(d0, d1, d2) -> (d0, d1)>,
                          affine_map<(d0, d1, d2) -> (d2)>,
                          affine_map<(d0, d1, d2) -> (d0, d1)>],
        iterator_types = [#linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>,
                          #linalg.iterator_type<reduction>],
        operandSegmentSizes = array<i32: 2, 1>}}> ({{
      ^bb0(%x: f32, %b: f32, %acc: i64):
        %less = "arith.cmpf"(%b, %x)
          <{{predicate = {12 if right else 11} : i64, fastmath = #arith.fastmath<none>}}> : (f32, f32) -> i1
        %increment = "arith.select"(%less, %one, %zero) : (i1, i64, i64) -> i64
        %count = "arith.addi"(%acc, %increment)
          <{{overflowFlags = #arith.overflow<none>}}> : (i64, i64) -> i64
        "linalg.yield"(%count) : (i64) -> ()
      }}) {{prov.aten = "aten.bucketize.Tensor"}}
        : (tensor<2x3xf32>, tensor<4xf32>, tensor<2x3xi64>) -> tensor<2x3xi64>
    func.return %out : tensor<2x3xi64>
  }}
}}'''
    return next(mq.walk(mq.parse(text), "linalg.generic"))


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
                        "id": "neutral_literal_bucketize",
                        "ops": ["aten.bucketize.Tensor"],
                        "placement": "host",
                        "signature": {
                            "family": "reduction",
                            "ordered_operand_dtypes": ["f32", "f32", "i64"],
                            "ordered_result_dtypes": ["i64"],
                        },
                        "source_body": {
                            "schema": STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA,
                            "operation": "f32_literal_boundary_count_i64",
                        },
                        "numerical_contract": {"status": "unreviewed"},
                    }
                ],
                "evidence": {"scope": "synthetic declaration screen only"},
            },
        }
    }


def _screen(op, *, source=True, rank=2, index_bits=64):
    selected_index = {
        "schema": "merlin.selected-index-lowering.v1",
        "compiler_requested": "neutral-clang",
        "compiler_resolved": "/neutral/clang",
        "compiler_sha256": "c" * 64,
        "cross_flags": ["--target=neutral"],
        "data_layout": f"e-p:{index_bits}:{index_bits}",
        "index_bits": index_bits,
        "scope": "neutral selected compiler observation",
    }
    return admit_host_operation(
        _selected(),
        {"mlir_operation": "linalg.generic", "frontend_op": "aten.bucketize.Tensor", "count": 1},
        {
            "family": "reduction",
            "ordered_operand_dtypes": ["f32", "f32", "i64"],
            "ordered_result_dtypes": ["i64"],
            "rank": rank,
        },
        source_operations=(op,) if source else None,
        source_context={"selected_index_observation": selected_index} if index_bits else None,
    )


def test_literal_bucketize_requires_closed_typed_source_and_keeps_numerics_unreviewed():
    validate_host_capabilities(_selected()["neutral"]["capability_spec"])
    for right in (False, True):
        verdict = _screen(_source(right=right))
        assert verdict["status"] == "admitted"
        assert verdict["source_body_proof"]["schema"] == STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA
        assert verdict["source_body_proof"]["patterns"][0]["right"] is right
    assert _screen(_source(), source=False)["status"] == "unknown"
    assert _screen(_source(), index_bits=0)["status"] == "unknown"
    assert _screen(_source(), rank=3)["status"] == "unsupported"


def test_dynamic_unsorted_or_nan_boundaries_cannot_receive_closed_host_proof():
    assert _screen(_source(dynamic=True))["status"] == "unsupported"
    assert _screen(_source(boundaries="0.0, 0.75, 0.5, 1.0"))["status"] == "unsupported"
    assert _screen(_source(boundaries="0x000000000000803E0000C07F0000803F"))["status"] == "unsupported"


def test_bucketize_declaration_cannot_expand_to_family_rank_or_numerical_claim():
    selected = _selected()["neutral"]["capability_spec"]
    row = selected["operations"][0]
    for changed in (
        {**row, "ops": ["linalg.generic"]},
        {**row, "families": ["reduction"]},
        {**row, "signature": {**row["signature"], "ranks": [2]}},
        {**row, "signature": {**row["signature"], "ordered_operand_dtypes": ["f32", "f32"]}},
        {**row, "numerical_contract": {"status": "reviewed"}},
        {**row, "source_body": {**row["source_body"], "right": False}},
    ):
        selected["operations"] = [changed]
        with pytest.raises(ValueError, match="bucketize|source_body"):
            validate_host_capabilities(selected)
