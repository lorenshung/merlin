"""Exact source occurrence and linked-build checks for opt-in host pointwise bodies."""

from __future__ import annotations

from hashlib import sha256

import pytest
from merlin_experiments.phase1.feedback import private_data_movement as movement
from merlin_experiments.phase1.feedback import private_pointwise_support as pointwise

from merlin.common import mlir_query as mq
from merlin.targetgen.host_capabilities import admit_host_operation

SOURCE = """builtin.module {
  func.func @f(%x: tensor<2x3xf32>, %y: tensor<2x3xf32>,
               %init: tensor<2x3xf32>) -> tensor<2x3xf32> {
    %a = "linalg.generic"(%x, %y, %init) <{indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [
      #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%u: f32, %v: f32, %old: f32):
        %sum = "arith.addf"(%u, %v) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "linalg.yield"(%sum) : (f32) -> ()
    }) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    %b = "linalg.generic"(%a, %y, %init) <{indexing_maps = [
      affine_map<(d0, d1) -> (d0, d1)>, affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>], iterator_types = [
      #linalg.iterator_type<parallel>, #linalg.iterator_type<parallel>],
      operandSegmentSizes = array<i32: 2, 1>}> ({
      ^bb0(%u: f32, %v: f32, %old: f32):
        %sum = "arith.addf"(%u, %v) <{fastmath = #arith.fastmath<none>}> : (f32, f32) -> f32
        "linalg.yield"(%sum) : (f32) -> ()
    }) : (tensor<2x3xf32>, tensor<2x3xf32>, tensor<2x3xf32>) -> tensor<2x3xf32>
    func.return %b : tensor<2x3xf32>
  }
}"""


def _source():
    module = mq.parse(SOURCE)
    parsed = tuple(mq.walk(module))
    ordinals = [ordinal for ordinal, op in enumerate(parsed) if mq.op_name(op) == "linalg.generic"]
    assert len(ordinals) == 2
    digest = sha256(SOURCE.encode()).hexdigest()
    row = {
        "operation": "linalg.generic",
        "mlir_operation": "linalg.generic",
        "disposition": "host_required",
        "count": 2,
        "ordinals": ordinals,
    }
    other = [
        {"mlir_operation": mq.op_name(op), "count": 1, "ordinals": [ordinal]}
        for ordinal, op in enumerate(parsed)
        if ordinal not in ordinals
    ]
    inventory = {
        "n_operations": len(parsed),
        "capture_sha256": digest,
        "capture_normalization": {"output_sha256": digest},
        "signatures": [row, *other],
    }
    joined = movement.source_inventory_by_ordinal(parsed, inventory, digest, digest)
    return parsed, row, joined, digest


def _admission(parsed, row):
    package, spec = "a" * 64, "b" * 64
    selected = {
        "host": {
            "package_sha256": package,
            "capability_spec_sha256": spec,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": package, "dtype_strategy": "int8_w8a8"},
                "operations": [
                    {
                        "id": "neutral_add",
                        "ops": ["linalg.generic"],
                        "placement": "host",
                        "signature": {
                            "ordered_operand_dtypes": ["f32", "f32", "f32"],
                            "ordered_result_dtypes": ["f32"],
                            "ranks": [2],
                        },
                        "source_body": {"schema": "merlin.static_pointwise_source_body.v1", "operation": "arith.addf"},
                    }
                ],
                "evidence": {"scope": "neutral declaration"},
            },
        }
    }
    observed = {
        "family": "elementwise_map",
        "ordered_operand_dtypes": ["f32", "f32", "f32"],
        "ordered_result_dtypes": ["f32"],
        "rank": 2,
    }
    return admit_host_operation(selected, row, observed, source_operations=tuple(parsed[i] for i in row["ordinals"]))


def test_all_exact_ordinals_join_same_linked_build():
    parsed, row, joined, digest = _source()
    admission = _admission(parsed, row)
    assert admission["status"] == "admitted" and admission["reviewed"] is True
    proof = pointwise.begin(digest, digest, len(parsed))
    pointwise.record(proof, row, admission, parsed, joined)
    assert [entry["ordinal"] for entry in proof["occurrences"]] == row["ordinals"]
    source = {
        "source_sha256": digest,
        "normalized_source_sha256": digest,
        "n_source_operations": len(parsed),
        "pointwise_host_support": proof,
    }
    entry = {"source_sha256": digest, "capture_tree_sha256": "c" * 64, "elf_sha256": "e" * 64}
    assert not pointwise.linked_complete(source, entry, "d" * 64)
    pointwise.link(source, {"candidate_tree_sha256": "d" * 64, "capture_tree_sha256": "c" * 64, "elf_sha256": "e" * 64})
    assert pointwise.linked_complete(source, entry, "d" * 64)
    assert not pointwise.linked_complete(source, entry, "f" * 64)
    entry["elf_sha256"] = "f" * 64
    assert not pointwise.linked_complete(source, entry, "d" * 64)


def test_missing_duplicate_or_changed_source_occurrence_refuses():
    parsed, row, joined, digest = _source()
    admission = _admission(parsed, row)
    for broken in (
        {**row, "ordinals": row["ordinals"][:1], "count": 2},
        {**row, "ordinals": [row["ordinals"][0]] * 2},
        {**row, "ordinals": [True, row["ordinals"][1]]},
    ):
        with pytest.raises(ValueError, match="occurrence"):
            pointwise.record(pointwise.begin(digest, digest, len(parsed)), broken, admission, parsed, joined)
    with pytest.raises(ValueError, match="occurrence"):
        pointwise.record(
            pointwise.begin(digest, digest, len(parsed)),
            row,
            admission,
            parsed,
            {**joined, row["ordinals"][1]: {**row}},
        )
    mutated = list(parsed)
    changed = mq.parse(SOURCE.replace('"arith.addf"', '"arith.subf"'))
    mutated[row["ordinals"][1]] = next(mq.walk(changed, "linalg.generic"))
    with pytest.raises(ValueError, match="occurrence"):
        pointwise.record(pointwise.begin(digest, digest, len(parsed)), row, admission, tuple(mutated), joined)


def test_unary_serialized_type_roster_and_empty_witness_are_exact():
    digest = "a" * 64
    source = {"source_sha256": digest, "normalized_source_sha256": digest, "n_source_operations": 3}
    entry = {"source_sha256": digest, "capture_tree_sha256": "b" * 64, "elf_sha256": "c" * 64}
    linked = {"candidate_tree_sha256": "d" * 64, "capture_tree_sha256": "b" * 64, "elf_sha256": "c" * 64}
    assert not pointwise.linked_complete(source, entry, "d" * 64)
    source["pointwise_host_support"] = pointwise.begin(digest, digest, 3)
    pointwise.link(source, linked)
    assert pointwise.linked_complete(source, entry, "d" * 64)
    unary = {
        "ordinal": 1,
        "profile": "host",
        "capability_spec_sha256": "e" * 64,
        "declaration": "typed_extui",
        "operation": "arith.extui",
        "predicate": None,
        "shape": [2, 3],
        "ordered_types": ["i1", "i64", "i64"],
    }
    proof = source["pointwise_host_support"]
    proof["count"] = 1
    proof["occurrences"] = [unary]
    assert pointwise.linked_complete(source, entry, "d" * 64)
    proof["occurrences"] = [{**unary, "ordered_types": ["i1", "i64"]}]
    assert not pointwise.linked_complete(source, entry, "d" * 64)
    proof["occurrences"] = [{**unary, "predicate": "eq"}]
    assert not pointwise.linked_complete(source, entry, "d" * 64)
    proof["occurrences"] = [{**unary, "declaration": ""}]
    assert not pointwise.linked_complete(source, entry, "d" * 64)
