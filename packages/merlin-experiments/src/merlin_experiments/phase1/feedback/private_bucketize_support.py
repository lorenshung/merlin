"""Build-only accounting for reviewed host bucketize source occurrences.

The source witness proves a closed count algorithm. This module additionally
checks literal boundary ordering, reviewed placement, selected index lowering,
and exact linked bytes. It proves neither original/prepared equivalence nor
whole-model numerical behavior.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from hashlib import sha256
from pathlib import Path
from typing import Any

from merlin.common import mlir_query as mq
from merlin.common.digest import is_sha256
from merlin.frontends.bucketize_source import screen_bucketize_source
from merlin.targetgen.application_inventory import verify_capture_receipt
from merlin_experiments.phase1.feedback.private_control_support import _record_matches_selected

FIELD = "bucketize_host_support"
PENDING = "source_bucketize_host_pending_build"
LINKED = "source_bucketize_host_linked"
SCOPE = "reviewed bucketize source and exact linked build only; no frontend or numerical equivalence"
_BUILD_KEYS = {"candidate_tree_sha256", "capture_tree_sha256", "elf_sha256"}
_BOUNDARY_PREMISE = "nondecreasing_non_nan_f32_boundaries"


def _need(condition: bool, reason: str) -> None:
    if not condition:
        raise ValueError(f"bucketize host support: {reason}")


def _digest(document: object) -> str:
    return sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _dense_boundary(op: Any, count: int) -> dict[str, Any]:
    """Prove the exact second SSA operand is a sorted non-NaN f32 literal.

    A dynamic boundary is explicitly unproved; its provenance tag or captured
    example values cannot establish the host precondition.
    """
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, f32
    from xdsl.traits import Pure

    boundary = op.inputs[1]
    owner = boundary.owner
    pending = {"status": "not_proved", "required_precondition": _BOUNDARY_PREMISE, "literal_sha256": None}
    if type(owner) is not arith.ConstantOp:
        return pending
    _need(type(owner).has_trait(Pure) and not owner.regions and not owner.successors, "boundary constant is impure")
    _need(
        not owner.operands
        and len(owner.results) == 1
        and owner.results[0] is boundary
        and set(owner.properties) == {"value"}
        and all(key.startswith("prov.") for key in owner.attributes),
        "boundary constant has unknown semantics",
    )
    value = owner.properties["value"]
    _need(
        isinstance(value, DenseIntOrFPElementsAttr)
        and isinstance(value.type, TensorType)
        and value.type == boundary.type
        and value.type.get_element_type() == f32
        and tuple(value.type.get_shape()) == (count,),
        "boundary literal has a different type or extent",
    )
    values = tuple(value.get_values())
    _need(len(values) == count and all(type(item) is float for item in values), "boundary literal is incomplete")
    if any(math.isnan(item) for item in values) or any(left > right for left, right in zip(values, values[1:])):
        return pending
    return {
        "status": "source_dense_non_decreasing_non_nan_proved",
        "required_precondition": _BOUNDARY_PREMISE,
        "literal_sha256": sha256(value.data.data).hexdigest(),
    }


def _bucketize_ordinals(parsed: tuple[Any, ...], trace: Mapping[str, Any]) -> tuple[int, ...]:
    """Join every prepared trace node to one actual parsed bucketize generic."""
    graphs = trace.get("graphs")
    prepared = graphs.get("prepared") if isinstance(graphs, Mapping) else None
    nodes = prepared.get("nodes") if isinstance(prepared, Mapping) else None
    _need(isinstance(nodes, list), "prepared frontend node roster is absent")
    ids = [
        node.get("id") for node in nodes if isinstance(node, Mapping) and node.get("target") == "aten.bucketize.Tensor"
    ]
    _need(
        all(isinstance(item, str) and item for item in ids) and len(ids) == len(set(ids)),
        "bucketize nodes are ambiguous",
    )
    ordinals = []
    observed_ids = []
    for ordinal, op in enumerate(parsed):
        if mq.op_name(op) != "linalg.generic" or mq.attr_str(op, "prov.aten") != "aten.bucketize.Tensor":
            continue
        from xdsl.dialects.builtin import ArrayAttr, StringAttr

        source_attr = op.attributes.get("prov.source_node_ids")
        source_ids = (
            [item.data for item in source_attr.data]
            if isinstance(source_attr, ArrayAttr) and all(isinstance(item, StringAttr) for item in source_attr.data)
            else []
        )
        _need(len(source_ids) == 1 and source_ids[0] in ids, "bucketize generic has no prepared-node owner")
        ordinals.append(ordinal)
        observed_ids.append(source_ids[0])
    _need(
        len(ordinals) == len(ids) and len(set(observed_ids)) == len(ids),
        "bucketize trace/source occurrences are incomplete",
    )
    return tuple(ordinals)


def begin(
    capture: Path,
    parsed: tuple[Any, ...],
    raw_sha256: str,
    normalized_sha256: str,
    selected_index_observation: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Recompute the mandatory source/trace roster before host admission."""
    capture = Path(capture)
    source = capture / "model.mlir"
    trace_path = capture / "frontend-trace.json"
    checked = verify_capture_receipt(source)
    _need(checked.get("status") == "verified_materialized", "capture artifact receipt is not verified")
    _need(is_sha256(raw_sha256) and is_sha256(normalized_sha256) and len(parsed) > 0, "source identity is malformed")
    _need(sha256(source.read_bytes()).hexdigest() == raw_sha256, "captured source bytes changed")
    try:
        trace = json.loads(trace_path.read_bytes())
    except (OSError, UnicodeError, ValueError) as exc:
        raise ValueError("bucketize host support: captured frontend trace is unreadable") from exc
    _need(isinstance(trace, Mapping), "captured frontend trace is malformed")
    ordinals = _bucketize_ordinals(parsed, trace)
    proof = None
    facts: list[dict[str, Any]] = []
    if ordinals:
        _need(
            isinstance(selected_index_observation, Mapping)
            and selected_index_observation.get("schema") == "merlin.selected-index-lowering.v1"
            and type(selected_index_observation.get("index_bits")) is int
            and 2 <= selected_index_observation["index_bits"] <= 128,
            "bucketize needs a selected producer-owned index observation",
        )
        observed = screen_bucketize_source(source, trace_path, ordinals)
        _need(
            observed.raw_sha256 == raw_sha256 and observed.normalized_sha256 == normalized_sha256,
            "bucketize source/trace differs from selected source",
        )
        proof = json.loads(json.dumps(asdict(observed)))
        for ordinal, pattern in observed.ordinals:
            _need(
                bool(pattern.input_shape)
                and all(
                    type(extent) is int and 0 < extent < 1 << (selected_index_observation["index_bits"] - 1)
                    for extent in pattern.input_shape
                )
                and pattern.boundary_count < 1 << (selected_index_observation["index_bits"] - 1),
                "source loop extent exceeds selected index width",
            )
            facts.append({"ordinal": ordinal, **_dense_boundary(parsed[ordinal], pattern.boundary_count)})
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "capture_receipt_sha256": checked.get("receipt_sha256"),
        "n_source_operations": len(parsed),
        "selected_index_observation": deepcopy(dict(selected_index_observation))
        if selected_index_observation is not None
        else None,
        "source_proof": proof,
        "source_proof_sha256": _digest(proof) if proof is not None else None,
        "original_to_prepared_equivalence": "not_proved",
        "boundary_facts": facts,
        "expected_ordinals": list(ordinals),
        "admissions": [],
    }


def record(
    witness: dict[str, Any],
    row: Mapping[str, Any],
    host_decision: Mapping[str, Any],
    source_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    """Require an existing exact reviewed host decision for every occurrence."""
    expected = set(witness["expected_ordinals"])
    ordinals = row.get("ordinals")
    _need(isinstance(ordinals, list) and all(type(item) is int for item in ordinals), "row ordinal roster is malformed")
    if not expected.intersection(ordinals):
        _need(
            row.get("mlir_operation") != "linalg.generic" or row.get("frontend_op") != "aten.bucketize.Tensor",
            "bucketize host row has no source proof",
        )
        return
    _need(
        witness.get("status") == PENDING
        and row.get("mlir_operation") == "linalg.generic"
        and row.get("frontend_op") == "aten.bucketize.Tensor"
        and type(row.get("count")) is int
        and row["count"] == len(ordinals)
        and len(ordinals) == len(set(ordinals))
        and bool(ordinals)
        and set(ordinals) <= expected
        and all(source_rows.get(item) is row for item in ordinals),
        "reviewed row has an omitted or unrelated bucketize ordinal",
    )
    _need(
        host_decision.get("status") == "admitted" and host_decision.get("reviewed") is True,
        "bucketize has no exact reviewed host admission",
    )
    facts = {item["ordinal"]: item for item in witness["boundary_facts"]}
    _need(
        all(facts[item]["status"] == "source_dense_non_decreasing_non_nan_proved" for item in ordinals),
        "boundary ordering/non-NaN premise is not proved by source; a reviewed host precondition is required",
    )
    raw_profiles = host_decision.get("profiles")
    _need(isinstance(raw_profiles, list), "reviewed host profile roster is malformed")
    profiles = [
        item
        for item in raw_profiles
        if isinstance(item, Mapping) and item.get("status") == "admitted" and item.get("reviewed") is True
    ]
    _need(len(profiles) == 1, "reviewed host profile is absent or ambiguous")
    profile = profiles[0]
    _need(
        isinstance(profile.get("profile"), str)
        and bool(profile["profile"])
        and is_sha256(profile.get("capability_spec_sha256")),
        "reviewed host profile has no selected capability identity",
    )
    seen = {item for admission in witness["admissions"] for item in admission["ordinals"]}
    _need(not seen.intersection(ordinals), "bucketize source ordinal was admitted twice")
    witness["admissions"].append(
        {
            "ordinals": sorted(ordinals),
            "profile": profile["profile"],
            "capability_spec_sha256": profile["capability_spec_sha256"],
        }
    )
    witness["admissions"].sort(key=lambda item: item["ordinals"][0])


def link(source: Mapping[str, Any], actual_index: Mapping[str, Any], linked_build: Mapping[str, str]) -> None:
    """Bind exact completed host source to the actual selected linked program."""
    witness = source.get(FIELD)
    _need(isinstance(witness, dict) and witness.get("status") == PENDING, "source roster is not pending")
    expected = witness["expected_ordinals"]
    admitted = [item for row in witness["admissions"] for item in row["ordinals"]]
    _need(sorted(admitted) == expected, "reviewed host admissions do not cover every bucketize occurrence")
    _need(
        witness["source_proof_sha256"]
        == (_digest(witness["source_proof"]) if witness["source_proof"] is not None else None),
        "source proof changed before the linked build",
    )
    selected = witness["selected_index_observation"]
    _need(_record_matches_selected(actual_index, selected), "actual index lowering differs from selected producer")
    _need(selected == source.get("selected_index_observation"), "selected source index observation changed")
    _need(witness["capture_receipt_sha256"] == source.get("capture_receipt_sha256"), "capture receipt differs")
    _need(
        set(linked_build) == _BUILD_KEYS and all(is_sha256(linked_build[key]) for key in _BUILD_KEYS),
        "linked candidate/capture/ELF identity is incomplete",
    )
    witness["status"] = LINKED
    witness["actual_index_observation"] = deepcopy(dict(actual_index))
    witness["linked_build"] = dict(linked_build)


def linked_source_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """Check BUILD-only source/host/index/ELF accounting, not numerical validity."""
    witness = source.get(FIELD)
    if not isinstance(witness, Mapping) or witness.get("status") != LINKED or witness.get("scope") != SCOPE:
        return False
    try:
        expected = witness["expected_ordinals"]
        admissions = witness["admissions"]
        facts = witness["boundary_facts"]
        proof = witness["source_proof"]
        linked = witness["linked_build"]
        selected = witness["selected_index_observation"]
        actual = witness["actual_index_observation"]
        if (
            set(witness)
            != {
                "status",
                "scope",
                "raw_source_sha256",
                "normalized_source_sha256",
                "capture_receipt_sha256",
                "n_source_operations",
                "selected_index_observation",
                "source_proof",
                "source_proof_sha256",
                "original_to_prepared_equivalence",
                "boundary_facts",
                "expected_ordinals",
                "admissions",
                "actual_index_observation",
                "linked_build",
            }
            or not is_sha256(witness["raw_source_sha256"])
            or not is_sha256(witness["normalized_source_sha256"])
            or not is_sha256(witness["capture_receipt_sha256"])
            or witness["original_to_prepared_equivalence"] != "not_proved"
            or witness["raw_source_sha256"] != source["source_sha256"]
            or witness["normalized_source_sha256"] != source["normalized_source_sha256"]
            or witness["capture_receipt_sha256"] != source["capture_receipt_sha256"]
            or witness["n_source_operations"] != source["n_source_operations"]
            or entry["source_sha256"] != source["source_sha256"]
            or set(linked) != _BUILD_KEYS
            or any(not is_sha256(linked[key]) for key in _BUILD_KEYS)
            or linked
            != {
                "candidate_tree_sha256": candidate_sha256,
                "capture_tree_sha256": entry["capture_tree_sha256"],
                "elf_sha256": entry["elf_sha256"],
            }
            or type(witness["n_source_operations"]) is not int
            or witness["n_source_operations"] < 1
            or not isinstance(expected, list)
            or any(type(item) is not int or item < 0 or item >= witness["n_source_operations"] for item in expected)
            or expected != sorted(set(expected))
            or not isinstance(admissions, list)
            or any(
                not isinstance(row, Mapping)
                or not isinstance(row.get("ordinals"), list)
                or any(type(item) is not int for item in row["ordinals"])
                for row in admissions
            )
            or sorted(item for row in admissions for item in row["ordinals"]) != expected
            or not isinstance(facts, list)
            or any(not isinstance(item, Mapping) or type(item.get("ordinal")) is not int for item in facts)
            or [item["ordinal"] for item in facts] != expected
            or any(
                not isinstance(item, Mapping)
                or set(item) != {"ordinal", "status", "required_precondition", "literal_sha256"}
                or item["required_precondition"] != _BOUNDARY_PREMISE
                for item in facts
            )
        ):
            return False
        if not expected:
            return (
                proof is None
                and witness["source_proof_sha256"] is None
                and not facts
                and not admissions
                and selected == source.get("selected_index_observation")
                and _record_matches_selected(actual, selected)
                and actual == entry.get("index_lowering")
            )
        if (
            not isinstance(proof, Mapping)
            or set(proof) != {"raw_sha256", "normalized_sha256", "trace_sha256", "ordinals", "trace_bindings"}
            or witness["source_proof_sha256"] != _digest(proof)
            or proof["raw_sha256"] != source["source_sha256"]
            or proof["normalized_sha256"] != source["normalized_source_sha256"]
            or not is_sha256(proof["trace_sha256"])
            or any(
                not isinstance(item, list) or len(item) != 2 or type(item[0]) is not int for item in proof["ordinals"]
            )
            or any(
                not isinstance(item, list) or len(item) != 2 or type(item[0]) is not int
                for item in proof["trace_bindings"]
            )
            or [item[0] for item in proof["ordinals"]] != expected
            or [item[0] for item in proof["trace_bindings"]] != expected
            or any(
                item[1]["ancestry_status"] != "original_quantized_prepared_flag_match_only"
                for item in proof["trace_bindings"]
            )
            or any(
                not isinstance(item[1], Mapping)
                or set(item[1])
                != {
                    "prepared_node_id",
                    "original_node_id",
                    "quantized_node_id",
                    "unresolved_ancestry_ids",
                    "ancestry_status",
                }
                or not isinstance(item[1]["unresolved_ancestry_ids"], list)
                or any(
                    not isinstance(item[1][key], str) or not item[1][key]
                    for key in ("prepared_node_id", "original_node_id", "quantized_node_id")
                )
                or any(not isinstance(node_id, str) or not node_id for node_id in item[1]["unresolved_ancestry_ids"])
                for item in proof["trace_bindings"]
            )
            or any(
                fact["status"] != "source_dense_non_decreasing_non_nan_proved" or not is_sha256(fact["literal_sha256"])
                for fact in facts
            )
            or selected != source.get("selected_index_observation")
            or not _record_matches_selected(actual, selected)
            or actual != entry.get("index_lowering")
            or any(
                not isinstance(item[1]["input_shape"], list)
                or not item[1]["input_shape"]
                or any(
                    type(extent) is not int or extent < 1 or extent >= 1 << (selected["index_bits"] - 1)
                    for extent in item[1]["input_shape"]
                )
                or item[1]["boundary_count"] >= 1 << (selected["index_bits"] - 1)
                for item in proof["ordinals"]
            )
            or any(
                not isinstance(item, list)
                or len(item) != 2
                or not isinstance(item[1], Mapping)
                or set(item[1]) != {"input_shape", "boundary_count", "right", "comparison", "count_range"}
                or type(item[1]["boundary_count"]) is not int
                or item[1]["boundary_count"] < 1
                or type(item[1]["right"]) is not bool
                or item[1]["comparison"] != ("ule" if item[1]["right"] else "ult")
                or item[1]["count_range"] != [0, item[1]["boundary_count"]]
                for item in proof["ordinals"]
            )
        ):
            return False
        return all(
            isinstance(row, Mapping)
            and set(row) == {"ordinals", "profile", "capability_spec_sha256"}
            and bool(row["ordinals"])
            and isinstance(row["profile"], str)
            and bool(row["profile"])
            and is_sha256(row["capability_spec_sha256"])
            for row in admissions
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False
