"""Verify frontend stage identities against exact captured MLIR bytes and typed SSA."""

from __future__ import annotations

import hashlib
import json
from collections import Counter

from merlin.common.digest import is_sha256

_CALLS = {"call_function", "call_module", "call_method"}
_STRUCTURAL = {"builtin.module", "func.func", "func.return", "linalg.yield", "scf.yield"}


def _digest(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _graph(snapshot: dict | None, stage: str, errors: list[str]) -> dict:
    unknown = {"status": "unknown", "call_count": None, "nodes": {}, "calls": set(), "by_target": {}}
    if not isinstance(snapshot, dict) or snapshot.get("status") != "complete":
        errors.append(f"{stage} graph is unavailable")
        return unknown
    if snapshot.get("schema") != "m2m.frontend_graph.v1" or snapshot.get("stage") != stage:
        errors.append(f"{stage} graph schema or stage identity is invalid")
        return unknown
    if not is_sha256(snapshot.get("sha256")) or snapshot["sha256"] != _digest(
        {key: value for key, value in snapshot.items() if key != "sha256"}
    ):
        errors.append(f"{stage} graph content SHA256 disagrees")
        return unknown
    nodes = snapshot.get("nodes")
    if not isinstance(nodes, list) or any(
        not isinstance(node, dict) or not isinstance(node.get("id"), str) for node in nodes
    ):
        errors.append(f"{stage} graph node roster is invalid")
        return unknown
    indexed = {node["id"]: node for node in nodes}
    if len(indexed) != len(nodes):
        errors.append(f"{stage} graph node identities are duplicated")
        return unknown
    calls = {node["id"] for node in nodes if node.get("op") in _CALLS}
    if type(snapshot.get("call_count")) is not int or snapshot["call_count"] != len(calls):
        errors.append(f"{stage} graph call count disagrees with its node roster")
        return unknown
    values = {
        result["id"]: result
        for node in nodes
        for result in node.get("results") or []
        if isinstance(result, dict) and isinstance(result.get("id"), str)
    }
    for edge in snapshot.get("edges") or []:
        if (
            not isinstance(edge, dict)
            or edge.get("producer_node_id") not in indexed
            or edge.get("consumer_node_id") not in indexed
        ):
            errors.append(f"{stage} graph has an edge without known node endpoints")
            return unknown
        value_id = edge.get("producer_value_id")
        if value_id is not None:
            result = values.get(value_id)
            if result is None or result.get("dtype") != edge.get("dtype") or result.get("shape") != edge.get("shape"):
                errors.append(f"{stage} graph typed edge disagrees with its producer value")
                return unknown
    return {
        "status": "verified",
        "call_count": len(calls),
        "nodes": indexed,
        "calls": calls,
        "sha256": snapshot["sha256"],
        "runtime_versions": snapshot.get("runtime_versions"),
        "by_target": dict(sorted(Counter(indexed[identity].get("target") for identity in calls).items())),
    }


def join_frontend_trace(trace: dict | None, application_graph: dict | None, *, capture_sha256: str) -> dict:
    """Return exact stage counts and digest-bound per-operation correspondence.

    Program names, FQNs, operator-family labels and matching row counts never
    establish identity. Missing original/quantized snapshots remain unknown.
    """
    base = {
        "status": "unknown",
        "counting_unit": "static_captured_call_sites",
        "original_invocation_count": None,
        "quantized_invocation_count": None,
        "prepared_invocation_count": None,
        "graphs": {},
        "normalized_operations": {},
        "raw_mlir_correspondence": {"status": "unknown"},
        "normalization_correspondence": {"status": "unknown"},
    }
    if trace is None:
        return {**base, "errors": ["no selected original/quantized frontend trace"]}
    if not isinstance(trace, dict) or trace.get("schema") != "m2m.frontend_trace.v1":
        return {**base, "status": "invalid", "errors": ["unsupported frontend trace schema"]}
    errors = []
    snapshots = trace.get("graphs") or {}
    graphs = {stage: _graph(snapshots.get(stage), stage, errors) for stage in ("original", "quantized", "prepared")}
    stages_valid = all(graph["status"] == "verified" for graph in graphs.values())
    for stage, graph in graphs.items():
        base[f"{stage}_invocation_count"] = graph["call_count"]
        base["graphs"][stage] = {key: value for key, value in graph.items() if key not in {"nodes", "calls"}}
    relations = trace.get("transformations")
    seen_transitions = set()
    if not isinstance(relations, list):
        errors.append("frontend transformation correspondence is absent")
        relations = []
    for transition in relations:
        source, destination = transition.get("from_stage"), transition.get("to_stage")
        if (source, destination) not in {("original", "quantized"), ("quantized", "prepared")}:
            errors.append("unknown frontend stage transition")
            continue
        seen_transitions.add((source, destination))
        consumed, produced = set(), set()
        for relation in transition.get("relations") or []:
            sources, destinations = relation.get("source_ids") or [], relation.get("destination_ids") or []
            if not set(sources) <= set(graphs[source]["nodes"]) or not set(destinations) <= set(
                graphs[destination]["nodes"]
            ):
                errors.append(f"{source} -> {destination} references unknown source identities")
            consumed.update(sources)
            produced.update(destinations)
        if (
            transition.get("status") != "complete"
            or not graphs[source]["calls"] <= consumed
            or not graphs[destination]["calls"] <= produced
        ):
            errors.append(f"{source} -> {destination} call-site correspondence is incomplete")
    if seen_transitions != {("original", "quantized"), ("quantized", "prepared")}:
        errors.append("original -> quantized -> prepared transition roster is incomplete")
    mlir = trace.get("mlir") or {}
    if not isinstance(application_graph, dict) or application_graph.get("schema") != "merlin.application_graph.v1":
        errors.append("selected typed MLIR graph is unavailable")
    elif (
        application_graph.get("capture_sha256") != capture_sha256
        or mlir.get("sha256") != capture_sha256
        or mlir.get("bytes") != application_graph.get("capture_bytes")
    ):
        errors.append("frontend trace does not correspond to the exact selected capture bytes")
    else:
        raw = application_graph.get("capture_graph") or {}
        raw_operations, recorded = raw.get("operations") or [], mlir.get("operations")
        prepared_nodes = graphs["prepared"]["nodes"]
        origin_nodes = {**graphs["original"]["nodes"], **graphs["quantized"]["nodes"]}
        valid = isinstance(recorded, list) and len(recorded) == len(raw_operations)
        if valid:
            for ordinal, (observed, record) in enumerate(zip(raw_operations, recorded, strict=True)):
                expected_role = observed.get("trace_role") or (
                    "structural" if observed["mlir_operation"] in _STRUCTURAL else "unresolved"
                )
                if (
                    record.get("ordinal") != ordinal
                    or observed.get("ordinal") != ordinal
                    or record.get("operation") != observed["mlir_operation"]
                    or record.get("operand_types") != [value["type"] for value in observed["operands"]]
                    or record.get("result_types") != [value["type"] for value in observed["results"]]
                    or record.get("source_node_ids") != observed.get("source_node_ids")
                    or record.get("origin_node_ids") != observed.get("origin_node_ids")
                    or record.get("role") != expected_role
                    or not set(record.get("source_node_ids") or []) <= set(prepared_nodes)
                    or not set(record.get("origin_node_ids") or []) <= set(origin_nodes)
                ):
                    valid = False
                    break
        if valid:
            mapped = {
                identity: [record["ordinal"] for record in recorded if identity in record["source_node_ids"]]
                for identity in graphs["prepared"]["calls"]
            }
            correspondence = mlir.get("source_correspondence")
            if not isinstance(correspondence, list) or len(correspondence) != len(mapped):
                valid = False
            else:
                seen = set()
                for receipt in correspondence:
                    identity = receipt.get("node_id")
                    if identity not in mapped or identity in seen or receipt.get("mlir_ordinals") != mapped[identity]:
                        valid = False
                        break
                    seen.add(identity)
                    if mapped[identity]:
                        if receipt.get("status") != "lowered":
                            valid = False
                            break
                    elif receipt.get("status") == "alias":
                        if (
                            receipt.get("alias_of_node_id") not in prepared_nodes
                            or type(receipt.get("result_index")) is not int
                            or receipt["result_index"] < 0
                        ):
                            valid = False
                            break
                    elif receipt.get("status") == "eliminated":
                        if not receipt.get("reason"):
                            valid = False
                            break
                    else:
                        valid = False
                        break
        if valid:
            base["raw_mlir_correspondence"] = {
                "status": "verified",
                "sha256": capture_sha256,
                "bytes": mlir["bytes"],
                "n_operations": len(recorded),
            }
            normalized = application_graph.get("normalization_correspondence") or {}
            normalization_valid = normalized.get("status") in {"identity", "serialization_equivalent"}
            base["normalization_correspondence"] = {
                **normalized,
                "status": "verified" if normalization_valid else "unknown",
            }
            if normalization_valid:
                base["normalized_operations"] = {
                    str(record["ordinal"]): {
                        "prepared_node_ids": record["source_node_ids"],
                        "origin_node_ids": record["origin_node_ids"],
                        "original_node_ids": [
                            identity
                            for identity in record["origin_node_ids"]
                            if identity in graphs["original"]["nodes"]
                        ],
                        "quantized_node_ids": [
                            identity
                            for identity in record["origin_node_ids"]
                            if identity in graphs["quantized"]["nodes"]
                        ],
                    }
                    for record in recorded
                }
            else:
                errors.append("normalization has no exact per-operation source correspondence")
        else:
            errors.append("frontend trace MLIR roster differs from exact parsed SSA/type/source identities")
    if trace.get("status") != "complete" or trace.get("blockers"):
        errors.extend(trace.get("blockers") or ["producer reports incomplete frontend trace"])
    base["status"] = "complete" if not errors and stages_valid else "partial"
    return {
        **base,
        "errors": sorted(set(errors)),
        "trace_document_sha256": _digest(trace),
        "scope": "exact selected static frontend call sites; not dynamic runtime invocation counts",
    }
