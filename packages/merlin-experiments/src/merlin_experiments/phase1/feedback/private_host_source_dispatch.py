"""Route closed source-body admissions to their independent mandatory witnesses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from merlin.frontends.bucketize_source import STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA
from merlin.frontends.linalg_reduction_source_body import STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA
from merlin.frontends.prepared_index_source_body import SCHEMA as PREPARED_INDEX_SOURCE_BODY_SCHEMA
from merlin_experiments.phase1.feedback import private_index_host_support as index_support
from merlin_experiments.phase1.feedback import private_integer_reduction_support as integer_support
from merlin_experiments.phase1.feedback import private_linalg_support as linalg_support


def source_context(row: Mapping[str, Any], index: Mapping[str, Any], selected: Mapping[str, Any] | None) -> dict:
    """Only prepared index roots receive the independently proved trace premise."""
    context = {"selected_index_observation": selected}
    if row.get("frontend_op") == "aten.index.Tensor" and row.get("mlir_operation") in {
        "linalg.generic",
        "linalg.reduce",
    }:
        context["verified_index_source"] = index["source_proof"]
    return context


def record(
    linalg: dict,
    integer: dict,
    index: dict,
    row: Mapping[str, Any],
    host_decision: Mapping[str, Any],
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
    bounded_control: Mapping[str, Any] | None,
) -> None:
    """Never send a new body schema through the legacy pointwise-only witness."""
    body = host_decision.get("source_body_proof")
    schema = body.get("schema") if isinstance(body, Mapping) else None
    if schema == STATIC_INTEGER_REDUCTION_SOURCE_BODY_SCHEMA:
        integer_support.record(integer, row, host_decision, parsed, source_rows)
        return
    if schema == PREPARED_INDEX_SOURCE_BODY_SCHEMA:
        index_support.record(index, row, host_decision, parsed, source_rows)
        return
    if schema == STATIC_BUCKETIZE_SOURCE_BODY_SCHEMA:
        # The caller's mandatory bucketize witness independently records the
        # verified trace, boundary literal and selected linked source below.
        return
    linalg_support.record(linalg, row, host_decision, parsed, source_rows, control_proof=bounded_control)
    integer_support.record(integer, row, host_decision, parsed, source_rows)
    index_support.record(index, row, host_decision, parsed, source_rows)
