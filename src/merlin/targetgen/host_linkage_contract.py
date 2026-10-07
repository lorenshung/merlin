"""Closed, opt-in host library-supplier requirement; never a numerical grant."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from merlin.common.digest import is_sha256
from merlin.llvmlower.link_supplier_trace import trace_symbol_flags

SCHEMA = "merlin.host_linkage_contract.v1"
SUPPLIER = "selected_math_archive"


def validate_linkage_contract(value: object) -> dict[str, Any]:
    """Require an exact caller-selected archive digest and bounded symbol roster."""
    if (
        not isinstance(value, Mapping)
        or set(value) != {"schema", "supplier", "archive_sha256", "symbols"}
        or value.get("schema") != SCHEMA
        or value.get("supplier") != SUPPLIER
        or not is_sha256(value.get("archive_sha256"))
        or not isinstance(value.get("symbols"), list)
    ):
        raise ValueError("host linkage contract is not a closed selected archive roster")
    symbols = list(value["symbols"])
    trace_symbol_flags(symbols)
    if symbols != sorted(symbols):
        raise ValueError("host linkage symbols must be in canonical order")
    return {"schema": SCHEMA, "supplier": SUPPLIER, "archive_sha256": value["archive_sha256"], "symbols": symbols}
