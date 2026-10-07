"""Explicit metadata request adapter for op/form coverage tools.

Reads only the requested document, never a corpus directory or hidden member.
The envelope reports source identities; these are not runtime qualifications.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from merlin.common.strict_json import loads
from merlin.targetgen import eligibility, op_form_regimes, opset_contract


def read_request(path: str | Path) -> tuple[dict, dict, op_form_regimes.FormBounds]:
    source = Path(path).resolve(strict=True)
    raw = source.read_bytes()
    doc = loads(raw)
    if not isinstance(doc, dict) or doc.get("schema") != "merlin.opset_coverage_input.v1":
        raise ValueError("explicit versioned coverage input required")
    target, contract, facts = (doc.get(k) for k in ("target", "contract", "facts"))
    if not isinstance(target, str) or not target or not isinstance(contract, dict) or not isinstance(facts, dict):
        raise ValueError("coverage requires explicit target, contract and source facts")
    for key in ("excluded", "windowed_ops", "epilogue_stages"):
        if key in doc and (
            not isinstance(doc[key], list)
            or any(not isinstance(v, str) or not v for v in doc[key])
            or len(set(doc[key])) != len(doc[key])
        ):
            raise ValueError(f"{key} must be a list of unique nonempty names")
    if doc.get("dtype") is not None and not isinstance(doc["dtype"], str):
        raise ValueError("dtype must be a format name")
    if not isinstance(doc.get("capsules"), list):
        raise ValueError("coverage requires an explicit public cohort")
    bounds = op_form_regimes.bounds_for_target(target, facts=facts, contract=contract, dtype=doc.get("dtype"))
    homes = opset_contract.homes_for_capabilities(
        eligibility.capability_map_from_contract(contract),
        undetermined=eligibility.undetermined_families_from_contract(contract),
    )
    report = opset_contract.cell_coverage(
        homes,
        bounds,
        doc["capsules"],
        capability_map=eligibility.capability_map_from_contract(contract),
        excluded=doc.get("excluded", ()),
        epilogue_stages=doc.get("epilogue_stages", ()),
        windowed_ops=doc.get("windowed_ops", ()),
    )
    report["input"] = {"path": str(source), "sha256": hashlib.sha256(raw).hexdigest()}
    report["bounds"] = bounds.to_dict()
    return doc, report, bounds
