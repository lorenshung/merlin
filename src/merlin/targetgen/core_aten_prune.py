"""Recorded, intentional cuts over the bounded Core ATen suite.

A cut ledger names its parent suite by digest and lists cuts in order. Each cut drops the selected
cases whose partition assignment matches its criterion. Applying the ledger yields a pruned suite in
the parent's schema, so the bounded runner consumes it unchanged, plus the state of the set after
every cut: cases by overload and partition axis, and which witnessed obligations remain covered.

An obligation lost because it names a dropped partition is lost by design. Any other lost obligation
is collateral: its only witnesses happened to sit in dropped cases. Collateral losses are listed in
full so a cut never removes coverage silently.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import Counter
from pathlib import Path

from merlin.common.yaml import load_yaml

AXES = ("dtype", "layout", "shape", "values")


def load_ledger(path: Path) -> dict:
    ledger = load_yaml(path)
    if ledger.get("schema_version") != 1:
        raise ValueError(f"unsupported cut ledger schema: {ledger.get('schema_version')!r}")
    ids = [cut["id"] for cut in ledger.get("cuts", [])]
    if len(ids) != len(set(ids)):
        raise ValueError("cut ids must be unique")
    for cut in ledger.get("cuts", []):
        if not cut.get("drop"):
            raise ValueError(f"cut {cut['id']!r} has no drop criterion")
    return ledger


def matches(case: dict, drop: dict) -> bool:
    """A case matches when every listed axis holds one of that axis's listed values."""
    assignment = case["partition_assignment"]
    return all(str(assignment.get(axis)) in {str(v) for v in values} for axis, values in drop.items())


def obligation_cells(obligation: str) -> tuple[str, str, dict]:
    """Split ``kind::overload::axis=value::...`` into its kind, overload and cells."""
    kind, overload, *cells = obligation.split("::")
    return kind, overload, dict(cell.partition("=")[::2] for cell in cells)


def names_dropped_partition(obligation: str, drop: dict) -> bool:
    _, _, cells = obligation_cells(obligation)
    return all(str(cells.get(axis)) in {str(v) for v in values} for axis, values in drop.items())


def state(cases: list[dict], overloads: list[str], obligations: list[str]) -> dict:
    covered = {o for case in cases for o in case["covered_obligations"]}
    per_overload = Counter(case["overload"] for case in cases)
    return {
        "case_count": len(cases),
        "per_overload": {o: per_overload[o] for o in overloads},
        "overloads_with_cases": sum(1 for o in overloads if per_overload[o]),
        "axes": {
            axis: dict(
                Counter(str(c["partition_assignment"][axis]) for c in cases if axis in c["partition_assignment"])
            )
            for axis in AXES
        },
        "obligations_covered": dict(Counter(o.split("::")[0] for o in obligations if o in covered)),
        "obligations_covered_count": sum(1 for o in obligations if o in covered),
    }


def apply_ledger(suite: dict, ledger: dict, suite_sha256: str) -> tuple[dict, list[dict]]:
    """Return the pruned suite and one state record per step (the parent first, then each cut)."""
    if suite_sha256 != ledger["parent"]["sha256"]:
        raise ValueError(f"parent suite digest {suite_sha256} does not match ledger {ledger['parent']['sha256']}")
    overloads = list(suite["overloads"])
    obligations = list(suite["witnessed_obligations"])
    cases = list(suite["selected_cases"])
    steps = [{"step": "parent", **state(cases, overloads, obligations)}]
    for cut in ledger["cuts"]:
        before = {o for case in cases for o in case["covered_obligations"]}
        kept = [case for case in cases if not matches(case, cut["drop"])]
        after = {o for case in kept for o in case["covered_obligations"]}
        lost = sorted(before - after)
        by_design = [o for o in lost if names_dropped_partition(o, cut["drop"])]
        emptied = sorted({c["overload"] for c in cases} - {c["overload"] for c in kept})
        steps.append(
            {
                "step": cut["id"],
                "drop": cut["drop"],
                "rationale": cut.get("rationale", ""),
                "cases_removed": len(cases) - len(kept),
                "obligations_lost_by_design": len(by_design),
                "obligations_lost_collateral": sorted(set(lost) - set(by_design)),
                "overloads_emptied": emptied,
                **state(kept, overloads, obligations),
            }
        )
        cases = kept
    pruned = copy.copy(suite)
    pruned["selected_cases"] = cases
    pruned["selected_count"] = len(cases)
    pruned["complete"] = False
    pruned["claim"] = f"{suite['claim']}; pruned by the recorded cuts below, so no longer complete"
    pruned["pruning"] = {
        "parent_sha256": suite_sha256,
        "ledger_sha256": hashlib.sha256(json.dumps(ledger, sort_keys=True).encode()).hexdigest(),
        "cuts": ledger["cuts"],
        "case_ids": [case["case_id"] for case in cases],
        "obligations_covered_count": steps[-1]["obligations_covered_count"],
    }
    return pruned, steps
