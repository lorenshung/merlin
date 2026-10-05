"""Reconcile an authored dialect-mode ledger with a selected ISA source census.

The decoder supplies the population, not instruction legality.  The ledger may
group several decoder modes under one parameterized dialect operation, but it
must account for every selected row and retain unresolved qualification work.
This audit deliberately does not turn a passing source comparison into a D gate.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any


def _controls(value: Any) -> tuple[str, ...] | None:
    if isinstance(value, str):
        fields = tuple(part.strip() for part in value.split(","))
    elif isinstance(value, list) and all(isinstance(part, str) for part in value):
        fields = tuple(part.strip() for part in value)
    else:
        return None
    return fields if len(fields) == 17 and all(fields) else None


def _mode_binding_problems(entry: dict[str, Any], plan_op: dict[str, Any] | None) -> list[str]:
    """Check that one decoder mode is expressible by a reviewed typed op.

    Machine placement attributes need not have a value in the mode ledger.
    Discriminating mode attributes do: otherwise two source modes can be
    accidentally merged into one operation with no checked way to choose one.
    """
    if plan_op is None:
        return ["dialect_operation_not_in_plan"]
    signature = plan_op.get("signature")
    if not isinstance(signature, dict) or not isinstance(signature.get("attributes"), list):
        return ["dialect_operation_untyped"]
    actual = entry.get("mode_attrs")
    if not isinstance(actual, dict):
        return ["mode_attributes_missing"]
    fields = {
        field["name"]: field
        for field in signature["attributes"]
        if isinstance(field, dict) and isinstance(field.get("name"), str) and field.get("role") == "mode"
    }
    problems = []
    if set(actual) - set(fields):
        problems.append("mode_attribute_not_in_plan")
    if set(fields) - set(actual):
        problems.append("required_mode_attribute_missing")
    for name in sorted(set(fields) & set(actual)):
        field = fields[name]
        value = actual[name]
        kind = field["type"]
        valid_type = (
            type(value) is str
            if kind == "string"
            else type(value) is bool
            if kind == "bool"
            else type(value) is int
            if kind in {"i32", "i64"}
            else False
        )
        if not valid_type:
            problems.append("mode_attribute_type_mismatch")
            continue
        if "choices" in field and value not in field["choices"]:
            problems.append("mode_attribute_out_of_domain")
        if kind in {"i32", "i64"} and (
            value < field.get("min", -(2 ** (31 if kind == "i32" else 63)))
            or value > field.get("max", 2 ** (31 if kind == "i32" else 63) - 1)
        ):
            problems.append("mode_attribute_out_of_domain")
    return sorted(set(problems))


def _discrepancy_key(kind: str, item: Any) -> str:
    return json.dumps([kind, item], sort_keys=True, separators=(",", ":"), allow_nan=False)


def _reviewed_resolutions(inventory: dict[str, Any], discrepancies: dict[str, list]) -> tuple[list[dict], list[dict]]:
    """Require an exact reviewed decision for each observed source disagreement.

    Resolutions are authored evidence pointers, not proof that the chosen RTL
    arithmetic is correct.  An extra or stale resolution is rejected so a
    changed source crosswalk cannot silently inherit an older decision.
    """
    observed = {
        _discrepancy_key(kind, item): {"kind": kind, "item": item}
        for kind, items in discrepancies.items()
        for item in items
    }
    provided: dict[str, dict] = {}
    rows = inventory.get("source_resolutions", [])
    if not isinstance(rows, list):
        raise ValueError("source_resolutions must be a list")
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("kind"), str):
            raise ValueError("source resolution requires a discrepancy kind")
        if row.get("authority") != "selected_rtl" or row.get("reviewed") is not True:
            raise ValueError("source resolution requires reviewed selected_rtl authority")
        if not isinstance(row.get("evidence"), str) or not row["evidence"].strip():
            raise ValueError("source resolution requires a nonempty evidence reference")
        key = _discrepancy_key(row["kind"], row.get("item"))
        if key not in observed or key in provided:
            raise ValueError("source resolution is stale, invented, or duplicated")
        provided[key] = {"kind": row["kind"], "item": row.get("item"), "evidence": row["evidence"]}
    return [provided[key] for key in sorted(provided)], [
        observed[key] for key in sorted(observed.keys() - provided.keys())
    ]


def audit_mode_inventory(
    census: dict[str, Any], inventory: dict[str, Any], *, dialect_plan: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return a complete mode denominator and source/plan discrepancies.

    ``inventory`` is an explicitly supplied OOT evidence index.  Boolean test
    claims in it are reported as declarations, never certified by this reader.
    A missing source row, changed decode or missing target op stays in the
    denominator.  Dialect plan absence is an explicit blocker.
    """
    if census.get("schema") != "merlin.isa_source_census.v1" or not isinstance(census.get("rows"), list):
        raise ValueError("selected ISA census has an unsupported schema")
    summary = census.get("summary")
    discrepancy_kinds = (
        "patterns_not_decoded",
        "decoder_rows_without_pattern",
        "model_classes_without_compatible_pattern",
        "overlapping_patterns",
        "dma_kind_conflicts",
    )
    if not isinstance(summary, dict) or any(not isinstance(summary.get(key), list) for key in discrepancy_kinds):
        raise ValueError("selected ISA census has no complete discrepancy summary")
    if not isinstance(inventory, dict) or not isinstance(inventory.get("variants"), list):
        raise ValueError("mode inventory requires a variants list")
    sources = inventory.get("selected_sources")
    if not isinstance(sources, dict) or not isinstance(sources.get("rtl_revision"), str):
        raise ValueError("mode inventory requires selected_sources.rtl_revision")
    revision_receipt = census.get("source_revision_verification") or {}
    revision_problems = []
    if revision_receipt.get("status") != "verified":
        revision_problems.append("selected_source_revisions_unverified")
    elif revision_receipt.get("rtl_revision") != census.get("rtl_revision") or revision_receipt.get(
        "model_revision"
    ) != sources.get("model_revision"):
        revision_problems.append("selected_source_revision_disagrees")
    parameter_domains = inventory.get("parameter_domains")
    if not isinstance(parameter_domains, dict):
        raise ValueError("mode inventory requires parameter_domains")
    if dialect_plan is not None and (
        not isinstance(dialect_plan, dict)
        or not isinstance(dialect_plan.get("dialect_name"), str)
        or not isinstance(dialect_plan.get("ops"), list)
    ):
        raise ValueError("dialect plan requires dialect_name and ops")

    selected_rows: dict[str, dict[str, Any]] = {}
    for row in census["rows"]:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str) or row["name"] in selected_rows:
            raise ValueError("selected ISA census contains malformed or duplicate rows")
        selected_rows[row["name"]] = row
    declared: dict[str, dict[str, Any]] = {}
    duplicate_ids: list[str] = []
    for row in inventory["variants"]:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"]:
            raise ValueError("mode inventory contains a variant without an id")
        if row["id"] in declared:
            duplicate_ids.append(row["id"])
        else:
            declared[row["id"]] = row

    plan_ops: set[str] = set()
    typed_plan_ops: dict[str, dict[str, Any]] = {}
    typed_plan_error = None
    if dialect_plan is not None:
        for op in dialect_plan["ops"]:
            if not isinstance(op, dict) or not isinstance(op.get("name"), str):
                raise ValueError("dialect plan has a malformed operation")
            plan_ops.add(f"{dialect_plan['dialect_name']}.{op['name']}")
        if any("signature" in op for op in dialect_plan["ops"]):
            from .generate.typed_mlir import validate

            checked = validate(dialect_plan)
            typed_plan_ops = {f"{checked['dialect_name']}.{op['name']}": op for op in checked["ops"]}
        else:
            typed_plan_error = "dialect plan has no reviewed typed signatures"
    else:
        typed_plan_error = "dialect plan unavailable"

    source_discrepancies = list(revision_problems)
    source_discrepancies.extend(f"duplicate_mode_id:{identity}" for identity in sorted(set(duplicate_ids)))
    mode_rows = []
    for identity in sorted(set(selected_rows) | set(declared)):
        actual = selected_rows.get(identity)
        entry = declared.get(identity)
        problems: list[str] = []
        if actual is None:
            problems.append("inventory_mode_not_in_selected_decoder")
        elif actual.get("decode_controls") is None:
            problems.append("selected_pattern_has_no_decode_row")
        if entry is None:
            problems.append("selected_mode_missing_from_inventory")
        else:
            if actual is not None and entry.get("rtl_bitpat") != actual.get("pattern_bits"):
                problems.append("rtl_pattern_changed")
            if actual is not None and _controls(entry.get("rtl_decode_controls")) != _controls(
                actual.get("decode_controls")
            ):
                problems.append("rtl_decode_controls_changed")
            declared_models = entry.get("model_classes")
            if declared_models is not None:
                if not isinstance(declared_models, list) or any(not isinstance(name, str) for name in declared_models):
                    problems.append("model_class_binding_malformed")
                elif actual is not None:
                    candidates = {
                        row.get("name") for row in actual.get("model_candidates", []) if isinstance(row, dict)
                    }
                    if not set(declared_models) <= candidates:
                        problems.append("model_encoding_disagrees")
            if not isinstance(entry.get("required"), bool):
                problems.append("required_selection_missing")
            if not isinstance(entry.get("mode_attrs"), dict):
                problems.append("mode_attributes_missing")
            if not isinstance(entry.get("dialect_op"), str) or "." not in entry["dialect_op"]:
                problems.append("dialect_operation_missing")
            elif dialect_plan is not None and entry["dialect_op"] not in plan_ops:
                problems.append("dialect_operation_not_in_plan")
            domains = entry.get("parameter_domains")
            if not isinstance(domains, list) or any(
                not isinstance(domain, str) or domain not in parameter_domains for domain in domains
            ):
                problems.append("parameter_domain_unresolved")
            if entry.get("required") is True:
                if entry.get("software_admitted") is not True:
                    problems.append("software_admission_missing")
                if not isinstance(entry.get("blocked"), list) or entry["blocked"]:
                    problems.append("mode_qualification_open")
                if dialect_plan is None:
                    problems.append("dialect_plan_unavailable")
        # An omitted row cannot remove a selected decoder mode from the
        # required denominator.  Exclusion must be explicit in the ledger.
        required = entry.get("required") if entry is not None else (actual is not None)
        dialect_op = entry.get("dialect_op") if entry is not None else None
        binding_problems = (
            _mode_binding_problems(entry, typed_plan_ops.get(dialect_op if isinstance(dialect_op, str) else ""))
            if entry is not None and required is True and typed_plan_error is None
            else []
        )
        mode_rows.append(
            {
                "id": identity,
                "required": required,
                "problems": problems,
                "typed_binding_problems": binding_problems,
            }
        )

    if sources["rtl_revision"] != census.get("rtl_revision"):
        source_discrepancies.append("selected_rtl_revision_changed")
    if duplicate_ids:
        for row in mode_rows:
            if row["id"] in duplicate_ids:
                row["problems"].append("duplicate_mode_id")
    problem_counts = Counter(problem for row in mode_rows for problem in row["problems"])
    source_problems = {
        "inventory_mode_not_in_selected_decoder",
        "selected_pattern_has_no_decode_row",
        "selected_mode_missing_from_inventory",
        "rtl_pattern_changed",
        "rtl_decode_controls_changed",
        "duplicate_mode_id",
        "selected_rtl_revision_changed",
    }
    source_bound = not source_discrepancies and not (source_problems & problem_counts.keys())
    census_discrepancies = {key: summary[key] for key in discrepancy_kinds if summary[key]}
    resolved, unresolved = _reviewed_resolutions(inventory, census_discrepancies)
    source_reconciled = source_bound and not unresolved
    required = sum(row["id"] in selected_rows and row["required"] is not False for row in mode_rows)
    bound = (
        sum(
            row["id"] in selected_rows
            and row["required"] is True
            and row["id"] in declared
            and not row["typed_binding_problems"]
            for row in mode_rows
        )
        if typed_plan_error is None
        else 0
    )
    binding_problem_counts = Counter(problem for row in mode_rows for problem in row["typed_binding_problems"])
    typed_mode_binding_ready = (
        source_reconciled
        and typed_plan_error is None
        and bound == required
        and not binding_problem_counts
        and not source_discrepancies
    )
    return {
        "schema": "merlin.isa_mode_audit.v1",
        "selected_rtl_revision": census.get("rtl_revision"),
        "inventory_rtl_revision": sources["rtl_revision"],
        "source_bound": source_bound,
        "source_reconciled": source_reconciled,
        "mode_inventory_ready": source_reconciled and not problem_counts and typed_mode_binding_ready,
        "typed_mode_binding_ready": typed_mode_binding_ready,
        "typed_plan_error": typed_plan_error,
        "counts": {
            "selected_decoder_modes": len(selected_rows),
            "inventory_modes": len(inventory["variants"]),
            "required_modes": required,
            "typed_mode_bindings": bound,
            "modes_with_open_obligations": sum(bool(row["problems"]) for row in mode_rows),
            "modes_with_open_bindings": sum(bool(row["typed_binding_problems"]) for row in mode_rows),
            "problem_kinds": dict(sorted(problem_counts.items())),
            "typed_binding_problem_kinds": dict(sorted(binding_problem_counts.items())),
        },
        "source_discrepancies": source_discrepancies,
        "census_discrepancies": census_discrepancies,
        "reviewed_source_resolutions": resolved,
        "unresolved_source_discrepancies": unresolved,
        "modes": mode_rows,
        "qualification": (
            "source/mode/typed-plan accounting and authored resolution declarations only; "
            "no legality, numerical, emission, or execution certification"
        ),
    }
