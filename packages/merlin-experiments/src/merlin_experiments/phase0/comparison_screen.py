"""Withdraw a comparison family whose unfused member the selected declarations refuse.

A comparison group measures a fused member against the standalone parts it replaces. When one of
those parts cannot exist on the target -- the declarations admit its operation only fused onto
another -- the group has nothing to compare against, and writing the part produces a capsule that
verified generation must refuse. The family is withdrawn at derivation instead, with the refusal
recorded, rather than emitted as a group that can never be completed.

A part is screened as what the group declares it to be: one operation, standalone, so it is observed
as composed with nothing and carrying no epilogue. Only a definite refusal withdraws the family; an
unresolved constraint is left to the written program's own screen.
"""

from __future__ import annotations

from typing import Any


def refused_part(variants: list[dict], base: dict, *, software_spec: dict | None, binding) -> dict | None:
    """The first ``part`` member every matching declaration refuses, with the decisions, or ``None``."""
    from merlin.targetgen.semantic_families import from_op
    from merlin.targetgen.software_spec import admit_operation

    if not software_spec:
        return None
    for variant in variants:
        group = variant.get("comparison_group")
        role = group.get("role") if isinstance(group, dict) else None
        op = variant.get("op") or base.get("op")
        family = from_op(op) if op else None
        if role != "part" or family is None:
            continue
        signature: dict[str, Any] = {
            "family": family,
            "operand_dtype": variant.get("operand_dtype") or base.get("operand_dtype") or binding.operand_dtype,
            "accum_dtype": binding.accum_dtype,
            "composed_with": [],
            "epilogues": [],
        }
        if family == "elementwise_map":
            # The unfused part of an epilogue fusion runs in the accumulator domain it would be fused in.
            signature["operand_dtype"] = binding.accum_dtype
        decisions = []
        for declaration in software_spec.get("operations") or []:
            decision = admit_operation(
                {**software_spec, "operations": [declaration]}, str(op), signature, declaration["placement"]
            )
            if "declaration" in decision:
                decisions.append(decision)
        if decisions and all(decision["status"] == "unsupported" for decision in decisions):
            return {
                "member": op,
                "family": family,
                "reason": "; ".join(sorted({decision["reason"] for decision in decisions})),
                "decisions": decisions,
            }
    return None
