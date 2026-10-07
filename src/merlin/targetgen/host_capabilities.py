"""Per-operation host support declarations beside an immutable compiler package.

A package precision lane is not an operation-support manifest. These declarations
are separately selected and digest-bound; no compiler payload is rewritten.
"""

from __future__ import annotations

import copy

from merlin.common.digest import is_sha256
from merlin.targetgen.host_linkage_contract import (
    required_composite_math_symbol,
    validate_linkage_contract,
    validate_source_linkage_contract,
)
from merlin.targetgen.software_spec import admit_operation, validate_quantization_parameters

SCHEMA = "merlin.host_capabilities.v1"


def validate_host_capabilities(
    document: dict, *, package_sha256: str | None = None, dtype_strategy: str | None = None
) -> dict:
    if not isinstance(document, dict) or document.get("schema") != SCHEMA:
        raise ValueError(f"host capability spec must declare schema {SCHEMA}")
    if document.get("status") not in {"reviewed", "unreviewed"}:
        raise ValueError("host capability spec must declare review status")
    compiler = document.get("compiler")
    if not isinstance(compiler, dict) or (
        not is_sha256(compiler.get("package_sha256"))
        and not (document["status"] == "unreviewed" and compiler.get("package_sha256") is None)
    ):
        raise ValueError("host capability spec requires an exact compiler package SHA256")
    if not isinstance(compiler.get("dtype_strategy"), str) or not compiler["dtype_strategy"]:
        raise ValueError("host capability spec requires an explicit precision lane")
    if (
        package_sha256 is not None
        and compiler["package_sha256"] is not None
        and compiler["package_sha256"] != package_sha256
    ):
        raise ValueError("host capability spec is bound to a different compiler package")
    if dtype_strategy is not None and compiler["dtype_strategy"] != dtype_strategy:
        raise ValueError("host capability spec is bound to a different precision lane")
    operations = document.get("operations")
    if not isinstance(operations, list):
        raise ValueError("host capability spec operations must be a list")
    seen = set()
    for row in operations:
        if not isinstance(row, dict) or not isinstance(row.get("id"), str) or not row["id"] or row["id"] in seen:
            raise ValueError("host capability operations require unique IDs")
        seen.add(row["id"])
        if row.get("status", "reviewed") not in {"reviewed", "unreviewed"}:
            raise ValueError(f"host capability {row['id']} has an invalid review status")
        if row.get("placement") != "host" or not isinstance(row.get("signature"), dict) or not row["signature"]:
            raise ValueError("host capability operations require host placement and typed signature constraints")
        if "source_body" in row["signature"]:
            raise ValueError("source_body must be an explicit host declaration field, not a signature constraint")
        if "linkage_contract" in row:
            body = row.get("source_body")
            validate_source_linkage_contract(
                body.get("schema") if isinstance(body, dict) else None,
                body.get("operation") if isinstance(body, dict) else None,
                row["linkage_contract"],
            )
        if "source_body" in row:
            from merlin.frontends.linalg_boolean_patterns import (
                DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA,
                STATIC_BOOLEAN_SOURCE_BODY_SCHEMA,
                validate_dynamic_boolean_cast_source_body,
                validate_static_boolean_source_body,
            )
            from merlin.frontends.linalg_composite_math import (
                STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA,
                validate_static_composite_math_source_body,
            )
            from merlin.frontends.linalg_math_patterns import (
                STATIC_F32_MATH_SOURCE_BODY_SCHEMA,
                validate_static_f32_math_source_body,
            )
            from merlin.frontends.linalg_patterns import (
                STATIC_PROJECTED_POINTWISE_BODY_SCHEMA,
                validate_static_pointwise_source_body,
                validate_static_projected_pointwise_source_body,
            )

            allowed = {
                "id",
                "ops",
                "families",
                "family",
                "placement",
                "signature",
                "status",
                "numerical_contract",
                "evidence",
                "description",
                "source_body",
                "linkage_contract",
            }
            if set(row) - allowed or not (row.get("ops") or row.get("families") or row.get("family")):
                raise ValueError(f"host capability {row['id']} has unsupported fields or no selector")
            body = row["source_body"]
            if isinstance(body, dict) and body.get("schema") == STATIC_BOOLEAN_SOURCE_BODY_SCHEMA:
                validate_static_boolean_source_body(body)
            elif isinstance(body, dict) and body.get("schema") == DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA:
                validate_dynamic_boolean_cast_source_body(body)
            elif isinstance(body, dict) and body.get("schema") == STATIC_F32_MATH_SOURCE_BODY_SCHEMA:
                validate_static_f32_math_source_body(body)
            elif isinstance(body, dict) and body.get("schema") == STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA:
                validate_static_composite_math_source_body(body)
                if required_composite_math_symbol(body["schema"], body["operation"]) and "linkage_contract" not in row:
                    raise ValueError("composite math source requires an exact selected linkage contract")
            elif isinstance(body, dict) and body.get("schema") == STATIC_PROJECTED_POINTWISE_BODY_SCHEMA:
                validate_static_projected_pointwise_source_body(body)
            else:
                validate_static_pointwise_source_body(body)
        if "quantization_parameters" in row["signature"]:
            validate_quantization_parameters(
                row["signature"]["quantization_parameters"], source=f"host capability {row['id']!r}"
            )
        for selector in ("ops", "families"):
            if selector in row and (
                not isinstance(row[selector], list)
                or any(not isinstance(value, str) or not value for value in row[selector])
            ):
                raise ValueError(f"host capability {selector} must be explicit string lists")
    if not isinstance(document.get("evidence"), dict):
        raise ValueError("host capability spec must record evidence and unresolved obligations")
    return document


def _screen_source_body(declaration: dict, row: dict, signature: dict, source_operations: tuple | None) -> dict:
    """Match every supplied parsed occurrence; never treat an omitted source as evidence."""
    from dataclasses import asdict

    from merlin.frontends.linalg_boolean_patterns import (
        DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA,
        STATIC_BOOLEAN_SOURCE_BODY_SCHEMA,
        recognize_dynamic_boolean_cast_body,
        recognize_static_boolean_body,
    )
    from merlin.frontends.linalg_composite_math import (
        STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA,
        recognize_static_composite_math_body,
    )
    from merlin.frontends.linalg_math_patterns import (
        STATIC_F32_MATH_SOURCE_BODY_SCHEMA,
        recognize_static_f32_math_body,
    )
    from merlin.frontends.linalg_patterns import (
        STATIC_PROJECTED_POINTWISE_BODY_SCHEMA,
        InvalidLinalgPattern,
        recognize_static_pointwise,
        recognize_static_projected_pointwise,
    )

    if source_operations is None or not source_operations:
        return {"status": "unknown", "reason": "source_body requires parsed source operations"}
    if (
        not isinstance(source_operations, tuple)
        or ("count" in row and (type(row["count"]) is not int or row["count"] != len(source_operations)))
        or row.get("mlir_operation") != "linalg.generic"
        or len({id(op) for op in source_operations}) != len(source_operations)
    ):
        return {"status": "unsupported", "reason": "source_body occurrence roster differs from source row"}
    try:
        body = declaration["source_body"]
        recognizer = {
            STATIC_BOOLEAN_SOURCE_BODY_SCHEMA: recognize_static_boolean_body,
            DYNAMIC_BOOLEAN_CAST_SOURCE_BODY_SCHEMA: recognize_dynamic_boolean_cast_body,
            STATIC_F32_MATH_SOURCE_BODY_SCHEMA: recognize_static_f32_math_body,
            STATIC_COMPOSITE_MATH_SOURCE_BODY_SCHEMA: recognize_static_composite_math_body,
            STATIC_PROJECTED_POINTWISE_BODY_SCHEMA: recognize_static_projected_pointwise,
        }.get(body["schema"], recognize_static_pointwise)
        patterns = tuple(recognizer(op) for op in source_operations)
    except InvalidLinalgPattern as exc:
        return {"status": "unsupported", "reason": f"source_body structural proof refused: {exc}"}
    if any(
        pattern.operation != body["operation"] or getattr(pattern, "predicate", None) != body.get("predicate")
        for pattern in patterns
    ):
        return {"status": "unsupported", "reason": "source_body operation or predicate differs"}
    operands, results, rank = (
        signature.get("ordered_operand_dtypes"),
        signature.get("ordered_result_dtypes"),
        signature.get("rank"),
    )
    if not isinstance(operands, list) or not isinstance(results, list) or type(rank) is not int:
        return {"status": "unknown", "reason": "source_body has no complete observed typed signature"}
    if any(
        operands != list(pattern.ordered_types[:-1])
        or results != [pattern.ordered_types[-1]]
        or rank != len(pattern.shape)
        for pattern in patterns
    ):
        return {"status": "unsupported", "reason": "source_body differs from observed typed signature"}
    return {
        "status": "admitted",
        "reason": "every parsed occurrence matches the declared static source body",
        "proof": {
            "schema": body["schema"],
            "declaration": declaration["id"],
            "operation": body["operation"],
            "predicate": body.get("predicate"),
            "patterns": [asdict(pattern) for pattern in patterns],
        },
    }


def admit_host_operation(
    selected: dict | None, row: dict, signature: dict, *, source_operations: tuple | None = None
) -> dict:
    """Screen every selected host profile independently from accelerator admission."""
    if selected is None:
        return {
            "status": "unknown",
            "review_status": "unknown",
            "reviewed": False,
            "reason": "no selected per-operation host capability spec",
            "profiles": [],
        }
    if not isinstance(selected, dict):
        raise ValueError("selected host capabilities must be a profile mapping")
    profiles = []
    for name, selection in sorted(selected.items()):
        if not isinstance(selection, dict):
            raise ValueError("selected host capability profile must be a mapping")
        document = selection.get("capability_spec")
        if document is None:
            profiles.append(
                {
                    "profile": name,
                    "status": "unknown",
                    "reviewed": False,
                    "review_status": "unknown",
                    "reason": "host profile has no operation capability declaration",
                }
            )
            continue
        validate_host_capabilities(
            document, package_sha256=selection.get("package_sha256"), dtype_strategy=selection.get("dtype_strategy")
        )
        pinned = (
            is_sha256(selection.get("package_sha256"))
            and is_sha256(selection.get("capability_spec_sha256"))
            and document["compiler"].get("package_sha256") == selection["package_sha256"]
        )
        decisions = []
        for declaration in document["operations"]:
            # A named selector is narrower than a semantic family. A package
            # schedule matching linalg.matmul must not admit an unrelated
            # linalg.generic merely because both describe contractions. Family
            # selectors remain available when no exact ops were declared.
            identities = (row.get("frontend_op"), row["mlir_operation"])
            exact_ops = declaration.get("ops") or []
            if exact_ops and not any(identity in exact_ops for identity in identities):
                continue
            operation = next(
                (identity for identity in identities if identity in exact_ops),
                row["mlir_operation"],
            )
            decision = admit_operation({**document, "operations": [declaration]}, operation, signature, "host")
            if "declaration" in decision:
                if "source_body" in declaration:
                    screened = _screen_source_body(declaration, row, signature, source_operations)
                    if decision["status"] == "admitted" or source_operations is None:
                        decision = {**decision, "status": screened["status"], "reason": screened["reason"]}
                        if screened.get("proof") is not None:
                            decision["source_body_proof"] = screened["proof"]
                            if "linkage_contract" in declaration and decision["status"] == "admitted":
                                decision["linkage_requirement"] = validate_linkage_contract(
                                    declaration["linkage_contract"]
                                )
                decisions.append(decision)
        verdict = next(
            (decision for decision in decisions if decision["status"] == "admitted"),
            next((decision for decision in decisions if decision["status"] == "unknown"), None),
        )
        status = verdict["status"] if verdict else "unsupported"
        reason = verdict["reason"] if verdict else "no host operation declaration admits the observed signature"
        if (
            document["status"] != "reviewed" or (verdict and verdict["review_status"] != "reviewed")
        ) and status == "unsupported":
            status, reason = "unknown", "unreviewed host declarations cannot establish absence of operation support"
        if not pinned:
            status, reason = "unknown", "host package and capability spec byte identities are not selected together"
        profiles.append(
            {
                "profile": name,
                "status": status,
                "reason": reason,
                "review_status": verdict["review_status"] if verdict else document["status"],
                "reviewed": document["status"] == "reviewed"
                and pinned
                and (verdict is None or verdict["review_status"] == "reviewed"),
                "package_sha256": selection.get("package_sha256"),
                "capability_spec_sha256": selection.get("capability_spec_sha256"),
                "dtype_strategy": selection.get("dtype_strategy"),
                "decisions": decisions,
                **(
                    {"source_body_proof": verdict["source_body_proof"]}
                    if verdict and "source_body_proof" in verdict
                    else {}
                ),
                **(
                    {"linkage_requirement": verdict["linkage_requirement"]}
                    if verdict and "linkage_requirement" in verdict and status == "admitted"
                    else {}
                ),
            }
        )
    verdict = next(
        (profile for profile in profiles if profile["status"] == "admitted"),
        next((profile for profile in profiles if profile["status"] == "unknown"), None),
    )
    if verdict is None:
        verdict = {
            "status": "unsupported" if profiles else "unknown",
            "review_status": "unknown",
            "reviewed": False,
            "reason": "no selected host profile admits the observed operation signature",
        }
    return {
        **copy.deepcopy({key: verdict[key] for key in ("status", "review_status", "reviewed", "reason")}),
        "profiles": profiles,
        "qualification": "selected declaration screen; host lowering remains unverified",
        **(
            {
                "source_body_proof": {
                    **verdict["source_body_proof"],
                    "profile": verdict["profile"],
                    "capability_spec_sha256": verdict["capability_spec_sha256"],
                }
            }
            if "source_body_proof" in verdict
            else {}
        ),
        **(
            {
                "linkage_requirement": {
                    "profile": verdict["profile"],
                    "capability_spec_sha256": verdict["capability_spec_sha256"],
                    "declaration": verdict["source_body_proof"]["declaration"],
                    "contract": copy.deepcopy(verdict["linkage_requirement"]),
                }
            }
            if verdict.get("status") == "admitted"
            and verdict.get("reviewed") is True
            and "source_body_proof" in verdict
            and "linkage_requirement" in verdict
            else {}
        ),
    }
