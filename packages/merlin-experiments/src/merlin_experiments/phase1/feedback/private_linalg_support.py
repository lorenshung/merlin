"""Exact source and linked-build accounting for opt-in static Linalg host bodies.

The witness is source-structural only. It neither grants a host rule nor proves
the numerical behavior of a translated program. Per-tensor index/byte bounds
do not prove a global arena allocation or pointer-base address range.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from typing import Any

from merlin.common.digest import is_sha256
from merlin.frontends.linalg_boolean_patterns import (
    STATIC_BOOLEAN_SOURCE_BODY_SCHEMA,
    recognize_static_boolean_body,
    static_boolean_ordered_types,
    validate_serialized_static_boolean_pattern,
    validate_static_boolean_source_body,
)
from merlin.frontends.linalg_math_patterns import (
    STATIC_F32_MATH_SOURCE_BODY_SCHEMA,
    recognize_static_f32_math_body,
    static_f32_math_ordered_types,
    validate_static_f32_math_source_body,
)
from merlin.frontends.linalg_patterns import (
    STATIC_POINTWISE_SOURCE_BODY_SCHEMA,
    InvalidLinalgPattern,
    recognize_static_pointwise,
    static_pointwise_ordered_types,
    validate_static_pointwise_source_body,
)
from merlin_experiments.phase1.feedback import private_control_support as control_support

PENDING = "source_linalg_host_support_pending_build"
LINKED = "source_linalg_host_support_linked"
SCOPE = "exact source bodies, selected per-tensor index span and linked build; no numerical equivalence"
_TOP = {
    "status",
    "scope",
    "raw_source_sha256",
    "normalized_source_sha256",
    "n_source_operations",
    "selected_index_observation",
    "actual_index_observation",
    "count",
    "occurrences",
    "linked_build",
}
_OCCURRENCE = {
    "ordinal",
    "profile",
    "capability_spec_sha256",
    "declaration",
    "schema",
    "operation",
    "predicate",
    "shape",
    "ordered_types",
    "input_shapes",
    "input_maps",
}


def _contract(schema: object, operation: object, predicate: object) -> tuple[dict[str, str], Any, tuple[str, ...]]:
    """Resolve a closed versioned declaration without crossing schema namespaces."""
    if schema == STATIC_BOOLEAN_SOURCE_BODY_SCHEMA:
        if predicate is not None:
            raise InvalidLinalgPattern("Boolean source body does not declare a predicate")
        declaration = validate_static_boolean_source_body({"schema": schema, "operation": operation})
        return declaration, recognize_static_boolean_body, static_boolean_ordered_types(declaration)
    if schema == STATIC_POINTWISE_SOURCE_BODY_SCHEMA:
        raw = {"schema": schema, "operation": operation}
        if predicate is not None:
            raw["predicate"] = predicate
        declaration = validate_static_pointwise_source_body(raw)
        return declaration, recognize_static_pointwise, static_pointwise_ordered_types(declaration)
    if schema == STATIC_F32_MATH_SOURCE_BODY_SCHEMA:
        if predicate is not None:
            raise InvalidLinalgPattern("unary f32 math source body does not declare a predicate")
        declaration = validate_static_f32_math_source_body({"schema": schema, "operation": operation})
        return declaration, recognize_static_f32_math_body, static_f32_math_ordered_types(declaration)
    raise InvalidLinalgPattern("unknown static source-body schema")


def begin(
    raw_sha256: str,
    normalized_sha256: str,
    n_operations: int,
    selected_index_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Make a mandatory empty-or-populated witness for one current source."""
    if (
        not is_sha256(raw_sha256)
        or not is_sha256(normalized_sha256)
        or type(n_operations) is not int
        or n_operations < 1
    ):
        raise ValueError("static Linalg source identity is malformed")
    return {
        "status": PENDING,
        "scope": SCOPE,
        "raw_source_sha256": raw_sha256,
        "normalized_source_sha256": normalized_sha256,
        "n_source_operations": n_operations,
        "selected_index_observation": (
            deepcopy(dict(selected_index_observation)) if selected_index_observation else None
        ),
        "count": 0,
        "occurrences": [],
    }


def record(
    witness: dict[str, Any],
    row: Mapping[str, Any],
    admission: Mapping[str, Any],
    parsed: tuple[Any, ...],
    source_rows: Mapping[int, Mapping[str, Any]],
) -> None:
    """Recompute and retain every exact source ordinal of an admitted row."""
    body = admission.get("source_body_proof")
    if body is None:
        return
    ordinals = row.get("ordinals")
    patterns = body.get("patterns") if isinstance(body, Mapping) else None
    if (
        witness.get("status") != PENDING
        or not isinstance(body, Mapping)
        or set(body)
        != {"schema", "declaration", "operation", "predicate", "patterns", "profile", "capability_spec_sha256"}
        or admission.get("status") != "admitted"
        or admission.get("reviewed") is not True
        or not is_sha256(body.get("capability_spec_sha256"))
        or not isinstance(body.get("profile"), str)
        or not body["profile"]
        or not isinstance(body.get("declaration"), str)
        or not body["declaration"]
        or not isinstance(ordinals, list)
        or not isinstance(patterns, list)
        or type(row.get("count")) is not int
        or row["count"] != len(ordinals)
        or len(patterns) != len(ordinals)
    ):
        raise ValueError("static Linalg host admission has no exact source occurrence roster")
    try:
        _, recognizer, expected_types = _contract(body.get("schema"), body.get("operation"), body.get("predicate"))
    except InvalidLinalgPattern as exc:
        raise ValueError("static Linalg host admission has an invalid source-body contract") from exc
    seen = {item["ordinal"] for item in witness["occurrences"]}
    for ordinal, pattern in zip(ordinals, patterns, strict=True):
        try:
            actual = (
                asdict(recognizer(parsed[ordinal])) if type(ordinal) is int and 0 <= ordinal < len(parsed) else None
            )
        except InvalidLinalgPattern as exc:
            raise ValueError("static Linalg proof differs from parsed source") from exc
        if (
            type(ordinal) is not int
            or ordinal in seen
            or source_rows.get(ordinal) is not row
            or not isinstance(pattern, Mapping)
            or actual != pattern
            or pattern["operation"] != body["operation"]
            or pattern.get("predicate") != body.get("predicate")
            or tuple(pattern["ordered_types"]) != expected_types
        ):
            raise ValueError("static Linalg proof differs from parsed source")
        if body["schema"] == STATIC_BOOLEAN_SOURCE_BODY_SCHEMA:
            try:
                validate_serialized_static_boolean_pattern(pattern)
            except InvalidLinalgPattern as exc:
                raise ValueError("static Boolean proof has an invalid serialized source pattern") from exc
        witness["occurrences"].append(
            {
                "ordinal": ordinal,
                "profile": body["profile"],
                "capability_spec_sha256": body["capability_spec_sha256"],
                "declaration": body["declaration"],
                "schema": body["schema"],
                "operation": pattern["operation"],
                "predicate": pattern.get("predicate"),
                "shape": pattern["shape"],
                "ordered_types": pattern["ordered_types"],
                "input_shapes": pattern.get("input_shapes", ()),
                "input_maps": pattern.get("input_maps", ()),
            }
        )
        seen.add(ordinal)
    witness["occurrences"].sort(key=lambda item: item["ordinal"])
    witness["count"] = len(witness["occurrences"])


def link(source: Mapping[str, Any], actual_index: Mapping[str, Any], linked_build: Mapping[str, str]) -> None:
    witness = source.get("linalg_host_support")
    if not isinstance(witness, dict) or witness.get("status") != PENDING:
        raise ValueError("static Linalg proof is not pending the selected build")
    selected = source.get("selected_index_observation")
    if (
        witness.get("selected_index_observation") != selected
        or not control_support._record_matches_selected(actual_index, selected)  # noqa: PLC2701 -- shared verifier
    ):
        raise ValueError("static Linalg proof has no exact selected index-lowering build")
    if not _all_shapes_fit(witness.get("occurrences"), selected["index_bits"]):
        raise ValueError("static Linalg shape exceeds selected signed index address span")
    witness["status"] = LINKED
    witness["actual_index_observation"] = deepcopy(dict(actual_index))
    witness["linked_build"] = dict(linked_build)


def _all_shapes_fit(occurrences: object, index_bits: int) -> bool:
    """Bound every tensor extent and byte span by the selected signed index."""
    if not isinstance(occurrences, list) or type(index_bits) is not int or not 2 <= index_bits <= 128:
        return False
    maximum = (1 << (index_bits - 1)) - 1
    element_bytes = {"i1": 1, "f32": 4, "i64": 8}

    def fits(shape: object, dtype: object, *, scalar: bool = False) -> bool:
        if (
            not isinstance(shape, (list, tuple))
            or (not shape and not scalar)
            or not isinstance(dtype, str)
            or dtype not in element_bytes
        ):
            return False
        elements = 1
        for dim in shape:
            if type(dim) is not int or not 0 < dim <= maximum:
                return False
            if elements > maximum // dim:
                return False
            elements *= dim
        return elements <= maximum // element_bytes[dtype]

    for item in occurrences:
        if not isinstance(item, Mapping):
            return False
        shape, types, inputs = item.get("shape"), item.get("ordered_types"), item.get("input_shapes")
        if not isinstance(types, (list, tuple)) or len(types) < 2 or not fits(shape, types[-1]):
            return False
        if item.get("schema") == STATIC_BOOLEAN_SOURCE_BODY_SCHEMA:
            if not isinstance(inputs, (list, tuple)) or len(inputs) != len(types) - 2:
                return False
            if any(
                not fits(input_shape, dtype, scalar=True) for input_shape, dtype in zip(inputs, types[:-2], strict=True)
            ) or not fits(shape, types[-2]):
                return False
        elif any(not fits(shape, dtype) for dtype in types[:-1]):
            return False
    return True


def linked_complete(source: Mapping[str, Any], entry: Mapping[str, Any], candidate_sha256: str) -> bool:
    """Require the mandatory witness to cite the same linked program bytes."""
    proof = source.get("linalg_host_support")
    if (
        not isinstance(proof, Mapping)
        or set(proof) != _TOP
        or proof.get("status") != LINKED
        or proof.get("scope") != SCOPE
    ):
        return False
    occurrences = proof.get("occurrences")
    if (
        proof.get("raw_source_sha256") != source.get("source_sha256")
        or proof.get("normalized_source_sha256") != source.get("normalized_source_sha256")
        or not is_sha256(proof.get("raw_source_sha256"))
        or not is_sha256(proof.get("normalized_source_sha256"))
        or entry.get("source_sha256") != source.get("source_sha256")
        or type(proof.get("n_source_operations")) is not int
        or proof["n_source_operations"] < 1
        or proof["n_source_operations"] != source.get("n_source_operations")
        or not isinstance(occurrences, list)
        or type(proof.get("count")) is not int
        or proof["count"] != len(occurrences)
    ):
        return False
    selected = source.get("selected_index_observation")
    actual = proof.get("actual_index_observation")
    if (
        proof.get("selected_index_observation") != selected
        or actual != entry.get("index_lowering")
        or not control_support._record_matches_selected(actual, selected)  # noqa: PLC2701 -- shared verifier
        or not _all_shapes_fit(occurrences, selected["index_bits"])
    ):
        return False
    ordinals = []
    for item in occurrences:
        if not isinstance(item, Mapping) or set(item) != _OCCURRENCE:
            return False
        ordinal, shape, types, maps = item["ordinal"], item["shape"], item["ordered_types"], item["input_maps"]
        if (
            type(ordinal) is not int
            or not 0 <= ordinal < proof["n_source_operations"]
            or not isinstance(shape, (list, tuple))
            or not shape
            or any(type(dim) is not int or dim < 0 for dim in shape)
            or not isinstance(types, (list, tuple))
            or any(not isinstance(dtype, str) or not dtype for dtype in types)
            or not isinstance(maps, (list, tuple))
            or any(not isinstance(mapping, str) or not mapping for mapping in maps)
            or not isinstance(item["input_shapes"], (list, tuple))
            or not isinstance(item["declaration"], str)
            or not item["declaration"]
            or not isinstance(item["profile"], str)
            or not item["profile"]
            or not is_sha256(item["capability_spec_sha256"])
        ):
            return False
        try:
            _, _, expected_types = _contract(item["schema"], item["operation"], item["predicate"])
        except InvalidLinalgPattern:
            return False
        if tuple(types) != expected_types:
            return False
        if item["schema"] in {STATIC_POINTWISE_SOURCE_BODY_SCHEMA, STATIC_F32_MATH_SOURCE_BODY_SCHEMA} and (
            maps or item["input_shapes"]
        ):
            return False
        if item["schema"] == STATIC_BOOLEAN_SOURCE_BODY_SCHEMA:
            try:
                validate_serialized_static_boolean_pattern(
                    {key: item[key] for key in ("operation", "shape", "ordered_types", "input_shapes", "input_maps")}
                )
            except InvalidLinalgPattern:
                return False
        ordinals.append(ordinal)
    if ordinals != sorted(set(ordinals)):
        return False
    linked = proof.get("linked_build")
    return bool(
        isinstance(linked, Mapping)
        and set(linked) == {"candidate_tree_sha256", "capture_tree_sha256", "elf_sha256"}
        and all(is_sha256(linked[key]) for key in linked)
        and linked["candidate_tree_sha256"] == candidate_sha256
        and linked["capture_tree_sha256"] == entry.get("capture_tree_sha256")
        and linked["elf_sha256"] == entry.get("elf_sha256")
    )
