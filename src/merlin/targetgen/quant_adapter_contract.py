"""Project an external quantization adapter's contract from a target's quantization contract.

An external adapter (an installed ``m2m.quantization_adapters`` entry point) receives the bytes
built here and nothing else about the target. Every value is read from the authored software spec,
the Phase 0 quantization contract or the numeric-format registry, and computed where it follows from
them (finite maximum, minimum normal, exponent bias). A value none of them states is written as
``unknown`` with its reason, and any unknown makes the whole contract ``diagnostic``: an adapter
that relies on one must state its assumption in its policy rather than inherit a default from here.

This module does not quantize, round or choose policy. It is JSON in, canonical JSON out.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping

from merlin.common import quant_formats

SCHEMA = "merlin.quantization_adapter_contract.v1"
UNKNOWN = "unknown"
NOT_APPLICABLE = "not_applicable"
_IEEE_ROUNDING = frozenset({"rne", "rmm", "rtz", "rdn", "rup"})
_ACCELERATOR_PLACEMENTS = frozenset({"accelerator", "fused_accelerator"})
_DOMAIN_FIELDS = frozenset({"exponent_range", "signed_zero", "reserved_codes"})


def canonical_bytes(contract: Mapping) -> bytes:
    """The exact bytes an adapter receives and a policy's ``contract_sha256`` names."""
    text = json.dumps(contract, sort_keys=True, separators=(",", ":"), allow_nan=False, ensure_ascii=False)
    return (text + "\n").encode("utf-8")


def _contract_digest(contract: Mapping) -> str:
    body = {key: value for key, value in contract.items() if key != "sha256"}
    return hashlib.sha256(json.dumps(body, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _known(value) -> bool:
    return value is not None and value != UNKNOWN


def operand_domain(operand_dtype: str, semantics: Mapping) -> tuple[dict, dict[str, str]]:
    """The operand element domain of a floating format, and the fields no input establishes.

    ``exponent_range`` is the inclusive range of biased exponent fields the hardware treats as
    finite normals; ``reserved_codes`` are element codes it does not treat as finite values.
    The finite maximum is the largest code inside that range that is not reserved, so a range
    ending one binade lower, or a reserved top code, each give a different maximum.
    """
    fmt = quant_formats.get(operand_dtype)
    if not fmt.is_float:
        raise ValueError(f"adapter contract projection covers floating operand formats only, not {fmt.name!r}")
    exponent_bits, mantissa_bits = fmt.exp_bits, fmt.mant_bits
    bias = (1 << (exponent_bits - 1)) - 1
    internal = semantics.get("internal_arithmetic") or {}
    domain = internal.get("operand_domain") if isinstance(internal, Mapping) else None
    domain = domain if isinstance(domain, Mapping) else {}
    foreign = set(domain) - _DOMAIN_FIELDS
    if foreign:
        raise ValueError(f"operand_domain has fields this projection does not interpret: {sorted(foreign)}")
    unknowns: dict[str, str] = {}
    prefix = "numerical_semantics.internal_arithmetic.operand_domain"

    exponent_range = domain.get("exponent_range")
    if exponent_range is None:
        unknowns["operand.exponent_range"] = f"{prefix}.exponent_range is not declared"
    elif (
        not isinstance(exponent_range, list)
        or len(exponent_range) != 2
        or any(type(value) is not int for value in exponent_range)
        or not 0 <= exponent_range[0] <= exponent_range[1] < (1 << exponent_bits)
    ):
        raise ValueError(f"{prefix}.exponent_range must be two ascending {exponent_bits}-bit exponent fields")

    reserved = domain.get("reserved_codes")
    if reserved is None:
        unknowns["operand.reserved_codes"] = f"{prefix}.reserved_codes is not declared (declare [] for none)"
    elif (
        not isinstance(reserved, list)
        or any(type(code) is not int or not 0 <= code < (1 << fmt.element_bits) for code in reserved)
        or len(set(reserved)) != len(reserved)
    ):
        raise ValueError(f"{prefix}.reserved_codes must be distinct {fmt.element_bits}-bit element codes")

    finite_max = min_normal = UNKNOWN
    if exponent_range is not None and reserved is not None:
        low, high = max(exponent_range[0], 1), exponent_range[1]
        excluded = set(reserved)
        for exponent in range(high, low - 1, -1):
            mantissa = next(
                (m for m in range((1 << mantissa_bits) - 1, -1, -1) if (exponent << mantissa_bits) | m not in excluded),
                None,
            )
            if mantissa is not None:
                finite_max = math.ldexp((1 << mantissa_bits) + mantissa, exponent - bias - mantissa_bits)
                break
        if finite_max == UNKNOWN:
            raise ValueError("operand_domain reserves every normal code in its exponent range")
        min_normal = math.ldexp(1, low - bias)
    else:
        unknowns["operand.finite_max"] = "needs the declared exponent range and reserved codes"
        unknowns["operand.min_normal"] = "needs the declared exponent range and reserved codes"

    signed_zero = domain.get("signed_zero", UNKNOWN)
    if signed_zero == UNKNOWN:
        unknowns["operand.signed_zero"] = f"{prefix}.signed_zero is not declared"
    elif type(signed_zero) is not bool:
        raise ValueError(f"{prefix}.signed_zero must be Boolean")

    flush = semantics.get("subnormal_operand_flush", UNKNOWN)
    if type(flush) is not bool:
        flush = UNKNOWN
        unknowns["operand.subnormal_flush"] = "numerical_semantics.subnormal_operand_flush is not declared"

    rounding = semantics.get("operand_rounding", UNKNOWN)
    if rounding not in _IEEE_ROUNDING:
        rounding = UNKNOWN
        unknowns["operand.rounding"] = (
            "numerical_semantics.operand_rounding (conversion into the operand format) is not declared"
        )

    return (
        {
            "dtype": fmt.name,
            "exponent_bits": exponent_bits,
            "mantissa_bits": mantissa_bits,
            "bias": bias,
            "exponent_range": list(exponent_range) if exponent_range is not None else UNKNOWN,
            "finite_max": finite_max,
            "min_normal": min_normal,
            "reserved_codes": sorted(reserved) if reserved is not None else UNKNOWN,
            "subnormal_flush": flush,
            "signed_zero": signed_zero,
            "rounding": rounding,
        },
        unknowns,
    )


def _selected_parameters(entry: Mapping) -> tuple[dict, str | None, dict[str, str]]:
    """Parameters of the one usable hardware match, else the unselected projection."""
    matches = entry.get("hardware_matches") or []
    usable = [match for match in matches if match.get("status") != "incompatible"]
    unknowns: dict[str, str] = {}
    if len(usable) == 1:
        match = usable[0]
        parameters = dict(match.get("parameters") or {})
        # Parameter unknowns are reported field by field by the caller; the rest are the match's own.
        for name in match.get("unknowns") or []:
            if name.partition(".")[0] not in parameters:
                unknowns[f"hardware_match.{name}"] = f"unresolved for unit {match.get('unit')!r}"
        return parameters, match.get("unit"), unknowns
    if not matches:
        unknowns["hardware_match"] = "no selected hardware quantization candidate matches this format"
    elif not usable:
        conflicts = [conflict for match in matches for conflict in match.get("conflicts") or []]
        unknowns["hardware_match"] = "every hardware candidate conflicts: " + "; ".join(conflicts)
    else:
        units = sorted(str(match.get("unit")) for match in usable)
        unknowns["hardware_match"] = f"several hardware units match ({units}); the contract does not choose one"
    return dict(entry.get("unselected_parameters") or {}), None, unknowns


def _parameter(parameters: Mapping, name: str, path: str, unknowns: dict[str, str]):
    record = parameters.get(name)
    if not isinstance(record, Mapping) or record.get("status") == "unknown":
        basis = record.get("basis") if isinstance(record, Mapping) else "no parameter record"
        unknowns[path] = f"{name} is unknown ({basis})"
        return UNKNOWN
    if record.get("status") == "not_applicable":
        return NOT_APPLICABLE
    return record.get("value")


def _families(software_spec: Mapping, entry: Mapping) -> dict:
    operations = {
        row["id"]: row
        for row in software_spec.get("operations") or []
        if isinstance(row, Mapping) and isinstance(row.get("id"), str)
    }
    placed: dict[str, list] = {"accelerator": [], "host": [], "unplaced": []}
    for decision in entry.get("operation_eligibility") or []:
        identity = decision.get("operation_id")
        row = operations.get(identity) or {}
        record = {
            "operation": identity,
            "families": sorted(row.get("families") or []),
            "ops": sorted(row.get("ops") or []),
        }
        placement = decision.get("placement")
        if placement in _ACCELERATOR_PLACEMENTS and decision.get("declared_eligible") is True:
            placed["accelerator"].append({**record, "eligibility": decision.get("status")})
        elif placement == "host":
            placed["host"].append(record)
        elif placement not in _ACCELERATOR_PLACEMENTS:
            placed["unplaced"].append({**record, "placement": placement})
    for rows in placed.values():
        rows.sort(key=lambda row: str(row["operation"]))
    return placed


def build(
    software_spec: Mapping,
    quantization_contract: Mapping,
    *,
    format_id: str,
    spec_source: Mapping,
) -> dict:
    """The adapter contract for one declared format of a Phase 0 quantization contract.

    ``spec_source`` names the spec bytes the caller read (``path`` and ``sha256``); the
    quantization contract must carry its own content digest, which is rechecked here.
    """
    if quantization_contract.get("schema") != "merlin.phase0.quantization_contract.v1":
        raise ValueError("expected a merlin.phase0.quantization_contract.v1 document")
    if quantization_contract.get("sha256") != _contract_digest(quantization_contract):
        raise ValueError("quantization contract content differs from its recorded digest")
    target = quantization_contract.get("target")
    if not isinstance(target, str) or not target or software_spec.get("target") != target:
        raise ValueError("quantization contract and software spec name different targets")
    entries = [row for row in quantization_contract.get("formats") or [] if row.get("id") == format_id]
    if len(entries) != 1:
        known = sorted(str(row.get("id")) for row in quantization_contract.get("formats") or [])
        raise ValueError(f"quantization contract has no single format {format_id!r}; declared: {known}")
    entry = entries[0]
    declaration = entry.get("declaration") or {}
    semantics = entry.get("numerical_semantics") or {}
    if not isinstance(spec_source.get("path"), str) or not isinstance(spec_source.get("sha256"), str):
        raise ValueError("spec_source must name the read spec's path and sha256")

    operand, unknowns = operand_domain(declaration["operand_dtype"], semantics)
    parameters, unit, match_unknowns = _selected_parameters(entry)
    unknowns.update(match_unknowns)

    encoding = _parameter(parameters, "scale_encoding", "scale.encoding", unknowns)
    if _known(encoding) and encoding != NOT_APPLICABLE:
        encoding = quant_formats.get(encoding).name
    value_rule = declaration.get("scale_rule", semantics.get("scale_rule", UNKNOWN))
    if not _known(value_rule) or not isinstance(value_rule, str):
        value_rule = UNKNOWN
        unknowns["scale.value_rule"] = "no scale_rule is declared for this format or its numerical semantics"
    scale = {
        "encoding": encoding,
        "weight_granularity": _parameter(parameters, "weight_granularity", "scale.weight_granularity", unknowns),
        "activation_granularity": _parameter(
            parameters, "activation_granularity", "scale.activation_granularity", unknowns
        ),
        "activation_mode": _parameter(parameters, "activation_mode", "scale.activation_mode", unknowns),
        "block_size": _parameter(parameters, "block_size", "scale.block_size", unknowns),
        "value_rule": value_rule,
    }

    accumulate = {}
    for key, value, source in (
        ("dtype", declaration.get("accumulator_dtype"), "quantization format accumulator_dtype"),
        ("readout", semantics.get("readout_dtype"), "numerical_semantics.readout_dtype"),
        ("rounding", semantics.get("rounding"), "numerical_semantics.rounding"),
        ("order", semantics.get("reduction_order"), "numerical_semantics.reduction_order"),
        ("cadence", semantics.get("reduction_cadence"), "numerical_semantics.reduction_cadence"),
        ("unit", declaration.get("unit", unit), "quantization format unit or its single hardware match"),
    ):
        if _known(value):
            accumulate[key] = value
        else:
            accumulate[key] = UNKNOWN
            unknowns[f"accumulate.{key}"] = f"{source} is not declared"

    families = _families(software_spec, entry)
    if not families["accelerator"]:
        unknowns["families.accelerator"] = "no eligible accelerator operation is declared for this format"

    return {
        "schema": SCHEMA,
        "target": target,
        "status": "diagnostic" if unknowns else "derived",
        "source": {
            "software_spec": {"path": spec_source["path"], "sha256": spec_source["sha256"]},
            "software_review": software_spec.get("status", "unreviewed"),
            "quantization_contract_sha256": quantization_contract["sha256"],
            "format_id": format_id,
        },
        "operand": operand,
        "scale": scale,
        "accumulate": accumulate,
        "families": families,
        "unknowns": dict(sorted(unknowns.items())),
    }
