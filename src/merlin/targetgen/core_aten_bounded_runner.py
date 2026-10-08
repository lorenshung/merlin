"""Resumable bounded case adapter using the existing capture and byte grader.

The byte ABI does not expose the full semantic observation protocol. A numeric
success is therefore explicitly ungradable, with the numeric verdict retained.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from merlin.targetgen.core_aten_capture import loader_source


def atomic_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def case_digest(case: dict) -> str:
    return hashlib.sha256(json.dumps(case, sort_keys=True).encode()).hexdigest()


def bundle_admission_reason(capture: dict, mlir_source: str | None = None) -> str | None:
    """Expose the native batch admission error before its empty-bundle wrapper."""
    from merlin.targetgen.core_aten_batch import _function_parts, _tensor_type

    meta = capture.get("capture_meta", {})
    try:
        for abi in [*meta.get("input_abi", []), *meta.get("output_abi", [])]:
            _tensor_type(abi)
        if mlir_source is not None:
            names, _, returned, _, _ = _function_parts(mlir_source)
            if len(names) != len(meta.get("input_abi", [])):
                raise ValueError("captured argument ABI or input count mismatch")
            contract = meta.get("result_contract")
            result_abi = contract["results"] if contract is not None else meta.get("output_abi", [])
            if len(returned) != len(result_abi):
                raise ValueError("captured result ABI count mismatch")
    except (KeyError, TypeError, ValueError) as exc:
        return f"{type(exc).__name__}: {exc}"
    return None


def bounded_loader_source(case: dict) -> str:
    """Retain recorded input layouts and seed in the standalone capture loader."""
    source = loader_source(case)
    old = 'return torch.tensor(_decode(value["values"]), dtype=dtype).reshape(value["shape"])'
    new = """logical = torch.tensor(_decode(value["values"]), dtype=dtype).reshape(value["shape"])
        shape, stride = value["shape"], value["stride"]
        offset = int(value.get("storage_offset", 0))
        size = offset if any(n == 0 for n in shape) else offset + sum((n - 1) * s for n, s in zip(shape, stride)) + 1
        storage = torch.empty(size, dtype=dtype)
        tensor = storage.as_strided(shape, stride, offset)
        if torch._debug_has_internal_overlap(tensor) == 0:
            tensor.copy_(logical)
        else:
            import itertools
            for index in itertools.product(*(range(n) for n in shape)):
                storage[offset + sum(i * s for i, s in zip(index, stride))] = logical[index]
        tensor.requires_grad_(bool(value.get("requires_grad", False)))
        return tensor"""
    if old not in source:
        raise ValueError("capture loader tensor decoder changed")
    source = source.replace(old, new)
    source = source.replace(
        'complex(value["real"], value["imag"])', 'complex(_decode(value["real"]), _decode(value["imag"]))'
    )
    source = source.replace(
        'arguments = _decode(_DOCUMENT["arguments"], inputs)',
        f'torch.manual_seed({int(case["rng_seed"])})\n        arguments = _decode(_DOCUMENT["arguments"], inputs)',
    )
    return source


def semantic_observation(verdict: dict, evidence: dict) -> dict:
    status = verdict["status"]
    reason = verdict.get("reason") or json.dumps(verdict.get("results", []), sort_keys=True)
    if status == "pass":
        status = "ungradable"
        reason = "numeric output passes; byte readback cannot observe post-arguments, mutations, aliases, or full output metadata"
    return {"status": status, "reason": reason, "evidence": evidence}


def grade_portable_outputs(batch: dict, output_bytes: list[bytes]) -> dict:
    """Decode portable nonfinite leaves, then use the unmodified byte comparator.

    The sealed case and its digest stay untouched. Only the transient numeric
    oracle representation is decoded, exactly as the portable value codec does.
    """
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    def decode(value):
        if isinstance(value, dict):
            if value.get("kind") == "float":
                return float(value["value"])
            return {key: decode(child) for key, child in value.items()}
        if isinstance(value, list):
            return [decode(child) for child in value]
        return value

    transient = copy.deepcopy(batch)
    for record in transient["cases"]:
        record["case"]["expected"] = decode(record["case"]["expected"])
    return grade_core_aten_batch(transient, output_bytes)


class SavedResponseAdapter:
    """The same response document accepted by JsonCommandSuiteAdapter."""

    name = "bounded-spike-byte-adapter"

    def __init__(self, response: dict):
        self.response = response

    def execute(self, corpus):
        return self.response["results"]
