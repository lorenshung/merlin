"""Grade every Core ATen batch result against the canonical case output documents."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

_DTYPES = {
    "f32": np.float32,
    "f64": np.float64,
    "f16": np.float16,
    "i64": np.int64,
    "i32": np.int32,
    "i16": np.int16,
    "i8": np.int8,
    "i1": np.uint8,
}
_EXPECTED_DTYPES = {
    "f32": "float32",
    "f64": "float64",
    "f16": "float16",
    "i64": "int64",
    "i32": "int32",
    "i16": "int16",
    "i8": "int8",
    "i1": "bool",
}


def _complex_values(value: Any) -> Any:
    if isinstance(value, Mapping) and value.get("kind") == "float":
        return float(value["value"])
    if isinstance(value, Mapping) and value.get("kind") == "complex":
        return complex(_complex_values(value["real"]), _complex_values(value["imag"]))
    if isinstance(value, list):
        return [_complex_values(item) for item in value]
    return value


def _leaves(document: Any) -> list[Any]:
    if isinstance(document, Mapping):
        if document.get("kind") == "tuple":
            return [leaf for item in document["items"] for leaf in _leaves(item)]
        if document.get("kind") in ("tensor", "float"):
            return [document]
        return [leaf for _, item in sorted(document.items()) for leaf in _leaves(item)]
    if isinstance(document, list):
        return [leaf for item in document for leaf in _leaves(item)]
    return [document]


def _grade_output(raw: bytes, abi: Mapping[str, Any], expected: Any, case: Mapping[str, Any]) -> dict[str, Any]:
    dtype = str(abi["dtype"])
    shape = tuple(int(dim) for dim in abi["shape"])
    if dtype in {"complex<f32>", "complex<f64>"}:
        complex_dtype = np.complex64 if dtype == "complex<f32>" else np.complex128
        count = int(np.prod(shape)) if shape else 1
        nbytes = count * np.dtype(complex_dtype).itemsize
        if len(raw) != nbytes:
            return {"status": "mismatch", "reason": f"complex result has {len(raw)} bytes; expected {nbytes}"}
        actual = np.frombuffer(raw, dtype=complex_dtype).reshape(shape)
        if not isinstance(expected, Mapping) or expected.get("kind") != "tensor":
            return {"status": "ungradable", "reason": "complex result lacks a canonical tensor document"}
        if expected.get("dtype") != np.dtype(complex_dtype).name or tuple(expected.get("shape", ())) != shape:
            return {"status": "mismatch", "reason": "output dtype or shape differs from canonical complex result"}
        reference = np.asarray(_complex_values(expected.get("values")), dtype=complex_dtype).reshape(shape)
        parameters = case.get("comparison_parameters") or {}
        equal = np.isclose(
            actual,
            reference,
            rtol=float(parameters.get("rtol", 0.0)),
            atol=float(parameters.get("atol", 0.0)),
            equal_nan=bool(parameters.get("equal_nan", False)),
        )
        if bool(np.all(equal)):
            return {"status": "pass", "elements": count}
        first = tuple(int(index) for index in np.argwhere(~equal)[0])
        return {
            "status": "mismatch",
            "elements": count,
            "mismatched_elements": int(np.size(equal) - np.count_nonzero(equal)),
            "first_index": list(first),
            "actual": str(np.asarray(actual[first]).item()),
            "expected": str(np.asarray(reference[first]).item()),
        }
    if dtype not in _DTYPES:
        return {"status": "ungradable", "reason": f"unsupported result ABI dtype {dtype!r}"}
    count = int(np.prod(shape)) if shape else 1
    actual = np.frombuffer(raw, dtype=_DTYPES[dtype])
    if len(actual) != count:
        return {
            "status": "mismatch",
            "reason": f"result has {len(raw)} bytes; expected {count * np.dtype(_DTYPES[dtype]).itemsize}",
        }
    actual = actual.reshape(shape)
    if isinstance(expected, Mapping) and expected.get("kind") == "tensor":
        if expected.get("dtype") != _EXPECTED_DTYPES[dtype] or tuple(expected.get("shape", ())) != shape:
            return {"status": "mismatch", "reason": "output dtype or shape differs from canonical result"}
        if "values" not in expected:
            if case.get("comparison") != "metadata":
                return {"status": "ungradable", "reason": "canonical case has no numeric expected values"}
            stride = 1
            contiguous_strides = [0] * len(shape)
            for axis in reversed(range(len(shape))):
                contiguous_strides[axis] = stride
                stride *= shape[axis]
            if list(expected.get("stride", ())) != contiguous_strides:
                return {
                    "status": "mismatch",
                    "reason": "output stride differs from contiguous runtime descriptor",
                    "actual_stride": contiguous_strides,
                    "expected_stride": expected.get("stride"),
                }
            if expected.get("requires_grad") is not False:
                return {"status": "mismatch", "reason": "bare-metal output has no autograd state"}
            return {
                "status": "pass",
                "comparison": "metadata",
                "evidence": {
                    "dtype": _EXPECTED_DTYPES[dtype],
                    "shape": list(shape),
                    "stride": contiguous_strides,
                    "requires_grad": False,
                    "descriptor_layout": "row_major_contiguous",
                    "values": "unspecified_by_case",
                },
            }
        expected_value = expected["values"]
    elif isinstance(expected, Mapping) and expected.get("kind") == "float":
        expected_value = float(expected["value"])
    else:
        expected_value = expected
    try:
        reference = np.asarray(_complex_values(expected_value), dtype=_DTYPES[dtype]).reshape(shape)
    except (TypeError, ValueError) as exc:
        return {"status": "ungradable", "reason": f"cannot decode canonical expected value: {exc}"}
    parameters = case.get("comparison_parameters") or {}
    if dtype.startswith("f"):
        rtol = float(parameters.get("rtol", 0.0))
        atol = float(parameters.get("atol", 0.0))
        equal = np.isclose(actual, reference, rtol=rtol, atol=atol, equal_nan=bool(parameters.get("equal_nan", False)))
    else:
        equal = np.equal(actual, reference)
    if bool(np.all(equal)):
        return {"status": "pass", "elements": count}
    first = tuple(int(index) for index in np.argwhere(~equal)[0])
    return {
        "status": "mismatch",
        "elements": count,
        "mismatched_elements": int(np.size(equal) - np.count_nonzero(equal)),
        "first_index": list(first),
        "actual": np.asarray(actual[first]).item(),
        "expected": np.asarray(reference[first]).item(),
    }


def grade_core_aten_batch(
    batch_map: Mapping[str, Any],
    output_bytes: Sequence[bytes] | None = None,
    *,
    execution_error: str | None = None,
    output_shapes: Sequence[Sequence[int]] | None = None,
    semantic_readback: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a complete pass/mismatch/unavailable ledger, including noncaptured cases.

    Lossless bytes alone earn numeric-only passes. Exported boundary metadata,
    pre-call snapshots and pointer alias frames additionally judge the full corpus contract.
    """
    verdicts: dict[str, dict[str, Any]] = {}
    for record in batch_map["cases"]:
        overload = str(record.get("case_id") or record["overload"])
        if record["status"] != "bundled":
            verdicts[overload] = {"status": record["status"], "reason": record.get("reason", "")}
            continue
        if execution_error is not None or output_bytes is None:
            verdicts[overload] = {"status": "execution_failed", "reason": execution_error or "no hardware output"}
            continue
        indices = record["output_indices"]
        abis = record["output_abi"]
        expected = _leaves(record["case"]["expected"])
        if len(indices) != len(abis) or len(expected) != len(abis):
            verdicts[overload] = {
                "status": "ungradable",
                "reason": "canonical result structure differs from captured result ABI",
            }
            continue
        if any(index >= len(output_bytes) for index in indices):
            verdicts[overload] = {"status": "execution_failed", "reason": "hardware did not emit all result indices"}
            continue
        details = []
        for index, abi, item in zip(indices, abis, expected):
            if output_shapes is not None:
                shape = list(output_shapes[index])
                declared = abi.get("declared_shape", abi["shape"])
                if len(shape) != len(declared) or any(d >= 0 and d != s for d, s in zip(declared, shape)):
                    details.append({"status": "mismatch", "reason": "runtime shape differs from declared result"})
                    continue
                abi = {**abi, "shape": shape}
            elif any(d < 0 for d in abi.get("declared_shape", abi["shape"])):
                details.append({"status": "ungradable", "reason": "dynamic result lacks runtime extent evidence"})
                continue
            details.append(_grade_output(output_bytes[index], abi, item, record["case"]))
        if all(item["status"] == "pass" for item in details):
            status = "pass"
        elif any(item["status"] == "mismatch" for item in details):
            status = "mismatch"
        else:
            status = "ungradable"
        verdicts[overload] = {
            "status": status,
            "results": details,
            "output_indices": indices,
            "semantic_scope": "numeric_only",
        }
        from merlin.targetgen.core_aten_semantics import validate_full

        full = validate_full(record, output_bytes, semantic_readback)
        if full is not None:
            verdicts[overload].update(full)
    for record in batch_map["cases"]:
        identifier = str(record.get("case_id") or record["overload"])
        verdict = verdicts[identifier]
        routing = dict(record.get("routing") or {"lane": "host", "routed": False, "reason": "scalar mode"})
        verdict.update(
            overload=record["overload"],
            routing=routing,
            routing_reason=routing.get("reason", ""),
            **{key: value for key, value in routing.items() if key != "reason"},
        )
        if record.get("source_preparation"):
            verdict["source_preparation"] = record["source_preparation"]
        if routing.get("lane") == "device" and record["status"] == "bundled":
            evidence = routing.get("execution_evidence") or {}
            count = evidence.get("executed_instructions")
            from merlin.targetgen.core_aten_device import verify_execution_evidence

            if (
                routing.get("routed") is not True
                or not routing.get("routed_operations")
                or sum(case["status"] == "bundled" for case in batch_map["cases"]) != 1
                or evidence.get("case_id") != identifier
                or evidence.get("status") != "measured"
                or evidence.get("attribution") != "single_case_shard"
                or type(count) is not int
                or count < 1
                or evidence.get("target") != routing.get("target")
                or not verify_execution_evidence(evidence)
            ):
                verdict.update(status="execution_failed", reason="device execution evidence missing or empty")
    counts: dict[str, int] = {}
    for item in verdicts.values():
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    from merlin.targetgen.core_aten_provenance import batch_provenance

    return {
        "provenance": batch_map.get("provenance") or batch_provenance(),
        "schema_version": 1,
        "case_count": len(verdicts),
        "passed_count": counts.get("pass", 0),
        "status_counts": dict(sorted(counts.items())),
        "cases": verdicts,
        "passed_by_lane": {
            lane: sum(v["status"] == "pass" and v["lane"] == lane for v in verdicts.values())
            for lane in ("host", "device")
        },
        "full_semantic_passed_count": sum(
            v["status"] == "pass" and v.get("semantic_scope") == "full" for v in verdicts.values()
        ),
        "numeric_only_passed_count": sum(
            v["status"] == "pass" and v.get("semantic_scope") != "full" for v in verdicts.values()
        ),
        "scope": "full exported call contract when boundary readback is present; otherwise numeric only",
    }
