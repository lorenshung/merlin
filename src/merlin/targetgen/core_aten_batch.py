"""Build one runnable MLIR bundle from independently captured Core ATen cases.

The corpus supplies the denominator and expected documents. A case is included whenever
its capture has a clean Linalg module, regardless of capture provenance labels. Cases
without such a module remain explicit unavailable records in the result map.
"""

from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path
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
    "i1": np.bool_,
}


def _json_bytes(document: Any) -> bytes:
    return (json.dumps(document, indent=2, sort_keys=True, allow_nan=True) + "\n").encode("utf-8")


def _tensor_type(abi: dict[str, Any]) -> str:
    shape = [*abi["shape"]]
    dtype = abi["dtype"]
    if dtype == "complex<f32>":
        shape.append(2)
        dtype = "f32"
    if dtype not in _DTYPES:
        raise ValueError(f"unsupported batch ABI dtype {dtype!r}")
    if any(not isinstance(dim, int) or dim < 0 for dim in shape):
        raise ValueError(f"unsupported batch ABI shape {shape!r}")
    return "tensor<" + "".join(f"{dim}x" for dim in shape) + dtype + ">"


def _tensor_documents(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, dict) and value.get("kind") == "tensor":
        return [value]
    if isinstance(value, dict):
        return [item for child in value.values() for item in _tensor_documents(child)]
    if isinstance(value, list):
        return [item for child in value for item in _tensor_documents(child)]
    return []


def _complex_value(value: Any) -> Any:
    if isinstance(value, dict) and value.get("kind") == "complex":
        return complex(value["real"], value["imag"])
    if isinstance(value, list):
        return [_complex_value(item) for item in value]
    return value


def _physical_abi(abi: dict[str, Any]) -> dict[str, Any]:
    if abi["dtype"] == "complex<f32>":
        return {"dtype": "f32", "shape": [*abi["shape"], 2]}
    return {"dtype": abi["dtype"], "shape": [*abi["shape"]]}


def _matching_brace(source: str, start: int) -> int:
    depth = 0
    quoted = False
    escaped = False
    for pos in range(start, len(source)):
        char = source[pos]
        if quoted:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
        elif char == '"':
            quoted = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return pos
    raise ValueError("unbalanced function braces")


def _function_parts(source: str) -> tuple[list[str], str, list[str]]:
    """Extract the one forward body and terminal return from model2MLIR's module."""
    mark = "func.func @forward("
    if source.count(mark) != 1:
        raise ValueError("capture must contain exactly one @forward function")
    start = source.index(mark) + len(mark)
    close = source.find(")", start)
    if close < 0:
        raise ValueError("missing @forward argument list")
    arguments = source[start:close]
    names = []
    if arguments.strip():
        for arg in arguments.split(","):
            name, sep, _ = arg.strip().partition(":")
            if not sep or not name.startswith("%"):
                raise ValueError(f"unsupported @forward argument {arg!r}")
            names.append(name)
    body_start = source.find("{", close)
    if body_start < 0:
        raise ValueError("missing @forward body")
    body_end = _matching_brace(source, body_start)
    body = source[body_start + 1 : body_end]
    marker = "func.return "
    if body.count(marker) != 1:
        raise ValueError("capture must have one terminal func.return")
    return_at = body.rfind(marker)
    terminal = body[return_at:].strip()
    if "\n" in terminal or ":" not in terminal:
        raise ValueError("unsupported terminal func.return")
    values = [item.strip() for item in terminal[len(marker) :].split(":", 1)[0].split(",")]
    if not values or any(not value.startswith("%") for value in values):
        raise ValueError("unsupported returned SSA values")
    return names, body[:return_at], values


def _rename_ssa(source: str, prefix: str, argument_names: dict[str, str]) -> str:
    """Rename SSA and block labels outside quoted attributes, preserving MLIR punctuation."""
    out: list[str] = []
    pos = 0
    quoted = False
    escaped = False
    while pos < len(source):
        char = source[pos]
        if quoted:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quoted = False
            pos += 1
            continue
        if char == '"':
            quoted = True
            out.append(char)
            pos += 1
            continue
        if char not in ("%", "^"):
            out.append(char)
            pos += 1
            continue
        end = pos + 1
        while end < len(source) and (source[end].isalnum() or source[end] in "_.$-"):
            end += 1
        token = source[pos:end]
        if len(token) == 1:
            out.append(token)
        elif char == "%":
            out.append(argument_names.get(token, "%" + prefix + token[1:]))
        else:
            out.append("^" + prefix + token[1:])
        pos = end
    return "".join(out)


def _write_npz(path: Path, arrays: list[np.ndarray]) -> None:
    """Use fixed ZIP metadata so identical bundles have identical bytes."""
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_STORED) as archive:
        for index, array in enumerate(arrays):
            stream = io.BytesIO()
            np.lib.format.write_array(stream, np.ascontiguousarray(array), allow_pickle=False)
            info = zipfile.ZipInfo(f"in{index}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            archive.writestr(info, stream.getvalue())


def build_core_aten_batch(
    corpus: dict[str, Any],
    capture_root: str | Path,
    bundle_root: str | Path,
    *,
    selected_overloads: set[str] | None = None,
) -> dict[str, Any]:
    """Write ``model.mlir`` and input/golden sidecars for every case in the corpus.

    The emitted @forward inlines each captured body in corpus order, giving all operands
    and results one positional ABI. The map identifies each case's contiguous input and
    output interval. An unavailable case has no interval and cannot earn a pass.
    """
    capture_root = Path(capture_root)
    bundle_root = Path(bundle_root)
    bundle_root.mkdir(parents=True, exist_ok=True)
    inputs: list[np.ndarray] = []
    arg_types: list[str] = []
    result_types: list[str] = []
    results: list[str] = []
    body_lines: list[str] = []
    records: list[dict[str, Any]] = []
    manifest: dict[str, dict[str, Any]] = {}
    input_order: dict[str, int] = {}

    for case in corpus["cases"]:
        overload = case["overload"]
        directory = capture_root / overload.replace(".", "__")
        capture_file = directory / "capture.json"
        mlir_file = directory / "capsule.linalg.mlir"
        record: dict[str, Any] = {
            "overload": overload,
            "case": case,
            "capture_directory": str(directory),
        }
        if selected_overloads is not None and overload not in selected_overloads:
            record.update(status="not_in_shard", reason="scheduled in a different batch shard")
            records.append(record)
            continue
        if not capture_file.is_file() or not mlir_file.is_file():
            record.update(status="capture_unavailable", reason="no clean Linalg capture artifact")
            if capture_file.is_file():
                capture = json.loads(capture_file.read_text(encoding="utf-8"))
                record["capture_status"] = capture.get("status")
                record["reason"] = capture.get("reason", record["reason"])
            records.append(record)
            continue
        capture = json.loads(capture_file.read_text(encoding="utf-8"))
        meta = capture["capture_meta"]
        if not meta.get("ok") or meta.get("opaque") != 0:
            record.update(status="capture_unavailable", reason="capture is not clean Linalg")
            records.append(record)
            continue
        try:
            names, body, returned = _function_parts(mlir_file.read_text(encoding="utf-8"))
            abi_in = meta["input_abi"]
            abi_out = meta["output_abi"]
            captured_inputs = json.loads((directory / "inputs.json").read_text(encoding="utf-8"))
            if len(names) != len(abi_in) or len(captured_inputs) != len(abi_in):
                raise ValueError("captured argument ABI or input count mismatch")
            if len(returned) != len(abi_out):
                raise ValueError("captured result ABI count mismatch")
            local_types = [_tensor_type(item) for item in abi_in]
            local_result_types = [_tensor_type(item) for item in abi_out]
            local_inputs = []
            tensor_documents = (
                _tensor_documents(case["arguments"]) if any(item["dtype"] == "complex<f32>" for item in abi_in) else []
            )
            for index, (value, item) in enumerate(zip(captured_inputs, abi_in)):
                physical = _physical_abi(item)
                if item["dtype"] == "complex<f32>":
                    logical = np.asarray(_complex_value(tensor_documents[index]["values"]), dtype=np.complex64)
                    array = np.stack((logical.real, logical.imag), axis=-1).astype(np.float32)
                else:
                    array = np.asarray(value, dtype=_DTYPES[item["dtype"]])
                if list(array.shape) != physical["shape"]:
                    raise ValueError(f"input shape {list(array.shape)} differs from ABI {physical['shape']}")
                local_inputs.append(array)
        except (KeyError, TypeError, ValueError, OSError) as exc:
            record.update(status="bundle_unavailable", reason=f"{type(exc).__name__}: {exc}")
            records.append(record)
            continue

        ordinal = len([item for item in records if item["status"] == "bundled"])
        prefix = f"c{ordinal}_"
        first_input = len(inputs)
        first_output = len(results)
        argument_names = {name: f"%arg{first_input + index}" for index, name in enumerate(names)}
        renamed_body = _rename_ssa(body, prefix, argument_names)
        renamed_returns = [_rename_ssa(name, prefix, argument_names) for name in returned]
        body_lines.append(f"    // {overload}\n" + renamed_body.rstrip() + "\n")
        inputs.extend(local_inputs)
        arg_types.extend(local_types)
        results.extend(renamed_returns)
        result_types.extend(local_result_types)
        for index, item in enumerate(abi_in):
            global_index = first_input + index
            physical = _physical_abi(item)
            input_order[f"arg{global_index}"] = global_index
            manifest[str(global_index)] = {
                "kind": "input",
                "name": f"arg{global_index}",
                "dtype": physical["dtype"],
                "shape": physical["shape"],
            }
        record.update(
            status="bundled",
            capture_status=capture.get("status"),
            input_indices=list(range(first_input, len(inputs))),
            output_indices=list(range(first_output, len(results))),
            input_abi=abi_in,
            output_abi=abi_out,
            capture_golden_observation=json.loads((directory / "golden.json").read_text(encoding="utf-8")),
        )
        records.append(record)

    if not results:
        raise ValueError("no clean captures to bundle")
    arguments = ", ".join(f"%arg{i}: {kind}" for i, kind in enumerate(arg_types))
    outputs = result_types[0] if len(result_types) == 1 else "(" + ", ".join(result_types) + ")"
    returns = ", ".join(results)
    return_types = ", ".join(result_types)
    module = (
        "builtin.module {\n"
        f"  func.func @forward({arguments}) -> {outputs} {{\n"
        + "".join(body_lines)
        + f"    func.return {returns} : {return_types}\n"
        + "  }\n}\n"
    )
    (bundle_root / "model.mlir").write_text(module, encoding="utf-8")
    _write_npz(bundle_root / "inputs.npz", inputs)
    (bundle_root / "weights.safetensors.manifest.json").write_bytes(_json_bytes(manifest))
    (bundle_root / "input_order.json").write_bytes(_json_bytes(input_order))
    report = {
        "schema_version": 1,
        "case_count": len(records),
        "bundled_count": sum(item["status"] == "bundled" for item in records),
        "input_count": len(inputs),
        "output_count": len(results),
        "denominator_sha256": corpus.get("denominator_sha256"),
        "cases": records,
    }
    (bundle_root / "core_aten_batch_map.json").write_bytes(_json_bytes(report))
    return report
