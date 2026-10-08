"""model2MLIR adapter for the compiler-independent Core ATen case corpus.

The source corpus is authoritative and remains complete when this adapter fails.  Each adapter result
is a separate observation: exact provenance retained, exact overload lost, or capture failed.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from merlin.targetgen.capsule_source import PytorchRefSource
from merlin.targetgen.core_aten_cover import _provenance_ops


def _split_tensor_documents(value: Any, tensors: list[dict[str, Any]]) -> Any:
    if isinstance(value, dict) and value.get("kind") == "tensor":
        index = len(tensors)
        tensors.append(value)
        return {"input_index": index}
    if isinstance(value, dict):
        return {key: _split_tensor_documents(child, tensors) for key, child in value.items()}
    if isinstance(value, list):
        return [_split_tensor_documents(child, tensors) for child in value]
    return value


def loader_source(case: dict[str, Any]) -> str:
    """Render a self-contained PyTorch module for one portable case document.

    Every tensor argument, including tensors nested in lists and keyword ``out`` values, is an
    external model input.  Scalars, dtype objects, memory formats, and container structure stay as
    static schema arguments.  The generated loader imports neither Merlin nor model2MLIR.
    """

    tensors: list[dict[str, Any]] = []
    arguments = _split_tensor_documents(case["arguments"], tensors)
    parameters = ", ".join(f"input_{index}" for index in range(len(tensors)))
    inputs = ", ".join(f"input_{index}" for index in range(len(tensors)))
    if len(tensors) == 1:
        inputs += ","
    expected = case["expected"]
    structured_result = isinstance(expected, list) or (
        isinstance(expected, dict) and expected.get("kind") in {"tensor", "tuple"}
    )
    scalar_result = not structured_result
    scalar_wrap = "result = torch.ops.aten.scalar_tensor.default(result)" if scalar_result else ""
    document = json.dumps(
        {"overload": case["overload"], "arguments": arguments, "tensor_inputs": tensors},
        sort_keys=True,
        separators=(",", ":"),
    )
    return f'''"""Generated from the versioned Core ATen case corpus; do not hand-edit."""
import json
import torch
from torch import nn

_DOCUMENT = json.loads({document!r})


def _decode(value, inputs=()):
    if isinstance(value, list):
        return [_decode(item, inputs) for item in value]
    if not isinstance(value, dict):
        return value
    if "input_index" in value:
        return inputs[value["input_index"]]
    kind = value.get("kind")
    if kind == "tuple":
        return tuple(_decode(item, inputs) for item in value["items"])
    if kind == "dtype" or kind == "memory_format":
        return getattr(torch, value["value"])
    if kind == "device":
        return torch.device(value["value"])
    if kind == "complex":
        return complex(value["real"], value["imag"])
    if kind == "float":
        return float(value["value"])
    if kind == "tensor":
        dtype = getattr(torch, value["dtype"])
        return torch.tensor(_decode(value["values"]), dtype=dtype).reshape(value["shape"])
    return {{key: _decode(child, inputs) for key, child in value.items()}}


def _op(name):
    namespace, packet, overload = name.split(".", 2)
    assert namespace == "aten"
    return getattr(getattr(torch.ops.aten, packet), overload)


class Model(nn.Module):
    def forward(self, {parameters}):
        inputs = ({inputs})
        arguments = _decode(_DOCUMENT["arguments"], inputs)
        result = _op(_DOCUMENT["overload"])(*arguments["args"], **arguments["kwargs"])
        {scalar_wrap}
        return result


def get_model_and_inputs():
    inputs = tuple(_decode(value) for value in _DOCUMENT["tensor_inputs"])
    return Model(), inputs
'''


def _git_revision(path: Path) -> str | None:
    process = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return process.stdout.strip() if process.returncode == 0 and process.stdout.strip() else None


def capture_case_corpus(
    corpus: dict[str, Any],
    output_root: str | Path,
    *,
    overloads: set[str] | None = None,
    m2m_dir: str | Path | None = None,
    python: str | Path | None = None,
) -> dict[str, Any]:
    """Capture canonical cases independently, retaining failures as auditable result records."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    source = PytorchRefSource(
        m2m_dir=Path(m2m_dir) if m2m_dir else None,
        python=Path(python) if python else None,
    )
    results: dict[str, dict[str, Any]] = {}
    for case in corpus["cases"]:
        overload = case["overload"]
        if overloads is not None and overload not in overloads:
            continue
        destination = root / overload.replace(".", "__")
        destination.mkdir(parents=True, exist_ok=True)
        loader = destination / "capsule.pytorch.py"
        loader.write_text(loader_source(case), encoding="utf-8")
        mlir = destination / "capsule.linalg.mlir"
        try:
            artifact = source.capture_loader(loader, "fp32", workdir=destination / "capture-work")
            mlir.write_text(artifact.linalg_mlir, encoding="utf-8")
            actual = sorted(_provenance_ops(mlir))
            exact = overload in actual
            record = {
                "schema_version": 1,
                "status": "captured_exact" if exact else "captured_lost_exact_provenance",
                "overload": overload,
                "captured_overloads": actual,
                "pytorch_version": artifact.meta.get("torch_version"),
                "model2mlir_revision": _git_revision(source.m2m_dir),
                "capture_meta": artifact.meta,
            }
            (destination / "inputs.json").write_text(json.dumps(artifact.inputs, indent=2) + "\n", encoding="utf-8")
            (destination / "golden.json").write_text(json.dumps(artifact.golden, indent=2) + "\n", encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 -- every frontend failure is a result, not an omission
            mlir.unlink(missing_ok=True)
            record = {
                "schema_version": 1,
                "status": "capture_failed",
                "overload": overload,
                "captured_overloads": [],
                "pytorch_version": corpus["pytorch_version"],
                "model2mlir_revision": _git_revision(source.m2m_dir),
                "reason": f"{type(exc).__name__}: {exc}",
            }
        (destination / "capture.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        results[overload] = record
    counts: dict[str, int] = {}
    for record in results.values():
        status = record["status"]
        counts[status] = counts.get(status, 0) + 1
    return {
        "schema_version": 1,
        "case_count": len(results),
        "status_counts": dict(sorted(counts.items())),
        "results": dict(sorted(results.items())),
    }
