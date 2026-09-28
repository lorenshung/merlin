"""Bound a captured matrix body for a finite, source-identified kernel diagnostic.

The source operation keeps its original geometry and dtype in the record. The
window is a *new synthetic integer operation*: projecting a float or bf16
capture does not establish that the model has an integerized contraction.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

_MATRIX = re.compile(r"tensor<([1-9][0-9]*)x([1-9][0-9]*)x([a-z][a-z0-9]*)>")


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _matrix_type(spelling: str) -> tuple[int, int, str]:
    match = _MATRIX.fullmatch(spelling)
    if match is None:
        raise ValueError(f"expected a static rank-2 tensor, got {spelling!r}")
    return int(match[1]), int(match[2]), match[3]


def derive_kernel_window(capture_dir: str | Path, source_node_id: str, *, tile_dim: int) -> dict:
    """Select one exact traced ``linalg.matmul`` and derive one bounded test window.

    The selected trace's MLIR hash must match the model bytes. A caller selects a
    source node ID, not a shape; ambiguous nodes fail closed. The K window has
    one full tile plus the parent's K remainder (or a second full tile), so a
    parent tail remains visible without simulating its full reduction.
    """
    if type(tile_dim) is not int or tile_dim < 1:
        raise ValueError("tile_dim must be a positive integer derived from selected hardware facts")
    capture = Path(capture_dir)
    model_path = capture / "model.mlir"
    trace_path = capture / "frontend-trace.json"
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    model_sha = _sha(model_path)
    if trace.get("mlir", {}).get("sha256") != model_sha:
        raise ValueError("frontend trace does not bind the selected model.mlir bytes")
    nodes = [node for node in trace["graphs"]["prepared"]["nodes"] if node.get("id") == source_node_id]
    if len(nodes) != 1:
        raise ValueError(f"source node {source_node_id!r} is absent or ambiguous in the prepared graph")
    operations = [
        op for op in trace["mlir"]["operations"]
        if op.get("operation") == "linalg.matmul" and source_node_id in op.get("source_node_ids", [])
    ]
    if len(operations) != 1:
        raise ValueError(f"source node {source_node_id!r} has {len(operations)} matrix bodies; expected one")
    op = operations[0]
    if len(op.get("operand_types", [])) != 3 or len(op.get("result_types", [])) != 1:
        raise ValueError("selected matrix body has an unsupported operand/result ABI")
    m, k, lhs_dtype = _matrix_type(op["operand_types"][0])
    wk, n, rhs_dtype = _matrix_type(op["operand_types"][1])
    om, on, out_dtype = _matrix_type(op["operand_types"][2])
    rm, rn, result_dtype = _matrix_type(op["result_types"][0])
    if (wk, om, on, rm, rn, out_dtype) != (k, m, n, m, n, result_dtype):
        raise ValueError("selected matrix body has inconsistent contraction geometry")
    remainder = k % tile_dim
    window = {
        "M": min(m, tile_dim),
        "K": min(k, tile_dim + (remainder if remainder else tile_dim)),
        "N": min(n, tile_dim),
    }
    metadata_path = capture / "meta.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    receipt_path = capture / "capture_receipt.json"
    return {
        "schema": "merlin.source-kernel-window.v1",
        "scope": "synthetic_i8_matrix_body_from_captured_geometry",
        "source": {
            "model_mlir_sha256": model_sha,
            "frontend_trace_sha256": _sha(trace_path),
            "capture_receipt_sha256": _sha(receipt_path) if receipt_path.is_file() else None,
            "trace_status": trace.get("status"),
            "source_node_id": source_node_id,
            "source_target": nodes[0].get("target"),
            "module_stack": nodes[0].get("module_stack", {}),
            "mlir_operation": op["operation"],
            "mlir_ordinal": op.get("ordinal"),
            "operand_types": op["operand_types"],
            "result_types": op["result_types"],
            "origin_node_ids": op.get("origin_node_ids", []),
            "geometry": {"M": m, "K": k, "N": n},
            "dtypes": {"lhs": lhs_dtype, "rhs": rhs_dtype, "result": result_dtype},
            "capture_dtype": metadata.get("dtype"),
            "recipe_sha256": metadata.get("recipe_sha256"),
        },
        "projection": {
            "tile_dim": tile_dim,
            "geometry": window,
            "dtype": {"lhs": "i8", "rhs": "i8", "result": "i32"},
            "rule": "M,N=min(parent,tile); K=min(parent,tile+(parent_K%tile or tile))",
        },
        "integer_body_observed": (lhs_dtype, rhs_dtype, result_dtype) == ("i8", "i8", "i32"),
        "integer_model_claim": "none_capture_quantization_unverified",
    }
