"""A projected kernel must remain tied to one exact captured MLIR body."""

from __future__ import annotations

import hashlib
import json

import pytest

from merlin.targetgen.source_kernel_probe import derive_kernel_window


def _capture(tmp_path, *, parent_k=147):
    model = b"module { one selected operation }\n"
    (tmp_path / "model.mlir").write_bytes(model)
    trace = {
        "status": "diagnostic",
        "graphs": {"prepared": {"nodes": [{"id": "g:prepared:root:n7", "target": "aten.convolution.default"}]}},
        "mlir": {
            "sha256": hashlib.sha256(model).hexdigest(),
            "operations": [
                {
                    "ordinal": 29,
                    "operation": "linalg.matmul",
                    "source_node_ids": ["g:prepared:root:n7"],
                    "origin_node_ids": ["g:original:root:n6"],
                    "operand_types": [
                        f"tensor<64x{parent_k}xf32>",
                        f"tensor<{parent_k}x12544xf32>",
                        "tensor<64x12544xf32>",
                    ],
                    "result_types": ["tensor<64x12544xf32>"],
                }
            ],
        },
    }
    (tmp_path / "frontend-trace.json").write_text(json.dumps(trace), encoding="utf-8")
    return tmp_path


def test_source_geometry_and_k_tail_survive_bounded_integer_projection(tmp_path):
    capture = _capture(tmp_path)
    projected = derive_kernel_window(capture, "g:prepared:root:n7", tile_dim=16)
    assert projected["source"]["geometry"] == {"M": 64, "K": 147, "N": 12544}
    assert projected["source"]["operand_types"][0] == "tensor<64x147xf32>"
    assert projected["projection"]["geometry"] == {"M": 16, "K": 19, "N": 16}
    assert projected["integer_body_observed"] is False
    assert projected["integer_model_claim"] == "none_capture_quantization_unverified"


def test_changed_model_bytes_refuse_stale_trace(tmp_path):
    capture = _capture(tmp_path)
    (capture / "model.mlir").write_text("changed", encoding="utf-8")
    with pytest.raises(ValueError, match="does not bind"):
        derive_kernel_window(capture, "g:prepared:root:n7", tile_dim=16)
