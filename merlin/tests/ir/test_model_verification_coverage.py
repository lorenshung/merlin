"""The three captured models must not be mislabeled as whole-module SMT proofs."""

from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.verify.model_coverage import audit_capture


@pytest.mark.parametrize(
    ("model", "required_op", "required_dtype"),
    [
        ("SY_model_tiny_llama", "quant_ext.dequantize_per_tensor", "f32"),
        # The verifier can encode a restricted linalg.generic; this capture's
        # linalg.reduce and mixed floating tensor types still force abstention.
        ("SY_model_smolvla", "linalg.reduce", "bf16"),
        ("SY_model_resnet50", "quant_ext.quantize_per_tensor", "f32"),
    ],
)
def test_actual_model_capture_abstains_with_explainable_gaps(model, required_op, required_dtype):
    path = merlin_dir() / "contract" / "capsules" / "model" / model / "capsule.interface.mlir"
    result = audit_capture(path)
    assert result["method"] == "textual_inventory_not_proof"
    assert result["capture_sha256"]
    assert not result["eligible_for_whole_module_smt_attempt"]
    assert result["unsupported_operation_counts"][required_op] > 0
    assert required_dtype in result["tensor_dtypes_observed"]
    assert result["unparsed_assignment_lines"] == 0
