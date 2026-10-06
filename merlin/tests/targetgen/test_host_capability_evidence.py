"""Exact host selectors may not turn a schedule into family-wide support."""

import pytest

from merlin.targetgen.host_capabilities import admit_host_operation, validate_host_capabilities


def _selection():
    return {
        "host": {
            "package_sha256": "a" * 64,
            "capability_spec_sha256": "b" * 64,
            "dtype_strategy": "int8_w8a8",
            "capability_spec": {
                "schema": "merlin.host_capabilities.v1",
                "status": "reviewed",
                "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "int8_w8a8"},
                "operations": [
                    {
                        "id": "named_matmul",
                        "ops": ["linalg.matmul"],
                        "families": ["contraction"],
                        "placement": "host",
                        "signature": {"operand_dtypes": ["int8"]},
                    }
                ],
                "evidence": {"scope": "declaration screen only"},
            },
        }
    }


def test_exact_host_operation_selector_does_not_expand_to_generic_contraction():
    selected = _selection()
    signature = {"family": "contraction", "operand_dtype": "int8"}
    generic = admit_host_operation(selected, {"mlir_operation": "linalg.generic"}, signature)
    assert generic["status"] == "unsupported"
    assert generic["profiles"][0]["decisions"] == []
    named = admit_host_operation(selected, {"mlir_operation": "linalg.matmul"}, signature)
    assert named["status"] == "admitted"
    assert named["qualification"].startswith("selected declaration screen")


def test_family_only_host_selector_remains_available_when_explicitly_declared():
    selected = _selection()
    selected["host"]["capability_spec"]["operations"][0].pop("ops")
    generic = admit_host_operation(
        selected, {"mlir_operation": "linalg.generic"}, {"family": "contraction", "operand_dtype": "int8"}
    )
    assert generic["status"] == "admitted"


def test_unreviewed_operation_inside_reviewed_host_document_remains_unknown():
    selected = _selection()
    declaration = selected["host"]["capability_spec"]["operations"][0]
    declaration["status"] = "unreviewed"
    named = admit_host_operation(
        selected, {"mlir_operation": "linalg.matmul"}, {"family": "contraction", "operand_dtype": "int8"}
    )
    assert named["status"] == "unknown"
    assert named["review_status"] == "unreviewed"
    assert named["reviewed"] is False
    declaration["status"] = "accepted"
    with pytest.raises(ValueError, match="invalid review status"):
        validate_host_capabilities(selected["host"]["capability_spec"])
