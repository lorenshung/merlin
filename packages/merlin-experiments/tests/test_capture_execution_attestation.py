"""A current source inventory cannot upgrade a historical capture."""

import hashlib
import json

import pytest
from merlin_experiments.phase0.capture_execution_attestation import (
    AttestationNotVerified,
    diagnose_capture,
    require_verified_execution,
    write_diagnostic,
)


def _materialized_capture(root):
    root.mkdir()
    files = {
        "model.mlir": b"module {}\n",
        "weights.safetensors": b"weights",
        "weights.safetensors.manifest.json": b"{}\n",
    }
    for name, raw in files.items():
        (root / name).write_bytes(raw)
    receipt = {
        "schema": "m2m.capture-receipt.v1",
        "materialized_abi": {"complete": True},
        # Even a claimed true value in an M2M receipt cannot mint a Merlin execution attestation.
        "source_closure_verified": True,
        "artifacts": {
            name: {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()} for name, raw in files.items()
        },
    }
    (root / "capture_receipt.json").write_text(json.dumps(receipt))


def test_diagnostic_binds_current_bytes_but_never_admits_old_capture(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "loader.py").write_bytes(b"print('first')\n")
    capture = tmp_path / "capture"
    _materialized_capture(capture)

    first = diagnose_capture(capture, source, ["loader.py"])
    assert first["capture"]["materialized_receipt"]["status"] == "verified_materialized"
    assert first["source_closure_verified"] is False
    assert first["fresh_execution"] is False
    assert first["selected_source"]["members"]["loader.py"]["sha256"] == hashlib.sha256(b"print('first')\n").hexdigest()
    output = tmp_path / "diagnostic.json"
    write_diagnostic(output, first)
    with pytest.raises(FileExistsError):
        write_diagnostic(output, first)
    with pytest.raises(AttestationNotVerified, match="no verified fresh sealed execution"):
        require_verified_execution(json.loads(output.read_text()))

    (source / "loader.py").write_bytes(b"print('second')\n")
    second = diagnose_capture(capture, source, ["loader.py"])
    assert first["selected_source"]["inventory_sha256"] != second["selected_source"]["inventory_sha256"]
    forged = dict(
        first,
        status="verified_sealed_execution",
        source_closure_verified=True,
        fresh_execution=True,
        issuer="self-reported",
    )
    with pytest.raises(AttestationNotVerified, match="no supported Merlin sealed execution issuer"):
        require_verified_execution(forged)


def test_diagnostic_rejects_ambiguous_source_selection(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "loader.py").write_text("pass\n")
    (source / "linked.py").symlink_to(source / "loader.py")
    capture = tmp_path / "capture"
    _materialized_capture(capture)
    with pytest.raises(ValueError, match="symlink"):
        diagnose_capture(capture, source, ["linked.py"])
    with pytest.raises(ValueError, match="unsafe selected source path"):
        diagnose_capture(capture, source, ["../loader.py"])
    with pytest.raises(ValueError, match="overlap"):
        diagnose_capture(capture, source, ["loader.py", "loader.py/child"])
    document = diagnose_capture(capture, source, ["loader.py"])
    with pytest.raises(ValueError, match="outside"):
        write_diagnostic(capture / "attestation.json", document)
