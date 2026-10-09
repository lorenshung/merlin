"""A materialized capture must bind its external quantization manifest end to end."""

import hashlib
import json

import pytest

from merlin.targetgen.application_inventory import verify_capture_receipt


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _manifest(schema="m2m.quantization_manifest.v1", sites=None):
    return {
        "schema": schema,
        "adapter_id": "fixture",
        "contract_sha256": "a" * 64,
        "policy_sha256": "b" * 64,
        "sites": [{"site_id": "one", "status": "host"}] if sites is None else sites,
    }


def _write_capture(root, *, change=None, manifest=None):
    manifest = _manifest() if manifest is None else manifest
    digest = _sha(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    mlir = f'module attributes {{prov.quantization_manifest_sha256 = "{digest}"}} {{}}\n'.encode()
    manifest_bytes = (json.dumps(manifest, sort_keys=True) + "\n").encode()
    pointer = {"path": "quantization-manifest.json", "sha256": _sha(manifest_bytes), "manifest_sha256": digest}
    if change == "pointer":
        pointer["manifest_sha256"] = "0" * 64
    if change == "mlir":
        mlir = mlir.replace(digest.encode(), b"0" * 64)
    if change == "manifest":
        manifest_bytes = b'{"schema":"m2m.quantization_manifest.v1","sites":[]}\n'
        pointer["sha256"] = _sha(manifest_bytes)
    metadata = {"quantization_manifest": pointer}
    contents = {
        "model.mlir": mlir,
        "weights.safetensors": b"weights",
        "weights.safetensors.manifest.json": b"{}",
        "meta.json": json.dumps(metadata).encode(),
        "quantization-manifest.json": manifest_bytes,
    }
    if change == "missing":
        contents.pop("quantization-manifest.json")
        metadata.pop("quantization_manifest")
        contents["meta.json"] = json.dumps(metadata).encode()
    for name, raw in contents.items():
        (root / name).write_bytes(raw)
    (root / "capture_receipt.json").write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "materialized_abi": {"complete": True},
                "artifacts": {name: {"bytes": len(raw), "sha256": _sha(raw)} for name, raw in contents.items()},
            }
        )
    )


@pytest.mark.parametrize("change", [None, "pointer", "mlir", "manifest", "missing"])
def test_external_manifest_binding(change, tmp_path):
    _write_capture(tmp_path, change=change)
    observed = verify_capture_receipt(tmp_path / "model.mlir")
    assert observed["status"] == ("verified_materialized" if change is None else "unverified")
    assert observed["source_closure_verified"] is False


_PRESERVED = {"site_id": "two", "status": "preserved", "execution_route": "float_unit"}


def test_v2_manifest_with_a_preserved_route_is_verified(tmp_path):
    manifest = _manifest("m2m.quantization_manifest.v2", [{"site_id": "one", "status": "quantized"}, _PRESERVED])
    _write_capture(tmp_path, manifest=manifest)
    assert verify_capture_receipt(tmp_path / "model.mlir")["status"] == "verified_materialized"


@pytest.mark.parametrize(
    ("schema", "sites", "fields", "reason"),
    [
        ("m2m.quantization_manifest.v2", [{"site_id": "two", "status": "preserved"}], {}, "execution_route"),
        ("m2m.quantization_manifest.v1", [_PRESERVED], {}, "status 'preserved'"),
        ("m2m.quantization_manifest.v3", None, {}, "unsupported manifest schema"),
        ("m2m.quantization_manifest.v2", [], {}, "nonempty site census"),
        ("m2m.quantization_manifest.v2", [{"site_id": "one", "status": "host"}] * 2, {}, "duplicate site_id"),
        ("m2m.quantization_manifest.v2", [{"site_id": "one", "status": "unknown"}], {}, "status 'unknown'"),
        ("m2m.quantization_manifest.v2", None, {"adapter_id": ""}, "adapter_id"),
        ("m2m.quantization_manifest.v2", None, {"policy_sha256": "policy"}, "policy_sha256"),
    ],
)
def test_malformed_manifest_is_refused_with_its_reason(tmp_path, schema, sites, fields, reason):
    manifest = {**_manifest(schema, sites), **fields}
    _write_capture(tmp_path, manifest=manifest)
    observed = verify_capture_receipt(tmp_path / "model.mlir")
    assert observed["status"] == "unverified"
    assert any(reason in error for error in observed["errors"]), observed["errors"]
