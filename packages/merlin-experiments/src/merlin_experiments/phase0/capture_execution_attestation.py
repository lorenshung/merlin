"""Diagnostic evidence for the separate Phase 0 capture-execution boundary.

The model2MLIR receipt verifies materialized capture members. It does not prove
which source, checkpoint, Python dependencies, or ambient files the process read.
This module deliberately has no verified issuer: a diagnostic made after a run
cannot upgrade that run to a sealed execution, even if its bytes still match.
"""

from __future__ import annotations

import hashlib
import json
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from merlin.targetgen.application_inventory import verify_capture_receipt

SCHEMA = "merlin.capture_execution_attestation.v1"
_VERIFIED_ISSUERS: frozenset[str] = frozenset()
_REQUIRED_CONTROLS = (
    "fresh_private_source_snapshot",
    "complete_runtime_and_checkpoint_snapshot",
    "network_namespace_disabled",
    "ambient_checkout_home_and_cache_inaccessible",
    "source_and_runtime_mounted_read_only",
    "new_capture_output_directory",
    "post_execution_source_and_output_byte_verification",
)


class AttestationNotVerified(ValueError):
    """No supported fresh, sealed execution has earned source-closure admission."""


def _sha256(path: Path) -> tuple[int, str]:
    before = path.stat(follow_symlinks=False)
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"source member is not a regular file: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat(follow_symlinks=False)

    def identity(value):
        return value.st_dev, value.st_ino, value.st_size, value.st_mtime_ns, value.st_ctime_ns

    if identity(before) != identity(after):
        raise ValueError(f"source member changed while being inspected: {path}")
    return before.st_size, digest.hexdigest()


def _relative(value: str) -> PurePosixPath:
    if not isinstance(value, str):
        raise ValueError("selected source path must be a string")
    path = PurePosixPath(value)
    if (
        not value
        or value == "."
        or path.is_absolute()
        or ".." in path.parts
        or path.as_posix() != value
        or "\\" in value
        or "\x00" in value
    ):
        raise ValueError(f"unsafe selected source path: {value!r}")
    return path


def _inventory(root: Path, selections: Sequence[str]) -> dict[str, dict[str, Any]]:
    """Inventory exact selected members, including directory membership and empty dirs.

    This is only a diagnostic byte inventory. It does not discover Python imports
    or the files an earlier capture process actually opened.
    """
    if root.is_symlink() or not root.is_dir():
        raise ValueError(f"source root is absent or indirect: {root}")
    selected = [_relative(value) for value in selections]
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("source selection must be nonempty and unique")
    if any(a != b and (a.is_relative_to(b) or b.is_relative_to(a)) for a in selected for b in selected):
        raise ValueError("selected source paths overlap")
    inventory: dict[str, dict[str, Any]] = {}

    def visit(path: Path, relative: str) -> None:
        if path.is_symlink():
            raise ValueError(f"selected source contains a symlink: {relative}")
        mode = path.stat(follow_symlinks=False).st_mode
        if stat.S_ISDIR(mode):
            children = sorted(path.iterdir(), key=lambda member: member.name)
            inventory[relative] = {"kind": "directory", "members": [member.name for member in children]}
            for child in children:
                visit(child, f"{relative}/{child.name}")
        elif stat.S_ISREG(mode):
            size, digest = _sha256(path)
            inventory[relative] = {"kind": "file", "bytes": size, "sha256": digest}
        else:
            raise ValueError(f"selected source contains a nonregular member: {relative}")

    for relative in selected:
        path = root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError(f"selected source contains a symlink: {relative}")
        if not path.exists():
            raise ValueError(f"selected source is absent: {relative}")
        visit(path, relative.as_posix())
    return dict(sorted(inventory.items()))


def diagnose_capture(
    capture_dir: Path,
    source_root: Path,
    selected_sources: Sequence[str],
) -> dict[str, Any]:
    """Describe current bytes without asserting an earlier execution was sealed."""
    capture_dir, source_root = Path(capture_dir).absolute(), Path(source_root).absolute()
    source_members = _inventory(source_root, selected_sources)
    source_payload = json.dumps(source_members, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    materialized = verify_capture_receipt(capture_dir / "model.mlir")
    return {
        "schema": SCHEMA,
        "status": "diagnostic_only",
        "source_closure_verified": False,
        "fresh_execution": False,
        "issuer": None,
        "capture": {"path": str(capture_dir), "materialized_receipt": materialized},
        "selected_source": {
            "root": str(source_root),
            "members": source_members,
            "inventory_sha256": hashlib.sha256(source_payload).hexdigest(),
        },
        "execution": {"observed": False, "controls_required": list(_REQUIRED_CONTROLS), "controls_verified": []},
        "blockers": [
            "selected current source bytes do not identify the source bytes read by the capture process",
            "no fresh private source/runtime/checkpoint snapshot was executed",
            "network and ambient filesystem isolation were not observed",
            "materialized capture verification does not establish source closure",
        ],
    }


def write_diagnostic(path: Path, document: Mapping[str, Any]) -> None:
    """Write once to a separate evidence path; never edit a capture or its receipt."""
    if document.get("schema") != SCHEMA or document.get("status") != "diagnostic_only":
        raise ValueError("only diagnostic attestations may be written by this module")
    if document.get("source_closure_verified") is not False or document.get("fresh_execution") is not False:
        raise ValueError("diagnostic attestation cannot assert verified execution")
    path = Path(path)
    if path.is_symlink() or path.parent.is_symlink():
        raise ValueError("attestation destination is indirect")
    capture = Path(str((document.get("capture") or {}).get("path", ""))).absolute()
    source = Path(str((document.get("selected_source") or {}).get("root", ""))).absolute()
    destination = path.absolute()
    if destination.is_relative_to(capture) or destination.is_relative_to(source):
        raise ValueError("diagnostic attestation must live outside the capture and source trees")
    raw = (json.dumps(document, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()
    with path.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())


def require_verified_execution(document: Mapping[str, Any]) -> None:
    """Admission gate for a future Merlin sealed runner, closed until one exists.

    A model2MLIR receipt, an old capture, a diagnostic receipt, or edited JSON with
    ``source_closure_verified: true`` cannot pass this gate. Introducing a verified
    issuer requires a separate reviewed runner and verifier that bind actual source,
    runtime, checkpoint, command, isolation controls, and new output bytes.
    """
    if document.get("schema") != SCHEMA:
        raise AttestationNotVerified("unsupported capture execution attestation schema")
    if (
        document.get("status") != "verified_sealed_execution"
        or document.get("source_closure_verified") is not True
        or document.get("fresh_execution") is not True
    ):
        raise AttestationNotVerified("capture has no verified fresh sealed execution")
    if document.get("issuer") not in _VERIFIED_ISSUERS:
        raise AttestationNotVerified("no supported Merlin sealed execution issuer has verified this capture")
