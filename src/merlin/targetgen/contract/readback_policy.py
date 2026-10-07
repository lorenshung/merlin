"""Trusted, invocation-only whole-program output transport and build evidence.

The policy is never read from a candidate command buffer.  It changes only how
the selected harness returns already-computed output buffers, not the program's
inputs, lowering, or numerical comparison policy.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from math import prod
from pathlib import Path
from typing import Any

POLICY_SCHEMA = "merlin_readback_policy_v1"
FULL_VALUES_B64 = "out_b64_v1"
BUILD_RECEIPT = "readback_build.json"


@dataclass(frozen=True)
class ReadbackPolicy:
    """An explicit operator choice; ``None`` retains the historical transport."""

    transport: str
    schema: str = POLICY_SCHEMA

    def __post_init__(self) -> None:
        if type(self) is not ReadbackPolicy or self.schema != POLICY_SCHEMA or self.transport != FULL_VALUES_B64:
            raise ValueError("unsupported invocation-only readback policy")

    def record(self) -> dict[str, str]:
        return {"schema": self.schema, "transport": self.transport}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> ReadbackPolicy:
        if type(record) is not dict or set(record) != {"schema", "transport"}:
            raise ValueError("readback policy record must have the exact versioned fields")
        return cls(schema=record["schema"], transport=record["transport"])


def selected(policy: ReadbackPolicy | None) -> ReadbackPolicy | None:
    if policy is not None and type(policy) is not ReadbackPolicy:
        raise ValueError("readback policy must be an explicit trusted ReadbackPolicy")
    return policy


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha256(value: Any) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def selected_build_inputs(target: str, recipe: Any, build_service: Any = None) -> tuple[dict, list[dict[str, str]]]:
    """Re-read the selected recipe and declared Python/header source bytes.

    These are the build path's known inputs, not a claim about every transitive
    compiler, system header, or library byte.
    """

    from merlin.common.paths import runtime_dir
    from merlin.targetgen import build_cache

    codec = (runtime_dir() / "baremetal" / "out_b64.h").resolve(strict=True)
    token = build_cache.recipe_token(recipe)
    if token is None:
        raise ValueError("full-value build recipe has no exact selected input token")
    # The opt-in header is staged next to harness.c. Keep default recipe flags
    # byte-identical, while including these exact bytes in the opt-in identity.
    token = {**token, "readback_codec": {"path": str(codec), "sha256": file_sha256(codec)}}
    if build_service is None:
        sources = build_cache.build_path(target)
        if not sources:
            raise ValueError("full-value build has no selected renderer source path")
        pins = [(str(path), file_sha256(Path(path))) for path in sources]
    else:
        build_service.verify(target)
        pins = list(build_service.source_pins)
        if not pins:
            raise ValueError("full-value build service has no selected source pins")
        if any(file_sha256(Path(path)) != digest for path, digest in pins):
            raise ValueError("full-value build service source bytes changed")
    return token, [{"path": path, "sha256": digest} for path, digest in sorted(pins)]


def stage_codec_header(workdir: Path) -> Path:
    """Stage only the selected generic codec beside the opt-in C harness."""

    from merlin.common.paths import runtime_dir

    source = (runtime_dir() / "baremetal" / "out_b64.h").resolve(strict=True)
    target = workdir / "out_b64.h"
    if target.is_symlink():
        raise ValueError("readback codec staging path must not be a symlink")
    payload = source.read_bytes()
    target.write_bytes(payload)
    if file_sha256(source) != file_sha256(target):
        raise ValueError("readback codec bytes changed while staging")
    return target


def build_receipt(
    *, policy: ReadbackPolicy, cb: Mapping[str, Any], target: str, recipe_record: Mapping[str, Any],
    source_pins: list[dict[str, str]], object_path: Path, harness_path: Path, elf_path: Path,
) -> dict[str, Any]:
    """Bind selected bytes after linking; this is not full toolchain closure."""

    body: dict[str, Any] = {
        "schema": "merlin_readback_build_v1",
        "status": "completed",
        "target": target,
        "readback_policy": policy.record(),
        "command_buffer_sha256": canonical_sha256(cb),
        "recipe": dict(recipe_record),
        "source_pins": source_pins,
        "kernel_object_sha256": file_sha256(object_path),
        "harness_sha256": file_sha256(harness_path),
        "staged_codec_sha256": file_sha256(harness_path.parent / "out_b64.h"),
        "elf_sha256": file_sha256(elf_path),
        "scope": "selected build inputs and produced bytes; not complete toolchain closure or numerical correctness",
    }
    body["build_identity_sha256"] = canonical_sha256(body)
    return body


def require_build_receipt(
    path: Path, *, policy: ReadbackPolicy, cb: Mapping[str, Any], target: str,
    recipe_record: Mapping[str, Any], source_pins: list[dict[str, str]],
    object_path: Path, harness_path: Path, elf_path: Path,
) -> dict[str, Any]:
    """Recheck the actual harness/ELF before consuming a full-value result."""

    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("readback build receipt is malformed")
    identity = data.get("build_identity_sha256")
    body = {key: value for key, value in data.items() if key != "build_identity_sha256"}
    if (
        body.get("schema") != "merlin_readback_build_v1"
        or body.get("status") != "completed"
        or body.get("target") != target
        or body.get("readback_policy") != policy.record()
        or body.get("command_buffer_sha256") != canonical_sha256(cb)
        or body.get("recipe") != dict(recipe_record)
        or body.get("source_pins") != source_pins
        or body.get("kernel_object_sha256") != file_sha256(object_path)
        or body.get("harness_sha256") != file_sha256(harness_path)
        or body.get("staged_codec_sha256") != file_sha256(harness_path.parent / "out_b64.h")
        or body.get("staged_codec_sha256") != recipe_record.get("readback_codec", {}).get("sha256")
        or body.get("elf_sha256") != file_sha256(elf_path)
        or identity != canonical_sha256(body)
    ):
        raise ValueError("readback build receipt does not bind selected policy and produced bytes")
    return data


def require_full_value_roster(cb: Mapping[str, Any], console: str, outputs: Mapping[str, Any]) -> None:
    """Require one complete packed value frame for every declared output."""

    abi = cb.get("kernel_abi") or {}
    names = abi.get("outputs")
    tensors = cb.get("tensors") or {}
    if abi.get("kind") != "whole_program" or not isinstance(names, list) or not names or len(set(names)) != len(names):
        raise ValueError("full-value readback requires a closed whole-program output roster")
    frames: dict[str, tuple[int, int]] = {}
    for line in console.splitlines():
        parts = line.split()
        if parts and parts[0] == "OUTSUM":
            raise ValueError("digest-only output cannot certify full numerical values")
        if parts and parts[0] == "OUT_B64_BEGIN":
            if len(parts) != 7:
                raise ValueError("malformed full-value frame header")
            if parts[2] in frames:
                raise ValueError("full-value transport duplicated an output")
            try:
                frames[parts[2]] = (int(parts[3]), int(parts[4]))
            except ValueError as exc:
                raise ValueError("full-value frame has invalid geometry") from exc
    if len(frames) != len(names) or set(frames) != set(names) or set(outputs) != set(names):
        raise ValueError("full-value transport omitted or duplicated a declared output")
    for name in names:
        shape = tensors.get(name, {}).get("shape")
        if not isinstance(shape, list) or any(type(dim) is not int or dim < 0 for dim in shape):
            raise ValueError("full-value output has no static declared shape")
        rows = outputs[name]
        expected_rows = prod(shape[:-1]) if shape else 1
        expected_cols = shape[-1] if shape else 1
        if (
            frames[name] != (expected_rows, expected_cols)
            or not isinstance(rows, list)
            or len(rows) != expected_rows
            or any(not isinstance(row, list) or len(row) != expected_cols for row in rows)
        ):
            raise ValueError("full-value output size differs from declared tensor")
