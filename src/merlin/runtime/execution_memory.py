"""Admit loaded and absolute runtime intervals against a selected decoded map.

The execution provider owns the map and its derivation. A simulator backing
store size is not a decoded-map contract. This validator supports identity
mapped executables and disjoint, simultaneously live runtime reservations;
aliasing and virtual translation need separate ownership/translation proofs.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from . import elf_audit


class ExecutionMemoryError(ValueError):
    """Selected memory evidence or deployment intervals do not close."""


def _pin(path: str | Path) -> dict:
    path = Path(path)
    try:
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        return {"path": str(path), "sha256": digest, "bytes": path.stat().st_size}
    except OSError as exc:
        raise ExecutionMemoryError(f"memory evidence unavailable: {path}") from exc


def _digest(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def _extent(name: str, begin: object, size: object, bits: int, alignment: object = 1) -> tuple[int, int]:
    if type(begin) is not int or type(size) is not int or not 0 <= begin < 1 << bits or size <= 0:
        raise ExecutionMemoryError(f"{name}: unknown or invalid memory extent")
    if begin + size > 1 << bits:
        raise ExecutionMemoryError(f"{name}: address extent overflows")
    if type(alignment) is not int or alignment <= 0 or alignment & (alignment - 1) or begin % alignment:
        raise ExecutionMemoryError(f"{name}: invalid or unsatisfied alignment")
    return begin, begin + size


def _permissions(value: object, name: str) -> frozenset[str]:
    if not isinstance(value, str) or not value or set(value) - set("RWX") or len(set(value)) != len(value):
        raise ExecutionMemoryError(f"{name}: unknown memory permissions")
    return frozenset(value)


@dataclass(frozen=True)
class MemoryMapBinding:
    """Pin to the selected provider's complete decoded-map JSON artifact."""

    path: Path
    sha256: str
    execution_identity: str

    def validate(self) -> dict:
        """Close map bytes and all declared source/elaboration evidence."""
        if not _digest(self.sha256) or _pin(self.path)["sha256"] != self.sha256:
            raise ExecutionMemoryError("selected memory map changed or has an unknown identity")
        try:
            record = json.loads(Path(self.path).read_text())
        except (OSError, ValueError) as exc:
            raise ExecutionMemoryError("unreadable selected memory map") from exc
        if not isinstance(record, dict) or record.get("schema") != "merlin.execution-memory-map.v1":
            raise ExecutionMemoryError("unknown memory map schema")
        if record.get("status") != "complete" or record.get("identity_mapped") is not True:
            raise ExecutionMemoryError("memory map is incomplete or not identity mapped")
        if not isinstance(record.get("execution_identity"), str) or not record["execution_identity"]:
            raise ExecutionMemoryError("memory map lacks the selected execution identity")
        if (
            not isinstance(self.execution_identity, str)
            or not self.execution_identity
            or record["execution_identity"] != self.execution_identity
        ):
            raise ExecutionMemoryError("memory map does not match the selected execution identity")
        bits = record.get("address_bits")
        if type(bits) is not int or bits not in (32, 64):
            raise ExecutionMemoryError("memory map has unknown address width")
        evidence = record.get("file_pins")
        if not isinstance(evidence, list) or not evidence:
            raise ExecutionMemoryError("memory map lacks source/elaboration evidence")
        for declared in evidence:
            if (
                not isinstance(declared, dict)
                or not isinstance(declared.get("path"), str)
                or not _digest(declared.get("sha256"))
            ):
                raise ExecutionMemoryError("memory map has malformed evidence")
            actual = _pin(declared["path"])
            if actual["sha256"] != declared["sha256"]:
                raise ExecutionMemoryError(f"memory evidence changed: {declared['path']}")
            if "bytes" in declared and (type(declared["bytes"]) is not int or declared["bytes"] != actual["bytes"]):
                raise ExecutionMemoryError(f"memory evidence extent changed: {declared['path']}")
        regions = record.get("regions")
        if not isinstance(regions, list) or not regions:
            raise ExecutionMemoryError("memory map lacks decoded regions")
        intervals, names = [], set()
        for region in regions:
            if (
                not isinstance(region, dict)
                or not isinstance(region.get("name"), str)
                or not region["name"]
                or region["name"] in names
            ):
                raise ExecutionMemoryError("memory map has missing or duplicate region names")
            names.add(region["name"])
            begin, end = _extent(region["name"], region.get("begin"), region.get("bytes"), bits)
            _permissions(region.get("permissions"), region["name"])
            intervals.append((begin, end))
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ExecutionMemoryError("decoded memory regions overlap")
        if _pin(self.path)["sha256"] != self.sha256:
            raise ExecutionMemoryError("selected memory map changed during validation")
        return record


@dataclass(frozen=True)
class MemoryReservation:
    """One absolute runtime allocation, live for the complete execution.

    It must be disjoint from other reservations and loaded segments. Storage
    inside ELF allocations is already covered by the segment, and is not an
    additional reservation. This API does not grant aliasing or phase reuse.
    """

    name: str
    begin: int
    bytes: int
    permissions: str = "RW"
    alignment: int = 1


def _covered(begin: int, end: int, permissions: frozenset[str], regions: list[dict]) -> bool:
    cursor = begin
    for region in sorted(regions, key=lambda item: item["begin"]):
        if not permissions <= frozenset(region["permissions"]):
            continue
        lo, hi = region["begin"], region["begin"] + region["bytes"]
        if hi <= cursor:
            continue
        if lo > cursor:
            return False
        cursor = hi
        if cursor >= end:
            return True
    return False


def admit_execution_memory(
    elf: str | Path, binding: MemoryMapBinding, reservations: tuple[MemoryReservation, ...]
) -> dict:
    """Refuse unknown maps, gaps, overflow, missing permissions and collisions.

    The caller/provider remains responsible for complete demands and for tying
    map derivation to the actually selected execution engine. Passing this gate
    proves interval admission, not a numerical or performance result.
    """
    if not isinstance(binding, MemoryMapBinding):
        raise ExecutionMemoryError("an explicit selected memory map is required")
    record = binding.validate()
    elf = Path(elf)
    original_pin = _pin(elf)
    with elf.open("rb") as stream:
        ident = stream.read(20)
    if len(ident) != 20 or ident[:4] != b"\x7fELF" or ident[4] not in (1, 2) or ident[5] not in (1, 2) or ident[6] != 1:
        raise ExecutionMemoryError("unknown ELF address width")
    if int.from_bytes(ident[16:18], "little" if ident[5] == 1 else "big") != 2:
        raise ExecutionMemoryError("memory admission requires an absolute executable ELF")
    if (32 if ident[4] == 1 else 64) != record["address_bits"]:
        raise ExecutionMemoryError("ELF and selected memory map address widths disagree")
    entry, segments, _ = elf_audit.read_elf(elf, strict=True)
    intervals, loads = [], []
    for index, segment in enumerate(segments):
        name = f"LOAD[{index}]"
        begin, end = _extent(name, segment.vaddr, segment.memsz, record["address_bits"])
        permissions = _permissions(segment.flags, name)
        if segment.paddr != segment.vaddr:
            raise ExecutionMemoryError(f"{name}: identity physical mapping is unproved")
        if not _covered(begin, end, permissions, record["regions"]):
            raise ExecutionMemoryError(f"{name}: outside decoded map or required permissions")
        intervals.append((begin, end, name))
        loads.append({"name": name, "begin": begin, "bytes": segment.memsz, "permissions": segment.flags})
    if not any(segment.vaddr <= entry < segment.end and "X" in segment.flags for segment in segments):
        raise ExecutionMemoryError("entry is outside executable loaded memory")
    runtime, names = [], set()
    if not isinstance(reservations, tuple):
        raise ExecutionMemoryError("runtime reservations must be an immutable tuple")
    for use in reservations:
        if not isinstance(use, MemoryReservation) or not isinstance(use.name, str) or not use.name or use.name in names:
            raise ExecutionMemoryError("runtime reservation lacks a distinct name")
        names.add(use.name)
        begin, end = _extent(use.name, use.begin, use.bytes, record["address_bits"], use.alignment)
        permissions = _permissions(use.permissions, use.name)
        if not _covered(begin, end, permissions, record["regions"]):
            raise ExecutionMemoryError(f"{use.name}: outside decoded map or required permissions")
        intervals.append((begin, end, use.name))
        runtime.append(
            {
                "name": use.name,
                "begin": begin,
                "bytes": use.bytes,
                "permissions": use.permissions,
                "alignment": use.alignment,
            }
        )
    intervals.sort()
    for left, right in zip(intervals, intervals[1:]):
        if left[1] > right[0]:
            raise ExecutionMemoryError(f"{left[2]} and {right[2]} overlap; storage ownership is unproved")
    if _pin(elf) != original_pin:
        raise ExecutionMemoryError("ELF changed during memory admission")
    binding.validate()
    return {
        "schema": "merlin.execution-memory-admission.v1",
        "status": "pass",
        "elf": original_pin,
        "memory_map": _pin(binding.path),
        "execution_identity": record["execution_identity"],
        "address_bits": record["address_bits"],
        "file_pins": record["file_pins"],
        "loaded_intervals": loads,
        "runtime_reservations": runtime,
        "scope": (
            "Identity-mapped decoded intervals and complete-execution disjoint reservations only; "
            "no numerical or timing certification."
        ),
    }
