"""Explicit source-bound linked-artifact checks before simulator dispatch.

This transport validates unchanged artifact/report bytes and callback ownership.
Its caller supplies the independently evaluated policy. It grants no physical,
instruction-effect, numerical or timing authority and never resolves a backend.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from merlin.common.strict_json import loads

from .build_service import file_digest


@dataclass(frozen=True)
class LinkedElfAdmissionService:
    target: str
    evaluator: Callable
    source_pins: tuple[tuple[str, str], ...]

    def verify(self, target):
        if (
            type(self) is not LinkedElfAdmissionService
            or target != self.target
            or type(self.source_pins) is not tuple
            or not self.source_pins
            or len(dict(self.source_pins)) != len(self.source_pins)
        ):
            raise ValueError("linked ELF admission requires an exact target and source selection")
        for path, digest in self.source_pins:
            member = Path(path)
            if (
                not member.is_absolute()
                or any(parent.is_symlink() for parent in (member, *member.parents))
                or not member.is_file()
                or file_digest(member) != digest
            ):
                raise ValueError("linked ELF admission source pin changed: " + str(path))
        owner = inspect.getsourcefile(self.evaluator) if callable(self.evaluator) else None
        if owner is None or (str(Path(owner).resolve()), file_digest(Path(owner))) not in self.source_pins:
            raise ValueError("linked ELF admission evaluator has no pinned inspected source owner")
        return {"target": target, "source_pins": self.source_pins}

    def evaluate(self, *, elf, target, evidence_root):
        before = self.verify(target)
        artifact = Path(elf)
        digest = file_digest(artifact)
        result = self.evaluator(elf=artifact, evidence_root=Path(evidence_root))
        if type(result) is not dict or result.get("status") not in ("accepted", "refused"):
            raise ValueError("linked ELF admission returned no actual accepted/refused observation")
        if self.verify(target) != before:
            raise ValueError("linked ELF admission selection changed during evaluation")
        self.revalidate(elf=artifact, result=result, target=target)
        if result["elf_sha256"] != digest:
            raise ValueError("linked ELF changed during admission evaluation")
        return result

    def revalidate(self, *, elf, result, target):
        self.verify(target)
        report = Path(result.get("report_path", ""))
        if (
            not report.is_absolute()
            or any(parent.is_symlink() for parent in (report, *report.parents))
            or not report.is_file()
            or file_digest(report) != result.get("report_sha256")
        ):
            raise ValueError("linked ELF admission report has no unchanged actual file")
        stored = loads(report.read_text())
        if type(stored) is not dict or any(stored.get(key) != result.get(key) for key in ("status", "elf_sha256")):
            raise ValueError("linked ELF admission report disagrees with its evaluated result")
        if file_digest(Path(elf)) != result.get("elf_sha256"):
            raise ValueError("linked ELF changed after admission")
        return result["status"]
