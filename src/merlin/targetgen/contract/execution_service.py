"""Explicit functional execution transport without target-backend discovery.

The caller owns independent runtime qualification. Source pins and selected
callbacks here provide attribution and drift checks, not ISA semantics, hardware
timing or correctness authority. This transport cannot label counters as RTL.
"""

from __future__ import annotations

import inspect
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .build_service import file_digest


@dataclass(frozen=True)
class FunctionalExecutionService:
    target: str
    simulator: str
    runner: Callable
    parser: Callable
    source_pins: tuple[tuple[str, str], ...]
    engine_json: str

    def verify(self, target: str, simulator: str) -> dict:
        if (
            type(self) is not FunctionalExecutionService
            or target != self.target
            or simulator != self.simulator
            or not isinstance(self.source_pins, tuple)
            or not self.source_pins
            or len(dict(self.source_pins)) != len(self.source_pins)
        ):
            raise ValueError("functional execution requires an exact target/engine transport")
        for path, expected in self.source_pins:
            member = Path(path)
            if (
                not member.is_absolute()
                or any(parent.is_symlink() for parent in (member, *member.parents))
                or not member.is_file()
                or file_digest(member) != expected
            ):
                raise ValueError("functional execution source/tool pin changed: " + str(path))
        for callback in (self.runner, self.parser):
            owner = inspect.getsourcefile(callback) if callable(callback) else None
            if owner is None or (str(Path(owner).resolve()), file_digest(Path(owner))) not in self.source_pins:
                raise ValueError("functional execution callback has no pinned inspected source owner")
        from merlin.common.strict_json import loads

        engine = loads(self.engine_json)
        if type(engine) is not dict or not engine:
            raise ValueError("functional execution requires an explicit selected-engine citation")
        return {
            "target": target,
            "simulator": simulator,
            "engine": engine,
            "source_pins": [{"path": path, "sha256": digest} for path, digest in self.source_pins],
            "scope": "functional transport only; ISA semantics and hardware/cost authority unqualified",
        }

    def run_elf(self, elf, *, simulator, timeout, **kwargs):
        before = self.verify(self.target, simulator)
        result = self.runner(elf, simulator=simulator, timeout=timeout, **kwargs)
        if self.verify(self.target, simulator) != before:
            raise ValueError("functional execution selection changed during invocation")
        return result

    def parse_output(self, console):
        self.verify(self.target, self.simulator)
        return self.parser(console)

    @property
    def oracle(self) -> dict:
        return {
            "kind": "functional_diagnostic",
            "derived_from_rtl": False,
            "hardware_timing": "unqualified",
            "engine": json.loads(self.engine_json),
        }
