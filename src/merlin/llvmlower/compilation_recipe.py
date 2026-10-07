"""Observe owned compiler/link invocations and their actual file inputs/outputs.

This is compilation evidence, not a hermetic toolchain lock or a cache identity.
Headers, library resolution, imported providers and non-project environment need
their own closure. Callers name inputs; this module never infers a target ABI.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

from merlin.common.digest import sha256_file
from merlin.common.jsonio import write_pretty_json

from . import codegen_env

FILENAME = "compilation_recipe.json"


def _identity(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": sha256_file(path), "bytes": path.stat().st_size}


class CompilationRecipe:
    """One build's observed commands, completed only after the caller's audits."""

    def __init__(self, work: Path, *, producer: Path):
        self.path = work / FILENAME
        self.record = {
            "schema": "merlin.compilation_recipe.v1",
            "scope": "caller-owned compilation and link; provider compilation and toolchain dependency closure unknown",
            "producer": _identity(producer),
            "status": "prepared",
            "commands": [],
        }
        write_pretty_json(self.path, self.record)

    def bind_preparation(self, name: str, path: Path) -> None:
        """Bind a caller-selected numerical preparation input before compilation."""
        if self.record["status"] != "prepared" or not name or not path.is_file():
            raise ValueError("compilation preparation is absent or already invoked")
        self.record.setdefault("preparation", {})[name] = _identity(path)
        write_pretty_json(self.path, self.record)

    def run(
        self,
        argv: Sequence[str | Path],
        *,
        runner: Callable[..., Any],
        inputs: Iterable[Path],
        output: Path,
        cwd: Path | None = None,
    ) -> Any:
        """Launch the original command after retaining its explicit input bytes.

        A command error leaves an invoked record without a returned output,
        even when an older artifact remains at the requested output path.
        """
        executable = shutil.which(str(argv[0]))
        if executable is None:
            raise FileNotFoundError(str(argv[0]))
        environment = codegen_env.snapshot()
        command = {
            "argv": [str(argument) for argument in argv],
            "cwd": str((cwd or Path.cwd()).resolve()),
            "executable": _identity(Path(executable)),
            "environment": environment,
            "environment_digest": codegen_env.digest(environment),
            "inputs": [_identity(path) for path in inputs],
            "requested_output": str(output.resolve()),
            "status": "invoked",
        }
        self.record["commands"].append(command)
        self.record["status"] = "invoked"
        write_pretty_json(self.path, self.record)
        result = runner(argv, **({"cwd": cwd} if cwd is not None else {}))
        if result.returncode != 0:
            raise RuntimeError(f"compilation command returned {result.returncode}")
        command.update(status="returned", output=_identity(output))
        write_pretty_json(self.path, self.record)
        return result

    def completed(self, executable: Path) -> None:
        """Bind the final executable only after successful linking and audits."""
        if not self.record["commands"] or any(command["status"] != "returned" for command in self.record["commands"]):
            raise ValueError("compilation has no complete successful command sequence")
        if any(
            not Path(identity["path"]).is_file()
            or _identity(Path(identity["path"])) != identity
            for identity in self.record.get("preparation", {}).values()
        ):
            raise ValueError("compilation preparation changed before completion")
        self.record.update(status="completed", executable=_identity(executable))
        write_pretty_json(self.path, self.record)
