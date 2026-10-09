"""Inventory explicit reviewed system runtime files for component launch inputs.

This preparation command establishes file membership and byte identities only.
It does not establish namespace support, compiler correctness or model readiness.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2.component_experiment import RuntimeGrant

from .component_runtime_authority import (
    IndependentComponentRuntime as IndependentComponentRuntime,
)
from .component_runtime_authority import (
    IndependentRuntimeServices as IndependentRuntimeServices,
)
from .component_runtime_authority import (
    admit_independent_component_runtime as admit_independent_component_runtime,
)
from .component_runtime_authority import (
    require_independent_runtime as require_independent_runtime,
)


def inventory_runtime(*, files: tuple[tuple[Path, str], ...], trees: tuple[tuple[Path, str], ...],
                      executables: tuple[tuple[Path, str], ...]) -> tuple[RuntimeGrant, ...]:
    """Pin declared files and the actual dynamic-loader dependencies of trusted tools.

    Callers review the input executable owners before running ``ldd``. Extra
    resources (Python modules, certificates, simulator data) must be declared
    explicitly. The launch's actual readiness probes detect incomplete closure.
    """
    index: dict[str, RuntimeGrant] = {}

    def add(source, destination):
        source = Path(source).resolve(strict=True)
        grant = RuntimeGrant(source, destination, C.sha256_file(source))
        grant.verify()
        previous = index.get(destination)
        if previous is not None and previous != grant:
            raise C.StageGateError("runtime inventory has conflicting destination identities")
        index[destination] = grant

    for source, destination in files:
        add(source, destination)
    for root, destination in trees:
        root = Path(root).resolve(strict=True)
        if not root.is_dir():
            raise C.StageGateError("runtime tree is not an explicit reviewed directory")
        for path in sorted(root.rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts:
                add(path, str(Path(destination) / path.relative_to(root)))
    for executable, destination in executables:
        add(executable, destination)
        result = subprocess.run(["ldd", str(Path(executable).resolve(strict=True))], capture_output=True,
                                text=True, timeout=30, env={"PATH": "/usr/bin:/bin", "LC_ALL": "C"})
        # ldd's ordinary diagnostic for a trusted statically linked tool is
        # valid. Any unresolved dynamic dependency remains an inventory error.
        if "not found" in result.stdout:
            raise C.StageGateError("trusted runtime executable has an unresolved dynamic dependency")
        if result.returncode and not any(text in result.stderr + result.stdout for text in
                                        ("not a dynamic executable", "statically linked")):
            raise C.StageGateError("trusted runtime executable dependency inspection failed")
        for line in result.stdout.splitlines():
            # ldd names dependency paths as whitespace-delimited absolute
            # tokens; no target/tool names or pattern parsing are needed.
            for token in line.split():
                if token.startswith("/") and Path(token).is_file():
                    add(Path(token), token)
    return tuple(index[name] for name in sorted(index))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for flag in ("file", "tree", "executable"):
        parser.add_argument("--" + flag, action="append", default=[], metavar="SOURCE=DESTINATION")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    def values(name):
        result = []
        for value in getattr(args, name):
            source, separator, destination = value.partition("=")
            if not separator or not Path(source).is_absolute() or not Path(destination).is_absolute():
                parser.error("runtime membership requires absolute SOURCE=DESTINATION")
            result.append((Path(source), destination))
        return tuple(result)

    if args.output.exists() or args.output.is_symlink():
        parser.error("runtime inventory destination must be fresh")
    grants = inventory_runtime(files=values("file"), trees=values("tree"), executables=values("executable"))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps([{"source": str(row.source), "destination": row.destination,
                                       "sha256": row.sha256} for row in grants], indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
