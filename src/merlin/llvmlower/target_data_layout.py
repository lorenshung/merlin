"""The LLVM data layout of a cross target, read from the compiler that compiles for it.

MLIR translates a module to LLVM IR under the module's ``llvm.data_layout``, or under LLVM's default
layout when it carries none -- and the default aligns 64-bit integers and doubles to 4 bytes. Every
load and store the translation emits then states ``align 4``, which the cross compiler keeps even after
it substitutes its own layout; on a vector host an under-aligned 64-bit access has no legal vector form,
so a loop over int64 or f64 data is not vectorized at all. The layout is a property of the target
(triple and ISA flags), so it is ASKED of the compiler with the build's own cross flags -- never written
here. Nothing here names a target.
"""

from __future__ import annotations

import subprocess
import threading
from collections.abc import Sequence
from pathlib import Path

_PREFIX = 'target datalayout = "'
_KNOWN: dict[tuple[str, ...], str] = {}
_LOCK = threading.Lock()


def parse(llvm_ir: str) -> str | None:
    """The layout string of a ``target datalayout = "..."`` line in ``llvm_ir``, or ``None``."""
    for line in llvm_ir.splitlines():
        stripped = line.strip()
        if stripped.startswith(_PREFIX) and stripped.endswith('"'):
            return stripped[len(_PREFIX) : -1]
    return None


def of(clang: str | Path, cross_flags: Sequence[str]) -> str:
    """The data layout ``clang`` uses for ``cross_flags``; raises when it states none."""
    key = (str(clang), *map(str, cross_flags))
    with _LOCK:
        if key in _KNOWN:
            return _KNOWN[key]
    flags = [str(f) for f in cross_flags if str(f) not in ("-c",)]
    done = subprocess.run(
        [str(clang), *flags, "-S", "-emit-llvm", "-x", "c", "-", "-o", "-"],
        input="",
        capture_output=True,
        text=True,
        timeout=120,
    )
    layout = parse(done.stdout) if done.returncode == 0 else None
    if not layout:
        raise RuntimeError(f"{clang} states no data layout for {flags}: {done.stderr[-500:]}")
    with _LOCK:
        _KNOWN[key] = layout
    return layout
