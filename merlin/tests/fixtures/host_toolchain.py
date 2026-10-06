"""Skip a test whose SUBJECT runs the host LLVM/MLIR toolchain on a host that does not have it.

The lowering path runs upstream MLIR passes inside a compiler Python (``MERLIN_COMPILER_PYTHON``,
``MERLIN_COMPILER_VENV``, ``MERLIN_M2M_VENV`` or the model2MLIR checkout's ``.venv``) and compiles with
clang-23 (``MERLIN_CLANG``, ``MERLIN_MLIR_INSTALL`` or the checkout's ``third_party/llvm-install``). None of
them ships in this repo, and a hosted CI runner has none, so a test that lowers through them must skip
there rather than fail on a missing interpreter deep inside the library.

Resolution is :mod:`merlin.llvmlower.toolchain`'s own, so a skip names exactly the tool the library
would have run. Only ABSENCE skips: a toolchain that is present and then fails still fails the test.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest


def _present(tool: Path) -> bool:
    return tool.is_file() or (not tool.is_absolute() and shutil.which(str(tool)) is not None)


def missing_lowering() -> list[str]:
    """The pieces of the host lowering toolchain this host lacks, named as the library resolves them."""
    from merlin.llvmlower import toolchain

    absent = []
    python = toolchain.compiler_python()
    if not _present(python):
        absent.append(f"compiler Python {python} (MERLIN_COMPILER_PYTHON / MERLIN_M2M_VENV)")
    clang = toolchain.clang()
    if not _present(clang):
        absent.append(f"clang {clang} (MERLIN_CLANG / MERLIN_MLIR_INSTALL)")
    return absent


def requires_lowering():
    """A ``skipif`` marker for a test that lowers through the host compiler Python and clang."""
    absent = missing_lowering()
    return pytest.mark.skipif(bool(absent), reason=f"host lowering toolchain not available: {'; '.join(absent)}")


def missing_checkout_llvm(*tools: str) -> list[str]:
    """The ``tools`` absent from this checkout's own ``third_party/llvm-install/bin``.

    For the trusted graders that resolve their tools from the checkout ONLY, by design, and never from
    the environment: the skip must test the same location the grader will read."""
    from merlin.common.paths import repo_root

    bin_dir = repo_root() / "third_party" / "llvm-install" / "bin"
    return [str(bin_dir / tool) for tool in tools if not (bin_dir / tool).is_file()]


def requires_checkout_llvm(*tools: str):
    """A ``skipif`` marker for a test of a grader that reads the checkout's pinned LLVM install."""
    absent = missing_checkout_llvm(*tools)
    return pytest.mark.skipif(bool(absent), reason=f"checkout LLVM install not available: {', '.join(absent)}")
