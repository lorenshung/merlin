"""Explicit prerequisites for tests that invoke the external host compiler."""

import shutil

import pytest

from merlin.llvmlower import toolchain


@pytest.fixture
def upstream_host_tools():
    """Skip only an unavailable tool; compilation and runtime failures still fail."""
    tools = {
        "compiler interpreter": toolchain.compiler_python(),
        "Clang": toolchain.clang(),
        "MLIR translator": toolchain.mlir_translate(),
    }
    missing = [name for name, path in tools.items() if shutil.which(str(path)) is None]
    if missing:
        pytest.skip("external host compiler tools unavailable: " + ", ".join(missing))
