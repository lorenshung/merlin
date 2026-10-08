"""Give every plan the suite builds a compiler that resolves, unless a test says otherwise.

Preflight refuses a grading phase whose ``toolchain.clang`` is not an executable file, resolved
in the engine's checkout. Most plans here are rooted in a temporary checkout with no LLVM install,
so each would be refused for a reason unrelated to what it tests. ``MERLIN_CLANG`` is pointed at
the host's resolvable compiler when there is one, else at an inert stand-in executable; a test of
the refusal itself overrides it.
"""

from __future__ import annotations

import os

import pytest

# Unset, MERLIN_TARGET_PATH selects every vendored examples/*/support provider of the checkout. These
# tests build their own providers and selections, so the session starts with none selected unless the
# operator chose one (the same rule as merlin/tests/conftest.py).
os.environ.setdefault("MERLIN_TARGET_PATH", "")


@pytest.fixture(autouse=True)
def _resolvable_clang(monkeypatch, tmp_path_factory):
    if os.environ.get("MERLIN_CLANG"):
        return
    from merlin.llvmlower.toolchain import clang

    compiler = clang()
    if not (compiler.is_absolute() and compiler.is_file() and os.access(compiler, os.X_OK)):
        compiler = tmp_path_factory.mktemp("toolchain") / "clang"
        compiler.write_text("#!/bin/sh\nexit 1\n")
        compiler.chmod(0o755)
    monkeypatch.setenv("MERLIN_CLANG", str(compiler))


@pytest.fixture
def tmp_path(tmp_path):
    """pytest's own ``tmp_path``, handed back in a state its retention policy can remove.

    ``tmp_path_retention_policy = "failed"`` deletes a passed test's tree with
    ``shutil.rmtree(ignore_errors=True)``, which silently skips every directory left without owner
    write -- and the sealed snapshots these tests build (bundle inputs, frozen corpora, reviewed
    releases) clear write bits by design. Such a tree outlives the policy, so a multi-GB copy left
    read-only is retained after a passing run. Restore owner write on directories only: unlinking
    needs the parent writable, and files (some hard-linked into a shared content store) keep their
    modes.
    """
    yield tmp_path
    for directory, _subdirs, _files in os.walk(tmp_path):
        try:
            mode = os.lstat(directory).st_mode
            if not mode & 0o200:
                os.chmod(directory, mode | 0o200)
        except OSError:
            pass  # best effort, exactly as pytest's own removal is
