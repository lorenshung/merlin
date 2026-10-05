"""The native FNV-1a word helper answers exactly what the interpreter loop does, or is not used."""

from __future__ import annotations

import numpy as np
import pytest

from merlin.common import fnv
from merlin.perf.layer_bench import reference as REF


@pytest.fixture(autouse=True)
def _cache_root(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    fnv._LOADED.clear()
    yield
    fnv._LOADED.clear()


@pytest.mark.parametrize("count", [0, 1, 7, 4096, 100_003])
def test_the_native_digest_equals_the_interpreter_loop(count):
    words = np.random.default_rng(count).integers(0, 1 << 63, count, dtype=np.uint64) * np.uint64(2)
    want = fnv.python_words(words, REF.FNV_OFFSET, REF.FNV_PRIME)
    assert fnv.fnv1a64_words(words, REF.FNV_OFFSET, REF.FNV_PRIME) == want


def test_the_reference_digest_is_unchanged_with_or_without_the_helper(monkeypatch):
    data = np.arange(-5000, 5000, dtype="<i8").tobytes() + b"\x01\x02\x03"  # an unaligned tail is padded
    native = REF.fnv1a64_words(data)
    monkeypatch.setenv(fnv.NATIVE_ENV, "0")
    fnv._LOADED.clear()
    assert REF.fnv1a64_words(data) == native


def test_a_helper_that_disagrees_is_never_used(monkeypatch):
    """The self-check is what makes the native path admissible: a wrong helper falls back."""
    monkeypatch.setattr(fnv, "_SOURCE", fnv._SOURCE.replace("state ^= words[i];", "state += words[i];"))
    words = np.arange(64, dtype=np.uint64)
    assert fnv._native() is None
    assert fnv.fnv1a64_words(words, 5, 7) == fnv.python_words(words, 5, 7)


def test_no_compiler_is_the_interpreter_loop(monkeypatch):
    monkeypatch.setattr(fnv, "_compiler", lambda: None)
    assert fnv._native() is None
    assert fnv.fnv1a64_words(np.arange(3, dtype=np.uint64), 1, 3) == fnv.python_words(
        np.arange(3, dtype=np.uint64), 1, 3
    )
