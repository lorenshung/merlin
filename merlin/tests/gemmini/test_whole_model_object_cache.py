"""The whole-model build's per-group compiled-object cache (deliverable: incremental build).

A full whole-model build recompiles every package-answered group's LLVM artifact to a riscv object,
even when an edit changed one group's kernel. :func:`merlin.perf.whole_model_build._kernel_objects`
now reuses a PRIOR object for a group whose artifact text is byte-identical to one already compiled
(same target, same toolchain fingerprint) instead of recompiling it.

These tests fake the compile step (``llvm_mlir_to_object``, the same seam
``test_host_prepack_harness.py`` and ``test_stack_frame_preflight.py`` fake) so they need no LLVM
lowering, but the RENAME AND SYMBOL CHECK after it are real: each "compiled" object is a real riscv
object built once with the real toolchain, so ``objcopy``/``nm`` verify a real linked symbol exactly as
a production build does. The cache is exercised end to end (write in one build, read in the next),
isolated per test under ``MERLIN_OUT_ROOT``.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower import toolchain
from merlin.perf import whole_model_build as W
from merlin.runtime.backends import base as backends
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract import harness_abi

pytestmark = pytest.mark.target("gemmini")

_TARGET = "gemmini"


def _entry_symbol() -> str:
    return harness_abi.for_target(_TARGET).entry_symbol


def _stub_object(tmp_path: Path, name: str, *, returns: int) -> Path:
    """A real riscv64 object defining the target's entry symbol, built with the real toolchain once.

    Distinct ``returns`` values make distinct object bytes, so a test can tell "the cache served THIS
    content's object" from "it served some other one" by comparing ``object_sha256`` -- not just by
    counting compile calls.
    """
    src = tmp_path / f"{name}.c"
    src.write_text(f"int {_entry_symbol()}(void) {{ return {returns}; }}\n", encoding="utf-8")
    obj = tmp_path / f"{name}.o"
    subprocess.run(
        [
            str(toolchain.clang()),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-mcmodel=medany",
            "-O2",
            "-ffreestanding",
            "-fno-builtin",
            "-c",
            str(src),
            "-o",
            str(obj),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return obj


def _fake_compiler(mapping: dict[str, Path], calls: list[str]):
    """A deterministic stand-in for ``llvm_mlir_to_object``: same artifact text -> same object bytes,
    every time -- what a real compiler is assumed to do (no ``-g``, no embedded paths; see
    ``_kernel_objects``'s docstring). Records every text it was actually asked to compile."""

    def compile_one(text: str, workdir: Path, *, target=None, entry_symbol=None, _build_service=None):
        calls.append(text)
        workdir = Path(workdir)
        workdir.mkdir(parents=True, exist_ok=True)
        dest = workdir / "kernel.o"
        dest.write_bytes(Path(mapping[text]).read_bytes())
        return dest

    return compile_one


def _row(group: int, artifact: Path) -> dict:
    return {"group": group, "on": W.ON_PACKAGE, "artifact": str(artifact)}


def test_object_cache_key_is_a_pure_function_of_everything_that_decides_the_bytes(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    fp = W._object_cache_fingerprint(_TARGET)
    assert fp == W._object_cache_fingerprint(_TARGET), "same target, same process: the same fingerprint"

    key_a = W._object_cache_key("text-a", target=_TARGET, fingerprint=fp)
    assert key_a == W._object_cache_key("text-a", target=_TARGET, fingerprint=fp), "pure function of its inputs"
    assert key_a != W._object_cache_key("text-b", target=_TARGET, fingerprint=fp), "different text, different key"
    assert key_a != W._object_cache_key("text-a", target="other-target", fingerprint=fp), "target is part of the key"

    # A different march (read through the target's own recipe, never hardcoded) changes the fingerprint,
    # because it changes the bytes clang would emit for the same LLVM IR.
    import dataclasses

    real_recipe = backends.harness_build_recipe(_TARGET)
    monkeypatch.setattr(backends, "harness_build_recipe", lambda target: dataclasses.replace(real_recipe))
    assert W._object_cache_fingerprint(_TARGET) == fp, "an unchanged recipe reproduces the same fingerprint"

    other_march = "-march=rv64gcv" if real_recipe.march() != "-march=rv64gcv" else "-march=rv64gc"
    changed_march = dataclasses.replace(
        real_recipe, cflags=tuple(other_march if f.startswith("-march=") else f for f in real_recipe.cflags)
    )
    monkeypatch.setattr(backends, "harness_build_recipe", lambda target: changed_march)
    assert W._object_cache_fingerprint(_TARGET) != fp, "a different -march= must invalidate every cache entry"

    # A tighter stack-frame budget alone -- no source file touched, no march changed -- can turn on the
    # arena-rewrite repair path and change the emitted bytes for the identical input, so it must move
    # the fingerprint too (this was a real gap: an earlier version keyed only on march + two hand-picked
    # modules and would have kept serving an object validated against a stale, looser budget).
    tighter_policy = dataclasses.replace(
        real_recipe,
        kernel_stack_frame=dataclasses.replace(
            real_recipe.kernel_stack_frame, max_static_bytes=max(1, real_recipe.kernel_stack_frame.max_static_bytes - 1)
        ),
    )
    monkeypatch.setattr(backends, "harness_build_recipe", lambda target: tighter_policy)
    assert W._object_cache_fingerprint(_TARGET) != fp, "a tighter stack-frame budget must invalidate the cache"


def test_kernel_objects_reuses_a_byte_identical_artifact_and_recompiles_a_changed_one(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    monkeypatch.delenv(W.OBJECT_CACHE_DISABLE_ENV, raising=False)

    stub_a = _stub_object(tmp_path, "a", returns=1)
    stub_b = _stub_object(tmp_path, "b", returns=2)
    stub_b2 = _stub_object(tmp_path, "b2", returns=3)

    content_a = "kernel content A (group 1, unedited)"
    content_b = "kernel content B (group 2, before the edit)"
    content_b2 = "kernel content B, edited"

    calls: list[str] = []
    monkeypatch.setattr(
        compiler,
        "llvm_mlir_to_object",
        _fake_compiler({content_a: stub_a, content_b: stub_b, content_b2: stub_b2}, calls),
    )

    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    a1 = art_dir / "g1_build1.artifact.txt"
    a1.write_text(content_a, encoding="utf-8")
    b1 = art_dir / "g2_build1.artifact.txt"
    b1.write_text(content_b, encoding="utf-8")

    rows_build1 = [_row(1, a1), _row(2, b1)]
    W._kernel_objects(rows_build1, target=_TARGET, out=tmp_path / "build1" / "objects", jobs=1)

    assert calls == [content_a, content_b], "a cold cache: both groups compile"
    assert [r.get("object_cache") for r in rows_build1] == ["miss", "miss"]
    assert all(r.get("object_sha256") for r in rows_build1)

    # BUILD 2: group 1's artifact is byte-identical (same content, a fresh file -- a real second build
    # writes it to a new per-build path); group 2's kernel changed.
    a2 = art_dir / "g1_build2.artifact.txt"
    a2.write_text(content_a, encoding="utf-8")
    b2 = art_dir / "g2_build2.artifact.txt"
    b2.write_text(content_b2, encoding="utf-8")

    rows_build2 = [_row(1, a2), _row(2, b2)]
    W._kernel_objects(rows_build2, target=_TARGET, out=tmp_path / "build2" / "objects", jobs=1)

    assert calls == [content_a, content_b, content_b2], "only the CHANGED group's kernel is recompiled"
    assert [r.get("object_cache") for r in rows_build2] == ["hit", "miss"]

    # THE REUSED OBJECT IS BYTE-IDENTICAL to what a fresh build produces for the same input: compare
    # against a THIRD build of the unchanged content with the cache disabled (a forced fresh compile).
    monkeypatch.setenv(W.OBJECT_CACHE_DISABLE_ENV, "1")
    a3 = art_dir / "g1_build3_nocache.artifact.txt"
    a3.write_text(content_a, encoding="utf-8")
    rows_build3 = [_row(1, a3)]
    W._kernel_objects(rows_build3, target=_TARGET, out=tmp_path / "build3" / "objects", jobs=1)
    assert calls == [content_a, content_b, content_b2, content_a], "cache disabled: a fresh compile, always"
    assert rows_build3[0]["object_cache"] == "miss"

    assert rows_build2[0]["object_sha256"] == rows_build1[0]["object_sha256"] == rows_build3[0]["object_sha256"], (
        "the cached-and-reused group-1 object is the exact bytes a fresh build (cache disabled) produces "
        "for the same artifact text"
    )
    assert rows_build2[1]["object_sha256"] != rows_build1[1]["object_sha256"], "group 2's kernel really changed"

    # The renamed, linked symbol is checked on every group, cached or fresh -- never skipped.
    for row in (*rows_build1, *rows_build2, *rows_build3):
        listed = subprocess.run(
            [str(toolchain.nm()), "--defined-only", row["object"]], check=True, capture_output=True, text=True
        ).stdout
        names = {line.split()[-1] for line in listed.splitlines() if line.split()}
        assert row["symbol"] in names
        assert _entry_symbol() not in names


def test_object_cache_is_bounded_least_recently_used(tmp_path, monkeypatch):
    """A long tuning loop tries many distinct candidate kernels per group -- each a new key -- so the
    cache must not grow without bound on a disk-tight host: it evicts the coldest entry first."""
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    monkeypatch.delenv(W.OBJECT_CACHE_DISABLE_ENV, raising=False)

    stub_a = _stub_object(tmp_path, "a", returns=1)
    stub_b = _stub_object(tmp_path, "b", returns=2)
    one_object_bytes = stub_a.stat().st_size
    monkeypatch.setenv(W.OBJECT_CACHE_MAX_BYTES_ENV, str(one_object_bytes + 1))

    content_a, content_b = "cache-bound content A", "cache-bound content B"
    calls: list[str] = []
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", _fake_compiler({content_a: stub_a, content_b: stub_b}, calls))

    art_dir = tmp_path / "artifacts"
    art_dir.mkdir()
    a1 = art_dir / "g1.artifact.txt"
    a1.write_text(content_a, encoding="utf-8")
    W._kernel_objects([_row(1, a1)], target=_TARGET, out=tmp_path / "b1" / "objects", jobs=1)
    assert calls == [content_a]

    # A second, distinct group evicts group 1's cache entry (the budget holds at most one object).
    b1 = art_dir / "g2.artifact.txt"
    b1.write_text(content_b, encoding="utf-8")
    W._kernel_objects([_row(2, b1)], target=_TARGET, out=tmp_path / "b2" / "objects", jobs=1)
    assert calls == [content_a, content_b]

    root = W._object_cache_root()
    kept = sum(f.stat().st_size for f in root.glob("*/*.o"))
    assert kept <= one_object_bytes + 1, "the cache stays under its configured budget"

    # Group 1's artifact, asked again, is a cache MISS: its entry was evicted, not merely aged.
    a2 = art_dir / "g1_again.artifact.txt"
    a2.write_text(content_a, encoding="utf-8")
    rows = [_row(1, a2)]
    W._kernel_objects(rows, target=_TARGET, out=tmp_path / "b3" / "objects", jobs=1)
    assert calls == [content_a, content_b, content_a]
    assert rows[0]["object_cache"] == "miss"


def test_a_disabled_cache_never_reads_or_writes_it(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path))
    monkeypatch.setenv(W.OBJECT_CACHE_DISABLE_ENV, "1")
    stub = _stub_object(tmp_path, "solo", returns=7)
    content = "solo kernel"
    calls: list[str] = []
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", _fake_compiler({content: stub}, calls))
    art = tmp_path / "solo.artifact.txt"
    art.write_text(content, encoding="utf-8")

    for n in (1, 2):
        rows = [_row(1, art)]
        W._kernel_objects(rows, target=_TARGET, out=tmp_path / f"build{n}" / "objects", jobs=1)
        assert rows[0].get("object_cache") == "miss", "cache disabled: every build is reported a fresh compile"

    assert calls == [content, content], "every build recompiles; nothing was cached"
    assert W._object_cache_root() is None
