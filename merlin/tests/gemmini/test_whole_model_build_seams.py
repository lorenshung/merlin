"""The whole-model build's reusable seams: package identity, corpus binding, stage timing, compiles."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.common.tree_hash import hash_tree
from merlin.perf import whole_model_build as W


def test_the_package_digest_is_the_tree_hash_a_store_keys_packages_on(tmp_path):
    """ONE digest names a package: a measurement store and a harness package repository key it with
    `tree_hash.hash_tree`, so a build record must too (build/ and __pycache__ are not the package)."""
    (tmp_path / "manifest.yaml").write_text("name: p\n")
    (tmp_path / "__pycache__").mkdir()
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"cache")
    digest = W.package_digest(tmp_path)
    assert digest == hash_tree(tmp_path)["sha256"]
    (tmp_path / "__pycache__" / "x.pyc").write_bytes(b"other cache")
    assert W.package_digest(tmp_path) == digest, "an interpreter cache is not the package"
    (tmp_path / "manifest.yaml").write_text("name: q\n")
    assert W.package_digest(tmp_path) != digest
    with pytest.raises(W.WholeModelBuildError, match="does not exist"):
        W.package_digest(tmp_path / "absent")


def test_a_package_build_refuses_without_the_recipe_its_corpus_is_built_under():
    """A group is asked as the capsule the corpus writes for it, so the binding comes from the Phase 0
    recipe's datapath block -- explicitly named, never discovered or assumed."""
    with pytest.raises(W.WholeModelBuildError, match="phase0_recipe"):
        W.corpus_binder("gemmini", phase0_recipe=None)


def test_the_corpus_binding_record_names_the_recipe_it_was_derived_from(tmp_path):
    recipe = repo_root() / "examples" / "gemmini" / "phase0" / "recipe.yaml"
    bound = W.corpus_binder("gemmini", phase0_recipe=recipe)
    assert bound.record["phase0_recipe_sha256"] == W._sha256(recipe)
    assert bound.record["datapath"] and callable(bound.binder)
    empty = tmp_path / "recipe.yaml"
    empty.write_text("capsules: []\n")
    with pytest.raises(W.WholeModelBuildError, match="no datapath"):
        W.corpus_binder("gemmini", phase0_recipe=empty)


def test_every_stage_is_timed_in_the_order_it_ran():
    clock = W._StageClock()
    with clock("statement"):
        time.sleep(0.01)
    with clock("group_objects"):
        pass
    with clock("statement"):
        pass
    record = clock.record()
    assert list(record["stages"]) == ["statement", "group_objects"]
    assert record["stages"]["statement"] >= 0.01 and record["total"] >= record["stages"]["statement"]


def test_an_underivable_argument_order_is_a_named_refusal_not_a_guess(monkeypatch):
    from merlin.targetgen import rtl_checks

    def unavailable(target):
        raise rtl_checks.RtlChecksUnavailable(f"no checks for {target}")

    monkeypatch.setattr(rtl_checks, "selected_checks", unavailable)
    assert W._kernel_arg_order("nowhere", {}) == ([], "", "no checks for nowhere")


def test_distinct_group_artifacts_compile_concurrently_and_identical_ones_once(tmp_path, monkeypatch):
    """One lock per distinct artifact: two different groups' objects are built at the same time, and
    two groups with byte-identical artifacts share one compile."""
    from merlin.llvmlower import toolchain
    from merlin.targetgen.contract import compile as compile_mod
    from merlin.targetgen.contract import harness_abi

    monkeypatch.setenv(W.OBJECT_CACHE_DISABLE_ENV, "1")
    active, peak, calls = [0], [0], []
    lock = threading.Lock()

    def compile_one(text, work, *, target):
        with lock:
            active[0] += 1
            peak[0] = max(peak[0], active[0])
            calls.append(text)
        time.sleep(0.2)
        with lock:
            active[0] -= 1
        work.mkdir(parents=True, exist_ok=True)
        (work / "kernel.o").write_bytes(text.encode())
        return work / "kernel.o"

    monkeypatch.setattr(compile_mod, "llvm_mlir_to_object", compile_one)
    monkeypatch.setattr(harness_abi, "for_target", lambda target: type("A", (), {"entry_symbol": "k"})())

    def rename(obj, old, new, out):
        Path(out).write_bytes(Path(obj).read_bytes())

    monkeypatch.setattr(W, "_rename_kernel_symbol", rename, raising=False)
    monkeypatch.setattr(toolchain, "objcopy", lambda: Path("/bin/true"))
    rows = []
    for index, text in enumerate(["a", "b", "a"]):
        artifact = tmp_path / f"g{index}.artifact.txt"
        artifact.write_text(text)
        rows.append({"group": index, "on": W.ON_PACKAGE, "artifact": str(artifact), "symbol": f"k_g{index}"})
    try:
        W._kernel_objects(rows, target="gemmini", out=tmp_path / "objects", jobs=3)
    except Exception:  # noqa: BLE001 -- the symbol check after the compile is not what this measures
        pass
    assert sorted(calls) == ["a", "b"], "identical artifacts compile once"
    assert peak[0] == 2, "different artifacts compile at the same time"


def test_the_object_stage_reports_how_many_distinct_artifacts_it_compiled(tmp_path, monkeypatch):
    """A repeated model block emits byte-identical artifacts; the build states the dedupe it bought."""
    from merlin.llvmlower import toolchain
    from merlin.targetgen.contract import compile as compile_mod
    from merlin.targetgen.contract import harness_abi

    monkeypatch.setenv(W.OBJECT_CACHE_DISABLE_ENV, "1")

    def compile_one(text, work, *, target):
        work.mkdir(parents=True, exist_ok=True)
        (work / "kernel.o").write_bytes(text.encode())
        return work / "kernel.o"

    monkeypatch.setattr(compile_mod, "llvm_mlir_to_object", compile_one)
    monkeypatch.setattr(harness_abi, "for_target", lambda target: type("A", (), {"entry_symbol": "k"})())
    monkeypatch.setattr(toolchain, "objcopy", lambda: Path("/bin/true"))
    rows = []
    for index, text in enumerate(["a", "b", "a", "a"]):
        artifact = tmp_path / f"g{index}.artifact.txt"
        artifact.write_text(text)
        rows.append({"group": index, "on": W.ON_PACKAGE, "artifact": str(artifact)})
    assert W._kernel_objects(rows, target="gemmini", out=tmp_path / "objects", jobs=2) == {
        "unique_signatures": 2,
        "total_groups": 4,
    }
