"""A package's own whole-model passes: declared in its manifest, run in order, never trusted past a
failure -- and never executed at all for a package that declares none.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.perf import whole_model_passes as WMPass

pytestmark = pytest.mark.target("gemmini")


def _package(*, passes=(), commands=("parse",)):
    manifest = {"target": "gemmini", "commands": {name: {"argv": ["{tool}", "{input_mlir}"]} for name in commands}}
    if passes:
        manifest["whole_model_passes"] = list(passes)
    return SimpleNamespace(manifest=manifest, tool=Path("/nonexistent/tool"), directory=Path("/nonexistent"))


def test_a_package_with_no_pass_list_declares_none():
    assert WMPass.declared_passes(_package()) == ()


def test_a_declared_pass_must_also_be_a_declared_command():
    package = _package(passes=["fuse_casts"], commands=["parse"])
    with pytest.raises(WMPass.PassDeclarationError, match="fuse_casts"):
        WMPass.declared_passes(package)


def test_declared_passes_run_in_the_manifests_own_order(tmp_path):
    package = _package(passes=["first", "second"], commands=["first", "second"])
    original = tmp_path / "model.mlir"
    original.write_text("module { }\n", encoding="utf-8")
    seen = []

    def run(pkg, name, source, destination=None, *, timeout):  # noqa: ARG001
        seen.append((name, Path(source).read_text(encoding="utf-8")))
        Path(destination).write_text(f"module {{ /* after {name} */ }}\n", encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    final_path, records, stopped = WMPass.apply_passes(original, package, work=tmp_path / "work", run=run)
    assert stopped == ""
    assert [name for name, _text in seen] == ["first", "second"]
    # The SECOND pass reads the FIRST pass's own output, not the original module.
    assert "after first" in seen[1][1]
    assert final_path.read_text(encoding="utf-8") == "module { /* after second */ }\n"
    assert [r["pass"] for r in records] == ["first", "second"]
    assert all(r["ok"] for r in records)


def test_a_pass_that_writes_nothing_stops_the_chain_and_keeps_the_last_good_module(tmp_path):
    package = _package(passes=["first", "second"], commands=["first", "second"])
    original = tmp_path / "model.mlir"
    original.write_text("module { }\n", encoding="utf-8")

    def run(pkg, name, source, destination=None, *, timeout):  # noqa: ARG001
        if name == "first":
            Path(destination).write_text("module { /* after first */ }\n", encoding="utf-8")
            return SimpleNamespace(returncode=0, stdout="", stderr="")
        return SimpleNamespace(returncode=1, stdout="", stderr="the pass crashed")

    final_path, records, stopped = WMPass.apply_passes(original, package, work=tmp_path / "work", run=run)
    assert "second" in stopped
    assert final_path.read_text(encoding="utf-8") == "module { /* after first */ }\n"
    assert records[-1]["pass"] == "second" and not records[-1]["ok"]


def test_the_first_passs_own_failure_keeps_the_original_module_untouched(tmp_path):
    package = _package(passes=["only"], commands=["only"])
    original = tmp_path / "model.mlir"
    original.write_text("module { /* original */ }\n", encoding="utf-8")

    def run(pkg, name, source, destination=None, *, timeout):  # noqa: ARG001
        raise RuntimeError("the entrypoint could not even start")

    final_path, records, stopped = WMPass.apply_passes(original, package, work=tmp_path / "work", run=run)
    assert final_path == original
    assert "could not be invoked" in stopped
    assert records == [{"pass": "only", "ok": False, "why": "RuntimeError: the entrypoint could not even start"}]


# ------------------------------------------------------------- semantic verification, oracle equality


def test_a_pass_that_preserves_the_final_digest_and_argmax_verifies(monkeypatch):
    import dataclasses

    from merlin.perf import whole_model_build as WMB
    from merlin.perf import whole_model_oracle as ORACLE

    same_oracle = {"final_digest": {"sum": 7, "fnv1a": 42}, "argmax": 3}

    def fake_oracle(capsule, *, target, digest):  # noqa: ARG001
        return dict(same_oracle), None

    monkeypatch.setattr(ORACLE, "_oracle", fake_oracle)
    capsule = WMB.ModelCapsule(Path("/x"), "m", Path("/x/i.mlir"), Path("/x/w"), Path("/x/wm"), {}, {})
    verdict = WMB.verify_passes_preserve_semantics(capsule, target="gemmini", transformed_interface="/x/t.mlir")
    assert verdict["ok"] is True and verdict["why"] == ""
    assert dataclasses.is_dataclass(capsule)


def test_a_pass_that_changes_the_final_digest_or_argmax_fails_closed(monkeypatch):
    from merlin.perf import whole_model_build as WMB
    from merlin.perf import whole_model_oracle as ORACLE

    calls = iter(
        [
            {"final_digest": {"sum": 7, "fnv1a": 42}, "argmax": 3},
            {"final_digest": {"sum": 7, "fnv1a": 999}, "argmax": 3},
        ]
    )

    def fake_oracle(capsule, *, target, digest):  # noqa: ARG001
        return next(calls), None

    monkeypatch.setattr(ORACLE, "_oracle", fake_oracle)
    capsule = WMB.ModelCapsule(Path("/x"), "m", Path("/x/i.mlir"), Path("/x/w"), Path("/x/wm"), {}, {})
    verdict = WMB.verify_passes_preserve_semantics(capsule, target="gemmini", transformed_interface="/x/t.mlir")
    assert verdict["ok"] is False
    assert "disagrees" in verdict["why"]


def test_a_transformed_model_that_cannot_even_be_stated_fails_closed(monkeypatch):
    from merlin.perf import whole_model_build as WMB
    from merlin.perf import whole_model_oracle as ORACLE

    def fake_oracle(capsule, *, target, digest):  # noqa: ARG001
        if str(capsule.interface).endswith("t.mlir"):
            raise WMB.WholeModelBuildError("the transformed module has an open host region")
        return {"final_digest": {"sum": 1, "fnv1a": 1}, "argmax": 0}, None

    monkeypatch.setattr(ORACLE, "_oracle", fake_oracle)
    capsule = WMB.ModelCapsule(Path("/x"), "m", Path("/x/i.mlir"), Path("/x/w"), Path("/x/wm"), {}, {})
    verdict = WMB.verify_passes_preserve_semantics(capsule, target="gemmini", transformed_interface="/x/t.mlir")
    assert verdict["ok"] is False and "transformed model could not be stated" in verdict["why"]
