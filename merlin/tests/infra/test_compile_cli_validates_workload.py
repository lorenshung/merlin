"""`merlin-compile --run none` must not report `compiled` for a workload it never looked at.

WHY THIS EXISTS. `compile_oot` built the BACKEND PACKAGE, set `status: compiled`, and returned on
`run == "none"` -- and the check that the named workload is actually a capsule sat BELOW that early
return. So the compile-only path never read `workload` at all.

MEASURED 2026-09-10, both reporting `status: compiled`:
  --workload definitely_not_a_real_workload_xyz --target gemmini --run none
  --workload tiny_llama                          --target gemmini --run none

The second is the damaging one. `tiny_llama` is a whole MODEL, and this function's own docstring says
"Accelerators run capsules/kernels, not whole VLA models" -- so a caller asking for a model got
`compiled` back, wrote no files, and had every reason to believe a model had been compiled for
gemmini. A status that cannot be false is not a status.
"""

from __future__ import annotations

import json

import pytest
import yaml


@pytest.fixture
def compiler_package(tmp_path, monkeypatch):
    """A schema-checked local package; building is observed, not delegated to host tools."""
    from merlin.targetgen import oot_runner

    package = tmp_path / "compiler"
    package.mkdir()
    manifest = {
        "artifact_type": "mlir_oot_target_backend",
        "target": "fixture",
        "language": "python",
        "authoring": {"mode": "hand_curated"},
        "integrity_exempt": False,
        "entrypoints": {"tool": "compiler.py"},
        "commands": {
            command: {"argv": ["python3", "compiler.py"]}
            for command in ("parse", "lower_interface_to_target", "emit_command_buffer", "lower_target_to_llvm")
        },
    }
    (package / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    (package / "compiler.py").write_text("raise RuntimeError('this unit fixture must never be executed')\n")
    built = []
    monkeypatch.setattr(oot_runner, "build_package", lambda pkg, **kwargs: built.append((pkg, kwargs)))
    corpus = tmp_path / "fixture-corpus" / "isa"
    capsule = corpus / "A0_config_smoke"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text(yaml.safe_dump({"name": "A0_config_smoke", "kind": "isa", "label": "public"}))
    (capsule / "capsule.interface.mlir").write_text("module {}\n")
    descriptor = tmp_path / "fixture-descriptor.yaml"
    descriptor.write_text(yaml.safe_dump({"target": "fixture", "capsule_corpus": str(corpus)}))
    return package, built, descriptor


def _compile_oot(workload: str, compiler_package, run: str = "none"):
    from merlin.compile_cli import compile_oot

    package, _, descriptor = compiler_package
    return compile_oot(
        workload,
        target="fixture",
        run=run,
        verify=False,
        package=str(package),
        timeout=600,
        corpus_descriptor=descriptor,
    )


def test_a_nonexistent_workload_is_not_reported_as_compiled(compiler_package):
    """THE REGRESSION. Moving the capsule check back below the early return fails here."""
    out = _compile_oot("definitely_not_a_real_workload_xyz", compiler_package)
    assert out["status"] == "not_run"
    assert "no capsule" in out["reason"]


def test_a_whole_model_name_is_refused_and_says_where_to_go(compiler_package):
    """`tiny_llama` is a model, not a capsule; the refusal must name the right path, not just fail."""
    out = _compile_oot("tiny_llama", compiler_package)
    assert out["status"] == "not_run"
    assert "whole models" in out["reason"]
    assert "--model-preflight" in out["reason"]


def test_the_refusal_happens_before_any_package_build(compiler_package):
    """The check must precede the build: validating after it wastes a full toolchain build to
    produce a refusal, and (the original defect) never runs at all on the compile-only path."""
    _, built, _ = compiler_package
    out = _compile_oot("definitely_not_a_real_workload_xyz", compiler_package)
    assert out["status"] == "not_run"
    assert "no capsule" in out["reason"], "an absent default package must not short-circuit this test"
    assert built == [], "the backend package was built before the workload was validated"


def test_non_gem_selected_corpus_compiles_without_legacy_tree(compiler_package, tmp_path, capsys):
    """An explicit non-Gem descriptor, not the checkout's Gem ISA tree, owns selection."""
    from merlin.compile_cli import main

    package, built, _ = compiler_package
    corpus = tmp_path / "saturn-corpus" / "isa"
    capsule = corpus / "SAT_fixture"
    capsule.mkdir(parents=True)
    (capsule / "capsule.yaml").write_text(yaml.safe_dump({"name": "SAT_fixture", "kind": "isa", "label": "public"}))
    (capsule / "capsule.interface.mlir").write_text("module {}\n")
    descriptor = tmp_path / "saturn-descriptor.yaml"
    descriptor.write_text(yaml.safe_dump({"target": "saturn", "capsule_corpus": str(corpus)}))

    assert (
        main(
            [
                "--target",
                "saturn",
                "--workload",
                "SAT_fixture",
                "--corpus-descriptor",
                str(descriptor),
                "--package",
                str(package),
                "--run",
                "none",
                "--timeout",
                "600",
                "--json",
            ]
        )
        == 0
    )
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "compiled"
    assert len(built) == 1
    assert built[0][0].directory == package
    assert built[0][1] == {"timeout": 600}


def test_oot_cli_refuses_missing_corpus_selection_before_build(compiler_package, capsys):
    from merlin.compile_cli import main

    package, built, _ = compiler_package
    assert (
        main(
            [
                "--target",
                "saturn",
                "--workload",
                "A0_config_smoke",
                "--package",
                str(package),
                "--run",
                "none",
                "--json",
            ]
        )
        == 1
    )
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "not_run"
    assert "--corpus-descriptor" in out["reason"]
    assert built == []


def test_oot_cli_refuses_foreign_target_descriptor_before_build(compiler_package, capsys):
    from merlin.compile_cli import main

    package, built, descriptor = compiler_package
    assert (
        main(
            [
                "--target",
                "saturn",
                "--workload",
                "A0_config_smoke",
                "--corpus-descriptor",
                str(descriptor),
                "--package",
                str(package),
                "--run",
                "none",
                "--json",
            ]
        )
        == 1
    )
    out = json.loads(capsys.readouterr().out)
    assert out["status"] == "not_run"
    assert "differs from requested target" in out["reason"]
    assert built == []
