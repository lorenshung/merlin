"""Public geometry probes expose emitted-text growth, never compiler certification."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest


@pytest.fixture
def probe(monkeypatch):
    from merlin_experiments.phase1.feedback import codegen_scalability as S

    from merlin.targetgen import capability_probes, corpora, corpus_spec, lowering_coverage

    monkeypatch.setattr(corpora, "experiment_for", lambda _target: object())
    monkeypatch.setattr(
        corpus_spec,
        "derive_binding",
        lambda _experiment, _profile: SimpleNamespace(
            operand_dtype="int8", accum_dtype="i32", mlir_dtype=lambda token: "i8" if token == "int8" else "i32"
        ),
    )
    monkeypatch.setattr(capability_probes, "tile_edge", lambda _target, **_kwargs: 32)
    seen = []

    def emit(_package, **kwargs):
        seen.append(kwargs)
        scale = kwargs["m"] // 32
        return "lowered", None, 20 * scale**3

    monkeypatch.setattr(lowering_coverage, "probe_shape", emit)
    return S, seen


def test_cases_are_derived_only_from_public_geometry_and_formats(probe):
    S, seen = probe
    result = S.run(
        "candidate", target="synthetic", contract="selected-contract", timeout=12, additional_forbidden=("extra",)
    )
    assert [(row["m"], row["k"], row["n"]) for row in seen] == [
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
    ]
    assert all(row["operand_mlir"] == "i8" and row["accum_mlir"] == "i32" for row in seen)
    assert all(row["contract"] == "selected-contract" and row["additional_forbidden"] == ("extra",) for row in seen)
    assert all(row["timeout"] == 12 for row in seen)
    assert [row["contraction_volume_ratio"] for row in result["samples"]] == [1, 8, 64, 512]
    assert [row["line_ratio_vs_baseline"] for row in result["samples"]] == [1, 8, 64, 512]
    assert result["scope"] == "public_emit_only"
    assert result["observations_complete"] is True
    assert "all_pass" not in result and "all_covered" not in result
    assert "not compilation" in result["note"]


def test_constant_size_looped_code_is_not_rejected(probe, monkeypatch):
    from merlin.targetgen import lowering_coverage

    S, _seen = probe
    monkeypatch.setattr(lowering_coverage, "probe_shape", lambda *_args, **_kwargs: ("lowered", None, 17))
    result = S.run("candidate", target="synthetic")
    assert result["observations_complete"] is True
    assert all(row["line_ratio_vs_baseline"] == 1 for row in result["samples"])
    assert "verdict" not in result


def test_smaller_looped_emission_does_not_revoke_a_passing_round(probe, monkeypatch, tmp_path):
    from merlin_experiments.phase1.feedback import loop_grading as L

    from merlin.targetgen import lowering_coverage as LC

    monkeypatch.setattr(LC, "tile_edge", lambda _target: 32)
    monkeypatch.setattr(
        LC,
        "probe_shape",
        lambda _package, **kw: ("lowered", None, 20 if (kw["m"], kw["k"], kw["n"]) == (64, 32, 32) else 30),
    )
    verdict = {"all_pass": True}
    L._attach_shape_generalization(
        verdict, "candidate", tmp_path, 0, timeout=30, context=SimpleNamespace(target="synthetic", repo=tmp_path)
    )
    assert verdict["all_pass"] is True
    assert verdict["shape_coverage"]["scope"] == "public_emit_only"
    assert verdict["shape_coverage"]["smaller_emitted_artifacts"] == ["m_2tiles"]
    assert verdict["shape_coverage"]["multi_tile_axes_uncovered"] == []


def test_declined_and_failed_baseline_are_measurements_not_passes(probe, monkeypatch):
    from merlin.targetgen import lowering_coverage

    S, _seen = probe
    monkeypatch.setattr(lowering_coverage, "probe_shape", lambda *_args, **_kwargs: ("declined", "unsupported", 0))
    result = S.run("candidate", target="synthetic")
    assert result["observations_complete"] is False
    assert all(row["line_ratio_vs_baseline"] is None for row in result["samples"])
    assert all(row["outcome"] == "declined" for row in result["samples"])


def test_round_feedback_persists_public_probe_without_changing_grade(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import codegen_scalability as S
    from merlin_experiments.phase1.feedback import loop_grading as L

    observation = {"scope": "public_emit_only", "samples": [{"extent_scale": 8, "artifact_nonempty_lines": 7982}]}
    seen = []

    def run(_candidate, **kwargs):
        seen.append(kwargs)
        return observation

    monkeypatch.setattr(S, "run", run)
    verdict = {"all_pass": False}
    L._attach_codegen_scalability(
        verdict, "candidate", tmp_path, 3, timeout=300, context=SimpleNamespace(target="synthetic"), contract="selected"
    )
    assert verdict["all_pass"] is False
    assert verdict["codegen_scalability"] == {"ran": True, **observation}
    assert seen[0]["timeout"] == 30
    assert seen[0]["contract"] == "selected"
    assert json.loads((tmp_path / "qa_history/codegen_scalability_round_03.json").read_text()) == observation


def test_missing_scalability_measurement_is_explicit_but_not_a_numerical_verdict(tmp_path, monkeypatch):
    from merlin_experiments.phase1.feedback import codegen_scalability as S
    from merlin_experiments.phase1.feedback import loop_grading as L

    def fail(*_args, **_kwargs):
        raise RuntimeError("selected geometry unavailable")

    monkeypatch.setattr(S, "run", fail)
    verdict = {"all_pass": False}
    L._attach_codegen_scalability(
        verdict, "candidate", tmp_path, 0, timeout=60, context=SimpleNamespace(target="synthetic")
    )
    assert verdict["all_pass"] is False
    assert verdict["codegen_scalability"]["ran"] is False
    assert "unavailable" in verdict["codegen_scalability"]["error"]
