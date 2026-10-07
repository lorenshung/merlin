"""Phase-1 runs live at out/runs/<target>/phase1/<run-id>/; legacy runs still resume where they began."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.session import orchestration_owned, phase_run_dir


def _context(tmp_path):
    return SimpleNamespace(
        target="alpha",
        runs=tmp_path / "out/runs/alpha/capsule-bench",
        phase_runs=tmp_path / "out/runs/alpha/phase1",
    )


def test_new_runs_use_the_phase_address(tmp_path):
    context = _context(tmp_path)
    run = phase_run_dir(context, "merlin_assisted", "20260930T000000Z_merlin_assisted_abc1234", resume=False)
    assert run == tmp_path / "out/runs/alpha/phase1/20260930T000000Z_merlin_assisted_abc1234"
    with pytest.raises(ValueError):
        phase_run_dir(context, "merlin_assisted", "../escape", resume=False)


def test_a_legacy_run_resumes_in_place(tmp_path):
    context = _context(tmp_path)
    legacy = context.runs / "merlin_assisted" / "old-run"
    legacy.mkdir(parents=True)
    assert phase_run_dir(context, "merlin_assisted", "old-run", resume=True) == legacy
    assert phase_run_dir(context, "merlin_assisted", "old-run", resume=False) == context.phase_runs / "old-run"


def test_only_an_orchestration_record_is_adopted(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    assert not orchestration_owned(run)
    (run / "resolved-plan.json").write_text("{}")
    (run / "orchestration.json").write_text("{}")
    (run / "phase1-attempt-0001.log").write_text("")
    assert orchestration_owned(run)
    (run / "environment.yaml").write_text("{}")  # engine state: never adopted as fresh
    assert not orchestration_owned(run)


def test_capsule_bench_engine_output_is_the_phase_run(tmp_path, monkeypatch):
    from merlin_experiments.adapters import ADAPTERS

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    adapter = ADAPTERS["capsule_bench"]
    spec = SimpleNamespace(target="alpha", id="functional", path=tmp_path / "x.yaml", resolve=lambda v: tmp_path / v)
    run_dir = tmp_path / "out/runs/alpha/phase1/20260930T000000Z_functional_abc1234"
    try:
        command = adapter.resolve(spec, {"descriptor": "d.yaml", "arm": "merlin_assisted"}, tmp_path, run_dir)
    except Exception as exc:  # the entrypoint may be unresolvable outside a full install
        pytest.skip(f"adapter resolution needs an installed entrypoint: {exc}")
    assert command["engine_output"] == str(run_dir)


def test_capsule_catalog_maps_explicit_bounded_object_build_budget(tmp_path, monkeypatch):
    from merlin_experiments.adapters import ADAPTERS
    from merlin_experiments.spec import SpecError

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    adapter = ADAPTERS["capsule_bench"]
    spec = SimpleNamespace(target="alpha", id="functional", path=tmp_path / "x.yaml", resolve=lambda v: tmp_path / v)
    config = {
        "descriptor": "d.yaml",
        "arm": "merlin_assisted",
        "model": "neutral",
        "effort": "low",
        "max_wall_s": 60,
        "round_timeout": 30,
        "public_object_build_budget_s": 7,
    }
    option = adapter.options["public_object_build_budget_s"]
    option.validate("public_object_build_budget_s", 7)
    command = adapter.resolve(spec, config, tmp_path, tmp_path / "out/run")
    assert command["argv"].count("--public-object-build-budget-s") == 1
    assert command["argv"][command["argv"].index("--public-object-build-budget-s") + 1] == "7"
    for bad in (0, 121, True):
        with pytest.raises(SpecError):
            option.validate("public_object_build_budget_s", bad)


def test_phase2_finds_a_functional_run_under_the_phase_root(tmp_path):
    from merlin_experiments.phase2.campaign import CampaignGateError, inspect_functional_run

    phase1 = tmp_path / "out/runs/alpha/phase1"
    (phase1 / "run-a").mkdir(parents=True)
    # Resolves the phase address (and so reaches the evidence checks), instead of refusing the path.
    with pytest.raises(CampaignGateError, match="submission is absent"):
        inspect_functional_run(phase1, "run-a", "0" * 64)
    with pytest.raises(CampaignGateError, match="does not resolve safely"):
        inspect_functional_run(phase1, "missing", "0" * 64)


def test_a_context_without_a_phase_root_keeps_the_legacy_layout(tmp_path):
    context = _context(tmp_path)
    legacy = SimpleNamespace(target=context.target, runs=context.runs, phase_runs=None)
    assert phase_run_dir(legacy, "raw_baseline", "one", resume=False) == context.runs / "raw_baseline" / "one"
