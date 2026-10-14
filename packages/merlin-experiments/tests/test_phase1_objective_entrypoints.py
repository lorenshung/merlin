"""Actual installed CLI and programmatic admission share objective policy."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1 import __main__ as entrypoint
from merlin_experiments.phase1 import context, controller, session
from merlin_experiments.phase1.options import RunOptions, parse_options

INVALID = (
    ({"max_wall_s": 0}, ["--max-wall-s", "0"]),
    ({"max_wall_s": -1}, ["--max-wall-s", "-1"]),
    ({"schedule": "rounds"}, ["--schedule", "rounds"]),
    ({"continuous": True}, ["--continuous"]),
    ({"plateau_rounds": 2}, ["--plateau-rounds", "2"]),
    ({"qualify_submission": "/selected"}, ["--qualify-submission", "/selected"]),
    ({"seal_current": True}, ["--seal-current"]),
)


def cli_inputs(tmp_path):
    manifest = tmp_path / "bundle.yaml"
    manifest.write_text("bundle_id: synthetic\n")
    return [
        "--run-id",
        "probe",
        "--bundle",
        "synthetic",
        "--descriptor",
        str(tmp_path / "descriptor.yaml"),
        "--repo",
        str(tmp_path),
        "--bundle-manifest",
        str(manifest),
        "--oracle-timing",
        str(tmp_path / "timing.json"),
    ]


@pytest.mark.parametrize("updates,args", INVALID)
def test_real_installed_cli_refuses_before_context_or_treatment_inputs(tmp_path, monkeypatch, capsys, updates, args):
    from merlin_experiments.phase1.feedback import rtlchecks

    def forbidden(*a, **k):
        pytest.fail("invalid objective reached resource discovery or admission")

    monkeypatch.setattr(context, "load_context", forbidden)
    monkeypatch.setattr(controller, "run", forbidden)
    monkeypatch.setattr(rtlchecks, "prepare_arguments", forbidden)
    command = (
        cli_inputs(tmp_path)
        + [
            "--objective",
            "device-coverage",
            "--schedule",
            "continuous",
            "--max-wall-s",
            "10",
            "--plateau-rounds",
            "0",
            "--treatment",
            "rtlchecks",
        ]
        + args
    )
    with pytest.raises(SystemExit) as error:
        entrypoint.main(command)
    assert error.value.code == 2
    assert "device-coverage requires" in capsys.readouterr().err


@pytest.mark.parametrize("updates,args", INVALID)
def test_programmatic_construction_and_session_refuse_identical_invalid_objectives(updates, args):
    values = dataclasses.asdict(parse_options(["--run-id", "probe"], environ={}))
    values.update(objective="device-coverage", schedule="continuous", max_wall_s=10, plateau_rounds=0)
    values.update(updates)
    with pytest.raises(ValueError, match="device-coverage requires"):
        RunOptions(**values)
    with pytest.raises(ValueError, match="device-coverage requires"):
        session.validate_options(SimpleNamespace(**values))


@pytest.mark.parametrize("budget", [True, 1.5, "10", float("nan"), float("inf"), -float("inf")])
def test_programmatic_coverage_budget_is_a_positive_integer(budget):
    values = dataclasses.asdict(parse_options(["--run-id", "probe"], environ={}))
    values.update(objective="device-coverage", schedule="continuous", max_wall_s=budget, plateau_rounds=0)
    with pytest.raises(ValueError, match="device-coverage requires"):
        RunOptions(**values)
    with pytest.raises(ValueError, match="device-coverage requires"):
        session.validate_options(SimpleNamespace(**values))


@pytest.mark.parametrize("objective", ["unknown", True, [], {}])
def test_programmatic_unknown_objective_is_rejected(objective):
    values = dataclasses.asdict(parse_options(["--run-id", "probe"], environ={}))
    values["objective"] = objective
    with pytest.raises(ValueError, match="unknown authoring objective"):
        RunOptions(**values)
    with pytest.raises(ValueError, match="unknown authoring objective"):
        session.validate_options(SimpleNamespace(**values))


@pytest.mark.parametrize("coverage", [False, True])
def test_real_installed_cli_preserves_legacy_default_and_forwards_valid_coverage(tmp_path, monkeypatch, coverage):
    observed = []
    monkeypatch.setattr(context, "load_context", lambda *a, **k: SimpleNamespace(target="synthetic"))

    def admit(selected_context, options, **kwargs):
        observed.append(options)
        return 77

    monkeypatch.setattr(controller, "run", admit)
    args = cli_inputs(tmp_path)
    if coverage:
        args += [
            "--objective",
            "device-coverage",
            "--schedule",
            "continuous",
            "--max-wall-s",
            "10",
            "--plateau-rounds",
            "0",
        ]
    assert entrypoint.main(args) == 77
    assert len(observed) == 1 and observed[0].objective == ("device-coverage" if coverage else "correctness")
    if not coverage:
        defaults = parse_options(["--run-id", "probe"], environ={})
        assert observed[0].schedule == defaults.schedule and observed[0].max_wall_s == defaults.max_wall_s
