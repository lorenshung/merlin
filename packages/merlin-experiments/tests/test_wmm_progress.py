"""A measured run's status and its changes, read from its own records -- the run, its launch record, its
stop request, its rounds (ended and still open), its stores' jobs and holds -- and the commands that
show them: ``status``, ``follow`` and their ``merlin experiment`` forms; plus ``resume`` of one named
run onto another seed."""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import cli as MCLI
from merlin_experiments.phase2.whole_model_measured import config as C
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import launch as LAUNCH
from merlin_experiments.phase2.whole_model_measured import progress as P
from merlin_experiments.phase2.whole_model_measured import rounds as R
from merlin_experiments.phase2.whole_model_measured import runs as RUNS
from merlin_experiments.phase2.whole_model_measured import service as S
from merlin_experiments.phase2.whole_model_measured import sessions as SES
from merlin_experiments.phase2.whole_model_measured.identity import package_digest


def _run(tmp_path: Path) -> Path:
    run_dir = tmp_path / "out" / "runs" / "toy" / "phase2" / "20261005T000000Z_m_nofsm_abcdef0"
    rounds = run_dir / "stage" / "rounds"
    rounds.mkdir(parents=True)
    store = tmp_path / "store" / "screen"
    for key, state in (("a" * 64, J.DONE), ("b" * 64, J.PENDING)):
        (store / key).mkdir(parents=True)
        (store / key / "job.json").write_text(json.dumps({"state": state, "package_sha256": key}))
    (store / S.DISK_HOLD).write_text(json.dumps({"reason": "infra_disk_low: the store filesystem ..."}))
    (store / SES.PLATEAU_FILE).write_text(json.dumps({"last_improvement": {"hours_since": 2.5, "sessions_since": 3}}))
    (run_dir / "run.json").write_text(
        json.dumps(
            {"schema": RUNS.RUN_SCHEMA, "target": "toy", "method": "m_nofsm", "prohibited_instruction_roles": []}
        )
    )
    (run_dir / "resumed_seed.json").write_text(
        json.dumps({"seed_package_sha256": "c" * 64, "lineage_kind": "resumed", "store_roots": {"screen": str(store)}})
    )
    (rounds / f"round_00{R.ROUND_SUFFIX}").write_text(
        json.dumps({"round": 0, "status": "refused", "why": "audit hit", "requested": ["a" * 64]})
    )
    (rounds / f"round_01{R.OPEN_SUFFIX}").write_text(json.dumps({"round": 1, "pid": 1, "requested": ["b" * 64]}))
    return run_dir


class _Objective:
    def __init__(self, best=None):
        self.polled = 0
        self.best = best

    def poll(self):
        self.polled += 1

    def summary(self):
        return {
            "bar": {"screen_whole_window_cycles": 1000},
            "best": self.best,
            "history": [{"package_sha256": "a" * 64, "replicate": 0, "state": "done", "objective_cycles": 900}],
        }


def test_the_status_is_the_runs_own_records(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    run_dir = _run(tmp_path)
    SES.request_stop(run_dir, why="hold the board")
    document = P.run_status(run_dir)
    assert document["method"] == "m_nofsm" and document["launcher"]["alive"] is None
    assert document["stop_requested"]["why"] == "hold the board"
    assert [r["status"] for r in document["rounds"]] == ["refused"]
    assert document["open_rounds"] == [{"round": 1, "started_at": None, "pid": 1, "requested": 1}]
    screen = document["stores"]["screen"]
    assert screen["jobs"] == {J.DONE: 1, J.PENDING: 1} and screen["disk_hold"]["reason"].startswith("infra_disk_low")
    assert screen["plateau"]["sessions_since"] == 3 and "objective" not in document
    text = P.format_status(document)
    assert "STOP REQUESTED: hold the board" in text and "DISK HOLD" in text and "round 1: OPEN" in text


def test_the_objectives_view_and_an_asked_for_poll(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    objective = _Objective(best={"package_sha256": "a" * 64, "screen_whole_window_cycles": 900})
    assert P.run_status(_run(tmp_path), objective=objective)["objective"]["best"]["screen_whole_window_cycles"] == 900
    assert objective.polled == 0  # read-only unless asked
    P.run_status(
        tmp_path / "out" / "runs" / "toy" / "phase2" / "20261005T000000Z_m_nofsm_abcdef0",
        objective=objective,
        poll=True,
    )
    assert objective.polled == 1


def test_follow_prints_each_change_and_ends_when_the_run_is_over(tmp_path):
    base = {
        "jobs": {"screen/" + "a" * 64: J.RUNNING},
        "rounds": {},
        "open_rounds": [0],
        "holds": {"screen disk hold": False},
    }
    ticks = iter(
        [
            {**base, "launcher_alive": True, "stop_requested": False, "stopped": None, "best": None},
            {
                **base,
                "jobs": {"screen/" + "a" * 64: J.DONE},
                "rounds": {0: "authored"},
                "open_rounds": [],
                "launcher_alive": False,
                "stop_requested": True,
                "stopped": "operator",
                "best": ("a" * 64, 900),
            },
        ]
    )
    lines = []
    ended = P.follow(tmp_path, take=lambda: next(ticks), out=lines.append, sleep=lambda s: None, clock=lambda: 0.0)
    assert ended["ended"] == "the run is over"
    text = "\n".join(lines)
    for expected in (
        "running -> done",
        "round 0 ended: authored",
        "best aaaaaaaaaaaa at 900 cycles",
        "launcher: False",
    ):
        assert expected in text


def test_follow_stops_at_its_deadline_while_the_run_goes_on(tmp_path):
    tick = {
        "jobs": {"s/k": J.PENDING},
        "rounds": {},
        "open_rounds": [],
        "holds": {},
        "launcher_alive": True,
        "best": None,
    }
    clock = iter([0.0, 0.0, 30.0, 30.0, 61.0, 61.0])
    ended = P.follow(
        tmp_path,
        take=lambda: tick,
        out=lambda line: None,
        sleep=lambda s: None,
        clock=lambda: next(clock),
        max_seconds=60,
    )
    assert ended["ended"] == "the follow deadline passed"


def test_the_command_lines_show_a_measured_runs_status(tmp_path, monkeypatch, capsys):
    from merlin_experiments import cli as TOP

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    run_dir = _run(tmp_path)
    assert MCLI.main(["status", str(run_dir), "--json"]) == 0
    document = json.loads(capsys.readouterr().out)
    assert document["stores"]["screen"]["jobs"] == {J.DONE: 1, J.PENDING: 1}
    assert "objective_error" in document  # the synthetic run has no objective config; the records still show
    assert TOP.main(["status", str(run_dir)]) == 0
    assert json.loads(capsys.readouterr().out)["method"] == "m_nofsm"
    assert TOP.main(["watch", str(run_dir), "--max-seconds", "0"]) == 0
    assert "the follow deadline passed" in capsys.readouterr().out
    assert TOP.main(["measured", "stop", str(run_dir), "--why", "via the top-level command"]) == 0
    assert json.loads((run_dir / SES.OPERATOR_STOP_FILE).read_text())["why"] == "via the top-level command"
