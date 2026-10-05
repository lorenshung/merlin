"""The fast tiers of a measured run, held to what makes them safe to show an agent: they run detached on
the exact bytes asked about, answer once and keep that answer, never claim to be the objective, refuse
rather than guess, and time against an ATTRIBUTABLE best only."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import wmm_fixtures as FX
from merlin_experiments.phase2.whole_model_measured import fast as FAST
from merlin_experiments.phase2.whole_model_measured import jobs as J
from merlin_experiments.phase2.whole_model_measured import objective as O
from merlin_experiments.phase2.whole_model_measured import service as S
from merlin_experiments.phase2.whole_model_measured.identity import package_digest

from merlin.perf import whole_model_group_timing as T
from merlin.perf import whole_model_screen as SCREEN

#: A driver's line spellings, as data: the tier reads whatever the target's driver declares.
UART = {
    "local": "GM_LOCAL {group} mismatches={mismatches} of={elements} first={first}",
    "local_map": "GM_LOCAL_MAP {group} axis={axis} extent={extent} wrong={wrong}",
    "local_sample": "GM_LOCAL_SAMPLE {group} index={index} row={row} col={col} channel={channel} got={got} want={want}",
    "local_sign": "GM_LOCAL_SIGN {group} over={over} under={under} max_abs={max_abs}",
    "bounded": "GM_BOUND {group} max_abs={max_abs} over={over} bound={bound}",
}


@pytest.fixture
def run(tmp_path, monkeypatch):
    spawned: list[list[str]] = []

    def spawn(argv, **kw):
        spawned.append(list(argv))
        return SimpleNamespace(pid=999_999_999)

    monkeypatch.setattr(S, "spawn", spawn)
    monkeypatch.setattr(S, "alive", lambda pid, owner: pid == 999_999_999)
    spec, pin = FX.write_builder(tmp_path)
    machine = {"kind": "paired", "timing": FX.spike_machine(tmp_path), "local": _local_machine(tmp_path)}
    screen = S.MeasurementService(
        tmp_path / "store" / "screen",
        target="toy",
        builder=spec,
        builder_sha256=pin,
        machine=FX.spike_machine(tmp_path),
        build_options={"model_capsule": "m", "machine": "board", "header": "h.h"},
    )
    screen.machine = machine
    objective = O.WholeModelObjective(screen=screen, screen_reference=None, repeats_on_best=1)
    return SimpleNamespace(objective=objective, screen=screen, spawned=spawned, tmp=tmp_path)


def _local_machine(root: Path) -> dict:
    script = root / "console_sim.py"
    script.write_text("import sys\nprint(open(sys.argv[-1] + '.console').read())\n")
    return {"kind": "spike", "target": "toy", "command": [sys.executable, str(script)], "environment": {}}


def _finish(run, kind_dir: Path) -> dict:
    """What the detached half does, run here."""
    return FAST.run(kind_dir)


def test_a_tier_runs_detached_on_the_exact_bytes_and_keeps_its_answer(run, monkeypatch):
    def screened(elf, *, groups, target, out, **kw):
        Path(out).mkdir(parents=True, exist_ok=True)
        for name in ("console.txt", SCREEN.SCREEN_FILE):
            (Path(out) / name).write_text("{}")
        return {"status": "screened", "label": "structure", "groups": [], "all_groups_correct": True}

    monkeypatch.setattr(SCREEN, "structure_screen", screened)
    package = FX.package(run.tmp, "cand")
    first = FAST.request(run.objective, "structure", package)
    assert first["state"] == "running" and "never the measured objective" in first["not_the_objective"]
    (argv,) = run.spawned
    root = Path(argv[-1])
    assert argv[-2] == "fast" and package_digest(root / "package") == package_digest(package)
    assert FAST.request(run.objective, "structure", package)["state"] == "running"  # no second launch
    assert len(run.spawned) == 1
    result = _finish(run, root)
    assert result["status"] == "screened" and "THESE CYCLES ARE NOT THE BOARD'S" in result["text"]
    done = FAST.request(run.objective, "structure", package)
    assert done["state"] == "done" and done["status"] == "screened"
    assert not (root / "package").exists() and (root / "out" / "result.json").is_file()  # build tree dropped
    # The board's program is filed by its digest under the store base, so a board run makes it a pair.
    assert list((run.tmp / "store").glob("**/" + SCREEN.SCREEN_FILE))


def test_a_tier_that_raises_answers_refused_and_says_why(run):
    package = FX.package(run.tmp, "cand")
    FAST.request(run.objective, "group-check", package, group=2)
    root = Path(run.spawned[-1][-1])
    result = _finish(run, root)  # the toy target has no whole-model driver to build a group program with
    assert result["status"] == "refused" and result["refusal"] and "REFUSED" in result["text"]


def test_a_group_check_reads_the_local_verdict_off_its_own_console(run, monkeypatch):
    elf = run.tmp / "g2.elf"
    elf.write_text("program")
    (run.tmp / "g2.elf.console").write_text(
        "\n".join(
            [
                "GM_LOCAL 2 mismatches=3 of=64 first=5",
                "GM_LOCAL_MAP 2 axis=channel extent=8 wrong=1:2,4:1",
                "GM_LOCAL_SIGN 2 over=2 under=1 max_abs=7",
                "GM_LOCAL 1 mismatches=0 of=64 first=-1",
            ]
        )
    )
    monkeypatch.setattr(
        T, "build_group_programs", lambda *a, **k: {2: {"elf": str(elf), "elf_sha256": "e" * 64, "on": "package"}}
    )
    from merlin.runtime.backends import base as backends

    monkeypatch.setattr(
        backends, "whole_model_driver", lambda target: SimpleNamespace(program=SimpleNamespace(UART=UART))
    )
    package = FX.package(run.tmp, "cand")
    FAST.request(run.objective, "group-check", package, group=2)
    result = _finish(run, Path(run.spawned[-1][-1]))
    assert result["status"] == "incorrect" and result["local"] == {"mismatches": 3, "elements": 64, "first": 5}
    assert result["maps"]["channel"] == {"extent": 8, "wrong": {1: 2, 4: 1}}
    assert "INCORRECT: 3 of 64" in result["text"] and "1,4" in result["text"]


def test_group_timing_against_best_never_times_against_unattributable_bytes(run):
    package = FX.package(run.tmp, "cand")
    with pytest.raises(ValueError, match="no correct measured best"):
        FAST.request(run.objective, "group-timing", package)
    other = FX.package(run.tmp, "other", argmax=4)
    digest = run.screen.request(other, label="agent", attribution={"state": J.ATTRIBUTION_UNAUTHORED, "round": 0})[
        "package_sha256"
    ]
    assert not run.screen.attributable(digest)
    with pytest.raises(ValueError, match="no correct measured best"):
        FAST.request(run.objective, "group-timing", package)
    view = FAST.request(run.objective, "group-timing", package, baseline="seed")
    assert view["state"] == "running" and view["baseline_sha256"]
    spec = json.loads((Path(run.spawned[-1][-1]) / "spec.json").read_text())
    assert spec["baseline"].endswith("/package") and spec["screen_options"]["machine"] == "board"


def test_at_most_a_few_tiers_of_a_kind_run_at_once(run):
    views = [FAST.request(run.objective, "structure", FX.package(run.tmp, f"p{i}", argmax=i)) for i in range(4)]
    assert [v["state"] for v in views] == ["running"] * FAST.FAST_CONCURRENCY + ["busy"] * (4 - FAST.FAST_CONCURRENCY)
    assert len(run.spawned) == FAST.FAST_CONCURRENCY
