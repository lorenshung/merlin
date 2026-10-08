"""A study's status is derived from disk, and its arms are scored only against a like-for-like reference.

The register is hand-maintained and drifts, so the board must read numbers from the run matrices and
the baseline record, never from the register. The scoring rules are the ones that have been got wrong
before: an incorrect result earns no performance credit, a speedup never crosses fidelities, a
voided run stays in the spend, and a cycle count comes from a certifying tier alone.
"""

from __future__ import annotations

import json

import pytest

from merlin.benchharness import study_status as SS
from merlin.common.paths import repo_root

REGISTER = """\
# Task register

## Phase 1 — Library

| id | task | state |
|---|---|---|
| 1.2 | kernel library | **DONE** — 21 tests |
| 1.3 | task basis | **PARTIAL** — runs on the census |
| 1.4 | ledger | **OPEN** |
| 1.5 | typo row | **DONNE** |

## Critical path to a defensible headline

1. **1.3 → 3.3** — materialise the basis.
2. **1.2** again, then 1.4; version 2.0.1 of a tool is not a task id.
"""


def _matrix(tmp_path, tag, rows):
    path = tmp_path / f"matrix_{tag}.json"
    path.write_text(json.dumps({"jobs": len(rows), "wall_seconds": 60, "results": rows}))
    return path


def _row(method, task, solved, cycles=None, tokens=100, billed=1.0, **extra):
    return {
        "method": method,
        "task": task,
        "solved": solved,
        "best_cycles": cycles,
        "model": f"{method}-model",
        "cost": {"tokens_total": tokens, "billed_usd": billed, "notional_usd": 0.0, "agent_seconds": 3600},
        **extra,
    }


def test_register_states_are_parsed_structurally_and_a_typo_is_not_progress():
    tasks = SS.read_register(REGISTER)
    assert [(t["id"], t["state"]) for t in tasks] == [
        ("1.2", "DONE"),
        ("1.3", "PARTIAL"),
        ("1.4", "OPEN"),
        ("1.5", SS.UNRECOGNISED),
    ]
    assert tasks[0]["note"] == "21 tests" and tasks[0]["phase"] == "Phase 1 — Library"


def test_the_critical_path_is_read_from_the_register_in_its_own_order():
    tasks = SS.read_register(REGISTER)
    spine = SS.critical_path(REGISTER, tasks)
    assert [row["id"] for row in spine] == ["1.3", "3.3", "1.2", "1.4"]
    # An id the plan names but no row defines is reported, never dropped.
    assert spine[1]["state"] == SS.UNRECOGNISED and spine[1]["task"] == "(no row defines it)"


def test_voided_runs_stay_in_the_spend_but_out_of_every_rate(tmp_path):
    _matrix(tmp_path, "s3", [_row("arm_a", "T0", True, 900, tokens=500, billed=5.0)])
    _matrix(tmp_path, "s4", [_row("arm_a", "T0", True, 800), _row("arm_a", "T0", False, 10)])
    void = tmp_path / "void.yaml"
    void.write_text("voided:\n  s3:\n    arm_a: harness defect\n")
    rolled = SS.rollup_matrices(tmp_path.glob("matrix_*.json"), SS.read_void(void))
    live, voided = rolled["by_method"]["arm_a"], rolled["voided"]["arm_a"]
    assert (live["runs"], live["solved"]) == (2, 1)
    # The unsolved run's speed is not a result; only the solved run's cycles are credited.
    assert live["cycles"] == {"T0": [800]}
    assert (voided["runs"], voided["tokens"], voided["billed_usd"]) == (1, 500, 5.0)
    assert voided["void_reasons"] == ["s3: harness defect"]


def test_a_void_entry_without_a_reason_is_refused(tmp_path):
    void = tmp_path / "void.yaml"
    void.write_text("voided:\n  s3:\n    arm_a: ''\n")
    with pytest.raises(ValueError, match="reason"):
        SS.read_void(void)


def test_an_unreadable_matrix_is_reported_not_skipped(tmp_path):
    (tmp_path / "matrix_bad.json").write_text("{not json")
    rolled = SS.rollup_matrices(tmp_path.glob("matrix_*.json"), {})
    assert rolled["matrices"][0]["tag"] == "bad" and "error" in rolled["matrices"][0]


def _result(status, tiers):
    return {"status": status, "tiers": tiers, "numeric": {"status": status}}


def test_baseline_cycles_come_from_a_certifying_tier_of_a_passing_result_only():
    results = {
        "T0": _result("pass", {"L1": {"status": "pass", "cycles": 5}, "L2": {"status": "pass", "cycles": 1000}}),
        "T1": _result("pass", {"L1": {"status": "pass", "cycles": 7}}),
        "T2": _result("fail", {"L2": {"status": "fail", "cycles": 400}}),
    }
    record = SS.baseline_record(
        results, label="ref", package="pkg", target="any", fidelity="fast", certifying_tiers=["L2"]
    )
    assert (record["tasks"]["T0"]["cycles"], record["tasks"]["T0"]["cycles_tier"]) == (1000, "L2")
    # An execution-only tier cannot supply a quotable latency, and a failing reference is never a divisor.
    assert record["tasks"]["T1"]["cycles"] is None
    assert record["tasks"]["T2"]["cycles"] is None
    with pytest.raises(ValueError):
        SS.baseline_record(results, label="r", package="p", target="t", fidelity="f", certifying_tiers=[])


def test_speedups_never_cross_fidelities_and_report_a_geomean(tmp_path):
    _matrix(
        tmp_path,
        "s4",
        [
            _row("arm_a", "T0", True, 500, fidelity="fast"),
            _row("arm_a", "T0", True, 400, fidelity="fast"),
            _row("arm_a", "T0", True, 250, fidelity="fast"),
            _row("arm_a", "T1", True, 100, fidelity="cert"),
            _row("arm_b", "T0", True, 1000),
        ],
    )
    baseline = {
        "available": True,
        "label": "ref",
        "target": "any",
        "fidelity": "fast",
        "certifying_tiers": ["L2"],
        "tasks": {"T0": {"cycles": 1000}, "T1": {"cycles": 300}},
    }
    rolled = SS.rollup_matrices(tmp_path.glob("matrix_*.json"), {})
    perf = SS.reference_relative(rolled["by_method"], baseline)
    arm_a = perf["by_arm"]["arm_a"]
    assert arm_a["tasks"]["T0"]["median_cycles"] == 400 and arm_a["tasks"]["T0"]["speedup_median"] == 2.5
    assert arm_a["tasks"]["T0"]["speedup_best"] == 4.0 and arm_a["geomean_speedup"] == pytest.approx(2.5)
    assert "not the baseline's 'fast'" in arm_a["set_aside"]["T1"]
    # A row that records no fidelity is UNKNOWN, so it is set aside rather than assumed comparable.
    assert perf["by_arm"]["arm_b"]["set_aside"]["T0"].startswith("arm fidelity ['UNKNOWN']")
    declared = SS.rollup_matrices(tmp_path.glob("matrix_*.json"), {}, declared_fidelity="fast")
    scored = SS.reference_relative(declared["by_method"], baseline)["by_arm"]["arm_b"]["tasks"]["T0"]
    assert scored["speedup_median"] == 1.0 and scored["fidelity_source"] == "declared"


def test_the_board_reads_numbers_from_disk_and_states_a_missing_baseline(tmp_path, capsys):
    register = tmp_path / "TASKS.md"
    register.write_text(REGISTER)
    _matrix(tmp_path, "s4", [_row("arm_a", "T0", True, 500, fidelity="fast")])
    assert SS.main(["board", "--register", str(register), "--runs-root", str(tmp_path)]) == 0
    out = capsys.readouterr().out
    assert "REGISTER  4 tasks" in out and "!! 1 row(s) have no recognised state" in out
    assert "PERFORMANCE  unavailable" in out and "merlin-study-status baseline" in out

    graded = tmp_path / "T0.json"
    graded.write_text(json.dumps(_result("pass", {"L2": {"status": "pass", "cycles": 1000}})))
    code = SS.main(
        [
            "baseline",
            "--label",
            "ref",
            "--package",
            "pkg",
            "--target",
            "any",
            "--fidelity",
            "fast",
            "--certifying-tier",
            "L2",
            "--out",
            str(tmp_path / "baselines.json"),
            f"T0={graded}",
        ]
    )
    assert code == 0
    state_json = tmp_path / "state.json"
    SS.main(["board", "--register", str(register), "--runs-root", str(tmp_path), "--json", str(state_json)])
    state = json.loads(state_json.read_text())
    assert state["performance"]["by_arm"]["arm_a"]["tasks"]["T0"]["speedup_median"] == 2.0


def test_the_kernel_vs_compiler_study_register_and_void_list_parse():
    study = repo_root() / "merlin/experiments/llm_kernel_vs_compiler_v0"
    text = (study / "TASKS.md").read_text()
    tasks = SS.read_register(text)
    assert tasks and not [t["id"] for t in tasks if t["state"] == SS.UNRECOGNISED]
    spine = SS.critical_path(text, tasks)
    assert spine and all(row["state"] != SS.UNRECOGNISED for row in spine)
    assert SS.read_void(study / "voided_runs.yaml")
