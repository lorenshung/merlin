"""When a measured run stops and what a session ran: the store-kept plateau, an exhausted account, the
model a round really ran, the infra circuit breaker; and the frozen source a run executes."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2.whole_model_measured import sessions as SES
from merlin_experiments.phase2.whole_model_measured import snapshot as SNAP


class _Objective:
    """A best that improves only when told to, over a store directory."""

    def __init__(self, root: Path, *, hours=6, min_sessions=2, history=()):
        self.screen = SimpleNamespace(root=root, history=lambda: list(history))
        root.mkdir(parents=True, exist_ok=True)
        self.config = {"plateau_hours": hours, "plateau_min_sessions": min_sessions, "stop_at_bar": False}
        self.cycles = 1000
        self.marked = 0

    def poll(self):
        return None

    def summary(self):
        return {"best": {"package_sha256": "b", "screen_whole_window_cycles": self.cycles}}

    def mark_session(self):
        self.marked += 1


def test_the_plateau_needs_both_its_hours_and_its_sessions(tmp_path):
    objective = _Objective(tmp_path / "store")
    hour = 3600.0
    assert SES.record_session(objective, session=1, run="r", now=0.0) is None
    assert SES.record_session(objective, session=2, run="r", now=7 * hour) is None  # 1 session since
    stop = SES.record_session(objective, session=3, run="r", now=8 * hour)
    assert stop["kind"] == "plateau" and stop["sessions_since_improvement"] == 2


def test_an_improvement_or_an_explicit_reset_restarts_the_clock(tmp_path):
    objective = _Objective(tmp_path / "store")
    hour = 3600.0
    SES.record_session(objective, session=1, run="r", now=0.0)
    objective.cycles = 900
    SES.record_session(objective, session=2, run="r", now=7 * hour)
    assert SES.record_session(objective, session=3, run="r", now=8 * hour) is None
    with pytest.raises(ValueError):
        SES.reset_plateau(SES.plateau_path(objective), reason=" ")
    SES.reset_plateau(SES.plateau_path(objective), reason="builder fix changed every measurement", now=20 * hour)
    assert SES.record_session(objective, session=4, run="r", now=21 * hour) is None


def test_the_plateau_survives_a_relaunch_because_it_lives_in_the_store(tmp_path):
    hour = 3600.0
    SES.record_session(_Objective(tmp_path / "store"), session=1, run="first", now=0.0)
    SES.record_session(_Objective(tmp_path / "store"), session=1, run="second", now=7 * hour)
    stop = SES.record_session(_Objective(tmp_path / "store"), session=2, run="second", now=8 * hour)
    assert stop is not None and stop["counted_from"] == "the last eligible improvement"


def test_an_exhausted_account_abandons_its_session_and_stops_rather_than_plateauing(tmp_path):
    """61 rounds once exited in seconds on a quota error and ended the run 'on evidence'."""
    objective = _Objective(tmp_path / "store", hours=0.0001, min_sessions=1)
    summary = tmp_path / "round_01.summary.json"
    summary.write_text(json.dumps({"errors": ["You've hit your usage limit. Try again later."]}))

    def run_round(*, session, stage_root):
        return {"status": "refused", "summaries": [str(summary)]}

    document = SES.run_sessions(
        objective, run_round=run_round, stage_root=tmp_path / "stage", run="r", max_sessions=5, total_seconds=1e9
    )
    assert document["stopped"]["kind"] == SES.INFRA_ACCOUNT_EXHAUSTED and document["stopped_on_evidence"] is False
    trace = json.loads(SES.plateau_path(objective).read_text())["trace"]
    assert trace[0]["abandoned"].startswith(SES.INFRA_ACCOUNT_EXHAUSTED)
    assert objective.marked == 1


def test_a_round_that_ran_another_model_is_refused(tmp_path):
    transcript = tmp_path / "t.jsonl"
    transcript.write_text(json.dumps({"model": "other-model", "model_requested": "asked-model"}) + "\n")
    assert "not the requested" in SES.verify_round_model(transcript, "asked-model")
    assert SES.verify_round_model(transcript, "other-model") is None
    (tmp_path / "empty.jsonl").write_text("")
    assert SES.verify_round_model(tmp_path / "empty.jsonl", "asked-model") is None
    objective = _Objective(tmp_path / "store")
    objective.config = {}

    def run_round(*, session, stage_root):
        return {"status": "authored", "transcript": str(transcript)}

    document = SES.run_sessions(
        objective,
        run_round=run_round,
        stage_root=tmp_path / "stage",
        run="r",
        max_sessions=3,
        total_seconds=1e9,
        model="asked-model",
    )
    assert document["stopped"]["kind"] == "model_mismatch" and len(document["sessions"]) == 1


def test_an_infra_streak_stops_the_run_and_says_so_beside_the_stage(tmp_path):
    infra = {"refusal": "build: ModuleNotFoundError: No module named x"}
    objective = _Objective(tmp_path / "store", history=[infra, infra, infra])
    document = SES.run_sessions(
        objective,
        run_round=lambda **k: pytest.fail("ran"),
        stage_root=tmp_path / "stage",
        run="r",
        max_sessions=3,
        total_seconds=1e9,
    )
    assert document["stopped"]["kind"] == "infra_circuit_breaker"
    assert (tmp_path / "stage" / "infra_circuit_breaker.json").is_file()


def test_a_backfill_never_duplicates_a_recorded_session(tmp_path):
    path = tmp_path / "plateau.json"
    SES.backfill_plateau(path, [{"epoch": 1.0, "run": "r", "session": 1, "cycles": 5}], reason="older store")
    added = SES.backfill_plateau(
        path,
        [{"epoch": 1.0, "run": "r", "session": 1, "cycles": 5}, {"epoch": 2.0, "run": "r", "session": 2}],
        reason="x",
    )
    assert [row["session"] for row in added] == [2]


# --------------------------------------------------------------- the frozen source
def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", *args], cwd=repo, check=True, capture_output=True
    )


def _repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    (repo / "src" / "pkg").mkdir(parents=True)
    (repo / "src" / "pkg" / "mod.py").write_text("X = 1\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-q", "-m", "init")
    (repo / ".env").write_text("HOST_ONLY=1\n")
    return repo


def test_a_dirty_tracked_file_or_a_stray_module_refuses_the_snapshot(tmp_path):
    repo = _repo(tmp_path)
    (repo / "src" / "pkg" / "mod.py").write_text("X = 2  # another session's edit\n")
    with pytest.raises(SNAP.SnapshotError, match="mod.py"):
        SNAP.committed_source(repo, tmp_path, source_roots=["src"])
    _git(repo, "checkout", "--", "src/pkg/mod.py")
    (repo / "src" / "pkg" / "stray.py").write_text("")
    with pytest.raises(SNAP.SnapshotError, match="stray.py"):
        SNAP.committed_source(repo, tmp_path, source_roots=["src"])


def test_the_snapshot_is_the_commit_with_the_live_trees_host_files(tmp_path):
    repo = _repo(tmp_path)
    (repo / "notes.txt").write_text("untracked, outside the source roots")
    checkout, sha = SNAP.committed_source(repo, tmp_path / "tmp", source_roots=["src"])
    assert (checkout / "src" / "pkg" / "mod.py").read_text() == "X = 1\n"
    assert (checkout / ".env").read_text() == "HOST_ONLY=1\n" and not (checkout / "notes.txt").exists()
    assert len(sha) == 40


def test_claim_dir_moves_a_leftover_aside_and_never_deletes_it(tmp_path):
    leftover = tmp_path / "iteration_0057"
    leftover.mkdir()
    (leftover / "partial").write_text("x")
    assert SNAP.claim_dir(leftover) == leftover and not leftover.exists()
    moved = [p for p in tmp_path.iterdir() if p.name.startswith("iteration_0057.interrupted_")]
    assert len(moved) == 1 and (moved[0] / "partial").is_file()


def _fake_snapshot(tmp_path: Path, clang: Path) -> Path:
    source = tmp_path / "snap"
    (source / "src" / "merlin" / "llvmlower").mkdir(parents=True)
    (source / "src" / "merlin" / "__init__.py").write_text("")
    (source / "src" / "merlin" / "llvmlower" / "__init__.py").write_text("")
    (source / "src" / "merlin" / "llvmlower" / "toolchain.py").write_text(
        f"from pathlib import Path\ndef clang():\n    return Path({str(clang)!r})\n"
    )
    return source


def test_the_preflight_refuses_a_snapshot_that_cannot_run_its_own_screen(tmp_path):
    catalog = tmp_path / "capsules" / "GR2"
    catalog.mkdir(parents=True)
    (catalog / "capsule.yaml").write_text("name: GR2\n")
    clang = tmp_path / "clang"
    source = _fake_snapshot(tmp_path, clang)
    common = dict(python=sys.executable, import_roots=["src"], capsule_catalog=tmp_path / "capsules")
    with pytest.raises(SNAP.SnapshotError, match="not a file"):
        SNAP.preflight(source, required_capsules=["GR2"], **common)
    clang.write_text("")
    with pytest.raises(SNAP.SnapshotError, match="GR3"):
        SNAP.preflight(source, required_capsules=["GR2", "GR3"], **common)
    assert SNAP.preflight(source, required_capsules=["GR2"], **common)["ok"]


def test_a_sealed_snapshot_from_a_commit_records_its_commit(tmp_path):
    from merlin_experiments import source_snapshot

    repo = _repo(tmp_path)
    destination = tmp_path / "run" / "stage.source"
    record = SNAP.create_from_commit(
        repo,
        destination,
        tmp_root=tmp_path / "tmp",
        source_roots=["src"],
        output_root=tmp_path / "out",
        python_roots=("src",),
        legacy_roots=(),
    )
    assert source_snapshot.verify(destination)["schema"] == source_snapshot.SCHEMA
    assert (destination / "src" / "pkg" / "mod.py").read_text() == "X = 1\n"
    assert (destination / ".env").read_text() == "HOST_ONLY=1\n"
    assert json.loads(Path(str(destination) + ".commit.json").read_text())["commit_sha"] == record["commit_sha"]
    assert not list((tmp_path / "tmp").iterdir())  # the scratch checkout is gone


def test_a_round_without_a_clean_audit_does_not_end_the_run(tmp_path):
    """A timed-out or killed agent once ended the live loop overnight: the next session must start."""
    objective = _Objective(tmp_path / "store")
    objective.config = {}
    outcomes = iter(
        [
            {"status": "refused", "audit_clean": False, "failure": "the round reached its deadline"},
            RuntimeError("agent process killed (signal 9)"),
            {"status": "authored"},
        ]
    )

    def run_round(*, session, stage_root):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    clock = iter(float(t) for t in range(0, 10_000, 100))
    document = SES.run_sessions(
        objective,
        run_round=run_round,
        stage_root=tmp_path / "stage",
        run="r",
        max_sessions=3,
        total_seconds=1e9,
        clock=lambda: next(clock),
    )
    assert [row["status"] for row in document["sessions"]] == ["refused", SES.ROUND_FAILED, "authored"]
    assert "signal 9" in document["sessions"][1]["failure"] and document["stopped"]["kind"] == "budget"


def test_a_crash_loop_is_the_one_round_failure_that_stops(tmp_path):
    objective = _Objective(tmp_path / "store")
    objective.config = {}

    def run_round(*, session, stage_root):
        raise RuntimeError("driver cannot start")

    document = SES.run_sessions(
        objective, run_round=run_round, stage_root=tmp_path / "stage", run="r", max_sessions=20, total_seconds=1e9
    )
    assert document["stopped"]["kind"] == "crash_loop" and len(document["sessions"]) == SES.CRASH_LOOP[0]
