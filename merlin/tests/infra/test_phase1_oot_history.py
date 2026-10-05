"""A dry phase-1 run writes its compiler history to out/runs/<target>/phase1/<run>/oot.

The real controller runs with the offline transport double and a dummy author (no provider, no paid
execution): two authored rounds, a weekly quota stop, then an operator seal through the official
freeze. The harness -- never the author -- commits each graded round and tags `frozen`.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import test_phase1_controller_integration as controller
import yaml

from merlin.common import oot_repo as O
from merlin.common.tree_hash import hash_tree

RUN_ID = "20260930T000000Z_raw_baseline_abc1234"


def _command(*extra: str) -> list[str]:
    argv = controller._command()
    argv[argv.index("--run-id") + 1] = RUN_ID
    argv[argv.index("--max-rounds") + 1] = "3"
    return argv + list(extra)


def _run(tmp_path):
    root = tmp_path / "controller"
    env = controller._project(root)
    env["PHASE1_FIXTURE_MODE"] = "history"
    fresh = subprocess.run(_command(), cwd=root, env=env, capture_output=True, text=True, timeout=180)
    assert fresh.returncode == 42, fresh.stdout + fresh.stderr
    sealed = subprocess.run(
        _command("--resume", "--seal-current"), cwd=root, env=env, capture_output=True, text=True, timeout=300
    )
    run_dir = root / "generated" / "runs" / "fixture" / "phase1" / RUN_ID
    assert run_dir.is_dir(), sealed.stdout + sealed.stderr
    # The offline fixture declares no formal simulator, so the official grader fails closed before its
    # freeze step (correctly). Run that step -- the same freeze() the formal grader calls -- directly.
    assert "cannot resolve formal whole-model simulator" in sealed.stderr, sealed.stdout + sealed.stderr
    frozen = subprocess.run(
        [
            sys.executable,
            "-m",
            "merlin_experiments.phase1.feedback.freeze",
            "--run-dir",
            str(run_dir),
            "--repo",
            env["MERLIN_REPO_ROOT"],
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert frozen.returncode == 0, frozen.stdout + frozen.stderr
    return root, env, run_dir, sealed


def test_dry_phase1_run_commits_each_graded_round_and_tags_frozen(tmp_path):
    root, env, run_dir, sealed = _run(tmp_path)
    repo = run_dir / "oot"
    assert (repo / ".git").is_dir(), sealed.stdout + sealed.stderr
    graded = sorted((run_dir / "qa_history").glob("verdict_round_*.json"))
    rows = [json.loads(line) for line in (run_dir / "oot_commits.jsonl").read_text().splitlines()]
    assert not [row for row in rows if "error" in row], rows
    history = [rec for rec in O.history(repo) if rec.label != "freeze"]
    # One commit per graded round, in order, each naming the round it graded.
    assert len(history) == len(graded) >= 2
    assert [rec.metadata["key"] for rec in history] == [p.stem.rsplit("_", 1)[1] for p in graded]
    assert [row["oot"]["commit"] for row in rows] == [rec.commit for rec in history]
    for rec in history:
        snapshot = run_dir / "_qa_work" / f"cand_{rec.metadata['key']}" / "submission"
        assert O.tree_digest(repo, rec.commit) == hash_tree(snapshot)["sha256"] == rec.package_digest
    # The round records carry the sha, not a copy of the package.
    summary = yaml.safe_load((run_dir / "qa_loop_summary.yaml").read_text())
    assert [r["oot_commit"] for r in summary["rounds"]] == [rec.commit for rec in history]
    # `frozen` names exactly the submission the official freeze hashed.
    freeze = json.loads((run_dir / "freeze.json").read_text())
    frozen = O.resolve(repo, O.FROZEN_TAG)
    assert freeze["oot"] == {"repo": str(repo), "frozen_commit": frozen, "package_digest": freeze["submission_sha256"]}
    assert O.tree_digest(repo, frozen) == hash_tree(run_dir / "submission")["sha256"] == freeze["submission_sha256"]
    # Phase 2 can start from it.
    clone = O.init_from(tmp_path / "phase2" / "oot", repo)
    assert O.resolve(clone, "HEAD") == frozen


def test_the_agent_sandbox_cannot_write_the_history(tmp_path):
    root, env, run_dir, _ = _run(tmp_path)
    calls = [json.loads(line) for line in (root / "provider_calls.jsonl").read_text().splitlines()]
    workspace = Path(calls[0]["workspace"])
    repo = (run_dir / "oot").resolve()
    assert not repo.is_relative_to(workspace.parent.resolve())
    # Compose the real isolation argv for that workspace and check what it mounts.
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            "import json, sys, yaml; from pathlib import Path\n"
            "from merlin.targetgen.sandbox import bwrap as B\n"
            "from merlin.targetgen.target_experiment import load_target_experiment as L\n"
            "ws, repo, bundle, te = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4])\n"
            "argv = B.full_argv(L(te), ws, yaml.safe_load(bundle.read_text()))\n"
            "writable = [argv[i + 2] for i, op in enumerate(argv) if op in ('--bind', '--bind-try', '--dev-bind')]\n"
            "print(json.dumps({'exposed': B.is_exposed(argv, repo), 'writable': writable}))\n",
            str(workspace),
            str(repo),
            str(run_dir / "input_bundle_manifest.yaml"),
            str(root / "target_experiment.yaml"),
        ],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert probe.returncode == 0, probe.stderr
    view = json.loads(probe.stdout)
    assert view["exposed"] is False
    for destination in view["writable"]:
        assert not repo.is_relative_to(Path(destination).resolve()), destination
    # And the history refuses to live inside the agent's writable root at all.
    try:
        O.init(workspace.parent / "oot", sandbox_roots=[workspace.parent])
    except O.OotRepoError as exc:
        assert "agent-writable" in str(exc)
    else:
        raise AssertionError("a repo inside the agent workspace was accepted")
