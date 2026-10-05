"""The harness-owned OOT history: one commit per graded package, digest-exact, reproducible."""

from __future__ import annotations

import subprocess
from datetime import UTC, datetime
from pathlib import Path

import pytest

from merlin.common import oot_repo as O
from merlin.common.tree_hash import hash_tree


def _package(root: Path, *, body: str = "x = 1\n") -> Path:
    (root / "lowering").mkdir(parents=True)
    (root / "xdsl_dialects").mkdir()
    (root / "manifest.yaml").write_text("package_id: fixture\nentrypoints: {tool: fixture-opt}\n")
    tool = root / "fixture-opt"
    tool.write_text("#!/usr/bin/env python3\nprint('fixture')\n")
    tool.chmod(0o755)
    (root / "transforms.py").write_text(body)
    (root / "lowering" / "__init__.py").write_text("")
    (root / "xdsl_dialects" / "ops.py").write_text("OPS = ()\n")
    (root / "lowering-notes.md").write_text("ordering differs from git: a-b sorts after a/\n")
    # Generated subtrees are not the package and must not reach the commit.
    (root / "__pycache__").mkdir()
    (root / "__pycache__" / "x.pyc").write_bytes(b"\0")
    (root / "build").mkdir()
    (root / "build" / "out.o").write_bytes(b"\1")
    return root


def _git_log(repo: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), "log", "--format=%an <%ae>|%cn <%ce>|%ad|%cd", "--date=raw"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout


def test_round_trip_commit_digest_equals_package_digest(tmp_path):
    pkg = _package(tmp_path / "pkg")
    repo = O.init(tmp_path / "run" / "oot")
    rec = O.commit_candidate(repo, pkg, label="round 1", when="20260929T120000Z", run_id="r1", metadata={"n": 1})
    digest = hash_tree(pkg)["sha256"]
    assert rec.package_digest == digest == O.package_digest(pkg)
    assert O.tree_digest(repo, rec.commit) == digest
    O.verify(repo, rec.commit, digest)
    with pytest.raises(O.OotRepoError):
        O.verify(repo, rec.commit, "0" * 64)
    # The working tree mirrors the commit, without the generated subtrees.
    assert (repo / "transforms.py").read_text() == "x = 1\n"
    assert not (repo / "build").exists() and not (repo / "__pycache__").exists()
    # Export is the standalone tree: same digest, executable entry point, no .git.
    out = O.export(repo, rec.commit, tmp_path / "export")
    assert hash_tree(out)["sha256"] == digest
    assert not (out / ".git").exists()
    assert (out / "fixture-opt").stat().st_mode & 0o100
    assert O.history(repo)[0].as_record() == rec.as_record()


def test_commits_are_reproducible_and_carry_pinned_identity(tmp_path):
    pkg = _package(tmp_path / "pkg")
    when = datetime(2026, 9, 29, 12, 0, tzinfo=UTC)
    shas = []
    for name in ("a", "b"):
        repo = O.init(tmp_path / name)
        shas.append(O.commit_candidate(repo, pkg, label="round 1", when=when).commit)
    assert shas[0] == shas[1]
    log = _git_log(tmp_path / "a")
    assert log.strip() == (
        "Merlin Harness <harness@merlin.invalid>|Merlin Harness <harness@merlin.invalid>|"
        f"{int(when.timestamp())} +0000|{int(when.timestamp())} +0000"
    )
    with pytest.raises(O.OotRepoError, match="timezone"):
        O.commit_candidate(tmp_path / "a", pkg, label="round 2", when=datetime(2026, 9, 29))


def test_one_commit_per_round_even_when_unchanged(tmp_path):
    pkg = _package(tmp_path / "pkg")
    repo = O.init(tmp_path / "oot")
    first = O.commit_candidate(repo, pkg, label="round 1", when=1_000)
    second = O.commit_candidate(repo, pkg, label="round 2", when=2_000)
    assert second.parent == first.commit and second.tree == first.tree
    (pkg / "transforms.py").write_text("x = 2\n")
    third = O.commit_candidate(repo, pkg, label="round 3", when=3_000)
    assert third.tree != second.tree
    assert [r.label for r in O.history(repo)] == ["round 1", "round 2", "round 3"]


def test_phase2_clone_from_phase1_frozen(tmp_path):
    pkg = _package(tmp_path / "pkg")
    phase1 = O.init(tmp_path / "phase1" / "oot")
    O.commit_candidate(phase1, pkg, label="round 1", when=1_000)
    frozen = O.commit_candidate(phase1, pkg, label="round 2", when=2_000)
    O.tag(phase1, O.FROZEN_TAG)
    (pkg / "transforms.py").write_text("x = 99\n")
    O.commit_candidate(phase1, pkg, label="after freeze", when=3_000)  # never reaches phase 2

    phase2 = O.init_from(tmp_path / "phase2" / "oot", phase1)
    assert O.resolve(phase2, "HEAD") == O.resolve(phase2, "frozen") == frozen.commit
    assert O.tags(phase2) == {"frozen": frozen.commit}
    assert O.origin(phase2) == {"source": str(phase1.resolve()), "ref": "frozen", "commit": frozen.commit}
    remotes = subprocess.run(["git", "-C", str(phase2), "remote"], capture_output=True, text=True, check=True)
    assert remotes.stdout == ""
    (pkg / "transforms.py").write_text("x = 3\n")
    cand = O.commit_candidate(phase2, pkg, label="candidate 1", when=4_000)
    assert cand.parent == frozen.commit
    O.verify(phase2, cand.commit, hash_tree(pkg)["sha256"])
    O.tag(phase2, "measured/1", cand.commit)
    O.tag(phase2, O.BEST_TAG, cand.commit)
    O.tag(phase2, O.BEST_TAG, frozen.commit, move=True)
    assert O.resolve(phase2, "best") == frozen.commit


def test_historical_tags_never_move(tmp_path):
    pkg = _package(tmp_path / "pkg")
    repo = O.init(tmp_path / "oot")
    a = O.commit_candidate(repo, pkg, label="c1", when=1)
    (pkg / "transforms.py").write_text("changed\n")
    b = O.commit_candidate(repo, pkg, label="c2", when=2)
    O.tag(repo, "measured/1", a.commit)
    assert O.tag(repo, "measured/1", a.commit) == a.commit  # idempotent
    with pytest.raises(O.OotRepoError, match="refusing"):
        O.tag(repo, "measured/1", b.commit)
    with pytest.raises(O.OotRepoError, match="cannot move"):
        O.tag(repo, "frozen", b.commit, move=True)
    with pytest.raises(O.OotRepoError, match="invalid ref"):
        O.tag(repo, "bad..name", b.commit)


def test_repo_inside_an_agent_sandbox_is_refused(tmp_path):
    sandbox = tmp_path / "workspace"
    sandbox.mkdir()
    with pytest.raises(O.OotRepoError, match="agent-writable"):
        O.init(sandbox / "oot", sandbox_roots=[sandbox])
    with pytest.raises(O.OotRepoError, match="agent-writable"):
        O.init(tmp_path, sandbox_roots=[sandbox])
    repo = O.init(tmp_path / "run" / "oot", sandbox_roots=[sandbox])
    pkg = _package(sandbox / "pkg")
    # The package may live in the sandbox (it is what the agent wrote); the repo may not.
    O.commit_candidate(repo, pkg, label="round 1", when=1, sandbox_roots=[sandbox])


def test_symlinks_and_changed_packages_are_refused(tmp_path):
    pkg = _package(tmp_path / "pkg")
    (pkg / "link.py").symlink_to(pkg / "transforms.py")
    repo = O.init(tmp_path / "oot")
    with pytest.raises(O.OotRepoError, match="symlink"):
        O.commit_candidate(repo, pkg, label="round 1", when=1)
    with pytest.raises(O.OotRepoError, match="fresh"):
        O.init(repo)


def test_harness_config_ignores_hooks_and_user_config(tmp_path, monkeypatch):
    """A global hook or signing config must not run or change the commit."""
    home = tmp_path / "home"
    hooks = home / "hooks"
    hooks.mkdir(parents=True)
    (hooks / "pre-commit").write_text("#!/bin/sh\nexit 1\n")
    (home / ".gitconfig").write_text(f"[core]\n\thooksPath = {hooks}\n[commit]\n\tgpgSign = true\n")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("GIT_AUTHOR_NAME", "someone else")
    pkg = _package(tmp_path / "pkg")
    repo = O.init(tmp_path / "oot")
    rec = O.commit_candidate(repo, pkg, label="round 1", when=1)
    assert O.history(repo)[0].commit == rec.commit
    assert "Merlin Harness" in _git_log(repo)
