"""A worktree that cannot answer truthfully must say so, not answer.

MEASURED 2026-09-23. A `git worktree` of this repo carries only tracked files. Eight gitignored
inputs were needed before a run produced a true answer, and NOT ONE failed loudly when absent --
each degraded into a plausible wrong answer. The cost that day: `.env` absent gave 28 failures out
of 28 collected, all collection errors; the `rtl_introspect` cache absent made every capability
question answer UNKNOWN, and UNKNOWN is silently non-blocking, so a perf family reported itself
inapplicable rather than underivable; and `third_party/llvm-build` absent made a whole module SKIP,
which a failure-set diff cannot distinguish from a pass -- producing one phantom regression in a
base-vs-branch comparison that cost an hour to disbelieve.

So the roster is data with a stated consequence per entry, and the provisioner FAILS CLOSED: an
input it could not provide is named with the symptom it produces, never skipped quietly.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest
import yaml

from merlin.common.paths import repo_root


def _tool():
    spec = importlib.util.spec_from_file_location(
        "provision_worktree_under_test", repo_root() / "build_tools/scripts/provision_worktree.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _checkout(root: Path) -> Path:
    """A directory the provisioner will accept as a checkout."""
    root.mkdir(parents=True, exist_ok=True)
    (root / ".git").write_text("gitdir: elsewhere\n", encoding="utf-8")
    return root


def test_every_rostered_artifact_states_its_consequence() -> None:
    """The symptom is never 'file not found', so the roster has to carry what it IS."""
    doc = yaml.safe_load((repo_root() / "build_tools/scripts/worktree_provisioning.yaml").read_text(encoding="utf-8"))
    rows = doc["artifacts"]
    assert rows, "a roster that provisions nothing cannot fail, which is the defect it exists to prevent"
    for row in rows:
        assert row.get("consequence", "").strip(), f"{row} names no consequence"
        assert ("path" in row) ^ ("path_glob" in row), f"{row} must name exactly one of path/path_glob"


def test_a_missing_artifact_is_named_with_its_consequence(tmp_path: Path) -> None:
    tool = _tool()
    source, dest = _checkout(tmp_path / "src"), _checkout(tmp_path / "dst")
    (source / ".env").write_text("X=1\n", encoding="utf-8")
    result = tool.provision(dest, source)
    assert ".env" in result["linked"]
    assert result["missing"], "the source lacked every other artifact and the report said nothing"
    for _rel, why in result["missing"]:
        assert why.strip(), "a missing artifact was reported without the symptom it produces"


def test_it_links_rather_than_copies(tmp_path: Path) -> None:
    """A copy drifts from its source silently -- the failure this whole roster exists to prevent."""
    tool = _tool()
    source, dest = _checkout(tmp_path / "src"), _checkout(tmp_path / "dst")
    (source / ".env").write_text("X=1\n", encoding="utf-8")
    tool.provision(dest, source)
    assert (dest / ".env").is_symlink()
    (source / ".env").write_text("X=2\n", encoding="utf-8")
    assert (dest / ".env").read_text(encoding="utf-8") == "X=2\n"


def test_it_is_idempotent(tmp_path: Path) -> None:
    tool = _tool()
    source, dest = _checkout(tmp_path / "src"), _checkout(tmp_path / "dst")
    (source / ".env").write_text("X=1\n", encoding="utf-8")
    first = tool.provision(dest, source)
    second = tool.provision(dest, source)
    assert ".env" in first["linked"] and ".env" in second["present"]


def test_verify_changes_nothing(tmp_path: Path) -> None:
    tool = _tool()
    source, dest = _checkout(tmp_path / "src"), _checkout(tmp_path / "dst")
    (source / ".env").write_text("X=1\n", encoding="utf-8")
    tool.provision(dest, source, verify_only=True)
    assert not (dest / ".env").exists(), "--verify wrote into the destination"


def test_a_destination_that_is_not_a_checkout_is_refused(tmp_path: Path) -> None:
    tool = _tool()
    source = _checkout(tmp_path / "src")
    (source / ".env").write_text("X=1\n", encoding="utf-8")
    with pytest.raises(tool.ProvisionError, match="not a git checkout"):
        tool.provision(tmp_path / "plain", source)


def test_provisioning_a_checkout_from_itself_is_refused(tmp_path: Path) -> None:
    tool = _tool()
    source = _checkout(tmp_path / "src")
    with pytest.raises(tool.ProvisionError, match="same checkout"):
        tool.provision(source, source)


def test_a_source_with_nothing_is_an_error_not_a_clean_run(tmp_path: Path) -> None:
    """The shape this roster exists to stop: provisioning zero artifacts and reporting success."""
    tool = _tool()
    source, dest = _checkout(tmp_path / "src"), _checkout(tmp_path / "dst")
    with pytest.raises(tool.ProvisionError, match="nothing was provisioned"):
        tool.provision(dest, source)


def test_a_tracked_directory_does_not_hide_its_untracked_files(tmp_path) -> None:
    """`merge: per_file` compares FILE BY FILE, because the directory itself comes from git.

    `merlin/contract/capsules/model` is tracked, so a whole-path check calls it present while the
    gitignored weights inside it are absent. Measured: every model capsule in a live phase-1 run
    graded `incomplete` for the whole run on "external weights asset is missing or a symlink".
    """
    source, worktree = tmp_path / "src", tmp_path / "wt"
    (source / "d" / "cap").mkdir(parents=True)
    (source / "d" / "cap" / "w.safetensors").write_bytes(b"weights")
    _checkout(worktree)
    (worktree / "d" / "cap").mkdir(parents=True)  # the tracked directory, already there

    roster = tmp_path / "roster.yaml"
    roster.write_text(
        "artifacts:\n"
        "  - path: d\n"
        "    kind: dir\n"
        "    merge: per_file\n"
        '    files_glob: "*/w.safetensors"\n'
        "    link: hard\n"
        "    consequence: the weights are absent and every model capsule grades incomplete\n",
        encoding="utf-8",
    )
    PW = _tool()
    PW.ROSTER = roster
    result = PW.provision(worktree, source)

    placed = worktree / "d" / "cap" / "w.safetensors"
    assert "d/cap/w.safetensors" in result["linked"], result
    assert placed.exists() and placed.read_bytes() == b"weights"
    # A HARD link, not a symlink: the consumer refuses a symlink, so a symlink here would provision
    # nothing while the run reported success.
    assert not placed.is_symlink()
    assert placed.stat().st_nlink == 2


def test_the_default_placement_is_still_a_symlink(tmp_path) -> None:
    """Only an entry that declares `link: hard` gets one; everything else stays a cheap symlink."""
    source, worktree = tmp_path / "src", tmp_path / "wt"
    (source / "thing").mkdir(parents=True)
    _checkout(worktree)
    roster = tmp_path / "roster.yaml"
    roster.write_text(
        "artifacts:\n  - path: thing\n    kind: dir\n    consequence: absent and nothing says so\n",
        encoding="utf-8",
    )
    PW = _tool()
    PW.ROSTER = roster
    PW.provision(worktree, source)
    assert (worktree / "thing").is_symlink()


def test_provisioning_reports_a_shadowed_library(tmp_path) -> None:
    """The linked venv runs the SOURCE's merlin, and provisioning is what creates that link.

    A worktree that imports another checkout's library tests its own tests against somebody else's
    code. Measured twice: a 30-minute sweep against the wrong library, and a phase-1 run "pinned" to
    a commit whose pin covered only the scripts invoked by path, never the code they import.
    """
    PW = _tool()
    worktree = tmp_path / "wt"
    (worktree / "src" / "merlin").mkdir(parents=True)
    (worktree / "packages" / "merlin-extra" / "src" / "merlin_extra").mkdir(parents=True)
    venv_bin = worktree / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    other = tmp_path / "other" / "merlin" / "python" / "merlin" / "__init__.py"
    other.parent.mkdir(parents=True)
    other.write_text("", encoding="utf-8")

    fake = venv_bin / "python"
    fake.write_text(f'#!/bin/sh\necho "{other}"\n', encoding="utf-8")
    fake.chmod(0o755)

    report = PW.resolved_library(worktree)
    assert report["status"] == "SHADOWED"
    assert report["resolves"] == str(other)
    # the corrective PYTHONPATH names every source root of THIS tree, the core first
    roots = report["expected"].split(os.pathsep)
    assert roots == [str((worktree / "src").resolve()), str((worktree / "packages/merlin-extra/src").resolve())]


def test_a_worktree_importing_its_own_library_is_not_flagged(tmp_path) -> None:
    """The other direction: a tree whose python resolves its OWN merlin must not warn."""
    PW = _tool()
    worktree = tmp_path / "wt"
    own = worktree / "src" / "merlin" / "__init__.py"
    own.parent.mkdir(parents=True)
    own.write_text("", encoding="utf-8")
    venv_bin = worktree / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    fake = venv_bin / "python"
    fake.write_text(f'#!/bin/sh\necho "{own}"\n', encoding="utf-8")
    fake.chmod(0o755)

    assert PW.resolved_library(worktree)["status"] == "own"
    # the compatibility symlink into src/ is still this tree's own library
    (worktree / "merlin" / "python").mkdir(parents=True)
    (worktree / "merlin" / "python" / "merlin").symlink_to(worktree / "src" / "merlin")
    via_link = worktree / "merlin" / "python" / "merlin" / "__init__.py"
    fake.write_text(f'#!/bin/sh\necho "{via_link}"\n', encoding="utf-8")
    assert PW.resolved_library(worktree)["status"] == "own"


def test_a_worktree_without_a_venv_is_reported_not_guessed(tmp_path) -> None:
    worktree = tmp_path / "wt"
    worktree.mkdir()
    assert _tool().resolved_library(worktree)["status"] == "no venv"


def test_a_link_to_nothing_is_reported_not_counted_as_present(tmp_path: Path) -> None:
    """A dangling link from an earlier provisioning is not the input; it must be named, not skipped."""
    tool = _tool()
    source, dest = _checkout(tmp_path / "src"), _checkout(tmp_path / "dst")
    (source / ".env").write_text("X=1\n", encoding="utf-8")
    (source / ".compat_lib").mkdir()
    (dest / ".env").symlink_to(tmp_path / "gone")
    result = tool.provision(dest, source)
    assert ".compat_lib" in result["linked"]
    assert ".env" not in result["present"]
    assert any(rel == ".env" and "link to nothing" in why for rel, why in result["missing"])


def test_the_roster_names_paths_this_layout_has(tmp_path: Path) -> None:
    """Every tracked directory a per-file entry merges into exists in this checkout, and no entry names a
    retired root: provisioning a tree that has moved on would otherwise report present/missing for
    paths nothing reads."""
    doc = yaml.safe_load((repo_root() / "build_tools/scripts/worktree_provisioning.yaml").read_text(encoding="utf-8"))
    for row in doc["artifacts"]:
        name = row.get("path") or row.get("path_glob")
        assert name.split("/", 1)[0] not in ("build", "packages", "output", "runs", "artifacts"), name
        if row.get("merge") == "per_file" and "path" in row:
            assert (repo_root() / row["path"]).is_dir(), f"{row['path']} is not a tracked directory here"
