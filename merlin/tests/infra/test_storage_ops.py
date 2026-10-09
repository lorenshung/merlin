"""Moving, linking and surveying trees outside out/: verified, dry-run by default, never through a link.

These operations used to be throwaway shell scripts, and each one's safety rested on whoever ran it
remembering the checks: copy without following symlinks, verify by checksum before deleting, leave
a symlink so quoted paths still resolve, refuse a tree a live process is using, borrow and return
the write bit on a frozen directory, and never write to a worktree that is only being looked at.
Each of those is pinned here.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common import storage_cli as SC
from merlin.common import storage_ops as SO

needs_rsync = pytest.mark.skipif(shutil.which("rsync") is None, reason="rsync is not installed on this host")


class _Holders:
    """An open-file check that reports ``held`` (and anything under it) as in use."""

    tool = "test"

    def __init__(self, *held: Path) -> None:
        self.held = [Path(h) for h in held]

    def holders(self, path: Path) -> list[int]:
        path = Path(path)
        return [4242] if any(h == path or path in h.parents or h in path.parents for h in self.held) else []


def _tree(root: Path) -> Path:
    (root / "sub").mkdir(parents=True)
    (root / "a.bin").write_bytes(b"a" * 5000)
    (root / "sub" / "b.txt").write_text("hello\n")
    (root / "link").symlink_to("a.bin")
    return root


def _git(cwd: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(cwd), "-c", "user.email=t@t", "-c", "user.name=t", *args],
        check=True,
        capture_output=True,
        stdin=subprocess.DEVNULL,
    )


@pytest.fixture
def own_repository(tmp_path):
    """``tmp_path`` as its own empty git repository.

    The CLI's move and dedup refuse a tree that the enclosing git index tracks or cannot be read
    (``storage_cli._holds_tracked_files``). Which repository encloses pytest's temp directory is a fact
    about the host -- a checkout, none, or a directory another account owns whose index cannot be
    read -- so a test that drives the CLI brings its own and holds the same answer everywhere.
    """
    _git(tmp_path, "init", "-q")
    return tmp_path


# --- move ------------------------------------------------------------------------------------------


@needs_rsync
def test_move_copies_verifies_and_leaves_a_relative_symlink(tmp_path):
    source = _tree(tmp_path / "disk1" / "data")
    destination = tmp_path / "disk2" / "archive" / "data"

    record = SO.move(source, destination, apply=True, checker=_Holders())

    assert record["status"] == "moved", record
    assert source.is_symlink()
    assert not os.path.isabs(os.readlink(source)), "the symlink must be relative so the root can move"
    assert (source / "a.bin").read_bytes() == b"a" * 5000, "the old path no longer resolves"
    assert (destination / "sub" / "b.txt").read_text() == "hello\n"
    assert (destination / "link").is_symlink(), "a symlink inside the tree was followed instead of copied"
    assert os.readlink(destination / "link") == "a.bin"


@needs_rsync
def test_a_dry_run_move_changes_nothing(tmp_path):
    source = _tree(tmp_path / "data")
    destination = tmp_path / "elsewhere" / "data"

    record = SO.move(source, destination, checker=_Holders())

    assert record["status"] == "planned"
    assert record["files"] == 2 and record["symlinks"] == 1
    assert source.is_dir() and not source.is_symlink()
    assert not destination.exists() and not destination.parent.exists()


def test_a_symlink_source_is_refused(tmp_path):
    real = _tree(tmp_path / "real")
    alias = tmp_path / "alias"
    alias.symlink_to(real)

    record = SO.move(alias, tmp_path / "dest", apply=True, checker=_Holders())

    assert record["status"] == "refused"
    assert "symlink" in record["reasons"][0]
    assert alias.is_symlink() and real.is_dir()


@needs_rsync
def test_a_source_held_open_is_refused_and_kept(tmp_path):
    source = _tree(tmp_path / "data")

    record = SO.move(source, tmp_path / "dest", apply=True, checker=_Holders(source / "sub" / "b.txt"))

    assert record["status"] == "refused"
    assert "4242" in record["reasons"][0]
    assert source.is_dir() and not source.is_symlink()
    assert not (tmp_path / "dest").exists(), "a copy was started for a tree that was in use"


@needs_rsync
def test_a_holder_appearing_during_the_copy_keeps_the_source(tmp_path):
    """The check runs again after the copy: a large copy is long enough for a process to start."""
    source = _tree(tmp_path / "data")
    calls = []

    class Late(_Holders):
        def holders(self, path):
            calls.append(path)
            return [] if len(calls) == 1 else [7]

    record = SO.move(source, tmp_path / "dest", apply=True, checker=Late())

    assert record["status"] == "failed"
    assert source.is_dir() and not source.is_symlink()
    assert (tmp_path / "dest" / "a.bin").read_bytes() == b"a" * 5000


@needs_rsync
def test_post_copy_lsof_check_reopens_the_original_cached_listing(tmp_path, monkeypatch):
    source = _tree(tmp_path / "data")
    calls = []
    real_run = subprocess.run

    def listing(argv, **kwargs):
        if argv[0] != "/owned/lsof-fixture":
            return real_run(argv, **kwargs)
        calls.append(tuple(argv))
        opened = tmp_path / "unrelated" if len(calls) == 1 else source / "sub/b.txt"
        return subprocess.CompletedProcess(argv, 0, f"p7\nn{opened}\n", "")

    monkeypatch.setattr(SO.subprocess, "run", listing)
    check = SO._LsofIndex("/owned/lsof-fixture")
    result = SO.move(source, tmp_path / "dest", apply=True, checker=check)
    assert len(calls) == 2
    assert result["status"] == "failed" and "after the copy" in result["reasons"][0]
    assert source.is_dir() and not source.is_symlink()
    assert (tmp_path / "dest/sub/b.txt").read_bytes() == (source / "sub/b.txt").read_bytes()


@needs_rsync
def test_actual_process_opened_after_copy_is_seen_by_fresh_lsof(tmp_path, monkeypatch):
    if shutil.which("lsof") is None:
        pytest.skip("lsof is not installed on this host")
    source = _tree(tmp_path / "data")
    check = SO.open_file_checker(which=lambda name: shutil.which(name) if name == "lsof" else None)
    verify = SO._rsync_verify
    processes = []

    def open_after_copy(*args):
        difference = verify(*args)
        with (source / "sub/b.txt").open("rb") as stream:
            processes.append(subprocess.Popen(["sleep", "60"], stdin=stream))
        return difference

    monkeypatch.setattr(SO, "_rsync_verify", open_after_copy)
    try:
        result = SO.move(source, tmp_path / "dest", apply=True, checker=check)
        assert result["status"] == "failed" and str(processes[0].pid) in result["reasons"][0]
        assert source.is_dir() and not source.is_symlink()
        assert (tmp_path / "dest/sub/b.txt").is_file()
    finally:
        for process in processes:
            process.kill()
            process.wait()


def test_no_open_file_tool_means_refusal_unless_the_caller_opts_out(tmp_path, monkeypatch):
    source = _tree(tmp_path / "data")
    with pytest.raises(SO.OpenFileCheckUnavailable, match="no-open-file-check"):
        SO.open_file_checker(which=lambda name: None)
    real = SO.open_file_checker
    monkeypatch.setattr(SO, "open_file_checker", lambda: real(which=lambda name: None))
    record = SO.move(source, tmp_path / "dest")
    assert record["status"] == "refused" and "lsof" in record["reasons"][0]
    if shutil.which("rsync"):
        assert SO.move(source, tmp_path / "dest", check_open=False)["status"] == "planned"


def test_an_existing_destination_or_a_denied_path_is_refused(tmp_path):
    source = _tree(tmp_path / "data")
    (tmp_path / "taken").mkdir()

    assert SO.move(source, tmp_path / "taken", checker=_Holders())["status"] == "refused"
    denied = SO.move(source, tmp_path / "dest", deny=[tmp_path / "data"], checker=_Holders())
    assert denied["status"] == "refused" and "deny-list" in denied["reasons"][0]


def test_protection_reasons_from_the_caller_are_honoured(tmp_path):
    source = _tree(tmp_path / "data")

    record = SO.move(source, tmp_path / "dest", checker=_Holders(), protected=lambda p: ["retention pin"])

    assert record == {**record, "status": "refused", "reasons": ["retention pin"]}


@needs_rsync
def test_the_cli_move_is_a_dry_run_by_default(own_repository, monkeypatch, capsys):
    tmp_path = own_repository
    source = _tree(tmp_path / "data")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(SO, "open_file_checker", lambda: _Holders())

    assert SC.main(["move", str(source), str(tmp_path / "dest")]) == 0

    assert "dry run" in capsys.readouterr().out
    assert source.is_dir() and not source.is_symlink() and not (tmp_path / "dest").exists()


@needs_rsync
def test_the_cli_refuses_to_move_a_tree_its_repository_tracks(own_repository, monkeypatch, capsys):
    """The other direction of the fixture above: the same tree, once tracked, is not moved."""
    tmp_path = own_repository
    source = _tree(tmp_path / "data")
    _git(tmp_path, "add", "data/sub/b.txt")
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setattr(SO, "open_file_checker", lambda: _Holders())

    assert SC.main(["move", str(source), str(tmp_path / "dest")]) == 1

    assert "tracked files" in capsys.readouterr().out
    assert source.is_dir() and not source.is_symlink() and not (tmp_path / "dest").exists()


# --- linking duplicates ----------------------------------------------------------------------------


def _groups(*paths: Path) -> list[tuple[int, list[Path]]]:
    return [(paths[0].stat().st_size, list(paths))]


def test_identical_files_become_one_inode(tmp_path):
    first, second = tmp_path / "a" / "w.bin", tmp_path / "b" / "w.bin"
    for path in (first, second):
        path.parent.mkdir()
        path.write_bytes(b"w" * 4096)

    result = SO.link_duplicates(_groups(first, second), apply=True, checker=_Holders())

    assert result["linked"] == 1 and result["released_bytes"] == 4096
    assert first.stat().st_ino == second.stat().st_ino
    assert second.read_bytes() == b"w" * 4096
    assert not first.stat().st_mode & 0o222, "a shared inode must refuse in-place writes"


def test_changed_content_and_held_files_are_skipped(tmp_path):
    keep, changed, held = (tmp_path / n for n in ("keep", "changed", "held"))
    for path in (keep, changed, held):
        path.write_bytes(b"x" * 4096)
    changed.write_bytes(b"y" * 4096)  # rewritten after the (simulated) scan

    result = SO.link_duplicates(_groups(keep, changed, held), apply=True, checker=_Holders(held))

    assert result["linked"] == 0
    assert {why.split(" ")[0] for _, why in result["skipped"]} == {"content", "held"}
    assert len({p.stat().st_ino for p in (keep, changed, held)}) == 3


def test_peer_link_refreshes_liveness_after_hashing_and_cleans_staged_link(tmp_path, monkeypatch):
    keep, name = tmp_path / "keep", tmp_path / "name"
    keep.write_bytes(b"contents")
    name.write_bytes(keep.read_bytes())
    check = _Holders()
    digest = SO._digest

    def acquire_after_hash(path):
        result = digest(path)
        if path == name:
            check.held.append(name)
        return result

    monkeypatch.setattr(SO, "_digest", acquire_after_hash)
    result = SO.link_duplicates(_groups(keep, name), apply=True, checker=check)
    assert result["linked"] == 0 and "held open" in result["skipped"][0][1]
    assert name.stat().st_ino != keep.stat().st_ino
    assert keep.stat().st_mode & 0o200, "a refused swap must not change the kept inode's mode"
    assert not list(tmp_path.glob(".*.link"))


def test_store_adoption_rechecks_after_copy_and_preserves_original_name(tmp_path, monkeypatch):
    from merlin.common import content_store as CS

    path = tmp_path / "owned/name"
    path.parent.mkdir()
    path.write_bytes(b"original bytes")
    store = tmp_path / "store"
    check = _Holders()
    original = path.stat().st_ino
    object_for = CS.object_for

    def acquire_after_object_copy(*args):
        result = object_for(*args)
        check.held.append(path)
        return result

    monkeypatch.setattr(CS, "object_for", acquire_after_object_copy)
    monkeypatch.setattr(CS, "store_root", lambda: store)
    monkeypatch.setattr(SC, "store_root", lambda: store)
    monkeypatch.setattr(SC, "_holds_tracked_files", lambda _path: False)
    with pytest.raises(SO.OpenFileCheckUnavailable, match="before store adoption"):
        SC.dedup([(path.stat().st_size, [path])], checker=check)
    assert path.stat().st_ino == original and path.read_bytes() == b"original bytes"
    assert not list(path.parent.glob(".*.adopt"))


def test_a_read_only_directory_gets_its_exact_mode_back(tmp_path):
    first, second = tmp_path / "a" / "w.bin", tmp_path / "frozen" / "w.bin"
    for path in (first, second):
        path.parent.mkdir()
        path.write_bytes(b"f" * 4096)
    second.parent.chmod(0o550)
    try:
        result = SO.link_duplicates(_groups(first, second), apply=True, checker=_Holders())
        assert result["linked"] == 1
        assert first.stat().st_ino == second.stat().st_ino
        assert second.parent.stat().st_mode & 0o777 == 0o550, "the frozen directory was left writable"
    finally:
        second.parent.chmod(0o750)


def test_a_dry_run_link_changes_nothing(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.write_bytes(b"d" * 4096)
    second.write_bytes(b"d" * 4096)

    result = SO.link_duplicates(_groups(first, second), checker=_Holders())

    assert result["linked"] == 1 and not result["apply"]
    assert first.stat().st_ino != second.stat().st_ino
    assert first.stat().st_mode & 0o200


def test_a_symlink_is_never_linked_or_followed(tmp_path):
    real = tmp_path / "real"
    real.write_bytes(b"s" * 4096)
    other = tmp_path / "other"
    other.write_bytes(b"s" * 4096)
    alias = tmp_path / "alias"
    alias.symlink_to(other)

    result = SO.link_duplicates(_groups(real, alias), apply=True, checker=_Holders())

    assert result["linked"] == 0
    assert alias.is_symlink() and os.readlink(alias) == str(other)
    assert other.stat().st_ino != real.stat().st_ino


def test_dedup_peers_from_the_cli(own_repository, monkeypatch):
    """``dedup --peers`` finds the groups the store-based dedup finds and links them to each other."""
    tmp_path = own_repository
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    declared = dict(SC.contract(), scan_roots=[])
    monkeypatch.setattr(SC, "contract", lambda: declared)
    monkeypatch.setattr(SO, "checker_for", lambda enabled: _Holders())
    tree = tmp_path / "elsewhere"
    first, second = tree / "a" / "w.bin", tree / "b" / "w.bin"
    for path in (first, second):
        path.parent.mkdir(parents=True)
        path.write_bytes(b"p" * 4096)
    (tree / "c").mkdir()
    (tree / "c" / "w.bin").symlink_to(first)

    assert SC.main(["dedup", str(tree), "--min-bytes", "1024", "--peers", "--apply"]) == 0

    assert first.stat().st_ino == second.stat().st_ino
    assert not (tmp_path / "out" / "artifacts" / "cache").exists(), "peer linking wrote into the store"
    assert (tree / "c" / "w.bin").is_symlink()


# --- worktree census -------------------------------------------------------------------------------


def test_worktrees_are_classified_without_writing_to_them(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "f.txt").write_text("1\n")
    _git(repo, "add", "f.txt")
    _git(repo, "commit", "-q", "-m", "one")
    _git(repo, "worktree", "add", "-q", "-b", "merged", str(tmp_path / "wt-merged"))
    _git(repo, "worktree", "add", "-q", "-b", "dirty", str(tmp_path / "wt-dirty"))
    (tmp_path / "wt-dirty" / "f.txt").write_text("changed\n")
    (tmp_path / "wt-dirty" / "new.txt").write_text("untracked\n")
    _git(repo, "worktree", "add", "-q", "-b", "ahead", str(tmp_path / "wt-ahead"))
    (tmp_path / "wt-ahead" / "g.txt").write_text("2\n")
    _git(tmp_path / "wt-ahead", "add", "g.txt")
    _git(tmp_path / "wt-ahead", "commit", "-q", "-m", "two")
    _git(repo, "worktree", "add", "-q", "-b", "private", str(tmp_path / "wt-private"))
    index = repo / ".git" / "worktrees" / "wt-dirty" / "index"
    before = index.stat().st_mtime_ns

    census = SO.worktrees(repo, base="main", deny=[tmp_path / "wt-private"], checker=_Holders(tmp_path / "wt-ahead"))

    rows = {Path(r["path"]).name: r for r in census["worktrees"]}
    assert census["base"] == "main"
    assert rows["wt-merged"]["state"] == "clean" and rows["wt-merged"]["merged"] is True
    assert rows["wt-merged"]["ahead"] == 0 and rows["wt-merged"]["branch"] == "merged"
    assert rows["wt-dirty"]["state"] == "dirty"
    assert rows["wt-dirty"]["modified"] == 1 and rows["wt-dirty"]["untracked"] == 1
    assert rows["wt-ahead"]["state"] == "clean" and rows["wt-ahead"]["merged"] is False
    assert rows["wt-ahead"]["ahead"] == 1 and rows["wt-ahead"]["holders"] == [4242]
    assert rows["wt-private"] == {**rows["wt-private"], "state": "denied"}
    assert "merged" not in rows["wt-private"] and "size" not in rows["wt-private"], "a denied tree was read"
    assert rows["wt-merged"]["size"] and rows["wt-merged"]["size"] > 0
    assert index.stat().st_mtime_ns == before, "the census refreshed a worktree's index"


def test_the_default_base_is_the_repositorys_main(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q", "-b", "main")
    (repo / "f").write_text("x")
    _git(repo, "add", "f")
    _git(repo, "commit", "-q", "-m", "one")

    assert SO.default_base(repo) == "main"
    assert SO.worktrees(repo, sizes=False, check_open=False)["open_file_check"] == "disabled"


def test_the_deny_list_is_lexical_and_empty_by_default(tmp_path):
    assert SO.denied(tmp_path / "a" / "b", [tmp_path / "a"]) == tmp_path / "a"
    assert SO.denied(tmp_path / "ab", [tmp_path / "a"]) is None
    assert SO.denied(tmp_path / "a", []) is None


@pytest.mark.parametrize("tool", ["lsof", "fuser"])
def test_the_real_check_sees_a_process_holding_a_file_under_a_tree(tmp_path, tool):
    """Both tools, parsed for real: a mis-read listing reports every tree as free."""
    if shutil.which(tool) is None:
        pytest.skip(f"{tool} is not installed on this host")
    tree = _tree(tmp_path / "held")
    (tmp_path / "held-not").mkdir()
    with (tree / "sub" / "b.txt").open("rb") as stream:
        sleeper = subprocess.Popen(["sleep", "60"], stdin=stream)
    try:
        check = SO.open_file_checker(which=lambda name: shutil.which(name) if name == tool else None)
        assert check.tool == tool
        assert sleeper.pid in check.holders(tree), f"{tool} missed an open file under the tree"
        assert sleeper.pid not in check.holders(tmp_path / "held-not")
    finally:
        sleeper.kill()
        sleeper.wait()


def test_store_dedup_skips_a_name_held_open_and_refuses_without_a_check(own_repository, monkeypatch):
    """The store-based ``dedup`` takes the same open-file check: a writer holding a name would keep
    writing to the inode the name no longer points at."""
    tmp_path = own_repository
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "out" / "artifacts" / "cache" / "cas"))
    declared = dict(SC.contract(), scan_roots=[])
    monkeypatch.setattr(SC, "contract", lambda: declared)
    base = tmp_path / "out" / "artifacts" / "delivery"
    names = [base / d / "w.bin" for d in "abc"]
    for path in names:
        path.parent.mkdir(parents=True)
        path.write_bytes(b"h" * 4096)

    monkeypatch.setattr(SO, "checker_for", lambda enabled: _Holders(names[2]))
    assert SC.main(["dedup", "--min-bytes", "1024", "--apply"]) == 0
    assert names[0].stat().st_ino == names[1].stat().st_ino
    assert names[2].stat().st_ino != names[0].stat().st_ino, "a name held open was re-pointed"

    def unavailable(enabled):
        raise SO.OpenFileCheckUnavailable("no tool")

    monkeypatch.setattr(SO, "checker_for", unavailable)
    assert SC.main(["dedup", "--min-bytes", "1024", "--apply"]) == 1
