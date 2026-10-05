"""The doc-paths gate reads this checkout's docs only, and survives a file vanishing mid-scan.

It crashed with FileNotFoundError when an agent worktree under ``.claude/worktrees`` was deleted
while the Stop hook walked the tree; a nested worktree's docs were never the repository's to judge.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from merlin.common.paths import repo_root

STALE = "Write results to generated_targets/foo here.\n"


def _gate(root: Path):
    script = repo_root() / "build_tools/scripts/check_doc_paths.py"
    spec = importlib.util.spec_from_file_location("doc_paths_under_test", script)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.ROOT, module.ALLOW_FILE = root, root / "allow.txt"
    return module


def _write(root: Path, rel: str, text: str = STALE) -> Path:
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def test_nested_worktrees_and_checkouts_are_not_the_repos_docs(tmp_path):
    _write(tmp_path, "docs/guides/real.md")
    _write(tmp_path, ".claude/worktrees/agent-1/docs/guides/copy.md")
    _write(tmp_path, ".claude/worktrees/agent-1/AGENT.md")
    _write(tmp_path, "vendor/clone/.git", "gitdir: elsewhere\n")  # a checkout marks itself with .git
    _write(tmp_path, "vendor/clone/AGENT.md")
    gate = _gate(tmp_path)
    files = {p.relative_to(tmp_path).as_posix() for p in gate._doc_files()}
    assert files == {"docs/guides/real.md"}
    assert gate.scan() == ["docs/guides/real.md:1: retired generated_targets/ -> out/artifacts/targets/"]


def test_a_doc_deleted_between_the_walk_and_the_read_is_skipped(tmp_path, monkeypatch):
    kept = _write(tmp_path, "docs/guides/kept.md")
    gone = _write(tmp_path, "docs/guides/gone.md")
    gate = _gate(tmp_path)
    walked = gate._doc_files()
    gone.unlink()
    monkeypatch.setattr(gate, "_doc_files", lambda: walked)
    assert gate.scan() == [
        f"{kept.relative_to(tmp_path).as_posix()}:1: retired generated_targets/ -> out/artifacts/targets/"
    ]
