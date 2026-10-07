"""Optional external frameworks are selected explicitly, without a vendored fallback."""

from __future__ import annotations

from pathlib import Path

import pytest

from merlin.baselines import external_source
from merlin.common.paths import ExternalPathUnset


def test_unconfigured_optional_checkout_is_unavailable(monkeypatch):
    monkeypatch.setattr(
        external_source, "ext_path", lambda _name: (_ for _ in ()).throw(ExternalPathUnset("unset"))
    )
    assert external_source.optional_checkout("tvm") is None


def test_explicit_invalid_checkout_is_not_treated_as_unconfigured(monkeypatch, tmp_path):
    monkeypatch.setattr(external_source, "ext_path", lambda _name: tmp_path / "missing")
    with pytest.raises(FileNotFoundError, match="external Git checkout"):
        external_source.optional_checkout("tvm")


@pytest.mark.parametrize("git_kind", ["directory", "worktree_file"])
def test_selected_checkout_accepts_git_repository_root(monkeypatch, tmp_path, git_kind):
    source = tmp_path / "framework"
    source.mkdir()
    if git_kind == "directory":
        (source / ".git").mkdir()
    else:
        (source / ".git").write_text("gitdir: /external/worktrees/framework\n", encoding="utf-8")
    monkeypatch.setattr(external_source, "ext_path", lambda _name: source)

    assert external_source.checkout("framework") == Path(source)
