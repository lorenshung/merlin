"""A test of target-owned executable support SKIPS when no provider is selected on MERLIN_TARGET_PATH,
and only then: absence is not a passing qualification, and a selected provider is used as selected."""

from __future__ import annotations

from pathlib import Path

import pytest
import selected_driver

from merlin.targetgen import target_registry


def test_no_selected_provider_skips_with_the_selection_it_needs(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(tmp_path / "nothing-selected"))
    with pytest.raises(pytest.skip.Exception, match="MERLIN_TARGET_PATH"):
        selected_driver.require_support("some_target")


def test_a_selected_provider_is_returned_not_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(target_registry, "explicit_targets", lambda: {"some_target": tmp_path})
    assert selected_driver.require_support("some_target") == Path(tmp_path)
