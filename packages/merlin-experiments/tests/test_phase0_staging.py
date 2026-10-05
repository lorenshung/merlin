"""A corpus member is written in a stage and moved into place only when the writer succeeds."""

from __future__ import annotations

from pathlib import Path

import pytest
from merlin_experiments.phase0.staging import STAGING_PREFIX, write_staged


def _write(root: Path, name: str, *, golden: str = "outputs: {}\n", fail: bool = False) -> Path:
    member = root / "_perf" / name
    member.mkdir(parents=True)
    (member / "capsule.yaml").write_text(f"name: {name}\n")
    (member / "golden.yaml").write_text(golden)
    if fail:
        raise ValueError("unfalsifiable golden")
    return member


def _no_stage_left(root: Path) -> bool:
    return not [p for p in root.iterdir() if p.name.startswith(STAGING_PREFIX)]


def test_a_successful_member_lands_at_its_corpus_path(tmp_path):
    written = write_staged(lambda root: _write(root, "PK00"), tmp_path, member="_perf/PK00")
    assert written == tmp_path / "_perf" / "PK00"
    assert (written / "golden.yaml").is_file() and not written.is_symlink()
    assert _no_stage_left(tmp_path)


def test_a_failed_member_leaves_nothing_on_disk(tmp_path):
    write_staged(lambda root: _write(root, "PK00"), tmp_path, member="_perf/PK00")
    with pytest.raises(ValueError, match="unfalsifiable"):
        write_staged(lambda root: _write(root, "PW04", fail=True), tmp_path, member="_perf/PW04")
    assert not (tmp_path / "_perf" / "PW04").exists()
    assert sorted(p.parent.name for p in tmp_path.rglob("capsule.yaml")) == ["PK00"]
    assert _no_stage_left(tmp_path)


def test_existing_members_are_reused_not_rewritten(tmp_path):
    shared = _write(tmp_path, "group_a")
    (shared / "capsule.yaml").write_text("name: group_a\nsoftware_screen: admitted\n")  # post-processed

    def model(root: Path) -> Path:
        group = root / "_perf" / "group_a"
        if not (group / "capsule.yaml").is_file():  # a qualifier writes a group only when it is absent
            _write(root, "group_a")
        return _write(root, "model")

    assert write_staged(model, tmp_path, member="_perf/model") == tmp_path / "_perf" / "model"
    assert (shared / "capsule.yaml").read_text().endswith("software_screen: admitted\n")
    assert not shared.is_symlink()


def test_a_rerun_replaces_its_own_member_only_on_success(tmp_path):
    write_staged(lambda root: _write(root, "PK00", golden="outputs: {Y: [1]}\n"), tmp_path, member="_perf/PK00")
    with pytest.raises(ValueError):
        write_staged(lambda root: _write(root, "PK00", golden="new\n", fail=True), tmp_path, member="_perf/PK00")
    assert (tmp_path / "_perf" / "PK00" / "golden.yaml").read_text() == "outputs: {Y: [1]}\n"
    write_staged(lambda root: _write(root, "PK00", golden="outputs: {Y: [2]}\n"), tmp_path, member="_perf/PK00")
    assert (tmp_path / "_perf" / "PK00" / "golden.yaml").read_text() == "outputs: {Y: [2]}\n"
    assert _no_stage_left(tmp_path)
