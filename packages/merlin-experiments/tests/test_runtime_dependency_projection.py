"""Reviewed coordinate projections inventory only actual loader dependencies."""

from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_runtime as R
from merlin_experiments.phase2.contracts import StageGateError, sha256_file


def inventory(*, roots=(), files=(), executables=()):
    return R.inventory_runtime(files=files, trees=(), executables=executables, dependency_relocations=roots)


@pytest.fixture
def layout(tmp_path, monkeypatch):
    root = tmp_path / "sdk"
    (root / "bin").mkdir(parents=True)
    (root / "lib").mkdir()
    tool = root / "bin" / "selected"
    tool.write_text("reviewed loader diagnostic fixture; not an executable compiler")
    library = root / "lib" / "library.so.2"
    library.write_text("actual fixture dependency bytes")
    alias = root / "lib" / "library.so"
    alias.symlink_to(library.name)
    unrelated = root / "lib" / "not-reported.so"
    unrelated.write_text("must never be granted from prefix membership")
    observed = root / "bin" / ".." / "lib" / alias.name
    calls = []

    def inspect(argv, **kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=0, stderr="", stdout=f"library.so => {observed} (0x0)\n")

    monkeypatch.setattr(R.subprocess, "run", inspect)
    return root, tool, library, calls


def test_observed_alias_preserves_relocated_layout_without_granting_tree(layout):
    root, tool, library, calls = layout
    grants = inventory(roots=((root, "/usr/selected-sdk"),), executables=((tool, "/usr/selected-sdk/bin/selected"),))
    assert [(row.source, row.destination) for row in grants] == [
        (tool, "/usr/selected-sdk/bin/selected"),
        (library, "/usr/selected-sdk/lib/library.so"),
    ]
    assert grants[1].sha256 == sha256_file(library)
    assert calls == [["ldd", str(tool)]]
    for row in grants:
        row.verify()


def test_outside_system_dependency_has_no_implicit_projection(layout):
    _, tool, _, _ = layout
    with pytest.raises(StageGateError, match="system-tool"):
        inventory(executables=((tool, "/usr/bin/selected"),))


@pytest.mark.parametrize(
    "change",
    [
        "mutable",
        "missing",
        "relative",
        "symlink",
        "parent_symlink",
        "escaped_destination",
        "control_destination",
        "source_overlap",
        "destination_overlap",
    ],
)
def test_invalid_coordinate_selections_refuse_before_loading_or_constructing_grants(
    layout, tmp_path, monkeypatch, change
):
    root, tool, _, calls = layout
    rows = ((root, "/usr/selected-sdk"),)
    if change == "mutable":
        rows = list(rows)
    elif change == "missing":
        rows = ((tmp_path / "absent", "/usr/selected-sdk"),)
    elif change == "relative":
        rows = ((Path("sdk"), "/usr/selected-sdk"),)
    elif change == "symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        rows = ((alias, "/usr/selected-sdk"),)
    elif change == "parent_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(root, target_is_directory=True)
        rows = ((alias / "lib", "/usr/selected-sdk"),)
    elif change == "escaped_destination":
        rows = ((root, "/usr/../candidate"),)
    elif change == "control_destination":
        rows = ((root, "/usr/selected\npath"),)
    elif change == "source_overlap":
        rows = (*rows, (root / "lib", "/usr/another"))
    else:
        other = tmp_path / "other"
        other.mkdir()
        rows = (*rows, (other, "/usr/selected-sdk/lib"))
    monkeypatch.setattr(
        R,
        "RuntimeGrant",
        SimpleNamespace(
            verify_destination=R.RuntimeGrant.verify_destination,
        ),
    )
    with pytest.raises(StageGateError):
        inventory(roots=rows, executables=((tool, "/usr/bin/selected"),))
    assert not calls


def test_reported_alias_cannot_escape_reviewed_source_prefix(layout, tmp_path):
    root, tool, library, _ = layout
    outside = tmp_path / "outside.so"
    outside.write_text(library.read_text())
    alias = root / "lib" / "library.so"
    alias.unlink()
    alias.symlink_to(outside)
    with pytest.raises(StageGateError, match="escaped"):
        inventory(roots=((root, "/usr/selected-sdk"),), executables=((tool, "/usr/bin/selected"),))


def test_reported_dependency_directory_cannot_escape_reviewed_source_prefix(layout, tmp_path):
    root, tool, library, _ = layout
    outside = tmp_path / "outside-libraries"
    library.parent.rename(outside)
    (root / "lib").symlink_to(outside, target_is_directory=True)
    with pytest.raises(StageGateError, match="escaped"):
        inventory(roots=((root, "/usr/selected-sdk"),), executables=((tool, "/usr/bin/selected"),))


def test_conflicting_mapping_refuses_before_any_grant_construction(layout, tmp_path, monkeypatch):
    root, tool, _, _ = layout
    other = tmp_path / "other.so"
    other.write_text("different bytes at the proposed same destination")
    # Preserve only the validator so a data grant construction would fail the test.
    monkeypatch.setattr(R, "RuntimeGrant", SimpleNamespace(verify_destination=R.RuntimeGrant.verify_destination))
    with pytest.raises(StageGateError, match="conflicting"):
        inventory(
            roots=((root, "/usr/selected-sdk"),),
            files=((other, "/usr/selected-sdk/lib/library.so"),),
            executables=((tool, "/usr/bin/selected"),),
        )


def test_file_directory_collision_refuses_including_lexical_intervening_sibling(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.write_text("file bytes")
    monkeypatch.setattr(R, "RuntimeGrant", SimpleNamespace(verify_destination=R.RuntimeGrant.verify_destination))
    with pytest.raises(StageGateError, match="file and directory"):
        inventory(files=((source, "/usr/a"), (source, "/usr/a-other"), (source, "/usr/a/lib/library.so")))


def test_cli_projection_is_explicit_and_keeps_all_membership_hashes(layout, tmp_path):
    root, tool, library, _ = layout
    output = tmp_path / "inventory.json"
    assert (
        R.main(
            [
                "--executable",
                f"{tool}=/usr/selected-sdk/bin/selected",
                "--dependency-relocation",
                f"{root}=/usr/selected-sdk",
                "--output",
                str(output),
            ]
        )
        == 0
    )
    import json

    rows = json.loads(output.read_text())
    assert rows[1] == {
        "source": str(library),
        "destination": "/usr/selected-sdk/lib/library.so",
        "sha256": sha256_file(library),
    }
