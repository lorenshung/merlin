"""Phase runs, sealed-release and champion homes, and their declared retention."""

from __future__ import annotations

import pytest

from merlin.common import storage_cli as SC
from merlin.common import storage_lifecycle
from merlin.common.artifacts import declared_home, phase_run_id, start_phase_run
from merlin.common.paths import phase_runs_root, repo_root


@pytest.fixture
def rooted(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    for root in SC.out_roots():
        (tmp_path / "out" / root).mkdir(parents=True)
    return tmp_path / "out"


def test_the_suite_is_the_phase(rooted):
    handle = start_phase_run(target="fixture", phase=1, method="merlin_assisted", make_subdirs=())
    assert handle.run_dir.parent == phase_runs_root("fixture", 1) == rooted / "runs" / "fixture" / "phase1"
    stamp, method, sha = handle.run_id.split("_", 1)[0], "merlin_assisted", handle.git_sha
    assert handle.run_id == f"{stamp}_{method}_{sha}"
    assert SC.unit_timestamp(handle.run_id) == stamp
    with pytest.raises(ValueError):
        start_phase_run(target="fixture", phase=3, method="x")
    with pytest.raises(ValueError):
        start_phase_run(target="fixture", phase=1, method="x", run_id="chosen")
    with pytest.raises(ValueError):
        phase_run_id("a/b")


def test_homes_come_from_the_contract(rooted):
    assert declared_home("target-champions") == rooted / "artifacts" / "targets"
    assert declared_home("phase0-releases") == rooted / "artifacts" / "protocols"
    with pytest.raises(ValueError, match="declares no product root"):
        declared_home("undeclared-home")


def test_declared_retention_protects_champions_and_releases(rooted):
    champion = rooted / "artifacts" / "targets" / "fixture" / "champions" / "pkg_v1"
    release = rooted / "artifacts" / "protocols" / "fixture" / "phase0-20260929T000000Z-abc1234"
    loose = rooted / "artifacts" / "targets" / "fixture" / "scratch_pkg"
    for path in (champion, release, loose):
        path.mkdir(parents=True)
        (path / "manifest.yaml").write_text("x: 1\n")
        with storage_lifecycle.lease(path, owner="test"):
            pass
    assert SC.retention_reasons(champion)
    assert SC.retention_reasons(release)
    assert SC.retention_reasons(champion / ".merlin")  # inside a pinned unit
    assert SC.retention_reasons(rooted / "artifacts" / "targets" / "fixture" / "champions")  # holds one
    assert SC.retention_reasons(loose) == []
    assert SC.retention_reasons(rooted / "artifacts" / "protocols" / "other") == []  # holds none
    plan = SC.retention_plan(keep=0)
    dropped = {d["path"] for row in plan["groups"].values() for d in row["drops"]}
    assert str(champion) not in dropped and str(release) not in dropped


def test_retention_rules_are_stated_in_the_convention_document():
    """The public storage guide states the contract's retention rules."""
    text = (repo_root() / "docs/guides/storage.md").read_text(encoding="utf-8")
    rows = SC.declared_retention()
    assert rows, "the storage contract declares no retention rules"
    for row in rows:
        assert row["pattern"].split("/")[0] in SC.out_roots()
        assert row["pattern"].split("/")[1] in SC.declared_concerns()
        assert isinstance(row.get("reason"), str) and len(row["reason"].split()) >= 3
        assert row["doc"] in text, f"retention rule {row['pattern']!r} is absent from the storage guide"
    assert "out/runs/<target>/phase<N>/<TS>_<method>_<sha7>/" in text
