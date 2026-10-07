"""Offline source-venv recovery refuses unowned or contradictory payloads."""

from __future__ import annotations

import base64
import csv
import hashlib
import json
from pathlib import Path

import pytest
from merlin_experiments.capture_execution.runtime_rehydrate import RuntimeRecoveryError, _inventory


def _record_row(site: Path, member: str) -> list[str]:
    path = site / member
    digest = base64.urlsafe_b64encode(hashlib.sha256(path.read_bytes()).digest()).rstrip(b"=").decode()
    return [member, f"sha256={digest}", str(path.stat().st_size)]


def _distribution(site: Path, name: str, files: dict[str, bytes], *, editable: bool = False) -> None:
    dist = site / f"{name}-1.0.dist-info"
    dist.mkdir()
    (dist / "METADATA").write_text(f"Metadata-Version: 2.4\nName: {name}\nVersion: 1.0\n")
    if editable:
        (dist / "direct_url.json").write_text(json.dumps({"url": "file:///gone", "dir_info": {"editable": True}}))
    for relative, data in files.items():
        path = site / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    rows = [_record_row(site, relative) for relative in files]
    rows.append(_record_row(site, f"{dist.name}/METADATA"))
    if editable:
        rows.append(_record_row(site, f"{dist.name}/direct_url.json"))
    rows.append([f"{dist.name}/RECORD", "", ""])
    with (dist / "RECORD").open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)


def test_inventory_preserves_shared_equal_owner_and_omits_only_declared_editable(tmp_path: Path) -> None:
    site = tmp_path / "venv/lib/python3.12/site-packages"
    site.mkdir(parents=True)
    _distribution(site, "alpha", {"shared/__init__.py": b"shared", "alpha.py": b"alpha"})
    _distribution(site, "beta", {"shared/__init__.py": b"shared", "beta.py": b"beta"})
    _distribution(site, "old_editable", {"__editable__.old_editable.pth": b"/gone\n"}, editable=True)
    (site / "_virtualenv.pth").write_text("import _virtualenv\n")
    roster, owners, unowned = _inventory(tmp_path / "venv", omitted_editables={"old_editable": "source absent"})
    assert len(roster) == 3
    assert owners[Path("lib/python3.12/site-packages/shared/__init__.py")]["owners"] == ["alpha", "beta"]
    assert owners[Path("lib/python3.12/site-packages/__editable__.old_editable.pth")]["omitted"] is True
    assert [row["path"] for row in unowned] == ["_virtualenv.pth"]


def test_inventory_retains_record_owned_venv_entrypoint_outside_site_packages(tmp_path: Path) -> None:
    site = tmp_path / "venv/lib/python3.12/site-packages"
    site.mkdir(parents=True)
    _distribution(site, "alpha", {"../../../bin/alpha": b"#!/historical/python\n"})
    _, owners, _ = _inventory(tmp_path / "venv", omitted_editables={})
    assert owners[Path("bin/alpha")]["owners"] == ["alpha"]


@pytest.mark.parametrize(
    "change",
    [
        "undeclared_editable",
        "wrong_record_hash",
        "unowned_startup",
        "duplicate_record",
        "symlink_member",
        "symlink_parent_cancelled",
        "escape",
    ],
)
def test_inventory_refuses_unbound_package_files(tmp_path: Path, change: str) -> None:
    site = tmp_path / "venv/lib/python3.12/site-packages"
    site.mkdir(parents=True)
    _distribution(site, "alpha", {"alpha.py": b"original"})
    excluded = {}
    if change == "undeclared_editable":
        _distribution(site, "old_editable", {"__editable__.old_editable.pth": b"/gone\n"}, editable=True)
    elif change == "wrong_record_hash":
        (site / "alpha.py").write_bytes(b"changed")
    elif change == "duplicate_record":
        record = site / "alpha-1.0.dist-info/RECORD"
        with record.open("a", newline="") as stream:
            csv.writer(stream).writerow(_record_row(site, "alpha.py"))
    elif change == "symlink_member":
        (site / "alpha.py").rename(site / "original.py")
        (site / "alpha.py").symlink_to("original.py")
    elif change == "symlink_parent_cancelled":
        (site / "real").mkdir()
        (site / "alias").symlink_to("real", target_is_directory=True)
        record = site / "alpha-1.0.dist-info/RECORD"
        with record.open("a", newline="") as stream:
            csv.writer(stream).writerow(_record_row(site, "alias/../alpha.py"))
    elif change == "escape":
        outside = tmp_path / "escape.py"
        outside.write_bytes(b"outside")
        record = site / "alpha-1.0.dist-info/RECORD"
        with record.open("a", newline="") as stream:
            csv.writer(stream).writerow(_record_row(site, "../../../../escape.py"))
    else:
        (site / "arbitrary.pth").write_text("import outside\n")
    with pytest.raises(RuntimeRecoveryError) as error:
        _inventory(tmp_path / "venv", omitted_editables=excluded)
    if change == "symlink_parent_cancelled":
        assert "indirect" in str(error.value)


def test_inventory_refuses_conflicting_shared_owner(tmp_path: Path) -> None:
    site = tmp_path / "venv/lib/python3.12/site-packages"
    site.mkdir(parents=True)
    _distribution(site, "alpha", {"shared.py": b"first"})
    _distribution(site, "beta", {"shared.py": b"second"})
    with pytest.raises(RuntimeRecoveryError, match="RECORD"):
        _inventory(tmp_path / "venv", omitted_editables={})
