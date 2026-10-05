"""One content-addressed copy of a sealed capture's runtime, hard-linked into each private guest root.

Every sealed capture copied the selected venv, base interpreter and system libraries (about 9 GB)
into its own guest root, although those bytes are identical for every capture that selects them --
the same pattern that once cost a campaign 235 GiB of byte-identical run inputs. The runtime is now
materialized once per selection identity under the regenerable cache and hard-linked into each run.

Nothing about the seal is relaxed: the per-run guest root is still a private directory whose every
byte the issuer re-hashes against the plan before executing anything, and the replay re-hashes it
again. A store entry is published only after its own trees match the plan's selected digests, and
is built under a private name and renamed into place, so a half-built store is never linked.
File modes are kept exactly as copied, because the selected tree digests bind them.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import stat
from pathlib import Path
from typing import Any


def _identity(plan: dict[str, Any]) -> str:
    from .sealed_static import _digest, _json

    trees = plan["selected_trees"]
    return _digest(
        _json(
            {
                "venv": trees["venv"],
                "base": trees["base"],
                "base_path": plan["base"],
                "system_libs": plan["system_libs"],
            }
        )
    )


#: An explicit store location, for a process whose repository root is a frozen snapshot (whose own
#: cache would place the runtime inside the run). Must be on the run's filesystem.
STORE_ENV = "MERLIN_SEALED_RUNTIME_STORE"


def _store_root() -> Path:
    selected = os.environ.get(STORE_ENV)
    if selected:
        return Path(selected)
    from merlin.common.artifacts import cache_dir

    return Path(cache_dir("sealed-m2m-runtime"))


def _build(plan: dict[str, Any], destination: Path) -> None:
    from .sealed_m2m import SealedM2MError, _snapshot_tree
    from .sealed_static import _file_digest

    venv = Path(plan["venv"])
    venv_copy = destination / "opt/capture-venv"
    venv_copy.parent.mkdir(parents=True)
    shutil.copytree(
        venv, venv_copy, symlinks=False, ignore=lambda directory, names: {"lib64"} if Path(directory) == venv else set()
    )
    base = Path(plan["base"])
    base_copy = destination / base.relative_to("/")
    base_copy.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(base.resolve(), base_copy, symlinks=False)
    libraries = []
    for name in plan["system_libs"]:
        path = Path(name)
        target = destination / path.relative_to("/")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
        libraries.append({"path": name, "sha256": _file_digest(target)})
    if (
        _snapshot_tree(venv_copy) != plan["selected_trees"]["venv"]
        or _snapshot_tree(base_copy) != plan["selected_trees"]["base"]
    ):
        raise SealedM2MError("runtime store copy differs from the selected runtime trees")
    if any(_file_digest(Path(row["path"])) != row["sha256"] for row in libraries):
        raise SealedM2MError("runtime store library copy differs from the selected host bytes")


def store_entry(plan: dict[str, Any]) -> Path:
    """The published store directory for this plan's runtime, building it once if absent."""
    root = _store_root()
    root.mkdir(parents=True, exist_ok=True)
    key = _identity(plan)
    entry = root / key
    marker = root / f"{key}.complete.json"
    if entry.is_dir() and marker.is_file() and not entry.is_symlink():
        return entry
    staging = root / f"{key}.building-{secrets.token_hex(8)}"
    staging.mkdir()
    try:
        _build(plan, staging)
        if entry.exists():
            shutil.rmtree(entry)
        staging.rename(entry)
        marker.write_text(json.dumps({"identity": key, "base": plan["base"], "system_libs": plan["system_libs"]}))
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)
    return entry


def link_runtime(plan: dict[str, Any], runtime: Path) -> None:
    """Hard-link the plan's stored runtime into ``runtime`` with the stored directory modes."""
    entry = store_entry(plan)
    directories: list[tuple[Path, int]] = []
    for current, names, files in os.walk(entry, followlinks=False):
        here = Path(current)
        relative = here.relative_to(entry)
        target_dir = runtime / relative
        target_dir.mkdir(parents=True, exist_ok=True)
        directories.append((target_dir, stat.S_IMODE(here.stat().st_mode)))
        for name in names:
            if (here / name).is_symlink():
                raise ValueError(f"runtime store contains a directory link: {here / name}")
        for name in files:
            source = here / name
            if source.is_symlink() or not source.is_file():
                raise ValueError(f"runtime store contains a nonregular member: {source}")
            os.link(source, target_dir / name)
    # Modes last, deepest first, so a read-only directory never blocks linking its own members.
    for path, mode in sorted(directories, key=lambda row: len(row[0].parts), reverse=True):
        if path != runtime:
            path.chmod(mode)
