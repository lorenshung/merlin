"""A run's frozen source snapshot imports ITS OWN merlin, not the checkout it happens to sit inside.

Measured: a run's snapshot lives under the live repo's out/ root and is no git checkout of its own, so
the harness's `git rev-parse --show-toplevel` answered with the LIVE checkout and put that merlin first
on the path. When another session moved that checkout's API, every capsule screen of the run failed to
import, fell back to a stale capsule set and refused every candidate ("unknown capsule(s)") for a
whole night, with none of it naming the cause.

The stub trees mirror the layout: core under ``src/merlin`` (with ``merlin/python/merlin`` as the
compatibility symlink) and ``merlin.benchharness`` / ``merlin_experiments`` under
``packages/merlin-experiments/src``, so the probe fails if either root comes from the wrong tree.
"""

from __future__ import annotations

import shutil
import subprocess
import sys

from merlin.common.paths import merlin_dir

_HARNESS = "merlin/experiments/capsule_bench/harness"
_EXPORTS = ("hash_tree", "repo_sha", "reports_root", "runs_root", "sh")
_CONTEXT = """\
from pathlib import Path
from types import SimpleNamespace


def load_context(descriptor, *, repo, harness, _legacy_target_reader=None):
    return SimpleNamespace(
        experiment=Path(descriptor).parent, sourced_environment=[], target="stub",
        runs=Path(repo) / "runs", reports=Path(repo) / "reports", bundles=Path(repo) / "bundles",
    )
"""


def _stub_tree(root, marker: str) -> None:
    core = root / "src" / "merlin"
    core.mkdir(parents=True)
    (core / "__init__.py").write_text("from pkgutil import extend_path\n__path__ = extend_path(__path__, __name__)\n")
    (root / "merlin" / "python").mkdir(parents=True)
    (root / "merlin" / "python" / "merlin").symlink_to("../../src/merlin", target_is_directory=True)
    extension = root / "packages" / "merlin-experiments" / "src"
    bench = extension / "merlin" / "benchharness"
    bench.mkdir(parents=True)
    body = "".join(f"def {name}(*a, **k):\n    return None\n" for name in _EXPORTS)
    (bench / "__init__.py").write_text(f"MARKER = {marker!r}\n{body}")
    phase1 = extension / "merlin_experiments" / "phase1"
    phase1.mkdir(parents=True)
    (phase1.parent / "__init__.py").write_text(f"MARKER = {marker!r}\n")
    (phase1 / "__init__.py").write_text("")
    (phase1 / "context.py").write_text(_CONTEXT)


def _probe(harness, tmp_path, **env) -> subprocess.CompletedProcess:
    probe = (
        "import _common, merlin.benchharness as b, merlin_experiments as e; "
        "print(_common.REPO); print(b.MARKER); print(e.MARKER)"
    )
    return subprocess.run(
        [sys.executable, "-S", "-c", f"import sys; sys.path.insert(0, {str(harness)!r}); {probe}"],
        cwd=str(harness),
        env={"PATH": "/usr/bin:/bin", "HOME": str(tmp_path), **env},
        capture_output=True,
        text=True,
        timeout=60,
    )


def _snapshot_inside_live_checkout(tmp_path):
    live = tmp_path / "live"
    _stub_tree(live, "live")
    subprocess.run(["git", "init", "-q", str(live)], check=True)
    snapshot = live / "out" / "runs" / "run" / "stage.source"
    _stub_tree(snapshot, "snapshot")
    harness = snapshot / _HARNESS
    harness.mkdir(parents=True)
    shutil.copy(merlin_dir() / "experiments/capsule_bench/harness/_common.py", harness / "_common.py")
    return live, snapshot, harness


def test_a_snapshot_inside_another_checkout_imports_its_own_merlin(tmp_path) -> None:
    live, snapshot, harness = _snapshot_inside_live_checkout(tmp_path)
    # The shared editable install names the live checkout's roots; the snapshot must still win.
    installed = ":".join(str(live / root) for root in ("src", "packages/merlin-experiments/src"))
    done = _probe(harness, tmp_path, PYTHONPATH=installed)
    assert done.returncode == 0, done.stderr
    repo, core_marker, experiments_marker = done.stdout.split()
    assert repo == str(snapshot.resolve())
    assert (core_marker, experiments_marker) == ("snapshot", "snapshot")


def test_an_explicit_repo_root_still_wins_over_the_containing_tree(tmp_path) -> None:
    live, _, harness = _snapshot_inside_live_checkout(tmp_path)
    done = _probe(harness, tmp_path, MERLIN_REPO_ROOT=str(live))
    assert done.returncode == 0, done.stderr
    repo, core_marker, experiments_marker = done.stdout.split()
    assert repo == str(live.resolve())
    assert (core_marker, experiments_marker) == ("live", "live")
