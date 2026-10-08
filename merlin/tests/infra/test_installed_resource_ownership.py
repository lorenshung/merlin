"""Read-only compiler resources follow implementation identity, not output roots."""

from __future__ import annotations

import importlib.resources

import pytest

from merlin.common import paths

KINDS = (
    ("runtime", paths.runtime_dir, "MERLIN_RUNTIME_DIR"),
    ("schemas", paths.schemas_dir, "MERLIN_SCHEMAS_DIR"),
    ("contract", paths.contract_dir, "MERLIN_CONTRACT_DIR"),
    ("prompts", paths.prompts_dir, "MERLIN_PROMPTS_DIR"),
    ("benchmarks", paths.bench_dir, "MERLIN_BENCH_DIR"),
    ("targets", paths.targets_dir, "MERLIN_TARGETS_DIR"),
)


@pytest.fixture
def installed_owner(tmp_path, monkeypatch):
    for variable in ("MERLIN_REPO_ROOT", "MERLIN_WORK_DIR", *(row[2] for row in KINDS)):
        monkeypatch.delenv(variable, raising=False)
    monkeypatch.setattr(paths, "checkout_root", lambda: None)
    package = tmp_path / "installed/merlin"
    package.mkdir(parents=True)
    original = importlib.resources.files
    monkeypatch.setattr(
        importlib.resources,
        "files",
        lambda name: package if name == "merlin" else original(name),
    )
    return package


@pytest.mark.parametrize("kind,getter,override", KINDS)
@pytest.mark.parametrize("explicit_work", [False, True])
def test_incidental_cwd_or_work_tree_cannot_shadow_installed_resources(
    tmp_path, monkeypatch, installed_owner, kind, getter, override, explicit_work
):
    caller = tmp_path / "caller"
    caller.mkdir()
    monkeypatch.chdir(caller)
    work = tmp_path / "work" if explicit_work else caller
    if explicit_work:
        monkeypatch.setenv("MERLIN_WORK_DIR", str(work))
    stale = work / "merlin" / kind
    stale.mkdir(parents=True)
    (stale / "identity").write_text("ambient-old-resource")
    # Project-like markers in a caller still do not make it the implementation owner.
    (work / "pyproject.toml").write_text("[project]\nname='caller'\n")
    (work / "build_tools").mkdir()
    (work / "build_tools/package_resources.json").write_text("{}")
    expected = installed_owner / "_data" / kind
    if kind != "targets":
        expected.mkdir(parents=True)
        (expected / "identity").write_text("installed-current-resource")
    assert paths.repo_root() == work
    assert getter() == expected
    assert paths.data_path(kind, "identity") == expected / "identity"
    if kind == "targets":
        assert not expected.exists()  # Core does not invent optional target data.
    else:
        assert (getter() / "identity").read_text() == "installed-current-resource"


@pytest.mark.parametrize("kind,getter,override", KINDS)
def test_explicit_selected_repository_still_owns_available_data(
    tmp_path, monkeypatch, installed_owner, kind, getter, override
):
    selected = tmp_path / "selected"
    expected = selected / "merlin" / kind
    expected.mkdir(parents=True)
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(selected))
    assert getter() == expected


@pytest.mark.parametrize("kind,getter,override", KINDS)
def test_resource_specific_override_remains_authoritative_when_missing(
    tmp_path, monkeypatch, installed_owner, kind, getter, override
):
    missing = tmp_path / "explicit-missing" / kind
    monkeypatch.setenv(override, str(missing))
    assert getter() == missing
    assert not getter().exists()


def test_actual_source_owner_retains_live_resource_precedence(tmp_path, monkeypatch, installed_owner):
    checkout = tmp_path / "implementation"
    expected = checkout / "merlin/runtime"
    expected.mkdir(parents=True)
    monkeypatch.setattr(paths, "checkout_root", lambda: checkout)
    assert paths.runtime_dir() == expected


def test_missing_selected_tree_retains_bundled_fallback(tmp_path, monkeypatch, installed_owner):
    monkeypatch.setenv("MERLIN_REPO_ROOT", str(tmp_path / "absent-selected"))
    assert paths.runtime_dir() == installed_owner / "_data/runtime"


@pytest.mark.parametrize("error", [ModuleNotFoundError, FileNotFoundError, NotADirectoryError])
def test_resource_loader_failure_cannot_return_ambient_data(tmp_path, monkeypatch, installed_owner, error):
    stale = tmp_path / "merlin/runtime"
    stale.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)

    def unavailable(name):
        raise error("bundled resource unavailable")

    monkeypatch.setattr(importlib.resources, "files", unavailable)
    expected = paths.python_source_dir() / "merlin/_data/runtime"
    assert paths.runtime_dir() == expected
    assert expected != stale
