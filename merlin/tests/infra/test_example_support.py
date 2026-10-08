"""Vendored target support: identical to its recorded source, selected by default, and never agent-visible.

Each target's Merlin support provider is tracked at ``examples/<example>/support``, with a ``SOURCE.yaml``
beside it naming the companion commit the bytes were copied from. Everything here is derived from those
records and from the providers' own declarations, so no target is named and a newly vendored provider
is covered the day its record lands.

Three properties are held:

* the vendored bytes are the recorded source (its git tree id, recomputed from the files on disk);
* with ``MERLIN_TARGET_PATH`` unset, each target's executable support is its vendored provider, and any
  explicit value -- the empty string included -- replaces that default;
* the vendored trees stay experimenter-side: a sandbox that exposes the whole checkout, a bundle snapshot
  or clean room built from a grant over all of ``examples/``, the transcript audit and publication all
  withhold or refuse them.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.targetgen import plugins, target_registry
from merlin.targetgen.providers import ProviderError, read_provider

RECORD = "SOURCE.yaml"


def _records() -> list[tuple[Path, dict]]:
    out = []
    for path in sorted((repo_root() / "examples").glob(f"*/{RECORD}")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        out.append((path, doc))
    return out


RECORDS = _records()
IDS = [path.parent.name for path, _ in RECORDS]


def _support_root(record: Path, doc: dict) -> Path:
    return (record.parent / doc["path"]).resolve()


def _tracked(root: Path) -> list[str]:
    """Tracked members of ``root``, relative to it (the index, so untracked caches never count)."""
    repo = repo_root()
    out = subprocess.run(
        ["git", "ls-files", "-z", "--", str(root.relative_to(repo.resolve()))],
        cwd=repo,
        capture_output=True,
        check=True,
    ).stdout
    prefix = root.relative_to(repo.resolve()).as_posix() + "/"
    return sorted(item.decode()[len(prefix) :] for item in out.split(b"\0") if item)


def _git_tree_id(root: Path, members: list[str]) -> str:
    """The git tree id ``members`` would have, computed from the bytes and modes on disk."""
    tree: dict = {}
    for rel in members:
        node = tree
        *parents, leaf = rel.split("/")
        for part in parents:
            node = node.setdefault(part, {})
        node[leaf] = root / rel

    def digest(node: dict) -> bytes:
        entries = []
        for name, value in node.items():
            if isinstance(value, dict):
                entries.append((name + "/", b"40000", name, digest(value)))
                continue
            if value.is_symlink():
                mode, data = b"120000", os.readlink(value).encode()
            else:
                mode = b"100755" if value.stat().st_mode & 0o100 else b"100644"
                data = value.read_bytes()
            entries.append((name, mode, name, hashlib.sha1(b"blob %d\0" % len(data) + data).digest()))
        entries.sort(key=lambda entry: entry[0].encode())
        body = b"".join(mode + b" " + name.encode() + b"\0" + sha for _, mode, name, sha in entries)
        return hashlib.sha1(b"tree %d\0" % len(body) + body).digest()

    return digest(tree).hex()


def test_every_vendored_support_has_a_source_record():
    """No provider under ``examples/*/support`` without a record, and no record without its provider."""
    vendored = {root for root in target_registry.in_repo_support().values()}
    recorded = {_support_root(path, doc) for path, doc in RECORDS}
    assert vendored, "no vendored support provider was found; the default selection would be empty"
    assert vendored == recorded


@pytest.mark.parametrize(("record", "doc"), RECORDS, ids=IDS)
def test_vendored_support_is_its_recorded_source(record, doc):
    root = _support_root(record, doc)
    members = _tracked(root)
    assert len(members) == doc["file_count"]
    assert _git_tree_id(root, members) == doc["source"]["tree"], "vendored bytes differ from the recorded tree"
    provider = read_provider(root)
    assert (provider.id, provider.target) == (doc["provider_id"], doc["target"])
    records = doc["records"]
    for relative in (records["file_provenance"], *records["migrations"]):
        assert (record.parent / relative).is_file(), relative
    if doc["pinned"]["identical_to_source"]:
        assert doc["pinned"]["provider_tree"] == doc["source"]["tree"]


def test_the_migration_manifest_agrees_with_every_source_record():
    """``target_support.json`` lists each vendored copy with the same commit, tree and file count."""
    import json

    manifest = json.loads((repo_root() / "build_tools/upstreams/target_support.json").read_text(encoding="utf-8"))
    listed = {entry["vendored"]["source_record"]: entry for entry in manifest["companions"] if "vendored" in entry}
    assert set(listed) == {path.relative_to(repo_root()).as_posix() for path, _ in RECORDS}
    for path, doc in RECORDS:
        entry = listed[path.relative_to(repo_root()).as_posix()]
        vendored = entry["vendored"]
        assert entry["target"] == doc["target"]
        assert (vendored["commit"], vendored["tree"], vendored["file_count"]) == (
            doc["source"]["commit"],
            doc["source"]["tree"],
            doc["file_count"],
        )
        assert vendored.get("merge_parents") == doc["source"].get("merge_parents")
        assert entry["companion_commit"] == doc["pinned"]["companion_commit"]


@pytest.mark.parametrize(("record", "doc"), RECORDS, ids=IDS)
def test_unset_selection_is_the_targets_vendored_support(record, doc, monkeypatch):
    monkeypatch.delenv("MERLIN_TARGET_PATH", raising=False)
    root, target = _support_root(record, doc), doc["target"]
    assert target_registry.default_support_root(target) == root
    assert target_registry.explicit_targets()[target] == root
    resolved = target_registry.resolve(target)
    assert (resolved.kind, resolved.base) == ("external", root)
    assert plugins.resolve_support(target).base == root


def test_an_explicit_selection_replaces_the_default(tmp_path, monkeypatch):
    target = RECORDS[0][1]["target"]
    monkeypatch.setenv("MERLIN_TARGET_PATH", "")
    assert target_registry.explicit_targets() == {}
    assert target_registry.effective_target_path() == ""
    with pytest.raises(plugins.PluginError, match="explicit MERLIN_TARGET_PATH"):
        plugins.resolve_support(target)

    elsewhere = tmp_path / "pinned-support"
    (elsewhere / "contracts").mkdir(parents=True)
    (elsewhere / "contracts/target_contract.yaml").write_text(f"name: {target}\n", encoding="utf-8")
    monkeypatch.setenv("MERLIN_TARGET_PATH", str(elsewhere))
    assert target_registry.explicit_targets() == {target: elsewhere.resolve()}
    assert target_registry.resolve(target).base == elsewhere.resolve()

    monkeypatch.delenv("MERLIN_TARGET_PATH")
    spelled = target_registry.effective_target_path().split(os.pathsep)
    assert spelled == [str(root) for root in target_registry.in_repo_support().values()]


def test_in_repo_support_is_keyed_by_declaration_not_directory(tmp_path, monkeypatch):
    def provider(example: str, target: str) -> Path:
        root = tmp_path / "examples" / example / target_registry.IN_REPO_SUPPORT_DIR
        (root / "contracts").mkdir(parents=True)
        (root / "contracts/target_contract.yaml").write_text(f"name: {target}\n", encoding="utf-8")
        (root / "provider.yaml").write_text(
            f"schema: merlin.provider.v1\nid: {example}-support\ntarget: {target}\nrole: support\n",
            encoding="utf-8",
        )
        return root

    monkeypatch.setattr(target_registry, "checkout_root", lambda: tmp_path)
    declared = provider("folder_name", "declared_device")
    (tmp_path / "examples/metadata_only/support").mkdir(parents=True)  # not a provider: ignored
    assert target_registry.in_repo_support() == {"declared_device": declared.resolve()}

    provider("second_folder", "declared_device")
    with pytest.raises(target_registry.TargetCollisionError, match="declared_device"):
        target_registry.in_repo_support()

    (tmp_path / "examples/second_folder/support/provider.yaml").write_text("schema: wrong\n", encoding="utf-8")
    with pytest.raises(ProviderError):
        target_registry.in_repo_support()

    monkeypatch.setattr(target_registry, "checkout_root", lambda: None)  # an installed distribution
    assert target_registry.in_repo_support() == {}


# ------------------------------------------------------------------------------------- the boundary
DESCRIPTORS = sorted((repo_root() / "examples").glob("*/target/descriptor.yaml"))


def _private_members(root: Path) -> list[Path]:
    return [root / rel for rel in _tracked(root)]


@pytest.fixture(scope="module")
def sandbox():
    pytest.importorskip("merlin_experiments")
    import importlib

    # The package re-exports a function named ``answer_surfaces``; the module is the one wanted.
    modules = {
        name: importlib.import_module(f"merlin.targetgen.sandbox.{name}")
        for name in ("answer_surfaces", "bwrap", "cleanroom")
    }
    return SimpleNamespace(surfaces=modules["answer_surfaces"], bwrap=modules["bwrap"], cleanroom=modules["cleanroom"])


@pytest.mark.parametrize("descriptor", DESCRIPTORS, ids=[d.parent.parent.name for d in DESCRIPTORS])
def test_a_sandbox_exposing_the_whole_checkout_shows_no_vendored_support(descriptor, sandbox):
    """Whichever target an arm works on, every vendored support tree is masked, contracts included.

    A real bundle grants specific paths, none under ``examples/*/support``; exposing the entire checkout
    is the worst case those grants could add up to.
    """
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(descriptor)
    derived = sandbox.surfaces.answer_surfaces(te)
    repo = str(repo_root())
    exposed = ["--ro-bind", repo, repo]
    roots = list(target_registry.in_repo_support().values())
    assert roots and all(any(item.path == root for item in derived) for root in roots)
    assert all(sandbox.bwrap.is_exposed(exposed, _private_members(root)[0]) for root in roots)
    masked = sandbox.bwrap.apply_answer_masks(exposed, derived)
    assert sandbox.bwrap.coverage_gap(masked, derived) == []
    leaked = [member for root in roots for member in _private_members(root) if sandbox.bwrap.is_exposed(masked, member)]
    assert leaked == []


def test_a_bundle_snapshot_of_all_examples_withholds_vendored_support(sandbox, tmp_path, monkeypatch):
    """The agent-visible snapshot: a grant over ``examples/`` freezes the support bytes as private views."""
    from merlin.targetgen.target_experiment import load_target_experiment

    monkeypatch.setenv("MERLIN_BUNDLE_CAS", str(tmp_path / "cas"))
    te = load_target_experiment(DESCRIPTORS[0])
    repo = repo_root().resolve()
    ws = tmp_path / "run/workspace"
    ws.mkdir(parents=True)
    bundle = {"allowed": [{"path": "examples/"}]}
    manifest = sandbox.bwrap.materialize_bundle_inputs(ws, bundle, repo=repo)
    frozen = sandbox.bwrap.snapshot_input_paths(ws, bundle, [repo / "examples"], repo=repo)[0]
    views = sandbox.bwrap.snapshot_support_surfaces(ws, manifest)
    view = tmp_path / "agent-view"
    argv = sandbox.bwrap.apply_final_answer_masks(["--ro-bind", str(frozen), str(view)], te, ws, bundle, repo=repo)
    for root in target_registry.in_repo_support().values():
        for member in _private_members(root):
            relative = member.relative_to(repo / "examples")
            assert not sandbox.bwrap.snapshot_public_member(frozen / relative, frozen, views), relative
            assert not sandbox.bwrap.is_exposed(argv, view / relative), relative


def test_a_clean_room_from_all_examples_places_no_vendored_support(sandbox, tmp_path):
    """Placement, judged by path: a grant over every example copies none of the vendored support.

    The verifier's content sieve is off here on purpose. Some public bytes are byte-identical to files
    inside a support tree (a curated harness header the support also carries), and the sieve refuses a
    room for that whichever tree the bytes were placed from. That stricter rule is the clean room's
    own policy and is unchanged; this test isolates the question of what the builder places.
    """
    from merlin.targetgen.target_experiment import load_target_experiment

    te = load_target_experiment(DESCRIPTORS[0])
    room = sandbox.cleanroom.build_clean_room(
        te, tmp_path / "home", {"allowed": [{"path": "examples/"}]}, check_content=False
    )
    placed = room.inputs / "repo/examples"
    assert placed.is_dir(), "the grant placed nothing, so this proves nothing"
    examples = repo_root().resolve() / "examples"
    vendored = [placed / root.relative_to(examples) for root in target_registry.in_repo_support().values()]
    # A surface that declares a grantable sub-tree is walked into, so its directory may exist, empty.
    leaked = [path for root in vendored if root.exists() for path in root.rglob("*") if not path.is_dir()]
    assert leaked == []


def test_the_transcript_audit_names_every_vendored_support_entry(sandbox):
    from merlin.targetgen.target_experiment import load_target_experiment

    tokens = sandbox.surfaces.audit_tokens(load_target_experiment(DESCRIPTORS[0]))["answer"]
    for root in target_registry.in_repo_support().values():
        for child in root.iterdir():
            if child.name != sandbox.surfaces.PACKAGE_CONTRACT_SUBDIR:
                assert any(token in str(child / "x") for token in tokens), child


@pytest.mark.parametrize(("record", "doc"), RECORDS, ids=IDS)
def test_vendored_support_cannot_be_published_as_a_candidate(record, doc):
    from merlin.targetgen import publish

    with pytest.raises(publish.PublishError, match="not support"):
        publish._export_role(SimpleNamespace(package_dir=_support_root(record, doc), target=doc["target"]))


# --------------------------------------------------------------------------------- vendored suites
SUITES = [(path, doc) for path, doc in RECORDS if (_support_root(path, doc) / "tests").is_dir()]


@pytest.mark.integration
@pytest.mark.parametrize(("record", "doc"), SUITES, ids=[path.parent.name for path, _ in SUITES])
def test_vendored_support_suite_passes_from_its_new_home(record, doc, tmp_path):
    """Each provider's own tests, run from its in-repo home the way its README documents.

    The README selects the provider alone on ``MERLIN_TARGET_PATH``; so does this, with the vendored
    root, so the suite sees exactly the plugins it saw at its companion. Integration: a suite may read
    operator tool paths from ``.env`` (Gemmini's conformance recording needs ``MERLIN_EXT_CHIPYARD``).
    Each suite runs in its own interpreter because the suites import their provider's modules by bare
    name and share test-module basenames.
    """
    root = _support_root(record, doc)
    env = dict(os.environ, MERLIN_TARGET_PATH=str(root), PYTHONDONTWRITEBYTECODE="1")
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), os.environ.get("PYTHONPATH"))))
    # A test bound to the companion repository's own layout (a sibling directory the vendored tree
    # does not reproduce) is recorded, with its reason, in SOURCE.yaml rather than edited in place.
    repo = repo_root().resolve()
    deselected = [
        f"--deselect={(root / entry['test']).relative_to(repo).as_posix()}"
        for entry in (doc.get("tests") or {}).get("deselect", [])
    ]
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-p",
            "no:cacheprovider",
            f"--rootdir={repo}",
            f"--basetemp={tmp_path / 'basetemp'}",
            *deselected,
            str(root / "tests"),
        ],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout[-4000:] + result.stderr[-2000:]
