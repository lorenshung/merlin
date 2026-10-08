"""Independent file/graph contracts and fail-closed package protocol boundaries."""

from __future__ import annotations

import copy
import json
from dataclasses import replace

import pytest
import yaml

from merlin.common.digest import sha256_file
from merlin.common.paths import contract_dir
from merlin.targetgen import artifact_bundle as bundles
from merlin.targetgen.artifact_bundle import BundleError, SelectedArtifactProfile, verify_bundle
from merlin.targetgen.contract import schemas
from merlin.targetgen.package_runtime import (
    CertFailure,
    _resolve_argv,
    analysis_emission_entrypoints,
    load_package,
    run_entrypoint,
)


def _profile(root):
    path = root / "profile.yaml"
    path.write_text("name: independent-profile\n")
    return SelectedArtifactProfile.capture(path)


def _bundle(root):
    profile = _profile(root)
    image = root / "commands.bin"
    image.write_bytes(b"independent compiler artifact")
    document = {
        "schema": "merlin.artifact_bundle.v1",
        "profile": {"name": profile.name, "sha256": profile.sha256},
        "artifacts": [
            {
                "id": "producer",
                "kind": "command_stream",
                "path": image.name,
                "sha256": sha256_file(image),
                "unit": "first",
                "issuer": "host",
            },
            {
                "id": "consumer",
                "kind": "native_image",
                "path": image.name,
                "sha256": sha256_file(image),
                "unit": "second",
                "issuer": "runtime",
            },
        ],
        "buffers": [
            {"id": "input", "memory_space": "input-space", "initialized": True},
            {"id": "shared", "memory_space": "shared-space", "initialized": False},
            {"id": "out", "memory_space": "output-space", "initialized": False},
        ],
        "steps": [
            {"id": "write", "artifact": "producer", "after": [], "reads": ["input"], "writes": ["shared"]},
            {"id": "barrier", "artifact": "producer", "after": ["write"], "reads": [], "writes": []},
            {"id": "read", "artifact": "consumer", "after": ["barrier"], "reads": ["shared"], "writes": ["out"]},
        ],
    }
    return document, profile


def test_transitive_dag_and_independent_readers_are_valid(tmp_path):
    document, profile = _bundle(tmp_path)
    verify_bundle(document, tmp_path, profile=profile)
    document["steps"].append(
        {"id": "independent", "artifact": "consumer", "after": [], "reads": ["input"], "writes": []}
    )
    verify_bundle(document, tmp_path, profile=profile)


@pytest.mark.parametrize("hazard", ["raw", "war", "waw"])
def test_unordered_hazards_are_refused(tmp_path, hazard):
    document, profile = _bundle(tmp_path)
    document["buffers"][1]["initialized"] = True
    first, second = document["steps"][0], document["steps"][2]
    second["after"] = []
    first["reads"], first["writes"] = ([], ["shared"]) if hazard != "war" else (["shared"], [])
    second["reads"], second["writes"] = (["shared"], []) if hazard == "raw" else ([], ["shared"])
    with pytest.raises(BundleError, match="without an order edge"):
        verify_bundle(document, tmp_path, profile=profile)
    second["after"] = ["barrier"]
    verify_bundle(document, tmp_path, profile=profile)


@pytest.mark.parametrize(
    "mutation",
    [
        "artifact_id",
        "buffer_id",
        "step_id",
        "artifact_ref",
        "buffer_ref",
        "forward",
        "self",
        "cycle",
        "uninitialized",
        "inplace_uninitialized",
    ],
)
def test_incomplete_or_ambiguous_graph_is_refused(tmp_path, mutation):
    document, profile = _bundle(tmp_path)
    if mutation == "artifact_id":
        document["artifacts"].append(copy.deepcopy(document["artifacts"][0]))
    elif mutation == "buffer_id":
        document["buffers"].append(copy.deepcopy(document["buffers"][0]))
    elif mutation == "step_id":
        document["steps"].append(copy.deepcopy(document["steps"][0]))
    elif mutation == "artifact_ref":
        document["steps"][0]["artifact"] = "missing"
    elif mutation == "buffer_ref":
        document["steps"][0]["reads"] = ["missing"]
    elif mutation == "forward":
        document["steps"][0]["after"] = ["barrier"]
    elif mutation == "self":
        document["steps"][0]["after"] = ["write"]
    elif mutation == "cycle":
        document["steps"][0]["after"] = ["read"]
    elif mutation == "uninitialized":
        document["buffers"][0]["initialized"] = False
    else:
        document["steps"][0]["reads"] = ["shared"]
    with pytest.raises(BundleError):
        verify_bundle(document, tmp_path, profile=profile)


@pytest.mark.parametrize("mutation", ["absolute", "parent", "missing", "escape", "broken", "digest"])
def test_artifact_paths_and_bytes_are_bound(tmp_path, mutation):
    root = tmp_path / "bundle"
    root.mkdir()
    document, profile = _bundle(root)
    row = document["artifacts"][0]
    if mutation == "absolute":
        row["path"] = str(root / row["path"])
    elif mutation == "parent":
        row["path"] = "../commands.bin"
    elif mutation in ("escape", "broken"):
        outside = tmp_path / "outside.bin"
        if mutation == "escape":
            outside.write_bytes((root / row["path"]).read_bytes())
        (root / "alias.bin").symlink_to(outside)
        row["path"] = "alias.bin"
    elif mutation == "digest":
        (root / row["path"]).write_bytes(b"changed")
    else:
        row["path"] = "missing.bin"
    with pytest.raises(BundleError):
        verify_bundle(document, root, profile=profile)


@pytest.mark.parametrize("mutation", ["none", "dict", "name", "digest", "bytes", "current"])
def test_independent_current_profile_is_required(tmp_path, mutation):
    document, profile = _bundle(tmp_path)
    if mutation == "none":
        profile = None
    elif mutation == "dict":
        profile = document["profile"]
    elif mutation == "name":
        document["profile"]["name"] = "other"
    elif mutation == "digest":
        document["profile"]["sha256"] = "0" * 64
    elif mutation == "bytes":
        profile = replace(profile, bytes=True)
    else:
        profile.path.write_text("name: independent-profile\nchanged: true\n")
    with pytest.raises(BundleError):
        verify_bundle(document, tmp_path, profile=profile)


@pytest.mark.parametrize("mutation", ["profile", "artifact", "alias"])
def test_changes_during_verification_refuse(tmp_path, monkeypatch, mutation):
    document, profile = _bundle(tmp_path)
    image = tmp_path / "commands.bin"
    alias = tmp_path / "alias.bin"
    alias.symlink_to(image)
    document["artifacts"][0]["path"] = alias.name
    original = bundles.sha256_file
    calls = 0

    def observed(path):
        nonlocal calls
        digest = original(path)
        calls += 1
        if calls == 2:
            if mutation == "profile":
                profile.path.write_text("name: changed\n")
            elif mutation == "artifact":
                image.write_bytes(b"changed")
            else:
                other = tmp_path / "other.bin"
                other.write_bytes(image.read_bytes())
                alias.unlink()
                alias.symlink_to(other)
        return digest

    monkeypatch.setattr(bundles, "sha256_file", observed)
    with pytest.raises(BundleError):
        verify_bundle(document, tmp_path, profile=profile)


def _manifest(version):
    fourth = "lower_target_to_llvm" if version == "0.1" else "emit_target_artifact"
    return {
        "abi_version": version,
        "artifact_type": "mlir_oot_target_backend",
        "target": "independent",
        "language": "python",
        "integrity_exempt": False,
        "authoring": {"mode": "hand_curated"},
        "entrypoints": {"tool": "tool.py"},
        "commands": {
            name: {"argv": ["{tool}", name, "{input_mlir}"]}
            for name in ("parse", "lower_interface_to_target", "emit_command_buffer", fourth)
        },
    }


def test_manifests_share_all_common_constraints_and_keep_legacy_default():
    v1, v2 = _manifest("0.1"), _manifest("0.2")
    for document in (v1, v2):
        schemas.validate_manifest(document)
        for missing in ("target", "language", "authoring", "entrypoints", "integrity_exempt"):
            incomplete = copy.deepcopy(document)
            del incomplete[missing]
            with pytest.raises(schemas.ContractViolation):
                schemas.validate_manifest(incomplete)
        invalid = copy.deepcopy(document)
        invalid["commands"]["parse"]["argv"] = []
        with pytest.raises(schemas.ContractViolation):
            schemas.validate_manifest(invalid)
    del v1["abi_version"]
    schemas.validate_manifest(v1)
    base = schemas.load_schema("manifest")
    assert base["properties"]["commands"]["required"][-1] == "lower_target_to_llvm"
    common = copy.deepcopy(base)
    for key in ("$schema", "$id", "title", "description"):
        del common[key]
    del common["properties"]["commands"]["required"]
    assert schemas.load_schema("manifest_v2")["allOf"][0] == common
    del v2["commands"]["emit_target_artifact"]
    with pytest.raises(schemas.ContractViolation):
        schemas.validate_manifest(v2)


@pytest.mark.parametrize("version", ["0.3", None, True, 2, ""])
def test_unknown_manifest_abi_refuses(version):
    document = _manifest("0.1")
    document["abi_version"] = version
    with pytest.raises(schemas.ContractViolation, match="unsupported"):
        schemas.validate_manifest(document)


def _package(root, version):
    root.mkdir()
    manifest = _manifest(version)
    (root / "manifest.yaml").write_text(yaml.safe_dump(manifest))
    (root / "tool.py").write_text(
        "import json, pathlib, sys\n"
        "pathlib.Path('invoked.json').write_text(json.dumps({'cwd':str(pathlib.Path.cwd()), 'argv':sys.argv[1:]}))\n"
        "print('LLVM_TEXT' if sys.argv[1]=='lower_target_to_llvm' else '{\"schema\":\"structural-only\"}')\n"
    )
    return manifest


def test_actual_legacy_alias_and_structural_emission_have_distinct_admission(tmp_path):
    source = tmp_path / "input.mlir"
    source.write_text("module {}\n")
    old = tmp_path / "old"
    _package(old, "0.1")
    legacy = load_package(old)
    assert analysis_emission_entrypoints(legacy) == ("emit_command_buffer", "lower_target_to_llvm")
    assert run_entrypoint(legacy, "emit_target_artifact", source).stdout.strip() == "LLVM_TEXT"
    assert json.loads((old / "invoked.json").read_text())["argv"] == ["lower_target_to_llvm", str(source.resolve())]
    new = tmp_path / "new"
    _package(new, "0.2")
    profile = _profile(tmp_path)
    with pytest.raises(CertFailure):
        load_package(new)
    package = load_package(new, artifact_profile=profile)
    for name in ("parse", "emit_target_artifact", "emit_analysis_bundle", "lower_target_to_llvm"):
        with pytest.raises(CertFailure, match="structural emission only"):
            run_entrypoint(package, name, source)
        assert not (new / "invoked.json").exists()
    with pytest.raises(CertFailure):
        analysis_emission_entrypoints(package)
    assert analysis_emission_entrypoints(package, artifact_profile=profile) == (
        "emit_command_buffer",
        "emit_target_artifact",
    )
    with pytest.raises(CertFailure, match="legacy LLVM"):
        _resolve_argv(package, "lower_target_to_llvm", source, None, artifact_profile=profile)
    result = run_entrypoint(package, "emit_target_artifact", source, artifact_profile=profile)
    assert result.returncode == 0
    assert json.loads(result.stdout) == {"schema": "structural-only"}
    assert json.loads((new / "invoked.json").read_text()) == {
        "cwd": str(new),
        "argv": ["emit_target_artifact", str(source.resolve())],
    }


def test_profile_mutation_during_actual_emission_refuses_stdout(tmp_path):
    root = tmp_path / "pkg"
    _package(root, "0.2")
    profile = _profile(tmp_path)
    package = load_package(root, artifact_profile=profile)
    package.tool.write_text(
        "from pathlib import Path\nPath('../profile.yaml').write_text('name: changed\\n')\nprint('LLVM_TEXT')\n"
    )
    source = tmp_path / "input.mlir"
    source.write_text("module {}\n")
    with pytest.raises(CertFailure, match="identity changed"):
        run_entrypoint(package, "emit_target_artifact", source, artifact_profile=profile)


def test_declared_schema_inheritance_matches_public_resource():
    declared = json.loads((contract_dir() / "schemas/manifest_v2.schema.json").read_text())
    assert declared["allOf"][0] == {"$ref": "manifest.schema.json#/$defs/common"}


def test_structural_permission_cannot_bypass_legacy_capability_gate(tmp_path):
    root = tmp_path / "legacy"
    _package(root, "0.1")
    with pytest.raises(CertFailure, match="requires artifact ABI 0.2"):
        load_package(root, artifact_profile=_profile(tmp_path))


@pytest.mark.parametrize("mutation", ["empty_steps", "empty_artifacts", "bad_boolean", "duplicate_dependency"])
def test_malformed_structural_schema_refuses(tmp_path, mutation):
    document, profile = _bundle(tmp_path)
    if mutation == "empty_steps":
        document["steps"] = []
    elif mutation == "empty_artifacts":
        document["artifacts"] = []
    elif mutation == "bad_boolean":
        document["buffers"][0]["initialized"] = "true"
    else:
        document["steps"][2]["after"] = ["barrier", "barrier"]
    with pytest.raises(schemas.ContractViolation):
        verify_bundle(document, tmp_path, profile=profile)
