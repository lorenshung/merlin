"""Run-owned public grading and descriptor-policy views, sealed by the native snapshot.

The authored bundle and its candidate grants are unchanged. A private derived input is
declared in the run's effective bundle before freezing. Resume consumes that frozen input,
not the source tree or a shared publication link. Records belong in host-only environment
evidence; they are references into the existing native snapshot, not another seal.
"""

from __future__ import annotations

import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path

import yaml

from merlin.targetgen.capsule_common import discover_capsules
from merlin.targetgen.sandbox import bwrap

from ..corpus.admission import public_capsules_for
from ..corpus.preparation import copy_input, ordinary_tree, private_json
from .run_inputs import bundle_manifest_identity
from .source_inputs import fingerprint


@dataclass(frozen=True)
class CorpusView:
    public: Path
    policy: Path
    contract: Path
    workload_coverage: dict | None = None


@dataclass(frozen=True)
class PreparedBundle:
    bundle: dict
    corpus_record: dict
    authored_sha256: str
    effective_sha256: str


def prepare_bundle(
    run_dir: Path,
    te,
    authored_path: Path,
    bundle: dict,
    *,
    contract: Path,
    capsules_root: Path | None = None,
    environment: dict | None = None,
) -> PreparedBundle:
    """Archive fresh declarations or verify existing ones; resume never derives from live data."""
    authored = run_dir / "authored_input_bundle_manifest.yaml"
    effective = run_dir / "input_bundle_manifest.yaml"
    if environment is None:
        with os.fdopen(os.open(authored, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "wb") as stream:
            stream.write(authored_path.read_bytes())
        authored_sha = bundle_manifest_identity(authored, bundle)
        document, record = stage(run_dir, te, bundle, contract=contract, capsules_root=capsules_root)
        with os.fdopen(os.open(effective, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w") as stream:
            yaml.safe_dump(document, stream, sort_keys=False)
    else:
        authored_sha = bundle_manifest_identity(authored, bundle)
        if environment.get("authored_bundle_manifest_sha256") != authored_sha:
            raise RuntimeError("resume refused: authored input bundle identity changed")
        record = environment.get("public_corpus_input")
        if not isinstance(record, dict) or record.get("descriptor_sha256") != te.descriptor_sha256:
            raise RuntimeError("resume refused: frozen public corpus descriptor identity is absent or changed")
        expected_source = str(run_dir.absolute() / "private_corpus_input")
        if record.get("staging_path") != expected_source:
            raise RuntimeError("resume refused: public corpus input belongs to a different run")
        if effective.is_symlink() or not effective.is_file():
            raise RuntimeError("resume refused: effective input bundle is absent or indirect")
        document = yaml.safe_load(effective.read_text())
        expected_bundle = copy.deepcopy(bundle)
        expected_bundle.setdefault("host_inputs", []).append(
            {"path": expected_source, "note": "host-only run corpus view"}
        )
        if document != expected_bundle:
            raise RuntimeError("resume refused: effective bundle changed candidate grants or declared inputs")
    effective_sha = bundle_manifest_identity(effective, document)
    if environment is not None and environment.get("bundle_manifest_sha256") != effective_sha:
        raise RuntimeError("resume refused: effective input bundle bytes changed")
    return PreparedBundle(document, record, authored_sha, effective_sha)


def _resources(capsules: list[dict]) -> None:
    """Require declared file resources to travel with their capsule, without rewriting refs."""
    for capsule in capsules:
        directory = Path(capsule["__dir__"]).resolve()
        attributes = (capsule.get("operation") or {}).get("attributes") or {}
        references = {
            "interface_mlir": capsule.get("interface_mlir"),
            "linalg_mlir": capsule.get("linalg_mlir"),
            "pytorch_ref.loader": (capsule.get("pytorch_ref") or {}).get("loader"),
            **{key: attributes.get(key) for key in ("weights", "weights_manifest", "loader_dependencies")},
        }
        for field, value in references.items():
            if value is None:
                continue
            relative = Path(str(value))
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"run corpus cannot freeze non-capsule-local resource: {field}")
            member = directory / relative
            if not member.exists() or member.is_symlink() or not member.resolve().is_relative_to(directory):
                raise ValueError(f"run corpus has an absent or indirect resource: {field}")


def stage(run_dir: Path, te, bundle: dict, *, contract: Path, capsules_root: Path | None = None) -> tuple[dict, dict]:
    """Prepare a fresh private input from either the descriptor or one explicit corpus."""
    stage_root = Path(run_dir).absolute() / "private_corpus_input"
    stage_root.mkdir(mode=0o700, exist_ok=False)
    schema_source = Path(contract).absolute() / "schemas"
    copy_input(schema_source, stage_root / "contract/schemas", private=True)
    contract = stage_root / "contract"
    source_root = Path(capsules_root).absolute() if capsules_root is not None else None
    source_before = None
    if source_root is None:
        sources = list(te.graded_roots())
        coverage_root = te.capsule_corpus.parent
    else:
        ordinary_tree(source_root)
        source_before = fingerprint(source_root)
        categories = [
            child
            for child in sorted(source_root.iterdir())
            if child.is_dir()
            and not child.name.startswith(("_", "."))
            and child.name != "hidden"
            and next(child.glob("*/capsule.yaml"), None) is not None
        ]
        sources = categories or [source_root]
        coverage_root = source_root.parent if source_root == te.capsule_corpus else source_root
        if not discover_capsules(sources, labels={"public", "dev"}, contract=contract):
            raise ValueError("explicit public/dev corpus contains no capsules")
    staged_sources = []
    commitments = [{"original": str(schema_source), "staged": "contract/schemas", "role": "host_schema"}]
    for index, source in enumerate(sources):
        destination = stage_root / "policy" / str(index)
        copy_input(Path(source), destination, private=True)
        staged_sources.append(destination)
        commitments.append({"original": str(Path(source).absolute()), "staged": f"policy/{index}", "role": "corpus"})
    if source_root is None and list(te.graded_roots()) != sources:
        raise RuntimeError("descriptor corpus root membership changed during preparation")
    _resources(discover_capsules(staged_sources, labels={"public", "dev"}, contract=contract))
    public = stage_root / "public"
    if capsules_root is None:
        public_capsules_for(te, corpus_roots=staged_sources, destination=public)
        mode = "descriptor_cohort"
    else:
        selected = discover_capsules(staged_sources, labels={"public", "dev"}, contract=contract)
        _resources(selected)
        if not selected:
            raise ValueError("public/dev override contains no capsules")
        public.mkdir(mode=0o700)
        for capsule in selected:
            source = Path(capsule["__dir__"])
            # The copied policy view is the authority for both grading and the
            # raw override. Never re-read live source capsule bytes here.
            policy_relative = source.relative_to(stage_root / "policy")
            original = sources[int(policy_relative.parts[0])] / Path(*policy_relative.parts[1:])
            relative = original.relative_to(source_root)
            copy_input(source, public / relative, private=True)
            commitments.append(
                {
                    "original": str(original.absolute()),
                    "staged": (Path("public") / relative).as_posix(),
                    "role": "corpus",
                }
            )
        if fingerprint(source_root) != source_before:
            raise RuntimeError("public/dev override changed during preparation")
        mode = "public_dev_override"
    _resources(discover_capsules(public, labels={"public", "dev"}, contract=contract))
    from ..phase0.coverage_commitment import (
        INPUT_PATH,
        _digest,
        observe_cohort,
        read_inputs,
        requires_workload_coverage,
    )

    selected_inputs = read_inputs(coverage_root)
    if selected_inputs is not None:
        if selected_inputs.get("target") != te.target:
            raise ValueError("selected coverage inputs belong to a different target")
        copy_input(coverage_root / INPUT_PATH.parent, stage_root / "coverage", private=True)
        commitments.append(
            {
                "original": str((coverage_root / "_phase0").absolute()),
                "staged": "coverage",
                "role": "coverage_inputs",
            }
        )
    completeness = observe_cohort(selected_inputs, public, target=te.target, contract=contract)
    private_json(stage_root / "coverage-report.json", completeness)
    private_json(stage_root / "source_commitments.json", {"version": 1, "mode": mode, "sources": commitments})
    record = {
        "version": 1,
        "staging_path": str(stage_root),
        "content_sha256": fingerprint(stage_root),
        "mode": mode,
        "authority": "private_preparation_bound_to_native_snapshot_not_operator_approval",
        "descriptor_sha256": te.descriptor_sha256,
        "workload_coverage": {
            "required": requires_workload_coverage(te, selected_inputs),
            "status": completeness["status"],
            "report_sha256": _digest(completeness),
            "inputs_sha256": completeness["inputs_sha256"],
            "cohort_sha256": completeness["cohort"]["sha256"],
        },
    }
    # The native snapshot copies private bytes with independent inodes, never public CAS.
    # Seal the staging copy too; setup never modifies it after recording its identity.
    for member in [*stage_root.rglob("*"), stage_root]:
        member.chmod(member.stat().st_mode & ~0o222)
    effective = copy.deepcopy(bundle)
    effective.setdefault("host_inputs", []).append({"path": str(stage_root), "note": "host-only run corpus view"})
    return effective, record


def resolve(
    ws: Path,
    bundle: dict,
    record: dict,
    *,
    repo: Path,
    reviewed_roots: tuple[Path, ...] | None = None,
) -> CorpusView:
    """Return only the already-verified native snapshot view; never read live staging."""
    if not isinstance(record, dict) or record.get("version") != 1:
        raise RuntimeError("run corpus input record is missing or unsupported")
    source = record.get("staging_path")
    if not isinstance(source, str) or not Path(source).is_absolute():
        raise RuntimeError("run corpus input record has no absolute staging identity")
    [frozen] = bwrap.snapshot_input_paths(ws, bundle, [Path(source)], repo=repo)
    if fingerprint(frozen) != record.get("content_sha256"):
        raise RuntimeError("run corpus input differs from its prepared native snapshot identity")
    commitment = json.loads((frozen / "source_commitments.json").read_text())
    if reviewed_roots is not None and (
        record.get("mode") != "descriptor_cohort" or commitment.get("mode") != "descriptor_cohort"
    ):
        raise RuntimeError("reviewed run corpus requires the complete descriptor cohort; raw overrides are unverified")
    manifest = bwrap.verify_bundle_snapshot(ws, bundle, repo=repo)
    declared = [Path(row["destination"]) for row in [*manifest["grants"], *manifest.get("host_records", [])]]
    covered = []
    for row in commitment["sources"]:
        original = Path(row["original"])
        relative = Path(row["staged"])
        if relative.is_absolute() or ".." in relative.parts:
            raise RuntimeError("run corpus source commitment escapes its frozen input")
        reviewed_corpus = reviewed_roots is not None and row["role"] == "corpus"
        if reviewed_corpus and not any(original == root or original.is_relative_to(root) for root in reviewed_roots):
            raise RuntimeError("reviewed run corpus has a source outside its reviewed original roots")
        if any(original == path or original.is_relative_to(path) for path in declared):
            covered.append((original, frozen / relative))
        elif reviewed_corpus:
            raise RuntimeError("reviewed run corpus has a source outside its original declared snapshot")
    if covered:
        originals = bwrap.snapshot_input_paths(ws, bundle, [source for source, _ in covered], repo=repo)
        for original, (_, staged) in zip(originals, covered, strict=True):
            if fingerprint(original) != fingerprint(staged):
                raise RuntimeError("prepared corpus differs from original frozen source; cannot borrow its review")
    public, policy, contract = frozen / "public", frozen / "policy", frozen / "contract"
    if not public.is_dir() or not policy.is_dir() or not (contract / "schemas").is_dir():
        raise RuntimeError("run corpus snapshot is missing a declared view")
    from ..phase0.coverage_commitment import _digest, verify_cohort_binding

    completeness = None
    commitment_record = record.get("workload_coverage")
    if commitment_record is not None:
        selected_path = frozen / "coverage/coverage-inputs.json"
        selected_inputs = json.loads(selected_path.read_bytes()) if selected_path.is_file() else None
        saved = json.loads((frozen / "coverage-report.json").read_bytes())
        verify_cohort_binding(saved, selected_inputs, public, contract=contract)
        completeness = saved
        if (
            _digest(saved) != commitment_record.get("report_sha256")
            or saved.get("inputs_sha256") != commitment_record.get("inputs_sha256")
            or saved["cohort"]["sha256"] != commitment_record.get("cohort_sha256")
        ):
            raise RuntimeError("frozen whole-workload coverage differs from its actual admitted-cohort commitment")
    return CorpusView(public=public, policy=policy, contract=contract, workload_coverage=completeness)
