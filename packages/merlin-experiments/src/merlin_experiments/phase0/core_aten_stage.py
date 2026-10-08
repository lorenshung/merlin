"""Deterministic retained-call selection and Phase 0 packaging, without execution claims.

The capsule writer belongs to targetgen. This owner binds declared cohorts and
capture bytes, stages one run-owned corpus, and exposes only hidden commitments.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import yaml

from ..corpus.phase_selection import generate_phase_selections

SCHEMA = "merlin.phase0.core_aten_stage.v1"
COHORTS = ("public", "hidden", "host_guard")
REQUIRED_COHORTS = ("public", "hidden")
#: ``device`` requires retired accelerator instructions; ``any`` lets the grader route to the device when the
#: submission covers the call and otherwise run the host lane, recording the lane per capsule.
LANE_EXPECTATIONS = ("device", "any")


def _json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, indent=2) + "\n").encode()


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def declaration(recipe: Path) -> dict | None:
    document = yaml.safe_load(recipe.read_bytes()) or {}
    block = document.get("core_aten")
    if block is None:
        return None
    if document.get("capsule_policy") != "derived_only" or not isinstance(block, dict):
        raise ValueError("Core ATen requires an explicit derived_only recipe mapping")
    if any(document.get(key) for key in ("capsules", "sweeps", "performance")):
        raise ValueError("Core ATen stage cannot mix authored capsule or sweep membership")
    return block


def _path(recipe: Path, value: object) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Core ATen input requires an explicit path")
    lexical = (recipe.parent / value).absolute()
    if lexical != lexical.resolve(strict=True):
        raise ValueError("Core ATen inputs must be ordinary paths without aliases")
    return lexical


def selection(recipe: Path) -> tuple[list[dict], dict[str, str]]:
    """Read only explicitly selected cases/captures, never discover another cohort."""
    from merlin.targetgen.core_aten_capture import case_capture_name

    block = declaration(recipe)
    declared = set((block or {}).get("cohorts", {}))
    if block is None or not set(REQUIRED_COHORTS) <= declared <= set(COHORTS):
        raise ValueError("Core ATen stage requires public and hidden declarations (host_guard optional)")
    paths: dict[str, str] = {}
    overlay = _path(recipe, block.get("overlay"))
    paths[str(overlay)] = _sha(overlay)
    batches, identities = [], set()
    for cohort in COHORTS:
        if cohort not in block["cohorts"]:
            continue
        rows = block["cohorts"][cohort]
        if not isinstance(rows, list) or not rows:
            raise ValueError("every Core ATen cohort must declare at least one case selection")
        for row in rows:
            expectation = row.get("lane_expectation", "device")
            if expectation not in LANE_EXPECTATIONS or (cohort == "host_guard" and "lane_expectation" in row):
                raise ValueError("lane_expectation must be device or any, and never set on host_guard rows")
            cases_path = _path(recipe, row.get("cases"))
            captures = _path(recipe, row.get("captures"))
            if not captures.is_dir():
                raise ValueError("capture owner must be an ordinary directory")
            paths[str(cases_path)] = _sha(cases_path)
            corpus = json.loads(cases_path.read_bytes())
            if "cases" not in corpus and isinstance(corpus.get("selected_cases"), list):
                corpus = {**corpus, "cases": corpus["selected_cases"]}
            wanted = row.get("select")
            if not isinstance(wanted, list) or not wanted or len(wanted) != len(set(wanted)):
                raise ValueError("case selection must be a nonempty unique list of case_id or overload")
            by_id = {case.get("case_id", case["overload"]): case for case in corpus["cases"]}
            if len(by_id) != len(corpus["cases"]) or set(wanted) - set(by_id):
                raise ValueError("selected cases are missing or ambiguous")
            cases = [by_id[key] for key in sorted(wanted)]
            for case in cases:
                name = case_capture_name(case).replace("-", "_")
                if name in identities:
                    raise ValueError("PUBLIC, HIDDEN and HOST-GUARD capsule identities must be disjoint")
                identities.add(name)
                source = captures / case_capture_name(case)
                if source.is_symlink() or not source.is_dir():
                    raise ValueError("selected capture is missing or indirect")
                capture = json.loads((source / "capture.json").read_bytes())
                if (
                    capture.get("status") != "captured_exact"
                    or capture.get("overload") != case["overload"]
                    or capture.get("capture_meta", {}).get("opaque") != 0
                    or not capture.get("capture_meta", {}).get("result_contract")
                ):
                    raise ValueError(
                        "selected capture must be exact, nonopaque, overload-bound "
                        "and carry a full-call result contract"
                    )
                # The batch builder may read several ABI/input sidecars. Bind complete
                # selected-directory membership, not only its program/capture receipt.
                for member in sorted(source.rglob("*")):
                    if member.is_symlink() or not (member.is_dir() or member.is_file()):
                        raise ValueError("selected capture contains indirect or special entries")
                    if member.is_file():
                        paths[str(member)] = _sha(member)
            batches.append(
                {
                    "cohort": cohort,
                    "lane_expectation": expectation,
                    "corpus": {"cases": cases, "denominator_sha256": corpus.get("denominator_sha256")},
                    "captures": str(captures),
                }
            )
        del by_id, corpus  # Do not retain an unselected bounded suite while loading the next batch.
    return batches, dict(sorted(paths.items()))


def input_paths(recipe: Path) -> list[Path]:
    """Input inventory used by the existing run/resume receipt owner."""
    _, paths = selection(recipe)
    block = declaration(recipe)
    if block.get("derivation"):
        paths[str(_path(recipe, block["derivation"]))] = ""
    # Bind the writer's implementation with this stage, not an arbitrary callback.
    from merlin.common.paths import module_source_path

    for module in (
        "merlin.targetgen.core_aten_capsules",
        "merlin.targetgen.core_aten_batch",
        "merlin.targetgen.core_aten_cases",
        "merlin.targetgen.golden_store",
    ):
        paths[str(module_source_path(module).resolve())] = ""
    return [Path(path) for path in sorted(paths)]


def derive(recipe: Path, output: Path, *, target: str) -> dict:
    batches, paths = selection(recipe)
    output = output.absolute()
    if output.exists():
        raise ValueError("Core ATen derivation output already exists")
    output.mkdir(parents=True, mode=0o700)
    receipt = {
        "schema": SCHEMA,
        "target": target,
        "inputs": paths,
        "selection_sha256": hashlib.sha256(_json(batches)).hexdigest(),
        "scope": "retained full-call packaging; no numerical or hardware verdict",
    }
    (output / "selection.json").write_bytes(_json(receipt))
    (output / "selection.json").chmod(0o600)
    document = yaml.safe_load(recipe.read_bytes())
    # Resolve authored owners once; generated recipe never discovers siblings.
    block = document["core_aten"]
    block["overlay"] = str(_path(recipe, block["overlay"]))
    for rows in block["cohorts"].values():
        for row in rows:
            for key in ("cases", "captures"):
                row[key] = str(_path(recipe, row[key]))
    block["derivation"] = str(output / "selection.json")
    (output / "recipe.yaml").write_text(yaml.safe_dump(document, sort_keys=False))
    (output / "recipe.yaml").chmod(0o600)
    return {
        "schema": SCHEMA,
        "recipe": str(output / "recipe.yaml"),
        "selection_sha256": receipt["selection_sha256"],
        "counts": {
            cohort: sum(len(b["corpus"]["cases"]) for b in batches if b["cohort"] == cohort) for cohort in COHORTS
        },
    }


def generate(
    recipe: Path,
    output: Path,
    *,
    target: str,
    capability_contract: Path | None = None,
    rtl_facts: Path | None = None,
) -> list[Path]:
    from merlin.targetgen.core_aten_capsules import write_capsules

    batches, paths = selection(recipe)
    block = declaration(recipe)
    receipt_path = _path(recipe, block.get("derivation"))
    receipt = json.loads(receipt_path.read_bytes())
    if receipt != {
        "schema": SCHEMA,
        "target": target,
        "inputs": paths,
        "selection_sha256": hashlib.sha256(_json(batches)).hexdigest(),
        "scope": "retained full-call packaging; no numerical or hardware verdict",
    }:
        raise ValueError("Core ATen derivation selection changed; rerun corpus derive")
    output = output.absolute()
    if output != output.resolve():
        raise ValueError("Core ATen output cannot follow a directory alias")
    if output.name != "capsules" or output.parent.name != "phase0":
        raise ValueError("Core ATen output must be <run>/phase0/capsules")
    if output.exists():
        raise ValueError("Core ATen output already exists; select a fresh run")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".core-aten-", dir=output.parent))
    try:
        members, manifests = [], []
        program_sets = {cohort: set() for cohort in COHORTS}
        for batch in batches:
            cohort = batch["cohort"]
            category = stage / cohort
            # Each batch uses its own writer manifest; aggregate only our actual members.
            writer_manifest = category / "manifest.json"
            if writer_manifest.exists():
                writer_manifest.unlink()
            manifest = write_capsules(
                batch["corpus"],
                Path(batch["captures"]),
                category,
                stage / "_private" / cohort,
                label="hidden" if cohort == "hidden" else "public",
                tiers=("L2",),
            )
            manifests.append(manifest)
            for row in manifest["capsules"]:
                member = category / row["id"]
                declaration_path = member / "capsule.yaml"
                capsule = yaml.safe_load(declaration_path.read_bytes())
                capsule["cohort"] = "guard" if cohort == "host_guard" else cohort
                capsule["scored"] = cohort != "host_guard"
                if cohort == "host_guard":
                    capsule["lane_expectation"] = "host-guard"
                    capsule["semantic"] = {"must_accelerate": False}
                    capsule["lanes"] = {"forbid": ["on_mesh"]}
                elif batch["lane_expectation"] == "any":
                    capsule["lane_expectation"] = "any"
                    capsule["semantic"] = {"must_accelerate": False}
                    capsule.pop("lanes", None)
                else:
                    capsule["lane_expectation"] = "device"
                    capsule["semantic"] = {"must_accelerate": True}
                    capsule["lanes"] = {"require": ["on_mesh"]}
                declaration_path.write_text(yaml.safe_dump(capsule, sort_keys=False))
                row.setdefault("digests", {})["capsule.yaml"] = _sha(declaration_path)
                call_digest = hashlib.sha256()
                for filename in ("capsule.interface.mlir", "inputs.npz", "capsule.pytorch.py", "call.json"):
                    call_digest.update(filename.encode())
                    call_digest.update((member / filename).read_bytes())
                program_sets[cohort].add(call_digest.hexdigest())
                members.append(member.relative_to(stage).as_posix())
            writer_manifest.unlink()  # Per-batch evidence belongs to the private ledger.
        if program_sets["public"] & program_sets["hidden"]:
            raise ValueError("hidden and public selected program/input pairs overlap")
        private = stage / "_private"
        (private / "stage.json").write_bytes(_json({"receipt": receipt, "packaging": manifests}))
        for member in private.rglob("*"):
            member.chmod(0o700 if member.is_dir() else 0o600)
        public_members = sorted(m for m in members if not m.startswith("hidden/"))
        manifest = {
            "generated_by": "merlin/contract/capsules/generate_corpus.py",
            "generated": public_members,
            "hand_authored": [],
            "held_out": {"n_generated": sum(m.startswith("hidden/") for m in members), "n_hand_authored": 0},
            "phase_corpora": {target: generate_phase_selections(public_members, performance_category="_perf")},
            "core_aten_stage": {
                "schema": SCHEMA,
                "private_ledger_sha256": _sha(private / "stage.json"),
                "scope": receipt["scope"],
            },
        }
        if capability_contract is not None:
            contract = yaml.safe_load(capability_contract.read_bytes())
            if not isinstance(contract, dict) or contract.get("name") != target:
                raise ValueError("selected capability contract names a different target")
            exported = stage / "software" / "contract.json"
            exported.parent.mkdir()
            exported.write_bytes(_json(contract))
            manifest["core_aten_stage"]["capability_contract_sha256"] = _sha(capability_contract)
        if rtl_facts is not None:
            facts = json.loads(rtl_facts.read_bytes())
            payload = facts.get("facts", facts)
            if not isinstance(payload, dict) or payload.get("target") != target:
                raise ValueError("selected facts name a different target")
            exported = stage / "hardware" / "effective-views" / "loaded-facts.json"
            exported.parent.mkdir(parents=True)
            exported.write_bytes(_json(facts if "facts" in facts else {"facts": facts}))
            manifest["core_aten_stage"]["rtl_facts_sha256"] = _sha(rtl_facts)
        (stage / "MANIFEST.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False))
        stage.rename(output)
        return [output / m for m in sorted(members)]
    finally:
        if stage.exists():
            shutil.rmtree(stage)
