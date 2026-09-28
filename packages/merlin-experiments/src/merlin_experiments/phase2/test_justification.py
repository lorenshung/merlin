"""Byte-bound rationale for generated Phase 2 tests; no measurements are inferred.

The receipt connects each development capsule to the Phase 0 machine observation,
workload census, falsifiable claim and planned measurement. It is an input audit,
not a claim analyzer or a simulator result.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping
from pathlib import Path

from merlin_experiments.phase2 import contracts as C
from merlin_experiments.phase2.contracts import StageGateError

_GENERATOR = "merlin/contract/capsules/generate_corpus.py"
_FACTS = "hardware/effective-views/performance-facts.json"
_ACCOUNTING = "coverage/operation-accounting.json"


def requires_justification(provenance: Mapping) -> bool:
    """Current Phase 0 output is gated; old synthetic fixtures stay inspectable."""
    return provenance.get("generated_by") == _GENERATOR


def _selected_evidence(corpus) -> tuple[dict, dict, dict, dict]:
    root = Path(corpus.corpus_root) / "_evidence"
    if root.is_symlink() or not root.is_dir():
        raise StageGateError("Phase 2 test rationale has no real frozen Phase 0 evidence directory")
    manifest_path = root / "evidence-manifest.json"
    evidence = C.mapping_file(manifest_path)
    facts_path, accounting_path = root / _FACTS, root / _ACCOUNTING
    facts, accounting = C.mapping_file(facts_path), C.mapping_file(accounting_path)
    artifacts = evidence.get("artifacts")
    generation = corpus.performance_generation
    declared = generation.get("facts") if isinstance(generation, Mapping) else None
    template = generation.get("shared_template") if isinstance(generation, Mapping) else None
    if (
        evidence.get("target") != corpus.target
        or facts.get("target") != corpus.target
        or not isinstance(artifacts, Mapping)
        or not isinstance(declared, Mapping)
        or not isinstance(template, Mapping)
        or not isinstance(template.get("sha256"), str)
        or len(template["sha256"]) != 64
        or declared.get("target") != corpus.target
        or declared.get("sha256") != facts.get("sha256")
        or declared.get("raw_facts_sha256") != facts.get("raw_facts_sha256")
        or evidence.get("performance_facts_sha256") != facts.get("sha256")
        or evidence.get("raw_facts_sha256") != facts.get("raw_facts_sha256")
    ):
        raise StageGateError("Phase 2 test rationale lacks coherent frozen Phase 0 source/fact identity")
    for relative, path in ((_FACTS, facts_path), (_ACCOUNTING, accounting_path)):
        recorded = artifacts.get(relative)
        if (
            not isinstance(recorded, Mapping)
            or recorded.get("sha256") != C.sha256_file(path)
            or recorded.get("size_bytes") != path.stat().st_size
        ):
            raise StageGateError(f"Phase 2 test rationale evidence bytes changed: {relative}")
    if accounting.get("schema") != "merlin.phase0.operation_accounting.v1":
        raise StageGateError("Phase 2 test rationale lacks a Phase 0 operation census")
    return (
        evidence,
        facts,
        accounting,
        {
            "manifest_sha256": C.sha256_file(manifest_path),
            "performance_facts_sha256": C.sha256_file(facts_path),
            "operation_accounting_sha256": C.sha256_file(accounting_path),
            "shared_template_sha256": template["sha256"],
            "derived_performance_facts_sha256": facts["sha256"],
            "raw_rtl_facts_sha256": facts["raw_facts_sha256"],
        },
    )


def verify_live_justification(corpus, receipt: Mapping) -> None:
    """Recheck all original inputs before a selected cohort is copied."""
    _, _, _, source = _selected_evidence(corpus)
    expected = receipt.get("source") or {}
    if any(expected.get(key) != value for key, value in source.items()):
        raise StageGateError("Phase 2 test rationale source evidence changed before freeze")
    if expected.get("provenance_sha256") != C.sha256_file(corpus.provenance_manifest):
        raise StageGateError("Phase 2 test rationale provenance changed before freeze")
    if expected.get("performance_generation_sha256") != C.document_sha256(corpus.performance_generation):
        raise StageGateError("Phase 2 test rationale generation changed before freeze")
    for row in receipt.get("members") or []:
        relative = Path(str(row.get("relative_path") or ""))
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) != 2:
            raise StageGateError("Phase 2 test rationale contains an unsafe member path")
        if C.exact_tree_record(Path(corpus.corpus_root) / relative)["sha256"] != row.get("capsule_tree_sha256"):
            raise StageGateError(f"Phase 2 test rationale member changed before freeze: {relative}")


def derive_justification(corpus) -> dict:
    """Derive a deterministic plan for every admitted generated performance member.

    The caller supplies the *complete* discovered Phase 2 cohort. A selected subset
    cannot silently omit a sibling needed by a comparison group or negative control.
    """
    provenance = C.mapping_file(corpus.provenance_manifest, yaml_file=True)
    if not requires_justification(provenance):
        raise StageGateError("test justification requires current generated Phase 0 provenance")
    if C.sha256_file(corpus.provenance_manifest) != corpus.provenance_sha256:
        raise StageGateError("performance provenance changed before test justification")
    evidence, facts, accounting, source = _selected_evidence(corpus)
    phase = corpus.performance_generation.get("phase") or {}
    generated = {
        str(item) for item in provenance.get("generated") or [] if Path(str(item)).parts[:1] == (phase.get("category"),)
    }
    present = {member.source_relative_path for member in corpus.capsules}
    if not generated or present != generated:
        raise StageGateError("test justification requires every generated Phase 2 capsule")
    roles_by_group: dict[tuple[str, str], list[tuple[str, str]]] = defaultdict(list)
    family_claims: dict[str, str] = {}
    for member in corpus.capsules:
        performance = member.descriptor.get("performance") or {}
        stable_fields = (
            "level",
            "family",
            "lever",
            "member_class",
            "claim",
            "comparand",
            "falsifier",
            "gate",
            "regime",
            "cost",
            "acceptance",
            "noise_band",
        )
        claim_sha = C.document_sha256({key: performance.get(key) for key in stable_fields})
        if member.family in family_claims and family_claims[member.family] != claim_sha:
            raise StageGateError(f"generated performance family has divergent claim contracts: {member.family}")
        family_claims[member.family] = claim_sha
        group = member.descriptor.get("comparison_group") or {}
        if isinstance(group, Mapping) and group.get("name") and group.get("role"):
            roles_by_group[(member.family, str(group["name"]))].append((str(group["role"]), member.capsule))
    rows = []
    for member in sorted(corpus.capsules, key=lambda item: (item.family, item.capsule)):
        if C.exact_tree_record(member.source_dir)["sha256"] != member.source_sha256:
            raise StageGateError(f"performance member changed before test justification: {member.capsule}")
        descriptor = member.descriptor
        performance = descriptor.get("performance") or {}
        gate, comparand, falsifier = (performance.get(key) or {} for key in ("gate", "comparand", "falsifier"))
        if any(not isinstance(value, Mapping) for value in (gate, comparand, falsifier)):
            raise StageGateError(f"malformed test claim for {member.capsule}")
        names = gate.get("traits") or []
        if not names or not isinstance(names, list) or any(name not in (facts.get("traits") or {}) for name in names):
            raise StageGateError(f"test claim has no frozen RTL trait basis: {member.capsule}")
        selected_traits = {name: facts["traits"][name] for name in names}
        if any(row.get("satisfied") is not True for row in selected_traits.values()):
            raise StageGateError(f"generated performance member has an unsatisfied RTL gate: {member.capsule}")
        capabilities = gate.get("execution_capabilities") or []
        selected_execution = {name: (facts.get("execution_capabilities") or {}).get(name) for name in capabilities}
        if any(not isinstance(row, Mapping) or row.get("satisfied") is not True for row in selected_execution.values()):
            raise StageGateError(f"generated performance member has an unsatisfied execution gate: {member.capsule}")
        if (
            not all(comparand.get(key) for key in ("kind", "against", "demand_equal"))
            or not all(falsifier.get(key) for key in ("observation", "fires_when", "negative_control"))
            or not gate.get("instrument")
        ):
            raise StageGateError(f"test claim lacks a comparator, falsifier or instrument: {member.capsule}")
        group = descriptor.get("comparison_group") or {}
        siblings = (
            sorted(
                ({"role": role, "capsule": name} for role, name in roles_by_group[(member.family, str(group["name"]))]),
                key=lambda item: (item["role"], item["capsule"]),
            )
            if isinstance(group, Mapping) and group.get("name")
            else []
        )
        expected_ops = ((performance.get("emitter") or {}).get("knobs") or {}).get("members") or []
        sibling_ops = (
            {
                ((other.descriptor.get("operation") or {}).get("op"))
                for other in corpus.capsules
                if other.family == member.family
                and isinstance(group, Mapping)
                and (other.descriptor.get("comparison_group") or {}).get("name") == group.get("name")
            }
            if siblings
            else set()
        )
        matching_status = (
            "incomplete_group"
            if (siblings and (len(siblings) < 2 or set(expected_ops) - sibling_ops))
            else "planned_unverified"
        )
        # A prose counterfactual is a *declared* control until an analyzer verifies
        # equal demand and the paired programs/results. Never label it established.
        acceptance = performance.get("acceptance") or {}
        measure = acceptance.get("evidence") or {} if isinstance(acceptance, Mapping) else {}
        repeats = acceptance.get("replicates") or {} if isinstance(acceptance, Mapping) else {}
        geometry = performance.get("shape_geometry") or {}
        workload = {
            "status": "observed_shape_class"
            if geometry.get("in_census") is True
            else ("off_census" if geometry.get("in_census") is False else "not_established"),
            "geometry_class": geometry.get("geometry_class"),
            "census_mac_fraction": geometry.get("census_mac_fraction"),
            "selected_inventory_sha256": (accounting.get("selected_inventory") or {}).get("content_sha256"),
            "scope": accounting.get("scope"),
        }
        screen = descriptor.get("software_screen") or {}
        rows.append(
            {
                "family": member.family,
                "capsule": member.capsule,
                "relative_path": member.source_relative_path,
                "capsule_tree_sha256": member.source_sha256,
                "performance_sha256": C.document_sha256(performance),
                "purpose": {
                    "level": performance.get("level"),
                    "lever": performance.get("lever"),
                    "member_class": performance.get("member_class"),
                    "claim": performance.get("claim"),
                },
                "hardware_basis": {"traits": selected_traits, "execution_capabilities": selected_execution},
                "workload_need": workload,
                "hypothesis": {"observation": falsifier["observation"], "falsified_when": falsifier["fires_when"]},
                "matched_comparator": {
                    "declaration": comparand,
                    "group": group or None,
                    "candidate_members": siblings,
                    "matching_status": matching_status,
                },
                "negative_control": {"declaration": falsifier["negative_control"], "status": "planned_unverified"},
                "measurement": {
                    "instrument": gate["instrument"],
                    "timing_simulator": measure.get("timing_simulator"),
                    "timing_tier": measure.get("timing_tier"),
                    "status": "unmeasured",
                },
                "correctness": {
                    "required_oracle_tiers": descriptor.get("required_oracle_tiers") or [],
                    "simulator": measure.get("correctness_simulator"),
                    "tier": measure.get("correctness_tier"),
                    "status": "unverified",
                },
                "repeat_dispersion": {
                    "replicates": repeats,
                    "noise_band": performance.get("noise_band") or acceptance.get("band") or {},
                    "status": "planned_unmeasured",
                },
                "support": {
                    "software_screen": screen.get("status", "not_established"),
                    "status": "unsupported" if screen.get("status") == "unsupported" else "unverified",
                },
            }
        )
    return {
        "schema": "merlin.phase2.test_justification.v1",
        "target": corpus.target,
        "status": "diagnostic_unmeasured",
        "qualification": "test design and input identity only; no compiler correctness or measured speedup",
        "source": {
            **source,
            "provenance_sha256": corpus.provenance_sha256,
            "performance_generation_sha256": C.document_sha256(corpus.performance_generation),
            "evidence_status": evidence.get("status"),
        },
        "phase_roles": (provenance.get("phase_corpora") or {}).get(corpus.target),
        "families_not_generated": {
            "skipped_inapplicable": corpus.performance_generation.get("skipped_inapplicable") or [],
            "blocked_unimplemented": corpus.performance_generation.get("blocked_unimplemented") or [],
            "errors": corpus.performance_generation.get("errors") or [],
        },
        "members": rows,
    }
