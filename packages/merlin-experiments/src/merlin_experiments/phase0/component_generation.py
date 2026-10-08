"""Admission of independent component sweeps through the normal generator.

This declaration path does not certify hardware, timings or whole-model coverage.
The host reviews the selected SW declaration against byte-bound HW inputs; the
existing writer, numerical engines and written-program screen retain ownership.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

import yaml

from merlin.targetgen import corpus_spec, phase_policy, software_spec
from merlin.targetgen.semantic_families import from_op

SCHEMA = "merlin.phase0.component_generation.v1"
DECLARATION = "merlin.component_performance.v1"
_OBJECTIVE_FIELDS = {"metric", "unit", "direction", "basis"}
_CAPTURE_FIELDS = {"micro_model", "materialized_capture", "model", "capture", "capture_dir", "quant_recipe"}


def digest(document):
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    ).hexdigest()


def _source(path):
    path = Path(path).resolve()
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def require_inputs(*, descriptor, recipe, performance_template, profiles_root, evidence_mode, **sidecars):
    """Refuse model/hidden selectors before the ordinary loader can use them."""
    if descriptor is None or recipe is None or performance_template is None or profiles_root is not None:
        raise ValueError("component generation requires explicit descriptor, recipe and shared performance_template")
    if evidence_mode != "diagnostic":
        raise ValueError(
            "component generation requires explicit diagnostic evidence mode; whole coverage is not verified"
        )
    if any(value is not None for value in sidecars.values()):
        raise ValueError(
            "component generation refuses capture, synthesis, hidden and previously exported evidence inputs"
        )
    document = yaml.safe_load(Path(descriptor).read_bytes())
    if not isinstance(document, dict) or any(
        document.get(key) for key in ("workload_spec", "claim_boundary", "grading")
    ):
        raise ValueError(
            "component generation requires an independent descriptor without model/holdout grading selectors"
        )
    public = yaml.safe_load(Path(recipe).read_bytes()) or {}
    if not isinstance(public, dict) or public.get("capsules") or public.get("sweeps"):
        raise ValueError("component recipe may not author capsules or sweeps; use the shared independent template")
    template = yaml.safe_load(Path(performance_template).read_bytes()) or {}
    if not isinstance(template, dict) or any(row.get("requires_form_scope") for row in template.get("sweeps") or []):
        raise ValueError("component generation refuses capture-derived form-scope sweeps")


def bind_entries(entries, *, evidence, profile, recipe, performance_template):
    """Bind reviewed objectives and the actual selected generator/source identity."""
    if evidence is None or evidence.software_spec.get("status") != "reviewed":
        raise ValueError("component generation requires selected reviewed software and hardware evidence")
    spec = evidence.software_spec
    declaration = spec.get("component_performance")
    if (
        not isinstance(declaration, dict)
        or set(declaration) != {"schema", "status", "hardware", "objectives"}
        or declaration.get("schema") != DECLARATION
        or declaration.get("status") != "reviewed"
    ):
        raise ValueError("component generation requires an explicit reviewed component_performance declaration")
    hardware = {key: evidence.derivation_identity[key] for key in ("contract_sha256", "raw_facts_sha256")}
    if any(value is None for value in hardware.values()) or declaration["hardware"] != hardware:
        raise ValueError("component objective declaration differs from selected hardware contract/facts")
    snapshots = [row for row in evidence.source_snapshots if row.role == "software-spec"]
    if len(snapshots) != 1 or snapshots[0].sha256 != (profile.get("_software_spec_identity") or {}).get("sha256"):
        raise ValueError("component declaration is not bound to the selected software source bytes")
    if evidence.application_inventory or evidence.whole_program_admission:
        raise ValueError("component generation refuses application/capture evidence")
    owners = {row["id"]: row for row in spec["operations"] if row.get("status", "reviewed") == "reviewed"}
    rows = declaration["objectives"]
    if not isinstance(rows, list):
        raise ValueError("component objectives must be an explicit list")
    families = {(entry.get("performance") or {}).get("family") for entry in entries}
    selected = {}
    provenance = (
        f"software-spec:{snapshots[0].sha256}",
        f"component-declaration:{digest(declaration)}",
        *(f"{key}:{value}" for key, value in hardware.items()),
    )
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"family", "operations", "objective"}:
            raise ValueError("component objective requires exactly family, operations and objective")
        family, operations, objective = row["family"], row["operations"], row["objective"]
        if not isinstance(family, str) or family not in families or family in selected:
            raise ValueError("component objective names an unknown or duplicate generated family")
        if (
            not isinstance(operations, list)
            or not operations
            or any(not isinstance(name, str) or name not in owners for name in operations)
            or len(set(operations)) != len(operations)
        ):
            raise ValueError("component objective must name unique reviewed software operation declarations")
        if not isinstance(objective, dict) or set(objective) != _OBJECTIVE_FIELDS:
            raise ValueError("component objective needs only metric, unit, direction and basis; provenance is derived")
        typed = phase_policy.PerformanceObjective(
            **objective,
            provenance=(*provenance, *(f"operation:{name}:{digest(owners[name])}" for name in operations)),
        )
        selected[family] = {"operations": list(operations), "objective": typed.to_dict()}
    source_paths = [*Path(__file__).parent.glob("*.py")]
    source_paths += [Path(owner.__file__) for owner in (corpus_spec, phase_policy, software_spec)]
    identity = {
        "schema": SCHEMA,
        "status": "generated_from_reviewed_component_declarations",
        "scope": "independent_components_no_application_inputs",
        "evidence_status": evidence.status,
        "whole_coverage_verified": False,
        "timing_verified": False,
        "hardware": hardware,
        "software_spec_sha256": snapshots[0].sha256,
        "declaration_sha256": digest(declaration),
        "recipe": _source(recipe),
        "shared_template": _source(performance_template),
        "generator_sources": [_source(path) for path in sorted(source_paths)],
        "families": selected,
    }
    bound = []
    for entry in entries:
        if (
            entry.get("cat") != "_perf"
            or entry.get("label") != "dev"
            or entry.get("source_role") != "derived_sweep"
            or entry.get("kind") == "model"
            or any(entry.get(key) for key in _CAPTURE_FIELDS)
            or (entry.get("performance") or {}).get("global_objective")
        ):
            raise ValueError("component generation admits only independent derived dev sweeps")
        value = copy.deepcopy(entry)
        performance = value["performance"]
        if "component_generation_sha256" in performance or "objective" in performance:
            raise ValueError("component objective identity is generated, never authored by the sweep")
        performance["component_generation_sha256"] = digest(identity)
        family = performance["family"]
        if family in selected:
            op = entry.get("op")
            semantic_family = from_op(op or "")
            declarations = [owners[name] for name in selected[family]["operations"]]
            if not any(
                op in row.get("ops", []) or (semantic_family is not None and semantic_family in row.get("families", []))
                for row in declarations
            ):
                raise ValueError("component objective owner does not declare the generated operation/family")
            performance["objective"] = copy.deepcopy(selected[family]["objective"])
        bound.append(value)
    return bound, identity


def require_written(capsule):
    """Keep unknown work and unreviewed placement out of the component corpus."""
    screen = capsule.get("software_screen") or {}
    if screen.get("status") != "admitted":
        raise ValueError("component generated program lacks reviewed concrete software/hardware admission")
    objective = (capsule.get("performance") or {}).get("objective")
    typed = None
    if objective is not None:
        value = dict(objective)
        value["provenance"] = tuple(value["provenance"])
        typed = phase_policy.PerformanceObjective(**value)
    verdict = phase_policy.priceable(capsule, performance_objective=typed)
    if verdict.value != phase_policy.YES:
        raise ValueError("component work admission refused: " + verdict.reason)
