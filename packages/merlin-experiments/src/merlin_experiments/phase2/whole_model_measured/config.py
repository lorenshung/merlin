"""The objective config a ``whole_model_measured`` launch declares, and the objective it builds.

Everything a measurement is attributed through is named here, as data the run keeps (read-only)
beside its record: the builder (and its pin), each machine, each machine's build options, each
machine's reference result, where the store lives, and the experiment's instruction POLICY.  A
config that is underdetermined is refused AT LAUNCH -- not at the first comparison an hour in.

THE POLICY IS CHECKED, NOT TRUSTED.  ``prohibited_instruction_roles`` is the experiment's declared
rule (``policy.prohibited_instruction_roles`` in the experiment definition).  Every CANDIDATE section
(the screen, the certifier and every held-out model) must carry exactly those roles in its build
options, because the build option is what the builder routes by and what the whole-ELF gate checks:
a relaunch that silently dropped them once ran a no-FSM campaign without its prohibition.  The
reference arms are measured without the rule by construction (they are the bar).

A section's ``machine`` is either a full spec or a registry reference::

    machine: {registry: /abs/whole-model-machines.yaml, name: <machine>, overrides: {...}}

resolved here, once, so the spec a job records -- and the store's key -- is the resolved data.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.perf import whole_model_builder

from . import capabilities as CAP
from . import gates as G
from . import registry as R
from .identity import builder_identity, store_root_for
from .objective import WholeModelObjective
from .service import MeasurementService

CONFIG_SCHEMA = "merlin_whole_model_objective_config_v1"
DERIVED_MECHANISMS = "derive_from_capture"
#: The production builder (the core's service builder over the whole-model build); a config may name
#: another (a test double, a second target's).  Imported, not only named, so the dependency is real.
DEFAULT_BUILDER = f"{whole_model_builder.__name__}:{whole_model_builder.build.__name__}"
DEFAULT_REFERENCE_BUILDER = f"{whole_model_builder.__name__}:{whole_model_builder.build_reference.__name__}"
CANDIDATE_SECTIONS = ("screen", "certifier")


class ConfigError(ValueError):
    """The objective config is underdetermined or inconsistent."""


def resolve_machine(section: Mapping[str, Any], *, environment: Mapping[str, str] | None = None) -> dict[str, Any]:
    """A section's machine as a full spec (a registry reference resolved; a full spec returned as is)."""
    machine = dict(section.get("machine") or {})
    if "registry" in machine:
        if set(machine) - {"registry", "name", "overrides"}:
            raise ConfigError(f"a registry machine reference takes registry, name and overrides; got {sorted(machine)}")
        return R.resolve(
            machine["registry"],
            str(machine.get("name") or ""),
            environment=environment,
            overrides=machine.get("overrides"),
        )
    return machine


def declared_roles(document: Mapping[str, Any]) -> list[str]:
    roles = document.get("prohibited_instruction_roles")
    if roles is None:
        return []
    if not isinstance(roles, list) or any(not isinstance(r, str) or not r for r in roles):
        raise ConfigError("prohibited_instruction_roles must be a list of role names")
    return list(roles)


def check_policy(document: Mapping[str, Any]) -> list[str]:
    """The declared roles, after refusing any candidate section whose build options disagree."""
    roles = declared_roles(document)
    sections = [(name, document.get(name)) for name in CANDIDATE_SECTIONS if document.get(name)]
    sections += [(f"held_out.{name}", body) for name, body in (document.get("held_out") or {}).items()]
    for name, section in sections:
        carried = list(((section or {}).get("build_options") or {}).get(G.PROHIBITED_ROLES) or ())
        if sorted(carried) != sorted(roles):
            raise ConfigError(
                f"the {name} section's build options carry prohibited roles {carried!r} but the experiment "
                f"declares {roles!r}; the rule a build is shaped by and the rule it is judged by must agree"
            )
    return roles


def with_policy(document: Mapping[str, Any], roles: list[str]) -> dict[str, Any]:
    """``document`` with the declared ``roles`` written into every candidate section's build options."""
    out = json.loads(json.dumps(dict(document), default=str))
    out["prohibited_instruction_roles"] = list(roles)
    for name in CANDIDATE_SECTIONS:
        if out.get(name):
            options = dict(out[name].get("build_options") or {})
            if roles:
                options[G.PROHIBITED_ROLES] = list(roles)
            else:
                options.pop(G.PROHIBITED_ROLES, None)
            out[name]["build_options"] = options
    for body in (out.get("held_out") or {}).values():
        options = dict(body.get("build_options") or {})
        if roles:
            options[G.PROHIBITED_ROLES] = list(roles)
        body["build_options"] = options
    return out


def prepare_document(document: Mapping[str, Any], *, target: str) -> dict[str, Any]:
    """Resolve output location and candidate mechanisms from selected model inputs.

    The model's host/accelerator closure is computed by the same target-neutral
    analysis used by the builder. A closed model can exercise package-declared
    passes and regions; an open model cannot. The decision is frozen into the
    run config, never inferred from a target name or a hand-authored kernel.
    """
    from merlin.common.paths import artifacts_dir

    out = json.loads(json.dumps(dict(document), default=str))
    if not out.get("store"):
        if not target or target in (".", "..") or "/" in target or "\\" in target:
            raise ConfigError(f"invalid target name for a generated store: {target!r}")
        out["store"] = str((artifacts_dir() / "perf-studies" / "whole-model" / target).resolve())
    policy = out.get("mechanism_policy")
    if policy is None:
        if "mechanism_derivation" in out:
            raise ConfigError("mechanism_derivation is a preparation receipt, not an operator declaration")
        return out
    if policy != DERIVED_MECHANISMS:
        raise ConfigError(f"unknown mechanism_policy {policy!r}")

    from merlin.perf.whole_model_open import is_open_model

    decisions = {}
    sections = [(name, out.get(name)) for name in CANDIDATE_SECTIONS]
    sections += [(f"held_out.{name}", body) for name, body in (out.get("held_out") or {}).items()]
    for name, section in sections:
        if not section:
            continue
        options = dict(section.get("build_options") or {})
        capsule = options.get("model_capsule")
        if not isinstance(capsule, str) or not Path(capsule).is_dir():
            raise ConfigError(f"{name} needs a frozen model_capsule directory to derive mechanisms")
        open_model = is_open_model(capsule, target)
        enabled = not open_model
        for key in ("allow_passes", "allow_regions"):
            if key in options and options[key] is not enabled:
                raise ConfigError(f"{name}.{key} contradicts the selected model's derived closure")
            options[key] = enabled
        section["build_options"] = options
        decisions[name] = {
            "model_closure": "open" if open_model else "closed",
            "allow_passes": enabled,
            "allow_regions": enabled,
        }
    out["mechanism_derivation"] = {"source": "merlin.perf.whole_model_open.is_open_model", "sections": decisions}
    return out


def store_roots(document: Mapping[str, Any], *, environment: Mapping[str, str] | None = None) -> dict[str, Path]:
    """``{section: store root}`` a config's sections own, computed exactly as :func:`from_config` does."""
    builder = dict(document.get("builder") or {"spec": DEFAULT_BUILDER, "sha256": None})
    identity = builder_identity(str(builder["spec"]), builder.get("sha256"))
    base = Path(str(document.get("store") or ""))
    roots = {}
    for name in CANDIDATE_SECTIONS:
        section = document.get(name)
        if section:
            roots[name] = store_root_for(
                base,
                builder=builder,
                machine=resolve_machine(section, environment=environment),
                build_options=dict(section.get("build_options") or {}),
                builder_sha256=identity["sha256"],
            )
    return roots


def from_config(
    document: Mapping[str, Any], *, target: str, environment: Mapping[str, str] | None = None
) -> WholeModelObjective:
    """Build the objective a launch declares, refusing anything underdetermined AT LAUNCH."""
    if not isinstance(document, Mapping) or document.get("schema") != CONFIG_SCHEMA:
        raise ConfigError(f"a whole-model objective config must declare schema {CONFIG_SCHEMA}")
    builder = dict(document.get("builder") or {"spec": DEFAULT_BUILDER, "sha256": None})
    if not builder.get("spec"):
        raise ConfigError("the objective config names no builder")
    base = Path(str(document.get("store") or ""))
    if not base.is_absolute():
        raise ConfigError("the objective config's store must be an absolute path")
    if not document.get("screen"):
        raise ConfigError("the objective config declares no screen section")
    check_policy(document)
    identity = builder_identity(str(builder["spec"]), builder.get("sha256"))
    env = {str(k): str(v) for k, v in (document.get("environment") or {}).items()}

    def root_of(section: Mapping[str, Any], machine: Mapping[str, Any]) -> Path:
        return store_root_for(
            base,
            builder=builder,
            machine=machine,
            build_options=dict(section.get("build_options") or {}),
            builder_sha256=identity["sha256"],
        )

    certifier_root = None
    if document.get("certifier"):
        certifier_root = root_of(document["certifier"], resolve_machine(document["certifier"], environment=environment))

    def service(section: Mapping[str, Any], *, is_screen: bool) -> tuple[MeasurementService, Path | None]:
        machine = resolve_machine(section, environment=environment)
        if machine.get("target") != target:
            raise ConfigError(f"machine target {machine.get('target')!r} is not this run's {target!r}")
        R.validate_limits((machine.get("timing") or machine).get("cannot_express"), where="machine")
        reference = Path(str(section["reference"])) if section.get("reference") else None
        return (
            MeasurementService(
                root_of(section, machine),
                target=target,
                builder=str(builder["spec"]),
                builder_sha256=builder.get("sha256"),
                machine=machine,
                slots=int(section.get("slots") or 1),
                max_pending=int(section.get("max_pending") or 3),
                timeout_seconds=float(section.get("timeout_seconds") or 6 * 3600),
                python=document.get("python"),
                environment=env,
                build_options=dict(section.get("build_options") or {}),
                reference=reference,
                excuse_reference_failures=bool(section.get("excuse_reference_failures")),
                pre_measure_check=document.get("pre_measure_check"),
                retain=document.get("retain"),
                certifier_root=certifier_root if is_screen else None,
                min_build_free_bytes=section.get("min_build_free_bytes"),
                machine_capabilities=CAP.compact(CAP.section_report(section, environment=environment)),
            ),
            reference,
        )

    screen, screen_reference = service(document["screen"], is_screen=True)
    certifier, certifier_reference = (None, None)
    if document.get("certifier"):
        certifier, certifier_reference = service(document["certifier"], is_screen=False)
    # TRANSFER CHECK, OPT-IN: one measurement service per held-out model; absent, nothing changes.
    held_out = {name: service(section, is_screen=False) for name, section in (document.get("held_out") or {}).items()}
    objective = WholeModelObjective(
        screen=screen,
        screen_reference=screen_reference,
        certifier=certifier,
        certifier_reference=certifier_reference,
        repeats_on_best=int(document.get("repeats_on_best") or 2),
        primary_name=str(document.get("primary_name") or ""),
        held_out=held_out,
    )
    objective.config = json.loads(json.dumps(dict(document), default=str))
    return objective


__all__ = [
    "CONFIG_SCHEMA",
    "ConfigError",
    "DEFAULT_BUILDER",
    "DEFAULT_REFERENCE_BUILDER",
    "check_policy",
    "declared_roles",
    "from_config",
    "resolve_machine",
    "store_roots",
    "with_policy",
]
