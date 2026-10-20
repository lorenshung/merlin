"""Target-bound, operator-selected Phase 1 oracle timing input.

This record is produced only after a real cert-tier observation. Its path is a
resource owned by the selected target, never an alias through shared scripts.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path

#: Elaborated-RTL engines whose Chipyard build can carry an oracle-timing observation. A record that
#: names no engine is a legacy Verilator record and keeps its exact historical fields.
TIMING_ENGINES = ("verilator", "vcs")


def timing_path(resources: Path, target: str) -> Path:
    if not target or not all(char.isascii() and (char.isalnum() or char in "_-") for char in target):
        raise ValueError("timing target must be a nonempty path-safe identity")
    return Path(resources) / f".oracle_timing.{target}.json"


def requires_chipyard_timing(descriptor: Path | None, *, public_root: Path | None = None) -> bool:
    """Require RTL timing when the selected Chipyard corpus requests an RTL tier.

    Functional-only capsules need no elaborated-RTL calibration. Missing or
    malformed corpus declarations retain the historical fail-closed requirement.
    Hidden requirements and an explicit public override participate equally.
    """
    if descriptor is None:
        return True
    from merlin.targetgen.target_experiment import load_target_experiment

    try:
        selected = load_target_experiment(descriptor)
        if selected.sim_via != "chipyard":
            return False
        roots = selected.graded_roots()
        hidden = selected.capsule_corpus.parent / "hidden"
        if hidden.exists() or hidden.is_symlink():
            roots.append(hidden)
        if public_root is not None:
            roots.append(Path(public_root))
        if not roots or any(not root.is_dir() or root.is_symlink() for root in roots):
            return True
        files = {path for root in roots for path in root.rglob("capsule.yaml")}
        if not files:
            return True
        import yaml

        for path in files:
            if any(p.is_symlink() for p in (path, *path.parents)):
                return True
            try:
                capsule = yaml.safe_load(path.read_text())
            except yaml.YAMLError:
                return True
            if not isinstance(capsule, dict):
                return True
            tiers = capsule.get("required_oracle_tiers")
            if (
                not isinstance(tiers, list)
                or not tiers
                or any(not isinstance(tier, str) or tier not in {"L0", "L1", "L2"} for tier in tiers)
            ):
                return True
        return False
    except (OSError, ValueError, AttributeError, TypeError):
        return True


def record_engine(record: dict) -> str:
    """The engine a record was measured on; an untagged record is the legacy Verilator observation."""
    engine = record.get("engine")
    return "verilator" if engine is None else engine


def per_capsule_seconds(record: dict) -> float:
    """The measured per-capsule L3 wall time, whichever engine produced it.

    Legacy Verilator records carry ``verilator_per_capsule_s``; engine-tagged records carry the
    engine-neutral ``per_capsule_s`` so a VCS observation is never filed under Verilator's name.
    """
    key = "verilator_per_capsule_s" if record_engine(record) == "verilator" else "per_capsule_s"
    return record[key]


def selected_simulator(engine: str, target: str, config: str) -> Path:
    """The Chipyard simulator binary an engine runs for ``config`` (existence is the caller's check).

    Verilator: ``<chipyard>/sims/verilator/simulator-chipyard.harness-<config>``.
    VCS: ``MERLIN_<TARGET>_SIMV`` when set (the convention ``heavy_oracles.vcs_simv`` and the provider's
    ``vcs_path`` share), else ``<chipyard>/sims/vcs/simv-chipyard.harness-<config>``.
    """
    from merlin.common.paths import ext_path, target_env_name

    if engine == "verilator":
        return ext_path("chipyard") / "sims" / "verilator" / f"simulator-chipyard.harness-{config}"
    if engine == "vcs":
        explicit = os.environ.get(target_env_name(target, "SIMV"), "").strip()
        if explicit:
            return Path(explicit)
        return ext_path("chipyard") / "sims" / "vcs" / f"simv-chipyard.harness-{config}"
    raise ValueError(f"no oracle-timing simulator for engine {engine!r}")


def daidir_sha256(simv: Path) -> str:
    """Digest of a VCS simv's runtime database ``<simv>.daidir``.

    The simv executable is a small loader; the compiled design lives in the daidir (``*_archive_*.so``,
    ``rmapats.so`` and the elaboration databases), so the simv's own bytes alone do not pin the RTL.
    sha256 over sorted ``<relpath>\0<sha256(file)>\n`` lines of every file under the daidir.
    """
    from merlin.common.digest import sha256_file

    root = Path(str(simv) + ".daidir")
    if not root.is_dir():
        raise ValueError(f"VCS simv has no runtime database beside it: {root}")
    digest = hashlib.sha256()
    for directory, dirs, files in os.walk(root):
        dirs.sort()
        for name in sorted(files):
            item = Path(directory) / name
            digest.update(f"{item.relative_to(root)}\0{sha256_file(item)}\n".encode())
    return digest.hexdigest()


def simulator_identity(engine: str, simulator: Path) -> dict:
    """The byte identity a record binds: the binary's sha256, plus the daidir digest for VCS."""
    from merlin.common.digest import sha256_file

    identity = {"simulator_sha256": sha256_file(simulator)}
    if engine == "vcs":
        identity["simulator_daidir_sha256"] = daidir_sha256(simulator)
    return identity


def _is_sha256(value) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def read_timing(path: Path, *, target: str) -> dict:
    """Read a measured record, refusing aliases and foreign/invalid observations."""
    path = Path(path).absolute()
    if any(part.is_symlink() for part in (path, *path.parents)):
        raise ValueError(f"oracle timing path contains a symlink: {path}")
    if not path.is_file():
        raise ValueError(f"oracle timing record is absent: {path}")
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"oracle timing record is unreadable: {path}") from exc
    if not isinstance(record, dict) or record.get("target") != target:
        raise ValueError(f"oracle timing record is not bound to target {target!r}: {path}")
    if not isinstance(record.get("config"), str) or not record["config"].strip():
        raise ValueError(f"oracle timing record has no simulator config: {path}")
    if not isinstance(record.get("measured_by"), str) or not record["measured_by"].strip():
        raise ValueError(f"oracle timing record has no measurement producer: {path}")
    engine = record_engine(record)
    if engine not in TIMING_ENGINES:
        raise ValueError(f"oracle timing record names an unsupported engine {engine!r}: {path}")
    if not _is_sha256(record.get("simulator_sha256")):
        raise ValueError(f"oracle timing record has no simulator byte identity: {path}")
    if engine == "vcs" and not _is_sha256(record.get("simulator_daidir_sha256")):
        raise ValueError(f"oracle timing record has no VCS daidir identity: {path}")
    try:
        seconds = per_capsule_seconds(record)
    except KeyError:
        seconds = None
    if isinstance(seconds, bool) or not isinstance(seconds, (int, float)) or not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"oracle timing record has no positive finite observation: {path}")
    return record


def read_verified_timing(path: Path, *, descriptor: Path, target: str) -> dict:
    """Match the observation to this target's selected Chipyard simulator bytes."""
    from merlin.common.digest import sha256_file
    from merlin.targetgen.target_experiment import (
        declared_vs_resolved_contract,
        load_capability_manifest,
        load_target_experiment,
    )

    if descriptor is None:
        raise ValueError("oracle timing has no selected target descriptor")
    record = read_timing(path, target=target)
    try:
        selected = load_target_experiment(descriptor)
        if selected.target != target or selected.sim_via != "chipyard":
            raise ValueError("selected descriptor does not declare this target's Chipyard simulator")
        _, contract_path, agreement = declared_vs_resolved_contract(selected)
        if agreement != "agree" or contract_path is None:
            raise ValueError(f"selected target contract is not agreed and resolvable: {agreement}")
        config = (load_capability_manifest(target, contract_path=contract_path).contract.get("runtime") or {}).get(
            "rtl_sim_config"
        )
        if not isinstance(config, str) or record["config"] != config:
            raise ValueError("oracle timing config differs from the selected target contract")
        engine = record_engine(record)
        required = os.environ.get("MERLIN_REQUIRED_RTL_ENGINE", "").strip()
        if required in TIMING_ENGINES and engine != required:
            raise ValueError(f"oracle timing was measured on {engine!r}, but the run requires {required!r}")
        simulator = selected_simulator(engine, target, config)
        if not simulator.is_file():
            raise ValueError(f"measured simulator binary is absent: {simulator}")
        if record["simulator_sha256"] != sha256_file(simulator):
            raise ValueError(f"measured simulator bytes changed: {simulator}")
        if engine == "vcs" and record["simulator_daidir_sha256"] != daidir_sha256(simulator):
            raise ValueError(f"measured VCS daidir bytes changed: {simulator}.daidir")
    except (OSError, KeyError) as exc:
        raise ValueError(f"cannot verify selected simulator for oracle timing: {exc}") from exc
    return record
