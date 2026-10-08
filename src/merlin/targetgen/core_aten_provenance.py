"""Source and selected hardware attribution for Core ATen verdicts."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from merlin.common import provenance
from merlin.common.paths import repo_root


def batch_provenance(
    directory: Path | None = None,
    *,
    target: str | None = None,
    package: Path | None = None,
    runner_options: dict[str, Any] | None = None,
    rtl_facts: Path | None = None,
) -> dict[str, Any]:
    from merlin.integrations.model2mlir import root as m2m_root
    from merlin.runtime.backends.spike import spike_path
    from merlin.targetgen.core_aten_capture import _git_revision
    from merlin.targetgen.provenance import declared_pins
    from merlin.targetgen.target_registry import load_contract

    names = declared_pins(target)
    if target:
        try:
            contract = load_contract(target)
        except (FileNotFoundError, KeyError):
            contract = {}
        # Explicit descriptor dependencies select the hardware configuration.
        # Registry target membership can also include alternative bitstreams;
        # use that broader fallback only when the descriptor states no pins.
        if "hardware_pins" in contract:
            names = tuple(str(name) for name in contract["hardware_pins"])
    pins = {name: provenance.verify(name) for name in names}
    sources = [
        repo_root() / "src/merlin" / path
        for path in (
            "targetgen/core_aten_batch.py",
            "targetgen/core_aten_semantics.py",
            "targetgen/_m2m_capture_worker.py",
            "targetgen/_capture_result_contract.py",
            "targetgen/capsule_source.py",
            "llvmlower/semantic_io.py",
            "runtime/semantic_readback.py",
            "targetgen/core_aten_batch_grade.py",
            "targetgen/core_aten_device.py",
            "targetgen/core_aten_provenance.py",
            "llvmlower/result_buffers.py",
            "llvmlower/pipeline.py",
            "llvmlower/c_runtime.py",
            "runtime/backends/spike_model.py",
        )
    ]
    sources.append(repo_root() / "merlin/runtime/baremetal/spike/model_main.c")
    options = runner_options or {}
    artifacts = {}
    simulator = Path(options.get("spike_binary") or spike_path())
    if simulator.is_file():
        artifacts["spike"] = simulator
    if options.get("extlib"):
        artifacts["extension"] = Path(options["extlib"])
    if target:
        from merlin.targetgen.rtl.facts import ensure_facts

        facts = ensure_facts(target, explicit=rtl_facts)
        sources.append(facts)
        artifacts["rtl_facts"] = facts
    if directory:
        for name in ("model.mlir", "inputs.npz", "semantic_io.json", "spike-build/model.elf"):
            path = directory / name
            if path.is_file():
                artifacts[name] = path
    return provenance.record(
        pins=pins,
        sources=sources,
        artifacts=artifacts,
        extra={
            "target": target,
            "substrate": "spike_functional",
            "model2mlir_commit": _git_revision(m2m_root()),
            "oot_commit": _git_revision(package) if package else None,
        },
    )
