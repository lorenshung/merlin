"""Spike functional oracle for the Core ATen full-call capsule contract.

The adapter executes the *submitted* LLVM-dialect emission, never a trusted lowering: the
candidate text is translated to LLVM IR and replaces the lowered module inside the normal
bare-metal image build (``host_llvm_transform``).  The bundle supplies the call boundary
(``semantic_io.json``, ``inputs.npz``); the return value carries lossless output bytes and the
semantic readback for the round-2 full-call grader.

The ISA, simulator options and optional extension come from the caller (the target contract or
the provider), not from this module, so any plain-ISA Spike target can share it.
"""

from __future__ import annotations

import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_ARENA_MB = 256


def _translate_candidate(llvm_mlir: str):
    """A ``host_llvm_transform`` that returns the candidate, translated to LLVM IR."""

    def transform(_trusted: Path, workdir: Path) -> Path:
        from merlin.llvmlower import toolchain

        source = Path(workdir) / "candidate.mlir"
        source.write_text(llvm_mlir, encoding="utf-8")
        out = Path(workdir) / "candidate.ll"
        done = subprocess.run(
            [str(toolchain.mlir_translate()), "--mlir-to-llvmir", str(source), "-o", str(out)],
            capture_output=True,
            text=True,
        )
        if done.returncode != 0 or not out.is_file():
            raise ValueError(f"candidate emission is not translatable LLVM-dialect MLIR: {done.stderr[-800:]}")
        return out

    return transform


def full_call_adapter(
    target: str,
    *,
    isa: str,
    backend: str = "scalar",
    runner_options: Mapping[str, Any] | None = None,
    engine: str = "spike",
):
    """Build an oracle callable exposing ``run_full_call`` for one plain Spike target.

    ``isa`` is the Spike ISA string (derived from the target contract by the caller).
    ``runner_options`` may select ``spike_binary``, ``extension``, ``extlib`` and ``path_prepend``.
    """
    options = dict(runner_options or {})

    def run_full_call(*, bundle, llvm_mlir, command_buffer=None, target=target, timeout=600, **_ignored):
        from merlin.runtime.backends import spike_model
        from merlin.targetgen.core_aten_provenance import batch_provenance

        bundle = Path(bundle)
        result: dict[str, Any] = {"engine": engine, "derived_from_rtl": False}
        try:
            build = spike_model.build(
                bundle,
                bundle / "spike-build",
                inputs_npz=bundle / "inputs.npz",
                arena_mb=DEFAULT_ARENA_MB,
                dump_all_outputs=True,
                backend=backend,
                host_llvm_transform=_translate_candidate(llvm_mlir),
            )
            run = spike_model.run(
                build["elf"], mem_bytes=build["mem_bytes"], timeout=timeout, vlen=build.get("vlen"), isa=isa, **options
            )
            if build.get("build_hash") is not None and run.get("metrics", {}).get("build_hash") != build["build_hash"]:
                raise RuntimeError("console build hash does not match the linked executable")
            result.update(
                output_bytes=run["output_bytes"],
                output_shapes=run.get("output_shapes") or None,
                semantic_readback=run.get("semantic_readback"),
                execution_error=None,
            )
        except Exception as exc:  # noqa: BLE001 -- a failed execution is a verdict input, not a crash
            result.update(
                output_bytes=None,
                output_shapes=None,
                semantic_readback=None,
                execution_error=f"{type(exc).__name__}: {exc}"[:2000],
            )
        provenance = batch_provenance(bundle, runner_options=options)
        provenance["isa"] = isa
        provenance["target"] = target
        provenance["candidate_llvm_sha256"] = hashlib.sha256(llvm_mlir.encode()).hexdigest()
        result["provenance"] = provenance
        return result

    def run(cb, llvm_text, workdir, timeout):
        from merlin.targetgen.capsule_runner import OracleUnavailable

        raise OracleUnavailable("this oracle executes the full-call boundary only (run_full_call)")

    run.run_full_call = run_full_call
    run._merlin_simulator_engine = engine
    return run


def _contract_isa(target: str) -> str:
    """The Spike ISA string is a declared target fact (``isa.march``); never defaulted here."""
    from merlin.targetgen.target_experiment import load_capability_manifest

    isa = (load_capability_manifest(target).contract.get("isa") or {}).get("march")
    if not isinstance(isa, str) or not isa.strip():
        raise ValueError(f"target contract declares no isa.march for {target!r}")
    return isa.strip()


def sim_adapters(target: str) -> dict:
    """Oracle factory for a plain-ISA Spike target: the functional (L2) tier only."""
    return {"L2": full_call_adapter(target, isa=_contract_isa(target))}


def sim_available(target: str) -> tuple[bool, str]:
    from merlin.runtime.backends.spike import spike_path

    binary = Path(spike_path())
    if not binary.is_file():
        return False, f"{target!r}: spike binary is absent: {binary}"
    try:
        isa = _contract_isa(target)
    except Exception as exc:  # noqa: BLE001 -- unreadable contract means not gradeable
        return False, f"{target!r}: {exc}"
    return True, f"{target!r}: spike functional oracle available (isa {isa})"
