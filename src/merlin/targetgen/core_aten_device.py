"""Provider-selected offload and executed-instruction evidence for Core ATen shards."""

from __future__ import annotations

import hashlib
import json
import os
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root
from merlin.targetgen.plugins import load_module
from merlin.targetgen.rtl.facts import ensure_facts, load_facts, observed_facts


def load_execution_provider(target: str, path: Path | None = None):
    """Select target-edge support explicitly; no compiler policy lives in this adapter."""
    if not target or Path(target).name != target or target in {".", ".."}:
        raise ValueError("target must be a single name")
    selected = path or repo_root() / "examples" / target / "phase0" / "core_aten" / "execution_provider.py"
    module = load_module(
        selected.parent,
        selected.name,
        package_name="core_aten_execution_" + hashlib.sha256(str(selected.parent.resolve()).encode()).hexdigest(),
    )
    if getattr(module, "TARGET", None) != target:
        raise ValueError("execution provider does not declare the selected target")
    return module


@contextmanager
def selected_facts(target: str, explicit: Path | None):
    """Pin nested consumers and lowering subprocesses to the same selected facts file."""
    path = ensure_facts(target, explicit=explicit)
    document = load_facts(target, explicit=path)
    declared = document.get("facts", {}).get("target", document.get("target"))
    if declared is not None and declared != target:
        raise ValueError("selected facts name a different target")
    previous = os.environ.get("MERLIN_RTL_FACTS")
    os.environ["MERLIN_RTL_FACTS"] = str(path.resolve())
    try:
        with observed_facts(target, document, path):
            yield document
    finally:
        if previous is None:
            os.environ.pop("MERLIN_RTL_FACTS", None)
        else:
            os.environ["MERLIN_RTL_FACTS"] = previous


def routing_for_bundle(directory: Path, target: str, package: Path, provider, facts: dict):
    """Report structural refusals, then ask the provider to bind supported source operations."""
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.kernels.shapes import observe_contractions, zero_initialised
    from merlin.system.offload import device_contraction_ranks, facts_dtype_triples, why_not

    module = parse_mlir_file(directory / "model.mlir")
    triples, ranks = facts_dtype_triples(facts), device_contraction_ranks(target)
    eligible, declined = [], []
    for op, shape in observe_contractions(module):
        reason = why_not(shape, triples=triples, ranks=ranks)
        if reason is None and not zero_initialised(op):
            reason = "contraction destination is not proven zero initialized"
        if reason:
            declined.append(reason)
        else:
            eligible.append((op, shape))
    if not eligible:
        return None, "; ".join(declined) or "no eligible contraction in captured source"
    routing = provider.routing(directory, target=target, package=package, facts=facts, eligible=eligible)
    if routing is None:
        return None, "provider declines captured contraction semantics"
    if routing.device != target or Path(routing.package_dir).resolve() != package.resolve():
        raise ValueError("provider routing differs from selected target/package")
    if routing.select is not None or (routing.catalog_builder is None and routing.exact_selection is None):
        raise ValueError("batch device mode requires source-bound catalog or exact selection")
    return routing, "provider selected source-bound contractions"


def submitted_catalog_routing(directory, *, target, package, facts, eligible):
    """Bind a submitted catalog to the precision of the selected facts and source."""
    from merlin.llvmlower import toolchain
    from merlin.llvmlower.device_build import DeviceRouting
    from merlin.system.offload import facts_dtype_triples

    triples = {tuple(shape.dtypes) for _, shape in eligible}
    if len(triples) != 1 or not triples.issubset(set(facts_dtype_triples(facts))):
        raise ValueError("ambiguous or underivable device datapath")
    operand, weight, accum = next(iter(triples))
    if operand != weight:
        raise ValueError("catalog requires identical operand storage types")
    # Immutable snapshots at different roots must not share cached Python modules.
    backend = load_module(
        package,
        "mlir_oot.golden_device_catalog",
        package_name="core_aten_backend_" + hashlib.sha256(str(Path(package).resolve()).encode()).hexdigest(),
    )
    source = (Path(directory) / "model.mlir").read_text()
    _, inventory = backend.build_catalog(source)
    if not inventory["covered_contractions"]:
        return None
    return DeviceRouting(
        device=target,
        package_dir=package,
        operand_dtype=operand,
        accum_dtype=accum,
        catalog_builder=backend.merlin_builder(toolchain.llvm_install() / "bin"),
    )


def executed_device_instructions(trace: Path, elf: Path, target: str) -> dict[str, Any]:
    """Count retired-log instruction words in executable ELF ranges; static presence is insufficient."""
    from merlin.perf.isa_prohibition import MAJOR_OPCODE_MASK
    from merlin.targetgen.elf_lanes import accelerator_opcode, executable_sections

    opcode, source = accelerator_opcode(target)
    if opcode is None:
        raise ValueError(source)
    blob = elf.read_bytes()
    sections = executable_sections(blob)
    if not sections:
        raise ValueError("ELF has no executable sections")
    count = 0
    digest = hashlib.sha256()
    with trace.open("rb") as stream:
        for raw in stream:
            digest.update(raw)
            fields = raw.decode("utf-8", errors="replace").split()
            # Spike -l: core <hart>: <pc> (<instruction word>) <disassembly>.
            if len(fields) < 4 or fields[0] != "core" or not fields[1].endswith(":"):
                continue
            if not fields[3].startswith("(0x") or not fields[3].endswith(")"):
                continue
            try:
                pc, word = int(fields[2], 16), int(fields[3][1:-1], 16)
            except ValueError:
                continue
            if word & MAJOR_OPCODE_MASK != opcode:
                continue
            for _, offset, size, address in sections:
                if address <= pc and pc + 4 <= address + size:
                    at = offset + pc - address
                    # Match the actual instruction at the logged PC, not arbitrary trace text.
                    byteorder = "little" if blob[5] == 1 else "big"
                    if int.from_bytes(blob[at : at + 4], byteorder) != word:
                        raise ValueError("trace word differs from linked executable at logged PC")
                    count += 1
                    break
    facts_path = ensure_facts(target)
    return {
        "status": "measured",
        "facts_path": str(facts_path.resolve()),
        "facts_sha256": hashlib.sha256(facts_path.read_bytes()).hexdigest(),
        "elf": str(elf.resolve()),
        "facts_provenance": {key: value for key, value in load_facts(target).items() if key != "facts"},
        "target": target,
        "opcode": opcode,
        "opcode_source": source,
        "executed_instructions": count,
        "trace": str(trace),
        "trace_sha256": digest.hexdigest(),
        "elf_sha256": hashlib.sha256(blob).hexdigest(),
        "attribution": "single_case_shard",
    }


def verify_execution_evidence(evidence: dict) -> bool:
    """Regrade retained evidence, refusing deleted/changed traces, executables or fact pins."""
    try:
        facts_path = Path(evidence["facts_path"])
        if hashlib.sha256(facts_path.read_bytes()).hexdigest() != evidence["facts_sha256"]:
            return False
        with selected_facts(evidence["target"], facts_path):
            actual = executed_device_instructions(Path(evidence["trace"]), Path(evidence["elf"]), evidence["target"])
        return (
            all(
                actual[key] == evidence.get(key)
                for key in ("opcode", "trace_sha256", "elf_sha256", "executed_instructions", "attribution")
            )
            and actual["executed_instructions"] > 0
        )
    except (KeyError, OSError, ValueError):
        return False


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def run_spike_bundle(
    directory: Path,
    report: dict,
    *,
    arena_mb: int,
    timeout: int,
    target: str | None = None,
    device_package: Path | None = None,
    execution_provider: Path | None = None,
    rtl_facts: Path | None = None,
) -> tuple[dict, dict]:
    from merlin.llvmlower.device_offload import load_sidecar
    from merlin.runtime.backends import spike_model
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    output_bytes = None
    output_shapes = None
    semantic_readback = None
    options = {}
    try:
        bundled = [case for case in report["cases"] if case["status"] == "bundled"]
        if target:
            for case in bundled:
                case["routing"] = {
                    "lane": "host",
                    "routed": False,
                    "reason": "device routing not completed",
                    "target": target,
                }
        if target and (device_package is None or len(bundled) != 1):
            raise ValueError("device execution requires a package and exactly one case per shard")
        with selected_facts(target, rtl_facts) if target else nullcontext(None) as facts:
            routing, reason, options = None, "scalar mode", {}
            if target:
                provider = load_execution_provider(target, execution_provider)
                prepare = getattr(provider, "prepare_bundle", None)
                if prepare is not None:
                    preparation = prepare(directory, target=target, facts=facts)
                    report["source_preparation"] = preparation
                    bundled[0]["source_preparation"] = preparation
                routing, reason = routing_for_bundle(directory, target, device_package, provider, facts)
                options = provider.runner_options()
            for case in bundled:
                case["routing"] = {
                    "lane": "device" if routing is not None else "host",
                    "routed": False,
                    "reason": reason,
                    "target": target,
                }
            build = spike_model.build(
                directory,
                directory / "spike-build",
                inputs_npz=directory / "inputs.npz",
                arena_mb=arena_mb,
                dump_all_outputs=True,
                backend="scalar",
                device=routing,
            )
            sidecar = load_sidecar(directory / "spike-build") if routing is not None else {}
            if routing is not None and not sidecar.get("routed"):
                raise RuntimeError("selected device routing emitted no source-bound calls")
            if routing is not None:
                bundled[0]["routing"].update(routed=True, routed_operations=sidecar["routed"])
            trace = directory / "spike-trace.log" if routing is not None else None
            run = spike_model.run(
                build["elf"],
                mem_bytes=build["mem_bytes"],
                timeout=timeout,
                vlen=build.get("vlen"),
                trace_path=trace,
                **options,
            )
            if build.get("build_hash") is not None and run.get("metrics", {}).get("build_hash") != build["build_hash"]:
                raise RuntimeError("console build hash does not match the linked executable")
            if run.get("metrics", {}).get("memref_rank_mismatch", 0) != 0:
                raise RuntimeError("runtime refused a memref copy with mismatched ranks")
            if routing is not None:
                evidence = executed_device_instructions(trace, Path(build["elf"]), target)
                evidence["case_id"] = str(bundled[0].get("case_id") or bundled[0]["overload"])
                bundled[0]["routing"] = {
                    "lane": "device",
                    "routed": True,
                    "reason": reason,
                    "target": target,
                    "routed_operations": sidecar["routed"],
                    "execution_evidence": evidence,
                }
                if evidence["executed_instructions"] < 1:
                    raise RuntimeError("device calls routed but no target instruction executed")
        output_bytes = run["output_bytes"]
        if len(output_bytes) != report["output_count"]:
            raise RuntimeError(f"hardware emitted {len(output_bytes)} outputs; bundle expects {report['output_count']}")
        (directory / "spike-console.txt").write_text(run["console"], encoding="utf-8")
        semantic_readback = run.get("semantic_readback")
        if semantic_readback is not None:
            _write_json(directory / "spike-semantic-readback.json", semantic_readback)
        output_shapes = run.get("output_shapes") or None
        _write_json(directory / "spike-output-shapes.json", output_shapes)
        _write_json(directory / "spike-output-bytes.json", [item.hex() for item in output_bytes])
        execution = {
            "status": "ran",
            "elf": str(build["elf"]),
            "build_hash": build.get("build_hash"),
            "output_count": len(output_bytes),
        }
    except Exception as exc:  # each stage failure remains an explicit artifact
        execution = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        for case in report["cases"]:
            if case["status"] == "bundled":
                case["routing"]["failure_reason"] = execution["error"]
    from merlin.targetgen.core_aten_provenance import batch_provenance

    report["provenance"] = batch_provenance(
        directory, target=target, package=device_package, runner_options=options, rtl_facts=rtl_facts
    )
    execution["provenance"] = report["provenance"]
    _write_json(directory / "core_aten_batch_map.json", report)
    _write_json(directory / "spike-execution.json", execution)
    verdict = grade_core_aten_batch(
        report,
        output_bytes,
        output_shapes=output_shapes,
        semantic_readback=semantic_readback,
        execution_error=execution.get("error") if execution["status"] != "ran" else None,
    )
    for case_verdict in verdict["cases"].values():
        observation = case_verdict.get("observation")
        if observation is not None:
            observation["evidence"].update(
                target=target,
                lane=case_verdict["lane"],
                receipt=execution.get("build_hash"),
                elf=execution.get("elf"),
                artifact_digests=report["provenance"]["artifact_digests"],
            )
    _write_json(directory / "core_aten_batch_verdict.json", verdict)
    return execution, verdict
