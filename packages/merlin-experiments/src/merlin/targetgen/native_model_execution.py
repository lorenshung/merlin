"""Execute an emitted whole-model kernel on a frozen capsule's exact operands.

This is an evidence adapter, not a model compiler or a certification verdict.  The
caller supplies the candidate's already-emitted command buffer and LLVM-dialect
artifact; no reference compiler can substitute its own model implementation here.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Any


class NativeModelExecutionError(ValueError):
    """The frozen model, candidate artifact, and pointer ABI cannot be joined."""


def _digest(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "size_bytes": path.stat().st_size}


def _build_artifacts(output: Path) -> dict[str, dict[str, Any]]:
    """Pin partial compiler products too, so a failed build remains inspectable."""
    build = output / "build"
    if not build.is_dir() or build.is_symlink():
        return {}
    return {str(path.relative_to(build)): _digest(path)
            for path in sorted(build.rglob("*")) if path.is_file() and not path.is_symlink()}


def _read_pinned_payload(emission: Mapping[str, Any], role: str) -> bytes:
    pin = emission.get(role)
    if not isinstance(pin, Mapping) or not isinstance(pin.get("path"), str):
        raise NativeModelExecutionError(f"candidate {role} pin is absent")
    path = Path(pin["path"])
    if (not path.is_absolute() or path.is_symlink() or path.resolve() != path
            or not path.is_file()):
        raise NativeModelExecutionError(f"candidate {role} is not an ordinary absolute file")
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != pin.get("sha256") or len(raw) != pin.get("size_bytes"):
        raise NativeModelExecutionError(f"candidate {role} bytes differ from emitted pin")
    return raw


def independent_frozen_source_eligibility(capsule: Mapping[str, Any], *, target: str) -> dict[str, Any]:
    """Build only the trusted eligibility denominator from frozen model source.

    No host-dispatch compile, outline, tile ledger, or candidate placement is
    consulted. The router is used to enumerate demands; its *decisions* are
    not native-execution evidence. This can run even when the legacy model
    grade fails before producing a coverage certificate.
    """
    root_value = capsule.get("__dir__")
    interface_name = capsule.get("interface_mlir") or capsule.get("linalg_mlir")
    attrs = ((capsule.get("operation") or {}).get("attributes") or {})
    fmt = attrs.get("dtype")
    if (not isinstance(root_value, str) or not isinstance(interface_name, str)
            or not interface_name or Path(interface_name).is_absolute()
            or ".." in Path(interface_name).parts or not isinstance(fmt, str) or not fmt):
        raise NativeModelExecutionError("frozen model has no exact local interface and routing dtype")
    root = Path(root_value)
    if not root.is_dir() or root.is_symlink():
        raise NativeModelExecutionError("frozen model capsule directory is absent or indirect")
    source = root / interface_name
    if source.is_symlink() or not source.is_file():
        raise NativeModelExecutionError("frozen model interface is absent or indirect")
    raw = source.read_bytes()
    from merlin.targetgen.capsule_source import model_op_demands_checked
    from merlin.targetgen.routing import route_plan
    from merlin.targetgen.coverage_certificate import for_target
    from merlin.targetgen import target_registry
    import yaml

    source_text = raw.decode("utf-8")
    selected_contract = target_registry.resolve(target).capability_contract_path
    if (not selected_contract.is_absolute() or selected_contract.is_symlink()
            or not selected_contract.is_file()):
        raise NativeModelExecutionError("selected target capability contract is absent or indirect")
    contract_pin = _digest(selected_contract)
    contract = yaml.safe_load(selected_contract.read_text(encoding="utf-8"))
    if not isinstance(contract, dict) or contract.get("name") != target:
        raise NativeModelExecutionError("selected target capability contract names a different target")
    with target_registry.observed_contract(target, contract, source_path=selected_contract):
        demands = model_op_demands_checked(source_text, fmt)
        certificate = for_target(route_plan(demands, target), target,
                                 linalg_mlir=source_text, execution=None)
    if _digest(source)["sha256"] != hashlib.sha256(raw).hexdigest():
        raise NativeModelExecutionError("frozen source changed during independent eligibility census")
    if _digest(selected_contract) != contract_pin:
        raise NativeModelExecutionError("selected capability contract changed during eligibility census")
    if (certificate.get("target") != target
            or certificate.get("source_mlir_sha256") != hashlib.sha256(raw).hexdigest()
            or certificate.get("n_regions") != len(demands)):
        raise NativeModelExecutionError("independent source denominator lacks exact identity")
    certificate["source_inventory_scope"] = "frozen_source_only_no_legacy_execution"
    certificate["eligibility_capability_contract"] = contract_pin
    return certificate


def _logical_values(raw: bytes, *, tensor: str, dtype: str, shape: tuple[int, ...]) -> Any:
    """Decode only captured leaf bytes; the submitted ELF still computes every operation."""
    import numpy as np

    types = {"i1": "u1", "i8": "i1", "u8": "u1", "i16": "<i2", "u16": "<u2",
             "i32": "<i4", "u32": "<u4", "i64": "<i8", "u64": "<u8",
             "index": "<i8", "f16": "<f2", "f32": "<f4", "f64": "<f8", "bf16": "<u2"}
    if dtype not in types:
        raise NativeModelExecutionError(f"{tensor}: unsupported captured leaf dtype {dtype!r}")
    values = np.frombuffer(raw, dtype=np.dtype(types[dtype]))
    expected = math.prod(shape) if shape else 1
    if values.size != expected:
        raise NativeModelExecutionError(
            f"{tensor}: captured {values.size} element(s), ABI declares {expected} {dtype} element(s)")
    if dtype == "bf16":
        values = (values.astype(np.uint32) << 16).view(np.float32)
    return values.reshape(shape).tolist() if shape else values[0].item()


def _bind_inputs(command_buffer: Mapping[str, Any], capture_bundle: Path, *, target: str,
                 rtl_facts: str | Path | None = None, board_config: str | None = None):
    """Bind every read pointer by parsed source-argument index, never by heuristic order."""
    from merlin.targetgen.bundle_pack import plan
    from merlin.targetgen.capability_probes import tile_edge
    from merlin.targetgen.capture_source import capture_tensor_source

    manifest_path = capture_bundle / "weights.safetensors.manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        raise NativeModelExecutionError(f"frozen runtime bundle lacks ordinary manifest {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise NativeModelExecutionError("frozen weight manifest is not a mapping")
    # Prefer the explicitly selected RTL array's column count.  A named target
    # can otherwise resolve its own geometry through the existing capability
    # path, which refuses rather than guessing an unavailable fixed array.
    if rtl_facts is not None:
        facts_path = Path(rtl_facts)
        if facts_path.is_symlink() or not facts_path.is_file():
            raise NativeModelExecutionError("selected RTL facts must be an ordinary file")
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        if (facts.get("inputs") or {}).get("target") != target:
            raise NativeModelExecutionError("selected RTL facts name a different target")
        if board_config and ((facts.get("facts") or {}).get("source") or {}).get("config") != board_config:
            raise NativeModelExecutionError("selected RTL facts name a different board config")
        arrays = [row for row in ((facts.get("facts") or {}).get("arrays") or ())
                  if isinstance(row, dict) and row.get("name") == "mesh"]
        if len(arrays) != 1 or type(arrays[0].get("cols")) is not int or arrays[0]["cols"] <= 0:
            raise NativeModelExecutionError("selected RTL facts have no unique positive mesh column count")
        pitch = arrays[0]["cols"]
    else:
        pitch = tile_edge(target)
    packed = plan(command_buffer, row_pitch_elements=pitch, weight_manifest=manifest)
    source, source_report = capture_tensor_source(capture_bundle, packed, weight_manifest=manifest)
    inputs: dict[str, Any] = {}
    for row in packed.const:
        if row.role != "argument":
            raise NativeModelExecutionError("carried model state needs an explicit session execution contract")
        if not row.weight:
            raise NativeModelExecutionError(f"{row.tensor}: no frozen source key in weight manifest")
        inputs[row.tensor] = _logical_values(
            source(row.weight), tensor=row.tensor, dtype=row.dtype, shape=row.shape)
    read = [str(arg["tensor"]) for arg in command_buffer["kernel_abi"]["args"]
            if arg["access"] in ("read", "readwrite")]
    if set(inputs) != set(read) or len(inputs) != len(read):
        raise NativeModelExecutionError("captured leaf binding does not cover the exact read-pointer ABI")
    return inputs, {"pack": packed.to_dict(), "source": source_report.to_dict(),
                    "weight_manifest": _digest(manifest_path)}


def _build_service_for(target: str):
    """Pin the host-selected target renderer and bypass the model importer for finished LLVM."""
    from merlin.runtime.backends import base as backends
    from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest

    backend = backends.get_backend(target)
    root = Path(backend.__file__).resolve().parent
    providers = list(root.rglob("*.py"))
    extra = backend.build_source_paths() if callable(getattr(backend, "build_source_paths", None)) else ()
    declared = [*providers, *map(Path, extra)]
    if not declared or any(not p.is_absolute() or p.is_symlink() or p.resolve() != p or not p.is_file()
                           for p in declared):
        raise NativeModelExecutionError("selected target renderer has no stable source closure")
    sources = sorted(set(declared))

    def render(cb, *, inputs):
        return backend.render_harness(cb, target=target, inputs=inputs)

    return BuildOnlyService(
        target=target, recipe=backends.harness_build_recipe(target), renderer=render,
        source_pins=tuple((str(p), file_digest(p)) for p in sources),
    )


def _host_compute_report(command_buffer: Mapping[str, Any], lowered_mlir_text: str,
                         *, entry_symbol: str):
    """Audit only task-scoped host IR of the emitted entry, or report why it is unverified.

    The independent route-quality analyzer owns the dataflow decision.  Parsing
    may be narrower than upstream MLIR's LLVM metadata vocabulary; that never
    becomes a clean report merely because the executable build accepts it.
    """
    from merlin.runtime.route_quality import host_compute

    try:
        from xdsl.context import Context
        from xdsl.dialects import builtin, llvm
        from xdsl.parser import Parser

        # LLVM metadata that xDSL does not model (for example loop-unroll
        # hints) must not hide a task-scoped function that upstream MLIR can
        # compile.  Unregistered operations are still handled conservatively
        # by route_quality, which marks unknown dataflow incomplete.
        context = Context(allow_unregistered=True)
        context.load_dialect(builtin.Builtin)
        context.load_dialect(llvm.LLVM)
        module = Parser(context, lowered_mlir_text).parse_module()
        matches = [op for op in module.body.block.ops
                   if op.name == "llvm.func"
                   and getattr(op.properties.get("sym_name"), "data", None) == entry_symbol]
        if len(matches) != 1:
            raise NativeModelExecutionError(
                f"candidate LLVM has {len(matches)} parsed entry function(s) named {entry_symbol!r}")
        return host_compute(command_buffer, function=matches[0])
    except Exception as exc:  # noqa: BLE001 -- no parse/attribution is not a clean audit
        return host_compute(command_buffer, function=None,
                            absent_cause=f"candidate host IR is not task-scoped/auditable: "
                                         f"{type(exc).__name__}: {str(exc)[:350]}")


def audit_emitted_host_compute(emission: Mapping[str, Any] | None, *, entry_symbol: str) -> dict[str, Any]:
    """Reopen the exact runner-emitted bytes before admitting scoped host compute.

    A result-file claim of ``clean`` is never evidence.  Both files must still
    match the runner's digest, and the current independent analyzer must accept
    the actual command buffer and LLVM function.  Missing scope is unverified;
    known tensor arithmetic on an accelerator task is a violation.
    """
    from merlin.runtime.route_quality import (
        HostComputeUnverified, HostComputeViolation, require_clean_host_compute,
    )

    try:
        if not isinstance(emission, Mapping):
            raise NativeModelExecutionError("candidate whole-program emission is absent")
        payloads = {role: _read_pinned_payload(emission, role)
                    for role in ("command_buffer", "lowered_mlir")}
        cb = json.loads(payloads["command_buffer"].decode("utf-8"))
        if not isinstance(cb, dict) or (cb.get("kernel_abi") or {}).get("kind") != "whole_program":
            raise NativeModelExecutionError("candidate emission has no whole_program ABI")
        artifact = payloads["lowered_mlir"].decode("utf-8")
        report = _host_compute_report(cb, artifact, entry_symbol=entry_symbol)
        try:
            require_clean_host_compute(report)
        except HostComputeViolation as exc:
            return {"status": "violation", "detail": str(exc), "report": report.to_dict()}
        except HostComputeUnverified as exc:
            return {"status": "unverified", "detail": str(exc), "report": report.to_dict()}
        return {
            "status": "clean", "report": report.to_dict(),
            "candidate": {
                "command_buffer_sha256": hashlib.sha256(
                    json.dumps(cb, sort_keys=True).encode()).hexdigest(),
                "lowered_mlir_sha256": hashlib.sha256(artifact.encode()).hexdigest(),
            },
        }
    except Exception as exc:  # noqa: BLE001 -- absent/mutated evidence can never become clean
        return {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}


def audit_candidate_source_placement(
    emission: Mapping[str, Any] | None, certificate: Mapping[str, Any] | None,
    *, target: str,
) -> dict[str, Any]:
    """Join candidate task ownership to independent source-operation eligibility.

    The candidate chooses task kinds and source indices; it does *not* choose
    the frozen source or target eligibility.  The trusted runner's coverage
    certificate must describe those exact source bytes and every routable
    top-level operation in order.  A source index may belong to one task only.
    This is static placement accounting, never completed-dispatch evidence.
    """
    try:
        if not isinstance(emission, Mapping):
            raise NativeModelExecutionError("candidate whole-program emission is absent")
        source_path = _pinned_native_file(emission, "source_interface")
        source = _read_pinned_payload(emission, "source_interface")
        cb = json.loads(_read_pinned_payload(emission, "command_buffer").decode("utf-8"))
        if not isinstance(cb, dict) or (cb.get("kernel_abi") or {}).get("kind") != "whole_program":
            raise NativeModelExecutionError("candidate source plan has no whole_program ABI")
        if not isinstance(certificate, Mapping):
            raise NativeModelExecutionError("independent model coverage certificate is absent")
        # Recompute the denominator from the pinned frozen source and selected
        # reviewed contract. A serialized coverage row is not itself an oracle:
        # changing its eligible flags or newly captured operand formats must
        # not let a submitted compiler make its own placement denominator.
        import yaml

        declaration_path = _pinned_native_file(emission, "capsule_declaration")
        declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
        if (not isinstance(declaration, dict)
                or not isinstance(declaration.get("interface_mlir"), str)
                or Path(declaration["interface_mlir"]).is_absolute()
                or ".." in Path(declaration["interface_mlir"]).parts
                or (declaration_path.parent / declaration["interface_mlir"]).resolve() != source_path):
            raise NativeModelExecutionError("pinned interface is not the frozen capsule's declared source")
        fresh = independent_frozen_source_eligibility(
            {**declaration, "__dir__": str(declaration_path.parent)}, target=target)
        if dict(certificate) != fresh:
            raise NativeModelExecutionError("serialized source eligibility differs from fresh frozen-source census")
        source_sha = hashlib.sha256(source).hexdigest()
        if (certificate.get("target") != target
                or certificate.get("source_mlir_sha256") != source_sha
                or certificate.get("denominator_source")
                != "semantic_capabilities (independent eligibility oracle)"
                or certificate.get("source_inventory_scope")
                != "frozen_source_only_no_legacy_execution"):
            raise NativeModelExecutionError("independent eligibility is not bound to this target and source")
        from merlin.targetgen import target_registry

        contract_pin = certificate.get("eligibility_capability_contract")
        if not isinstance(contract_pin, Mapping):
            raise NativeModelExecutionError("independent eligibility lacks selected capability bytes")
        selected_contract = target_registry.resolve(target).capability_contract_path
        if (_pinned_native_file({"contract": contract_pin}, "contract") != selected_contract
                or not selected_contract.is_absolute()):
            raise NativeModelExecutionError("eligibility names a different selected target capability contract")

        from merlin.frontends.linalg_mlir import parse_mlir_text

        module = parse_mlir_text(source.decode("utf-8"))
        funcs = [op for op in module.body.block.ops if op.name == "func.func"]
        if len(funcs) != 1 or len(funcs[0].body.blocks) != 1:
            raise NativeModelExecutionError("frozen source has no unique single-block model entry")
        ops = [op for op in funcs[0].body.blocks[0].ops if op.name != "func.return"]

        def real_name(op: Any) -> str:
            return op.op_name.data if op.name == "builtin.unregistered" else op.name

        indexed = [(index, op) for index, op in enumerate(ops) if real_name(op) != "linalg.fill"]
        regions = certificate.get("regions")
        if (not isinstance(regions, list) or certificate.get("n_regions") != len(regions)
                or len(regions) != len(indexed)):
            raise NativeModelExecutionError("independent eligibility lacks a complete source-operation inventory")
        unknown_formats = sum(isinstance(row, Mapping) and row.get("precision_transform_required") is None
                              for row in regions)
        conversion_obligations = sum(isinstance(row, Mapping) and row.get("precision_transform_required") is True
                                     for row in regions)
        precision = certificate.get("precision_transform_verification")
        if (type(certificate.get("n_unknown_capture_formats")) is not int
                or certificate["n_unknown_capture_formats"] != unknown_formats
                or type(certificate.get("n_precision_transform_obligations")) is not int
                or certificate["n_precision_transform_obligations"] != conversion_obligations
                or not isinstance(precision, Mapping)
                or precision.get("status") != "not_required"
                or unknown_formats or conversion_obligations):
            raise NativeModelExecutionError(
                "independent source eligibility has unknown capture precision or an unverified precision transform")
        from merlin.common.quant_formats import get as quant_format

        def same_observed_format(observed: Any, admitted: Any) -> bool:
            if not isinstance(observed, str) or not isinstance(admitted, str):
                return False
            try:
                return quant_format(observed).name == quant_format(admitted).name
            except (KeyError, ValueError):
                return False

        eligible: set[int] = set()
        source_regions: dict[int, str | None] = {}
        source_families: dict[int, str] = {}
        for (index, op), row in zip(indexed, regions, strict=True):
            if not isinstance(row, Mapping) or type(row.get("target_eligible")) is not bool:
                raise NativeModelExecutionError("independent source eligibility row is malformed")
            rid = getattr(op.attributes.get("prov.region_id"), "data", None)
            if row.get("region_id") != rid or row.get("carrier_op") != real_name(op):
                raise NativeModelExecutionError(
                    f"independent source eligibility does not identify operation {index}")
            source_regions[index] = rid
            if row["target_eligible"]:
                if not same_observed_format(row.get("captured_input_format"),
                                            row.get("eligibility_input_format")):
                    raise NativeModelExecutionError(
                        f"eligible source operation {index} lacks exact observed input-format admission")
                weight = row.get("captured_weight_format")
                admitted_weight = row.get("eligibility_weight_format")
                if row.get("semantic_family") in {"contraction", "convolution"}:
                    if not same_observed_format(weight, admitted_weight):
                        raise NativeModelExecutionError(
                            f"eligible source operation {index} lacks exact observed weight-format admission")
                elif admitted_weight is not None and not same_observed_format(weight, admitted_weight):
                    raise NativeModelExecutionError(
                        f"eligible source operation {index} has an unverified observed weight format")
                if not isinstance(rid, str) or not rid:
                    raise NativeModelExecutionError(
                        f"eligible source operation {index} has no provenance region identity")
                if not isinstance(row.get("semantic_family"), str) or not row["semantic_family"]:
                    raise NativeModelExecutionError("eligible source operation lacks independent semantic family")
                source_families[index] = row["semantic_family"]
                eligible.add(index)
        for index, op in enumerate(ops):
            source_regions.setdefault(index, getattr(op.attributes.get("prov.region_id"), "data", None))

        receipt = (cb.get("params") or {}).get("global_program_plan")
        if not isinstance(receipt, Mapping):
            raise NativeModelExecutionError("candidate emitted no global_program_plan")
        problems: list[str] = []
        if receipt.get("schema") != "mixed_program_plan_v1":
            problems.append("candidate global-plan schema is not mixed_program_plan_v1")
        if receipt.get("source_sha256") != source_sha:
            problems.append("candidate global plan identifies different source bytes")
        if receipt.get("source_op_count") != len(ops):
            problems.append("candidate global plan has wrong source-operation count")
        tasks = receipt.get("tasks")
        if not isinstance(tasks, list) or not tasks:
            problems.append("candidate global plan has no tasks")
            tasks = []
        owners: dict[int, str] = {}
        task_regions: list[dict[str, Any]] = []
        for ordinal, task in enumerate(tasks):
            if not isinstance(task, Mapping) or task.get("task_index") != ordinal:
                problems.append(f"candidate task {ordinal} has a missing or duplicate task identity")
                continue
            indices = task.get("source_op_indices")
            kind = task.get("kind")
            if (not isinstance(indices, list) or not indices
                    or not isinstance(kind, str) or not kind.strip()):
                problems.append(f"candidate task {ordinal} lacks source indices or execution kind")
                continue
            derived: set[str] = set()
            for index in indices:
                if type(index) is not int or not 0 <= index < len(ops):
                    problems.append(f"candidate task {ordinal} names an absent source operation")
                elif index in owners:
                    problems.append(f"source operation {index} has duplicate candidate task owners")
                else:
                    owners[index] = kind
                    if source_regions[index] is not None:
                        derived.add(source_regions[index])
            task_regions.append({"task_index": ordinal, "kind": kind,
                                 "source_op_indices": list(indices),
                                 "source_region_ids": sorted(derived)})
            if "source_region_ids" in task and task["source_region_ids"] != sorted(derived):
                problems.append(f"candidate task {ordinal} region IDs disagree with frozen source")
        if set(owners) != set(range(len(ops))):
            problems.append("candidate tasks omit source operations")
        host_eligible = sorted(index for index in eligible if owners.get(index) == "host")
        if host_eligible:
            problems.append("independently eligible source operations are assigned to host tasks")
        if problems:
            return {"status": "violation", "problems": problems,
                    "eligible_host_source_op_indices": host_eligible,
                    "task_source_regions": task_regions, "source_sha256": source_sha}
        return {"status": "clean", "source_sha256": source_sha,
                "n_source_operations": len(ops), "eligible_source_op_indices": sorted(eligible),
                "eligible_source_regions": [
                    {"source_op_index": index, "region_id": source_regions[index],
                     "semantic_family": source_families[index]}
                    for index in sorted(eligible)],
                "eligible_host_source_op_indices": [], "task_source_regions": task_regions,
                "scope": "static source/task/eligibility join; no completed candidate dispatch proof"}
    except Exception as exc:  # noqa: BLE001 -- absent independent identity never permits placement
        return {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}


def _pinned_native_file(record: Mapping[str, Any], role: str) -> Path:
    pin = record.get(role)
    if not isinstance(pin, Mapping) or not isinstance(pin.get("path"), str):
        raise NativeModelExecutionError(f"native {role} has no file pin")
    path = Path(pin["path"])
    if (not path.is_absolute() or path.is_symlink() or path.resolve() != path
            or not path.is_file() or _digest(path) != pin):
        raise NativeModelExecutionError(f"native {role} bytes do not match their file pin")
    return path


def _frozen_model_policy(capsule_dir: Path, *, target: str) -> dict[str, Any]:
    """Read the selected reviewed corpus policy, not an ambient/default ISA rule."""
    import yaml
    from merlin.targetgen.target_experiment import load_target_experiment

    descriptor_name = os.environ.get("MERLIN_TARGET_EXPERIMENT")
    if not descriptor_name:
        raise NativeModelExecutionError("selected reviewed target experiment is absent")
    descriptor = Path(descriptor_name).resolve(strict=True)
    if descriptor.is_symlink() or not descriptor.is_file():
        raise NativeModelExecutionError("selected experiment is not an ordinary file")
    experiment = load_target_experiment(descriptor)
    if experiment.target != target:
        raise NativeModelExecutionError("selected experiment names a different target")
    manifest = experiment.capsule_corpus.parent / "MANIFEST.yaml"
    if manifest.is_symlink() or not manifest.is_file():
        raise NativeModelExecutionError("selected corpus has no ordinary instruction-policy manifest")
    corpus = yaml.safe_load(manifest.read_text(encoding="utf-8"))
    declaration_path = capsule_dir / "capsule.yaml"
    declaration = yaml.safe_load(declaration_path.read_text(encoding="utf-8"))
    coverage_path = capsule_dir / "expected_instruction_coverage.yaml"
    coverage = yaml.safe_load(coverage_path.read_text(encoding="utf-8"))
    if not all(isinstance(doc, dict) for doc in (corpus, declaration, coverage)):
        raise NativeModelExecutionError("frozen model instruction policy is malformed")
    instruction = corpus.get("instruction_policy") or {}
    roles = instruction.get("prohibited_instruction_roles")
    expected = (declaration.get("expected") or {}).get("instruction_classes")
    if (instruction.get("status") != "resolved" or not isinstance(roles, list)
            or any(not isinstance(role, str) or not role for role in roles)
            or not isinstance(expected, list) or not expected
            or any(not isinstance(name, str) or not name for name in expected)
            or coverage.get("instruction_classes") != expected):
        raise NativeModelExecutionError("frozen instruction coverage or prohibited-role policy is unresolved")
    required = declaration.get("required_oracle_tiers")
    if (not isinstance(required, list) or not required or len(required) != len(set(required))
            or any(tier not in {"L0", "L1", "L2", "L3"} for tier in required)):
        raise NativeModelExecutionError("frozen required oracle tiers are malformed")
    return {
        "descriptor": _digest(descriptor), "manifest": _digest(manifest),
        "capsule_declaration": _digest(declaration_path),
        "instruction_coverage": _digest(coverage_path),
        "required_tiers": required, "expected_instruction_classes": expected,
        "prohibited_instruction_roles": roles,
    }


def _functional_engine(target: str):
    """Bind a declared chipyard L2 direct-ELF runner and its extension bytes.

    A backend with no inspectable selected Spike executable/extension remains
    unavailable. The extension is located by the backend's own invocation, not
    by target spelling or a fallback host implementation.
    """
    from merlin.runtime.backends.base import get_backend
    from merlin.targetgen.oracle_policy import oracle_tier_plan, selected_sim_via

    via = selected_sim_via(target)
    plan = oracle_tier_plan(target, via)
    if via != "chipyard" or "L2" not in plan.tiers:
        raise NativeModelExecutionError("selected target metadata has no direct-ELF functional L2 tier")
    backend = get_backend(target)
    path_fn = getattr(backend, "spike_path", None)
    extension_fn = getattr(backend, "spike_extension", None)
    if (not callable(path_fn) or not callable(extension_fn)
            or not callable(getattr(backend, "run_elf", None))
            or not backend.available("spike")):
        raise NativeModelExecutionError("selected backend has no available inspectable Spike ELF runner")
    binary = Path(path_fn()).resolve(strict=True)
    if not binary.is_file() or binary.is_symlink():
        raise NativeModelExecutionError("selected Spike executable is absent or indirect")
    flags, library_dir = extension_fn()
    if not isinstance(flags, (tuple, list)) or not isinstance(library_dir, Path):
        raise NativeModelExecutionError("selected Spike extension invocation is malformed")
    names = [part.split("=", 1)[1] for part in flags if part.startswith("--extension=")]
    explicit = [part.split("=", 1)[1] for part in flags if part.startswith("--extlib=")]
    if len(names) != 1 or len(explicit) > 1:
        raise NativeModelExecutionError("selected Spike extension identity is ambiguous")
    if explicit:
        extension = Path(explicit[0]).resolve(strict=True)
    else:
        if not library_dir.is_dir() or library_dir.is_symlink():
            raise NativeModelExecutionError("selected Spike extension library directory is absent")
        matches = list(library_dir.glob(f"lib{names[0]}.so"))
        if len(matches) != 1:
            raise NativeModelExecutionError("selected default Spike extension bytes are not unique")
        extension = matches[0].resolve(strict=True)
    if not extension.is_file() or extension.is_symlink():
        raise NativeModelExecutionError("selected Spike extension is absent or indirect")
    citation = {"engine": "spike", "target": target, "binary": _digest(binary),
                "extension": _digest(extension), "flags": list(flags),
                "library_dir": str(library_dir)}

    def revalidate() -> None:
        if (Path(path_fn()).resolve(strict=True) != binary or _digest(binary) != citation["binary"]
                or tuple(extension_fn()[0]) != tuple(flags)
                or _digest(extension) != citation["extension"]):
            raise NativeModelExecutionError("selected L2 engine or extension changed during execution")

    return backend, citation, revalidate


def audit_candidate_static_tiers(
    emission: Mapping[str, Any] | None, certificate: Mapping[str, Any] | None,
    native: Mapping[str, Any] | None, *, target: str, entry_symbol: str,
) -> dict[str, Any]:
    """Recompute L0 source/IR structure and L1 exact-ELF ISA legality.

    These are candidate-artifact checks, not host reference numeric executions.
    An L0/L1 result never stands in for L2 or L3 functional execution.
    """
    from merlin.targetgen.bundle_harness import is_executable_emission

    tiers: dict[str, Any] = {}
    try:
        if not isinstance(emission, Mapping) or not isinstance(native, Mapping):
            raise NativeModelExecutionError("candidate emission/native build is absent")
        cb = json.loads(_read_pinned_payload(emission, "command_buffer").decode("utf-8"))
        lowered = _read_pinned_payload(emission, "lowered_mlir").decode("utf-8")
        source = _read_pinned_payload(emission, "source_interface").decode("utf-8")
        ok, reason = is_executable_emission(cb, artifact_text=lowered)
        if not ok:
            raise NativeModelExecutionError(reason)
        placement = audit_candidate_source_placement(emission, certificate, target=target)
        if placement.get("status") != "clean":
            raise NativeModelExecutionError("source placement is not independently verified")
        host = audit_emitted_host_compute(emission, entry_symbol=entry_symbol)
        if host.get("status") != "clean":
            raise NativeModelExecutionError("candidate task-scoped host compute is not clean")
        from xdsl.context import Context
        from xdsl.dialects import builtin, llvm
        from xdsl.parser import Parser
        from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan

        context = Context(allow_unregistered=True)
        context.load_dialect(builtin.Builtin)
        context.load_dialect(llvm.LLVM)
        module = Parser(context, lowered).parse_module()
        plan = verify_compiler_global_plan(
            source_text=source, lowered_text=lowered, command_buffer=cb,
            candidate_sha256=hashlib.sha256(lowered.encode()).hexdigest(),
            parsed_lowered_module=module)
        if (plan.get("status") != "verified"
                or (plan.get("control_flow") or {}).get("status") != "verified"):
            raise NativeModelExecutionError("candidate source/ABI/task CFG is not verified")
        candidate = native.get("candidate") or {}
        if (candidate.get("command_buffer_sha256")
                != hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest()
                or candidate.get("lowered_mlir_sha256") != hashlib.sha256(lowered.encode()).hexdigest()):
            raise NativeModelExecutionError("candidate build has different CB/LLVM bytes")
        elf = _pinned_native_file(native, "elf")
        policy = _frozen_model_policy(_pinned_native_file(native["source"], "capsule_declaration").parent,
                                      target=target)
        if native.get("frozen_policy") != policy:
            raise NativeModelExecutionError("frozen model instruction/tier policy changed after build")
        tiers["L0"] = {"status": "pass", "scope": "candidate source/CB/LLVM/ABI/task CFG and host-veto structure",
                       "plan_digest": plan.get("plan_digest"), "elf": _digest(elf)}
    except Exception as exc:  # noqa: BLE001 -- missing structure cannot be credited
        tiers["L0"] = {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}
        tiers["L1"] = {"status": "unverified", "detail": "L0 candidate identity/structure is incomplete"}
        return tiers
    try:
        from merlin.compile.model_execution_inputs import selected_firrtl
        from merlin.targetgen.rtl.facts import observed_facts
        from merlin.targetgen.rocc.decode import decode_module
        from merlin.perf.isa_prohibition import scan_elf
        from merlin.targetgen import elf_lanes
        from merlin.common.facts_view import interface as facts_interface

        selected = native.get("rtl_facts") or {}
        if (not isinstance(selected, Mapping) or not isinstance(selected.get("path"), str)
                or not isinstance(selected.get("config"), str)
                or selected_firrtl(selected["path"], target=target, config=selected["config"]) != selected):
            raise NativeModelExecutionError("L1 lacks selected byte-bound RTL ISA facts")
        facts_path = Path(selected["path"])
        facts = json.loads(facts_path.read_text(encoding="utf-8"))
        table = facts_interface(facts.get("facts"), "funct_decode_table") or {}
        opcode = table.get("custom_opcode")
        names = {int(key) for key in (table.get("names") or {})}
        if type(opcode) is not int or not names:
            raise NativeModelExecutionError("selected ISA has no complete custom command table")
        blob = elf.read_bytes()
        sections = elf_lanes.executable_sections(blob)
        if not sections:
            raise NativeModelExecutionError("linked candidate ELF has no executable sections")
        custom_count = 0
        for _name, offset, size, address in sections:
            for _at, word in elf_lanes.instruction_words(blob[offset:offset + size], address):
                if word & 0x7f == opcode:
                    custom_count += 1
                    if word >> 25 not in names:
                        raise NativeModelExecutionError("linked candidate ELF has an unknown custom selector")
        with observed_facts(target, facts, path=facts_path):
            decoded = decode_module(module, target=target)
            instructions = decoded.get("instructions")
            if (not isinstance(instructions, list) or not instructions
                    or any(row.get("class") == "UNKNOWN" for row in instructions)):
                raise NativeModelExecutionError("candidate LLVM has missing/unknown selected ISA decode")
            classes = {str(row.get("class")) for row in instructions}
            required = set(policy["expected_instruction_classes"])
            if not required.issubset(classes):
                raise NativeModelExecutionError(
                    f"candidate LLVM omits frozen required instruction classes {sorted(required - classes)}")
            if policy["prohibited_instruction_roles"]:
                prohibition = scan_elf(elf, target=target,
                                       roles=policy["prohibited_instruction_roles"])
                if prohibition.get("status") != "measured" or prohibition.get("clean") is not True:
                    raise NativeModelExecutionError("linked candidate ELF has prohibited or unmeasured ISA roles")
            else:
                prohibition = {"status": "not_applicable", "clean": True,
                               "reason": "frozen reviewed policy explicitly prohibits no roles"}
        tiers["L1"] = {"status": "pass", "scope": "selected ISA decode, frozen class coverage and whole-ELF prohibited roles",
                       "observed_classes": sorted(classes), "required_classes": sorted(required),
                       "prohibition": prohibition, "whole_elf_known_custom_commands": custom_count,
                       "rtl_facts": selected, "elf": _digest(elf)}
    except Exception as exc:  # noqa: BLE001 -- unknown ISA legality is not a pass
        tiers["L1"] = {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}
    return tiers


def audit_candidate_tiers(
    emission: Mapping[str, Any] | None, certificate: Mapping[str, Any] | None,
    native: Mapping[str, Any] | None, *, target: str, entry_symbol: str,
    completed_dispatch: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Audit each frozen mandatory tier against this candidate's own artifact.

    L0/L1 are recomputed; L2/L3 reopen console bytes and independent golden.
    Merely listing a tier in the receipt never counts as exercising it.
    """
    tiers = audit_candidate_static_tiers(
        emission, certificate, native, target=target, entry_symbol=entry_symbol)
    try:
        if not isinstance(native, Mapping):
            raise NativeModelExecutionError("native receipt is absent")
        policy = _frozen_model_policy(
            _pinned_native_file(native["source"], "capsule_declaration").parent, target=target)
        if native.get("frozen_policy") != policy:
            raise NativeModelExecutionError("native receipt names different frozen tier policy")
        required = policy["required_tiers"]
    except Exception as exc:  # noqa: BLE001 -- no policy means no tier may be admitted
        return {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}",
                "tiers": tiers, "required_tiers": None}
    try:
        from merlin.runtime.backends.base import get_backend
        from merlin.runtime.backends import base as backends
        from merlin.runtime.commandbuffer import declared_output_dtypes
        from merlin.targetgen.golden_store import load_golden
        from merlin.targetgen.capsule_golden import compare
        import yaml

        functional = (native.get("tiers") or {}).get("L2") or {}
        if functional.get("status") != "pass" or functional.get("engine") != "spike":
            raise NativeModelExecutionError("same candidate ELF has no passing functional L2 execution")
        backend, citation, revalidate = _functional_engine(target)
        if functional.get("engine_citation") != citation:
            raise NativeModelExecutionError("L2 engine citation differs from currently selected bytes")
        revalidate()
        elf = _pinned_native_file(native, "elf")
        if functional.get("elf") != _digest(elf):
            raise NativeModelExecutionError("L2 ran a different ELF")
        console = _pinned_native_file(functional, "console").read_text(encoding="utf-8")
        observed, _metrics = backend.parse_output(console)
        cb = json.loads(_read_pinned_payload(emission, "command_buffer").decode("utf-8"))
        observed = backends.decode_float_readback(observed, declared_output_dtypes(cb))
        capsule_dir = _pinned_native_file(native["source"], "capsule_declaration").parent
        declaration = yaml.safe_load((capsule_dir / "capsule.yaml").read_text(encoding="utf-8"))
        golden = load_golden(capsule_dir)
        numeric = compare(golden["outputs"], observed, declaration["numeric_policy"],
                          golden_source=str(golden.get("golden_source") or "independent_capsule"))
        if numeric.get("status") != "pass" or numeric != functional.get("numeric"):
            raise NativeModelExecutionError("L2 full output does not match independent golden")
        tiers["L2"] = {"status": "pass", "engine_citation": citation,
                       "elf": _digest(elf), "console": _digest(_pinned_native_file(functional, "console")),
                       "numeric": numeric}
    except Exception as exc:  # noqa: BLE001 -- unbound engine/console is unavailable, not a tier pass
        tiers["L2"] = {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}
    try:
        rtl = (native.get("tiers") or {}).get("L3") or {}
        if (not isinstance(completed_dispatch, Mapping)
                or completed_dispatch.get("status") != "verified"
                or rtl.get("status") != "pass"
                or rtl.get("elf") != _digest(_pinned_native_file(native, "elf"))
                or rtl.get("console") != _digest(_pinned_native_file(native, "console"))
                or rtl.get("numeric") != native.get("numeric")):
            raise NativeModelExecutionError("selected RTL tier lacks independently verified candidate completion")
        tiers["L3"] = {"status": "pass", "engine": rtl.get("engine"),
                       "elf": rtl["elf"], "console": rtl["console"],
                       "completed_dispatch": completed_dispatch}
    except Exception as exc:  # noqa: BLE001 -- no verified RTL completion is not an L3 pass
        tiers["L3"] = {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}
    missing = [tier for tier in required if (tiers.get(tier) or {}).get("status") != "pass"]
    return {"status": "pass" if not missing else "unverified", "required_tiers": required,
            "missing_tiers": missing, "tiers": tiers,
            "scope": "candidate-only mandatory tier evidence; L0/L1 structural/ISA, L2 functional, L3 RTL"}


def _mandatory_command_blocks(function: Any, commands: set[Any]) -> bool:
    """A return cannot be reached after removing blocks carrying real work commands.

    This proves one command issues on every *returning* path, including a loop's
    zero-trip edge. Nontermination and equivalence to source computation are
    separate obligations; neither is inferred from this CFG property.
    """
    from merlin.perf.host_cfg_index import prepare_host_cfg

    cfg = prepare_host_cfg(function)
    carrying = {block for block in cfg.blocks if any(op in commands for op in block.ops)}
    if not carrying or not cfg.blocks:
        return False
    pending = [cfg.blocks[0]]
    reached: set[Any] = set()
    while pending:
        block = pending.pop()
        if block in reached or block in carrying:
            continue
        reached.add(block)
        if block.last_op is not None and block.last_op.name == "llvm.return":
            return False
        pending.extend(cfg.successors[block])
    return True


def _kernel_command_inventory(path: Path, *, symbol: str,
                              table: Mapping[str, Any]) -> tuple[list[int], list[int]]:
    """Read only the named kernel symbol, never decoy opcodes in the harness."""
    import subprocess

    from merlin.kernels.decode.objdump import tokenize_text
    from merlin.kernels.decode.rocc import decode_stream, fields_of
    from merlin.kernels.decode.objdump import word_of
    from merlin.llvmlower import toolchain

    disassembler = toolchain.objdump()
    if not disassembler.is_file() or disassembler.is_symlink():
        raise NativeModelExecutionError("selected compiler has no matching LLVM objdump")
    run = subprocess.run(
        [str(disassembler), "-d", "--triple=riscv64", "-M", "no-aliases",
         f"--disassemble-symbols={symbol}", str(path)],
        capture_output=True, text=True, timeout=60)
    if run.returncode:
        raise NativeModelExecutionError(f"symbol-scoped disassembly failed: {run.stderr[-350:]}")
    rows = tokenize_text(run.stdout)
    if not rows or f"<{symbol}>:" not in run.stdout:
        raise NativeModelExecutionError(f"no disassembled {symbol!r} symbol in {path.name}")
    decoded = decode_stream(rows, table)
    opcode = table.get("custom_opcode")
    if type(opcode) is not int:
        raise NativeModelExecutionError("selected target has no derived accelerator opcode")
    all_custom: list[int] = []
    known: list[int] = []
    for row, instruction in zip(rows, decoded, strict=True):
        word = word_of(row.hexcode, width_bits=32)
        if word is None:
            continue
        fields = fields_of(word)
        if fields["opcode"] == opcode:
            all_custom.append(word)
            if instruction.from_endpoint:
                known.append(fields["funct"])
    if len(all_custom) != len(known):
        raise NativeModelExecutionError(f"{path.name} has an unrecognized custom command in {symbol}")
    return all_custom, known


_SOURCE_FAMILY_DEVICE_ROLES = {
    "contraction": ("accumulate",),
    "convolution": ("accumulate",),
    "elementwise_map": ("elementwise",),
    "movement": ("move", "operand_load", "readout", "dma"),
}


def _verified_work_functs_by_family(endpoints: list[Mapping[str, Any]],
                                    table: Mapping[str, Any]) -> dict[str, set[int]]:
    """Bind trusted source families to selected RTL-confirmed work roles.

    Config, wait, flush and generic loop-descriptor setup are never credited.
    Fused epilogues need an independent candidate-transform witness, not a
    broad role mapping that mistakes configuration for arithmetic.
    """
    if (len(endpoints) != 1 or endpoints[0].get("engine") != "spatial"
            or endpoints[0].get("exposure") != "rocc"):
        raise NativeModelExecutionError("selected target has no unique verified spatial endpoint")
    names = {int(key): value for key, value in (table.get("names") or {}).items()}
    roles = endpoints[0].get("roles") or {}
    if (not isinstance(roles, Mapping)
            or any(not isinstance(declared, list) or not declared
                   or any(name not in names.values() for name in declared)
                   for declared in roles.values())):
        raise NativeModelExecutionError("endpoint roles are not confirmed by selected RTL decode table")
    return {
        family: {funct for funct, name in names.items()
                 if any(name in roles.get(role, ()) for role in accepted)}
        for family, accepted in _SOURCE_FAMILY_DEVICE_ROLES.items()
    }


def _completed_eligible_tasks(function: Any, placement: Mapping[str, Any],
                              task_commands: Mapping[int, Mapping[str, set[Any]]],
                              task_region_commands: Mapping[int, Mapping[str, set[Any]]],
                              task_op_commands: Mapping[int, Mapping[int, set[Any]]] | None = None,
                              ) -> list[dict[str, Any]]:
    """Require a mandatory family-matched work command per eligible region."""
    eligible = set(placement["eligible_source_op_indices"])
    eligible_regions = {row["source_op_index"]: row["region_id"]
                        for row in placement["eligible_source_regions"]}
    eligible_families = {row["source_op_index"]: row["semantic_family"]
                         for row in placement["eligible_source_regions"]}
    op_commands = task_op_commands or {}
    completed: list[dict[str, Any]] = []
    for task in placement["task_source_regions"]:
        indices = task["source_op_indices"]
        task_eligible = sorted(eligible.intersection(indices))
        if not task_eligible:
            continue  # genuinely unsupported host glue is not an accelerator obligation
        ident = task["task_index"]
        by_family = task_commands.get(ident, {})
        commands = set().union(*(by_family.get(eligible_families[index], set())
                                 for index in task_eligible))
        for family in {eligible_families[index] for index in task_eligible}:
            if not _mandatory_command_blocks(function, by_family.get(family, set())):
                raise NativeModelExecutionError(
                    f"eligible source task {ident} has no unavoidable {family} device command")
        if len(task_eligible) > 1:
            # A repeated region ID can contain multiple linalg roots. One
            # command may not credit all of them: use exact source-op identity
            # when present, or region identity only when unique in this group.
            group_regions = [eligible_regions[index] for index in task_eligible]
            for index in task_eligible:
                rid = eligible_regions[index]
                attributed = op_commands.get(ident, {}).get(index, set())
                if not attributed and group_regions.count(rid) == 1:
                    attributed = task_region_commands[ident].get(rid, set())
                matched = attributed & by_family.get(eligible_families[index], set())
                if not _mandatory_command_blocks(function, matched):
                    raise NativeModelExecutionError(
                        f"eligible source operation {index} in region {rid} lacks an unavoidable "
                        "family-matched attributed command; fused groups need independent "
                        "candidate-transform evidence")
        else:
            named = task_region_commands[ident]
            if named and set(named) != {eligible_regions[task_eligible[0]]}:
                raise NativeModelExecutionError(
                    f"eligible source task {ident} commands name a different source region")
        completed.append({"task_index": ident, "source_op_indices": sorted(indices),
                          "eligible_source_region_ids": [eligible_regions[index]
                                                         for index in task_eligible],
                          "mandatory_device_work_commands": len(commands)})
    if not completed:
        raise NativeModelExecutionError("no independently eligible source task was joined")
    return completed


def audit_candidate_completed_dispatch(
    emission: Mapping[str, Any] | None, certificate: Mapping[str, Any] | None,
    native: Mapping[str, Any] | None, *, target: str, entry_symbol: str,
) -> dict[str, Any]:
    """Infer mandatory candidate dispatch from exact native completion and codegen.

    This is NOT a hardware commit trace. The trusted translator/compiler/linker
    form the bridge from a side-effecting LLVM command to object and linked ELF;
    each independently eligible source task must have a family-matched work command
    that no returning CFG path can bypass. The complete custom-command multiset
    must survive in the *kernel symbol* of both compiled artifacts. Unknown
    transforms or missing attribution stay unverified, not clean.
    """
    try:
        if not isinstance(emission, Mapping) or not isinstance(native, Mapping):
            raise NativeModelExecutionError("candidate emission or native execution is absent")
        placement = audit_candidate_source_placement(emission, certificate, target=target)
        if placement.get("status") != "clean":
            raise NativeModelExecutionError("independent source placement is not clean")
        if native.get("status") != "numeric_match_diagnostic" or native.get("simulator") is None:
            raise NativeModelExecutionError("exact candidate ELF has no completed native oracle run")
        source_text = _read_pinned_payload(emission, "source_interface").decode("utf-8")
        lowered_text = _read_pinned_payload(emission, "lowered_mlir").decode("utf-8")
        cb = json.loads(_read_pinned_payload(emission, "command_buffer").decode("utf-8"))
        candidate = native.get("candidate") or {}
        if (candidate.get("lowered_mlir_sha256") != hashlib.sha256(lowered_text.encode()).hexdigest()
                or candidate.get("command_buffer_sha256")
                != hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest()):
            raise NativeModelExecutionError("native build is not bound to exact candidate bytes")

        from xdsl.context import Context
        from xdsl.dialects import builtin, llvm
        from xdsl.parser import Parser
        from merlin.perf.compiler_plan_evidence import verify_compiler_global_plan
        from merlin.targetgen.rocc.decode import decode_module
        from merlin.common.facts_view import interface as facts_interface
        from merlin.targetgen.contract.schemas import contract_dir
        from merlin.compile.model_execution_inputs import native_engine, selected_firrtl
        from merlin.targetgen.oracle_policy import selected_l3_engine_report
        import yaml

        context = Context(allow_unregistered=True)
        context.load_dialect(builtin.Builtin)
        context.load_dialect(llvm.LLVM)
        module = Parser(context, lowered_text).parse_module()
        functions = [op for op in module.body.block.ops if op.name == "llvm.func"
                     and getattr(op.properties.get("sym_name"), "data", None) == entry_symbol]
        if len(functions) != 1:
            raise NativeModelExecutionError("emitted LLVM has no unique selected entry function")
        plan = verify_compiler_global_plan(
            source_text=source_text, lowered_text=lowered_text, command_buffer=cb,
            candidate_sha256=hashlib.sha256(lowered_text.encode()).hexdigest(),
            parsed_lowered_module=module)
        if (plan.get("status") != "verified"
                or (plan.get("control_flow") or {}).get("status") != "verified"):
            raise NativeModelExecutionError("independent source/ABI/task CFG proof is not verified: "
                                            + str(plan.get("problems") or plan.get("reason")))
        selected = native.get("rtl_facts") or {}
        if (not isinstance(selected, Mapping) or selected.get("target") != target
                or not isinstance(selected.get("config"), str)
                or not isinstance(selected.get("path"), str)
                or selected_firrtl(selected["path"], target=target,
                                   config=selected["config"]) != selected):
            raise NativeModelExecutionError("native RTL fact lineage changed or is absent")
        selection = selected_l3_engine_report(target)
        if (not selection.get("available") or selection.get("engine") != native["simulator"]
                or (native.get("simulator_provenance") or {}).get("selection") != selection):
            raise NativeModelExecutionError("native run is not on the selected mandatory RTL engine")
        _, citation, revalidate_engine, _ = native_engine(target, native["simulator"], dict(selected))
        if (native.get("simulator_provenance") or {}).get("citation") != citation:
            raise NativeModelExecutionError("native run cites different RTL engine bytes")
        revalidate_engine()
        selected_facts_path = Path(selected["path"])
        facts = json.loads(selected_facts_path.read_text(encoding="utf-8"))
        table = facts_interface(facts.get("facts"), "funct_decode_table") or {}
        spec = yaml.safe_load((contract_dir() / "compute_endpoints.yaml").read_text())
        endpoints = [row for row in (spec.get("endpoints") or {}).values()
                     if isinstance(row, Mapping) and row.get("target") == target
                     and row.get("exposure") == "rocc"]
        names = {int(key): value for key, value in (table.get("names") or {}).items()}
        work_functs = _verified_work_functs_by_family(endpoints, table)
        trace = decode_module(module, target=target)
        trace_opcode = (trace.get("abi") or {}).get("custom_opcode")
        if (type(table.get("custom_opcode")) is not int or not isinstance(trace_opcode, str)
                or int(trace_opcode, 16) != table["custom_opcode"]):
            raise NativeModelExecutionError("selected RTL opcode differs from host decoder")
        asm = [op for op in module.walk() if op.name == "llvm.inline_asm"]
        instructions = trace.get("instructions")
        if not isinstance(instructions, list) or len(instructions) != len(asm):
            raise NativeModelExecutionError("emitted inline assembly has incomplete structural decode")
        tasks = ((cb.get("params") or {}).get("global_program_plan") or {}).get("tasks") or []
        task_commands: dict[int, dict[str, set[Any]]] = {row["task_index"]: {} for row in tasks}
        task_region_commands: dict[int, dict[str, set[Any]]] = {
            row["task_index"]: {} for row in tasks}
        task_op_commands: dict[int, dict[int, set[Any]]] = {
            row["task_index"]: {} for row in tasks}
        allowed_regions = {row["region_id"] for row in placement["eligible_source_regions"]}
        region_by_index = {row["source_op_index"]: row["region_id"]
                           for row in placement["eligible_source_regions"]}
        all_functs: list[int] = []
        for operation, decoded in zip(asm, instructions, strict=True):
            funct = decoded.get("funct")
            if funct is None:
                if decoded.get("class") == "UNKNOWN":
                    raise NativeModelExecutionError("unrecognized inline assembly in candidate kernel")
                continue
            if type(funct) is not int or funct not in names:
                raise NativeModelExecutionError("candidate command lacks a derived ISA identity")
            all_functs.append(funct)
            families = [family for family, functs in work_functs.items() if funct in functs]
            if not families:
                continue
            if getattr(operation, "has_side_effects", None) is None:
                raise NativeModelExecutionError("device-work inline asm is not side-effecting")
            constraints = getattr(getattr(operation, "constraints", None), "data", "")
            if "~{memory}" not in constraints.split(","):
                raise NativeModelExecutionError("device-work inline asm lacks memory-ordering clobber")
            attr = operation.attributes.get("merlin.global_task")
            owner = getattr(getattr(attr, "value", None), "data", None)
            if type(owner) is not int or owner not in task_commands:
                raise NativeModelExecutionError("device-work command has no verified task owner")
            for family in families:
                task_commands[owner].setdefault(family, set()).add(operation)
            region = getattr(operation.attributes.get("prov.region_id"), "data", None)
            source_index_attr = operation.attributes.get("merlin.source_op_index")
            source_index = getattr(getattr(source_index_attr, "value", None), "data", None)
            if source_index_attr is not None:
                if (type(source_index) is not int or source_index not in region_by_index
                        or source_index not in tasks[owner]["source_op_indices"]):
                    raise NativeModelExecutionError("device-work command names an absent source operation")
                if region is not None and region != region_by_index[source_index]:
                    raise NativeModelExecutionError("device-work command source index/region disagree")
                task_op_commands[owner].setdefault(source_index, set()).add(operation)
            if region is not None:
                if region not in allowed_regions:
                    raise NativeModelExecutionError("device-work command names an absent source region")
                task_region_commands[owner].setdefault(region, set()).add(operation)
        completed = _completed_eligible_tasks(
            functions[0], placement, task_commands, task_region_commands, task_op_commands)

        artifacts = native.get("build_artifacts")
        if not isinstance(artifacts, Mapping):
            raise NativeModelExecutionError("native build artifact pins are absent")
        for name in ("kernel.llvm.mlir", "kernel.ll", "kernel.o", "kernel.stack_frame.json",
                     "kernel.abi.json"):
            _pinned_native_file(artifacts, name)
        if _pinned_native_file(artifacts, "kernel.llvm.mlir").read_bytes() != lowered_text.encode():
            raise NativeModelExecutionError("translated object used different candidate LLVM bytes")
        stack = json.loads(_pinned_native_file(artifacts, "kernel.stack_frame.json").read_text())
        if stack.get("status") != "passed" or stack.get("repair") is not None:
            raise NativeModelExecutionError("object transform needs separate command-preservation proof")
        obj = _pinned_native_file(artifacts, "kernel.o")
        if (stack.get("entry_symbol") != entry_symbol
                or stack.get("llvm_ir_sha256") != _digest(_pinned_native_file(artifacts, "kernel.ll"))["sha256"]
                or stack.get("object_sha256") != _digest(obj)["sha256"]):
            raise NativeModelExecutionError("kernel object is not bound to translated LLVM and entry")
        abi = json.loads(_pinned_native_file(artifacts, "kernel.abi.json").read_text())
        if abi.get("object_sha256") != _digest(obj)["sha256"]:
            raise NativeModelExecutionError("ABI receipt names a different kernel object")
        host_sources = native.get("host_build_sources")
        if not isinstance(host_sources, list) or not host_sources:
            raise NativeModelExecutionError("trusted harness renderer source closure is absent")
        for row in host_sources:
            if (not isinstance(row, Mapping) or not isinstance(row.get("path"), str)
                    or not isinstance(row.get("sha256"), str)):
                raise NativeModelExecutionError("trusted harness renderer source pin is malformed")
            host_path = Path(row["path"])
            if (not host_path.is_absolute() or host_path.is_symlink() or not host_path.is_file()
                    or _digest(host_path)["sha256"] != row["sha256"]):
                raise NativeModelExecutionError("trusted harness renderer source changed")
        elf = _pinned_native_file(native, "elf")
        object_words, object_functs = _kernel_command_inventory(
            obj, symbol=entry_symbol, table=table)
        elf_words, elf_functs = _kernel_command_inventory(
            elf, symbol=entry_symbol, table=table)
        if (Counter(all_functs) != Counter(object_functs)
                or Counter(object_words) != Counter(elf_words)
                or Counter(object_functs) != Counter(elf_functs)):
            raise NativeModelExecutionError("task command inventory changed across LLVM/object/linked ELF")
        console = _pinned_native_file(native, "console").read_text(encoding="utf-8")
        from merlin.runtime.backends.base import get_backend
        from merlin.runtime.commandbuffer import declared_output_dtypes
        from merlin.runtime.backends import base as backends
        from merlin.targetgen.golden_store import load_golden
        from merlin.targetgen.capsule_golden import compare
        sources = native.get("source") or {}
        # The model bundle is deliberately temporary and is removed when the
        # runner leaves its context. Reopen *stable frozen capsule assets* at
        # grade time; the build-time source pins for generated inputs/manifest
        # remain in the trusted execution receipt, but their deleted paths
        # cannot be misread as durable proof.
        for name in ("capsule_declaration", "golden"):
            _pinned_native_file(sources, name)
        if "golden_arrays" in sources:
            _pinned_native_file(sources, "golden_arrays")
        capsule_dir = _pinned_native_file(sources, "capsule_declaration").parent
        declaration = yaml.safe_load((capsule_dir / "capsule.yaml").read_text(encoding="utf-8"))
        attrs = ((declaration.get("operation") or {}).get("attributes") or {})
        for role, relative in (("interface", declaration.get("interface_mlir")),
                               ("weights", attrs.get("weights")),
                               ("weight_manifest", attrs.get("weights_manifest"))):
            if not isinstance(relative, str) or not relative or Path(relative).is_absolute() \
                    or ".." in Path(relative).parts:
                raise NativeModelExecutionError(f"frozen {role} has no capsule-local path")
            frozen = capsule_dir / relative
            if (frozen.is_symlink() or not frozen.is_file()
                    or _digest(frozen)["sha256"] != (sources.get(role) or {}).get("sha256")):
                raise NativeModelExecutionError(f"frozen {role} differs from bound runtime bundle")
        if _digest(capsule_dir / declaration["interface_mlir"])["sha256"] \
                != hashlib.sha256(source_text.encode()).hexdigest():
            raise NativeModelExecutionError("candidate source differs from frozen model interface")
        golden = load_golden(capsule_dir)
        observed, _ = get_backend(target).parse_output(console)
        observed = backends.decode_float_readback(observed, declared_output_dtypes(cb))
        numeric = compare(golden["outputs"], observed, declaration["numeric_policy"],
                          golden_source=str(golden.get("golden_source") or "independent_capsule"))
        if numeric.get("status") != "pass":
            raise NativeModelExecutionError("exact completed ELF output differs from independent golden")
        from merlin.llvmlower import toolchain

        return {"status": "verified", "scope": "completion-implied mandatory family-matched "
                "device-work command issue; "
                "not hardware commit trace or source-transform semantic equivalence",
                "source_sha256": placement["source_sha256"],
                "lowered_mlir_sha256": candidate["lowered_mlir_sha256"],
                "kernel_object": _digest(obj), "linked_elf": _digest(elf),
                "console": _digest(_pinned_native_file(native, "console")),
                "eligible_tasks": completed, "kernel_command_count": len(all_functs),
                "plan_digest": plan.get("plan_digest"),
                "disassembler": _digest(toolchain.objdump())}
    except Exception as exc:  # noqa: BLE001 -- no partial dispatch credit
        return {"status": "unverified", "detail": f"{type(exc).__name__}: {exc}"}


def execute_candidate_model(
    *, command_buffer: Mapping[str, Any], lowered_mlir_text: str,
    capsule_dir: str | Path, capture_bundle: str | Path, target: str,
    out_dir: str | Path, simulator: str | None = None, timeout: int = 600,
    numeric_policy: Mapping[str, Any] | None = None,
    rtl_facts: str | Path | None = None, board_config: str | None = None,
) -> dict[str, Any]:
    """Build one candidate whole-model ELF; optionally run and compare its entire output.

    ``capture_bundle`` is the caller's byte-bound, frozen runtime bundle from
    ``_model_runtime_bundle``.  ``capsule_dir`` supplies the independent,
    digest-checked golden.  A successful comparison is deliberately labelled a
    *diagnostic*: eligibility, transform replay, and dynamic source-region
    execution still need their independent mandatory gates.
    """
    from merlin.targetgen.bundle_harness import emitted_entry_arity, is_executable_emission
    from merlin.targetgen.golden_store import load_golden

    source = Path(capsule_dir)
    capture = Path(capture_bundle)
    output = Path(out_dir)
    if output.is_symlink():
        raise NativeModelExecutionError(f"artifact directory must not be a symlink: {output}")
    if output.exists() and any(output.iterdir()):
        raise NativeModelExecutionError(f"fresh artifact directory required: {output}")
    output.mkdir(parents=True, exist_ok=True)
    record: dict[str, Any] = {
        "schema": "merlin_candidate_native_model_execution_v1",
        "status": "incomplete", "target": target, "simulator": simulator,
        "scope": "candidate whole-program ELF output diagnostic; not a Phase-1 certification",
    }
    receipt = output / "result.json"
    try:
        if not source.is_dir() or source.is_symlink() or not capture.is_dir() or capture.is_symlink():
            raise NativeModelExecutionError("model capsule and frozen runtime bundle must be ordinary directories")
        cb = json.loads(json.dumps(command_buffer))
        if (cb.get("kernel_abi") or {}).get("kind") != "whole_program":
            raise NativeModelExecutionError("candidate did not declare a whole_program pointer ABI")
        ok, reason = is_executable_emission(cb, artifact_text=lowered_mlir_text)
        if not ok:
            raise NativeModelExecutionError(reason)
        abi = cb["kernel_abi"]
        outputs = abi.get("outputs")
        if not isinstance(outputs, list) or len(outputs) != 1 or not isinstance(outputs[0], str):
            raise NativeModelExecutionError("one declared whole-model output is required")
        output_name = outputs[0]
        if (cb.get("tensors") or {}).get(output_name, {}).get("role") != "output":
            raise NativeModelExecutionError("declared model output has no output tensor role")
        golden = load_golden(source)
        if not isinstance(golden, dict) or set(golden.get("outputs") or {}) != {output_name}:
            raise NativeModelExecutionError("independent golden outputs do not match the candidate's ABI")
        if numeric_policy is None:
            import yaml

            declaration = yaml.safe_load((source / "capsule.yaml").read_text(encoding="utf-8"))
            if not isinstance(declaration, dict):
                raise NativeModelExecutionError("frozen capsule declaration is not a mapping")
            policy = dict(declaration.get("numeric_policy") or {})
        else:
            policy = dict(numeric_policy)
        if policy.get("compare") not in ("exact_int", "tolerance_float"):
            raise NativeModelExecutionError("no explicit supported numeric comparison policy")
        if policy["compare"] == "exact_int" and str(cb["tensors"][output_name]["dtype"]).startswith("f"):
            raise NativeModelExecutionError("exact_int policy cannot certify a floating output")
        inputs, binding = _bind_inputs(
            cb, capture, target=target, rtl_facts=rtl_facts, board_config=board_config)
        record["binding"] = binding
        record["source"] = {
            "capsule_declaration": _digest(source / "capsule.yaml"),
            "interface": _digest(capture / "model.mlir"),
            "weights": _digest(capture / "weights.safetensors"),
            "weight_manifest": _digest(capture / "weights.safetensors.manifest.json"),
            "inputs": _digest(capture / "inputs.npz"),
            "golden": _digest(source / "golden.yaml"),
        }
        arrays = source / "golden.npz"
        if arrays.is_file():
            record["source"]["golden_arrays"] = _digest(arrays)
        record["frozen_policy"] = _frozen_model_policy(source, target=target)

        def revalidate_source() -> None:
            for role, pinned in record["source"].items():
                current = _digest(Path(pinned["path"]))
                if current["sha256"] != pinned["sha256"] or current["size_bytes"] != pinned["size_bytes"]:
                    raise NativeModelExecutionError(f"frozen {role} bytes changed during candidate execution")

        record["candidate"] = {
            "command_buffer_sha256": hashlib.sha256(json.dumps(cb, sort_keys=True).encode()).hexdigest(),
            "lowered_mlir_sha256": hashlib.sha256(lowered_mlir_text.encode()).hexdigest(),
            "entry_arity": emitted_entry_arity(lowered_mlir_text),
            "abi_args": len(abi["args"]),
        }
        from merlin.runtime.route_quality import (
            HostComputeUnverified, HostComputeViolation, require_clean_host_compute,
        )

        service = _build_service_for(target)
        entry_symbol = service.recipe.require_kernel_stack_frame().entry_symbol
        host_report = _host_compute_report(cb, lowered_mlir_text, entry_symbol=entry_symbol)
        record["host_compute"] = host_report.to_dict()
        try:
            require_clean_host_compute(host_report)
        except (HostComputeUnverified, HostComputeViolation) as exc:
            record["host_compute_qualification"] = {
                "status": "unverified" if isinstance(exc, HostComputeUnverified) else "violation",
                "detail": str(exc),
            }
        else:
            record["host_compute_qualification"] = {"status": "clean"}
        from merlin.targetgen.contract.compile import compile_lowered_to_elf

        record["host_build_sources"] = [{"path": p, "sha256": digest} for p, digest in service.source_pins]
        elf = compile_lowered_to_elf(
            cb, lowered_mlir_text, output / "build", target=target, inputs=inputs,
            _build_service=service)
        record["elf"] = _digest(Path(elf))
        revalidate_source()
        if rtl_facts is not None and board_config:
            from merlin.compile.model_execution_inputs import selected_firrtl

            record["rtl_facts"] = selected_firrtl(rtl_facts, target=target, config=board_config)
        if simulator is None:
            record["status"] = "compiled_not_run"
            return record
        if rtl_facts is None or not board_config:
            raise NativeModelExecutionError("native RTL execution requires selected facts and board config")
        from merlin.compile.model_execution_inputs import native_engine, selected_firrtl
        from merlin.runtime.backends import base as backends
        from merlin.runtime.commandbuffer import declared_output_dtypes
        from merlin.targetgen.capsule_golden import compare
        from merlin.targetgen.oracle_policy import selected_l3_engine_report

        selected = selected_l3_engine_report(target)
        if not selected.get("available") or selected.get("engine") != simulator:
            raise NativeModelExecutionError("requested simulator is not the selected RTL oracle")
        facts = selected_firrtl(rtl_facts, target=target, config=board_config)
        # The functional tier consumes the SAME already-linked candidate ELF.
        # It is independently optional as a diagnostic run, but a mandatory L2
        # remains unavailable in the grade if its engine cannot be byte-bound.
        try:
            functional_backend, functional_citation, revalidate_functional = _functional_engine(target)
            revalidate_functional()
            functional_console = functional_backend.run_elf(elf, simulator="spike", timeout=timeout)
            revalidate_functional()
            revalidate_source()
            if _digest(Path(elf)) != record["elf"]:
                raise NativeModelExecutionError("candidate ELF changed during L2 functional execution")
            functional_path = output / "console_l2.txt"
            functional_path.write_text(functional_console, encoding="utf-8")
            functional_observed, functional_metrics = functional_backend.parse_output(functional_console)
            functional_observed = backends.decode_float_readback(
                functional_observed, declared_output_dtypes(cb))
            functional_numeric = compare(
                golden["outputs"], functional_observed, policy,
                golden_source=str(golden.get("golden_source") or "independent_capsule"))
            record.setdefault("tiers", {})["L2"] = {
                "status": "pass" if functional_numeric.get("status") == "pass" else "fail",
                "engine": "spike", "engine_citation": functional_citation,
                "elf": record["elf"], "console": _digest(functional_path),
                "numeric": functional_numeric, "metrics": functional_metrics,
                "scope": "same candidate ELF on selected functional engine; independent full golden",
            }
        except Exception as exc:  # noqa: BLE001 -- no substitution for unavailable L2
            record.setdefault("tiers", {})["L2"] = {
                "status": "unavailable", "detail": f"{type(exc).__name__}: {exc}",
                "elf": record["elf"],
            }
        backend, engine, revalidate, prepare = native_engine(target, simulator, facts)
        record["rtl_facts"] = facts
        record["simulator_provenance"] = {"selection": selected, "citation": engine}
        command = (prepare(elf, expected_elf_sha256=record["elf"]["sha256"],
                           expected_engine_provenance=engine) if prepare is not None else None)
        if command is not None:
            record["native_command"] = command.to_evidence()
        revalidate()
        if simulator == "gsim":
            from merlin.targetgen.rtl_engine_policy import gsim_runtime_slot

            with gsim_runtime_slot(wait_timeout_s=timeout):
                console = backend.run_elf(elf, simulator=simulator, timeout=timeout)
        else:
            console = backend.run_elf(elf, simulator=simulator, timeout=timeout)
        if command is not None:
            command.revalidate()
        revalidate()
        revalidate_source()
        if _digest(Path(elf))["sha256"] != record["elf"]["sha256"]:
            raise NativeModelExecutionError("candidate ELF bytes changed during execution")
        console_path = output / "console.txt"
        console_path.write_text(console, encoding="utf-8")
        observed, metrics = backend.parse_output(console)
        observed = backends.decode_float_readback(observed, declared_output_dtypes(cb))
        comparison = compare(golden["outputs"], observed, policy,
                             golden_source=str(golden.get("golden_source") or "independent_capsule"))
        record.update(console=_digest(console_path), metrics=metrics, numeric=comparison,
                      status="numeric_match_diagnostic" if comparison["status"] == "pass"
                      else "numeric_mismatch_diagnostic")
        record.setdefault("tiers", {})["L3"] = {
            "status": "pass" if comparison.get("status") == "pass" else "fail",
            "engine": simulator, "engine_citation": engine, "selection": selected,
            "elf": record["elf"], "console": record["console"],
            "numeric": comparison, "metrics": metrics,
            "scope": "same candidate ELF on selected RTL engine; independent full golden",
        }
        return record
    except Exception as exc:  # noqa: BLE001 -- always retain a build/receipt on failure
        record["failure"] = {"type": type(exc).__name__, "detail": str(exc)[:2000]}
        return record
    finally:
        record["build_artifacts"] = _build_artifacts(output)
        receipt.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
