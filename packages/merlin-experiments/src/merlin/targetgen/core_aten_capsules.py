"""Materialize and grade captured single-call capsules with the Core ATen boundary contract.

The batch builder owns result projection, mutation returns and semantic I/O. This adapter
only separates public compiler inputs from private answers and joins the existing capsule
entrypoints to that boundary grader. It does not provide a candidate compiler.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

import yaml

from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.core_aten_batch import build_core_aten_batch, caller_boundary_layouts
from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch
from merlin.targetgen.core_aten_capture import case_capture_name
from merlin.targetgen.core_aten_cover import _provenance_ops
from merlin.targetgen.golden_store import FILES, load_golden, write_golden
from merlin.targetgen.package_runtime import CONTRACT_VERSION

CONTRACT = "full_call_v1"


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path, document) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def is_full_call(capsule: dict) -> bool:
    return (capsule.get("operation", {}).get("attributes", {})).get("core_aten_contract") == CONTRACT


def write_capsules(corpus: dict, captures: Path, public: Path, private: Path, *, label="public", tiers=("L2",)):
    """Write one capsule per case, using precisely the round-2 result projection.

    ``private`` must be outside the compiler-input tree. No seal or hardware admission is
    implied. Additive bounded case IDs retain the capture tool's content-derived names.
    """
    public, private, captures = public.resolve(), private.resolve(), captures.resolve()
    if private.is_relative_to(public) or public.is_relative_to(private):
        raise ValueError("public inputs and private answers require disjoint roots")
    if label not in {"public", "dev", "hidden"} or not tiers:
        raise ValueError("explicit label and nonempty oracle tiers required")
    capture_names = [case_capture_name(case) for case in corpus["cases"]]
    names = [name.replace("-", "_") for name in capture_names]
    if len(names) != len(set(names)):
        raise ValueError("duplicate capsule identifiers")
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True, mode=0o700)
    private.chmod(0o700)
    rows = []
    for case, name, capture_name in zip(corpus["cases"], names, capture_names):
        source = captures / capture_name
        destination, answers = public / name, private / name
        if destination.exists() or answers.exists():
            raise ValueError(f"refusing to overwrite capsule {name}")
        capture = json.loads((source / "capture.json").read_text())
        if capture.get("overload") != case["overload"]:
            raise ValueError(f"capture overload differs from case: {name}")
        if not capture["capture_meta"].get("result_contract"):
            raise ValueError(f"full-call capsule lacks same-conversion boundary: {name}")
        with tempfile.TemporaryDirectory(prefix="capsule-", dir=private) as scratch:
            bundle = Path(scratch)
            report = build_core_aten_batch({**corpus, "cases": [case]}, captures, bundle)
            if report["bundled_count"] != 1 or not (bundle / "semantic_io.json").is_file():
                raise ValueError(f"full-call capsule did not bundle: {name}: {report['cases']}")
            if case["overload"] not in _provenance_ops(bundle / "model.mlir"):
                raise ValueError(f"full-call interface lacks exact overload provenance: {name}")
            destination.mkdir(mode=0o700 if label == "hidden" else 0o755)
            answers.mkdir(mode=0o700)
            shutil.copyfile(bundle / "model.mlir", destination / "capsule.interface.mlir")
            for filename in ("inputs.npz", "input_order.json", "weights.safetensors.manifest.json"):
                shutil.copyfile(bundle / filename, destination / filename)
            shutil.copyfile(source / "capsule.pytorch.py", destination / "capsule.pytorch.py")
            # The public ABI contains no expected results, mutation observations or answers.
            record = report["cases"][0]
            _json(
                destination / "call.json",
                {
                    "schema_version": 1,
                    "overload": case["overload"],
                    "input_abi": record["input_abi"],
                    "output_abi": record["output_abi"],
                    "output_indices": record["output_indices"],
                    "post_indices": record["semantic_boundary"]["post_indices"],
                    "comparison": case["comparison"],
                    "comparison_parameters": case.get("comparison_parameters", {}),
                    "output_count": report["output_count"],
                },
            )
            public_files = {p.name: _sha(p) for p in sorted(destination.iterdir())}
            # Retain original evidence host-side, without copying capture-work trees.
            _json(answers / "capture.json", capture)
            write_golden(
                answers,
                {
                    "schema": "merlin.core_aten_capsule.v1",
                    "batch_map": report,
                    "semantic_io": json.loads((bundle / "semantic_io.json").read_text()),
                    "public_files": public_files,
                },
            )
            binding = {filename: _sha(answers / filename) for filename in FILES if (answers / filename).is_file()}
            parameters = case.get("comparison_parameters") or {}
            dtypes = [item["dtype"] for item in record["output_abi"]]
            floating = any(dtype.startswith(("f", "complex<")) for dtype in dtypes)
            policy_dtype = next((dtype for dtype in dtypes if dtype.startswith(("f", "complex<"))), dtypes[0])
            if policy_dtype.startswith("complex<"):
                policy_dtype = policy_dtype.removeprefix("complex<").removesuffix(">")
            declaration = {
                "name": name,
                "kind": "model_slice",
                "source_role": "pytorch_model_slice",
                "label": label,
                "source_reference": case["overload"],
                "interface_mlir": "capsule.interface.mlir",
                "operation": {
                    "op": "model",
                    "attributes": {
                        "core_aten_contract": CONTRACT,
                        "overload": case["overload"],
                    },
                },
                "numeric_policy": {
                    "compare": "tolerance_float" if floating else "exact_int",
                    "dtype": policy_dtype,
                    "rtol": float(parameters.get("rtol", 0)),
                    "atol": float(parameters.get("atol", 0)),
                },
                "expected": {"instruction_classes": []},
                "required_oracle_tiers": list(tiers),
                "full_call_golden": binding,
            }
            (destination / "capsule.yaml").write_text(yaml.safe_dump(declaration, sort_keys=False))
            load_capsule(destination)
            for path in answers.iterdir():
                path.chmod(0o600)
            if label == "hidden":
                for path in destination.iterdir():
                    path.chmod(0o600)
            rows.append(
                {
                    "id": name,
                    "overload": case["overload"],
                    "case_id": case.get("case_id", case["overload"]),
                    "digests": {**public_files, "capsule.yaml": _sha(destination / "capsule.yaml")},
                    "golden_binding": binding,
                    "provenance": {
                        "capture_sha256": _sha(source / "capture.json"),
                        "capture_source_sha256": _sha(source / "capsule.linalg.mlir"),
                        "model2mlir_revision": capture.get("model2mlir_revision"),
                        "captured_overloads": capture.get("captured_overloads"),
                        "capture_abi_version": capture["capture_meta"].get("capture_abi_version"),
                        "retained_capture_provenance": capture.get("provenance"),
                    },
                }
            )
    manifest = {
        "schema": "merlin.core_aten_capsules.v1",
        "label": label,
        "count": len(rows),
        "capsules": rows,
        "provenance": {
            "scope": "packaging retained captures; no execution or hardware verdict",
            "denominator_sha256": corpus.get("denominator_sha256"),
        },
    }
    _json(public / "manifest.json", manifest)
    if label == "hidden":
        (public / "manifest.json").chmod(0o600)
    return manifest


def load_private(capsule: dict, private_root: Path | None = None) -> dict:
    """Verify the answer commitment and every public compiler-input byte before grading."""
    root = Path(capsule["__dir__"]).resolve()
    selected = private_root or os.environ.get("MERLIN_CORE_ATEN_PRIVATE_ROOT")
    answers = (Path(selected) if selected else root.parent.parent / "_private" / root.parent.name) / capsule["name"]
    if answers.is_symlink() or not answers.is_dir():
        raise ValueError("private full-call answer directory is absent or indirect")
    binding = capsule.get("full_call_golden")
    if not isinstance(binding, dict) or "golden.yaml" not in binding or set(binding) - set(FILES):
        raise ValueError("full-call golden binding is absent or malformed")
    for filename, digest in binding.items():
        if (answers / filename).is_symlink() or _sha(answers / filename) != digest:
            raise ValueError("private full-call answer digest changed")
    golden = load_golden(answers)
    if golden.get("schema") != "merlin.core_aten_capsule.v1":
        raise ValueError("private full-call record schema differs")
    for filename, digest in golden["public_files"].items():
        if Path(filename).name != filename or (root / filename).is_symlink() or _sha(root / filename) != digest:
            raise ValueError(f"public full-call input changed: {filename}")
    report = golden["batch_map"]
    if len(report["cases"]) != 1 or report["cases"][0]["overload"] != capsule["operation"]["attributes"]["overload"]:
        raise ValueError("private case binding differs from declaration")
    return golden


def grade_readback(
    capsule: dict,
    output_bytes,
    *,
    output_shapes=None,
    semantic_readback=None,
    execution_error=None,
    private_root=None,
    provenance=None,
) -> dict:
    """The same numeric and full-call semantic checker used by the round-2 campaign."""
    golden = load_private(capsule, private_root)
    report = golden["batch_map"]
    if provenance is not None:
        report["provenance"] = provenance
    if output_bytes is not None and len(output_bytes) != report["output_count"]:
        execution_error = "full-call execution did not emit exactly the declared result count"
    if output_shapes and len(output_shapes) != report["output_count"]:
        execution_error = "full-call execution result shape frame count differs from the declared ABI"
    verdict = grade_core_aten_batch(
        report,
        output_bytes,
        output_shapes=output_shapes or None,
        semantic_readback=semantic_readback,
        execution_error=execution_error,
    )
    row = next(iter(verdict["cases"].values()))
    # An absent boundary can never silently downgrade this capsule to numeric-only.
    if row["status"] == "pass" and row.get("semantic_scope") != "full":
        raise ValueError("full-call capsule cannot earn a numeric-only pass")
    return verdict


def runtime_bundle(capsule: dict, destination: Path) -> dict:
    golden = load_private(capsule)
    root = Path(capsule["__dir__"])
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "capsule.interface.mlir", destination / "model.mlir")
    for filename in ("inputs.npz", "input_order.json", "weights.safetensors.manifest.json"):
        shutil.copyfile(root / filename, destination / filename)
    # Reconstruct caller layouts also for previously sealed normalized captures.
    # The sealed records remain immutable; only runtime boundary metadata changes.
    semantic_io = caller_boundary_layouts(
        (destination / "model.mlir").read_text(), golden["semantic_io"], golden["batch_map"]["cases"]
    )
    _json(destination / "semantic_io.json", semantic_io)
    return golden["batch_map"]


def run_capsule(capsule, package_dir, *, paths, config, adapters, pkg, contract, timeout, no_oracle):
    """Compile through the standard package ABI; compare executed candidate boundary bytes.

    An adapter may implement ``run_full_call(bundle, llvm_mlir, target, timeout)`` for
    its substrate, returning lossless output_bytes/output_shapes/semantic_readback plus
    provenance. An ordinary command-buffer adapter cannot certify this boundary ABI.
    """
    from merlin.common import provenance
    from merlin.targetgen.capsule_common import run_entrypoints
    from merlin.targetgen.tier_policy import tier_depth_order

    paths.run_path.mkdir(parents=True, exist_ok=True)
    result = {
        "capsule": capsule["name"],
        "kind": capsule["kind"],
        "label": capsule["label"],
        "contract_version": CONTRACT_VERSION,
        "cohort": "guard" if capsule.get("cohort") == "host_guard" else capsule.get("cohort", capsule["label"]),
        "scored": capsule.get("scored", True),
        "lane": "host",
        "executed_instructions": 0,
        "status": "incomplete",
        "tiers": {},
        "numeric": {"status": "skipped"},
        "trace_check": {"status": "skipped", "violations": []},
        "failure": None,
        "provenance": provenance.record(
            sources=[Path(__file__), Path(capsule["__dir__"]) / "capsule.yaml"],
            extra={"target": config.target, "executions": {}},
        ),
    }
    try:
        source_adapters = [a for a in (adapters or {}).values() if getattr(a, "compiles_source_bundle", False)]
        if source_adapters:
            cb, emission = None, ""
        else:
            pkg, cb, emission = run_entrypoints(
                pkg,
                package_dir,
                capsule,
                paths,
                contract=contract,
                timeout=timeout,
                fourth_output_name=config.fourth_output_name,
            )
        if not source_adapters:
            result["candidate_emission"] = {"sha256": hashlib.sha256(emission.encode()).hexdigest()}
        else:
            result["compilation_mode"] = "source_bound_submitted_device_catalog"
        bundle = paths.run_path / ".private_full_call_runtime"
        bundle.mkdir(mode=0o700, exist_ok=True)
        runtime_bundle(capsule, bundle)
        for tier in tier_depth_order(capsule["required_oracle_tiers"]):
            adapter = (adapters or {}).get(tier)
            execute = getattr(adapter, "run_full_call", None)
            if no_oracle or execute is None:
                result["tiers"][tier] = {
                    "status": "unavailable",
                    "mandatory": True,
                    "not_run_is_not_pass": True,
                    "detail": "selected oracle does not implement full-call boundary execution",
                    "reason": "selected oracle does not implement full-call boundary execution",
                }
                result["failure"] = {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS", "tier": tier}
                break
            kwargs = (
                {"package_dir": package_dir, "capsule": capsule}
                if getattr(adapter, "compiles_source_bundle", False)
                else {}
            )
            run = execute(
                bundle=bundle, llvm_mlir=emission, command_buffer=cb, target=config.target, timeout=timeout, **kwargs
            )
            result.update(lane=run.get("lane", "host"), executed_instructions=run.get("executed_instructions", 0))
            if run.get("execution_evidence") is not None:
                result["execution_evidence"] = run["execution_evidence"]
            if not isinstance(run.get("provenance"), dict):
                raise ValueError("full-call execution lacks provenance")
            result["provenance"]["executions"][tier] = run["provenance"]
            verdict = grade_readback(
                capsule,
                run.get("output_bytes"),
                output_shapes=run.get("output_shapes"),
                semantic_readback=run.get("semantic_readback"),
                execution_error=run.get("execution_error"),
                provenance=run["provenance"],
            )
            row = next(iter(verdict["cases"].values()))
            execution_failed = row["status"] == "execution_failed"
            passed = row["status"] == "pass"
            if capsule.get("lane_expectation") == "device":
                passed = passed and result["lane"] == "device" and result["executed_instructions"] > 0
            elif capsule.get("lane_expectation") == "host-guard":
                passed = passed and result["lane"] == "host" and result["executed_instructions"] == 0
            result["tiers"][tier] = {
                "status": "pass" if passed else "fail",
                "mandatory": True,
                "not_run_is_not_pass": False,
                "derived_from_rtl": bool(run.get("derived_from_rtl")),
                "engine": run.get("engine"),
                "semantic_scope": row.get("semantic_scope"),
                "provenance": run["provenance"],
                "reason": row["reason"]
                if execution_failed
                else None
                if passed
                else "compiled full-call boundary or execution lane differs from the call contract",
            }
            result["numeric"] = {
                "status": "skipped" if execution_failed else "pass" if passed else "fail",
                "semantic_scope": row.get("semantic_scope"),
            }
            if tier in config.rtl_tiers and run.get("derived_from_rtl") is not True:
                result["tiers"][tier]["status"] = "unavailable"
                result["tiers"][tier]["not_run_is_not_pass"] = True
                result["tiers"][tier]["reason"] = "selected oracle did not derive execution from RTL"
                result["failure"] = {"plane": "oracle_unavailable", "category": "NOT_RUN_IS_NOT_PASS", "tier": tier}
                break
            if not passed:
                result["status"] = "fail"
                # Full observations and expected values remain exclusively in the private runtime.
                result["failure"] = {
                    "plane": "compile" if execution_failed else "numeric",
                    "category": "PROTOCOL_VIOLATION" if execution_failed else "NUMERIC_MISMATCH",
                    "tier": tier,
                    "detail": row["reason"]
                    if execution_failed
                    else "compiled full-call boundary differs from the call contract",
                }
                _json(bundle / "private-verdict.json", verdict)
                break
        else:
            result["status"] = "pass"
    except Exception as exc:  # noqa: BLE001 -- every capsule gets a durable refusing verdict
        result["status"] = "fail"
        result["failure"] = {"plane": "compile", "category": "PROTOCOL_VIOLATION", "detail": str(exc)[:1000]}
    if no_oracle:
        result["status"] = "not_gradeable_no_oracle"
    from merlin.targetgen.contract import schemas

    schemas.validate(result, "capsule_result", contract=contract)
    _json(paths.run_path / "capsule_result.json", result)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--private", type=Path, required=True)
    parser.add_argument("--label", choices=("public", "dev", "hidden"), default="public")
    parser.add_argument("--tier", action="append", required=True)
    args = parser.parse_args(argv)
    manifest = write_capsules(
        json.loads(args.corpus.read_text()), args.captures, args.public, args.private, label=args.label, tiers=args.tier
    )
    print(f"Packaged {manifest['count']} full-call capsules")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
