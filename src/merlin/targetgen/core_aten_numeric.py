"""Opt-in Core ATen numeric policy preparation and separate approximation metrics."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def prepare_w8a8(directory: Path, report: dict, *, triples, ranks) -> dict:
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.contraction_quantization import SPEC, quantize_module, selected_contractions

    module = parse_mlir_file(directory / "model.mlir")
    selected = list(selected_contractions(module, triples, ranks))
    if not selected:
        return {"quantized_contractions": 0, "scales": []}
    # Single-case attribution is mandatory for both oracle replacement and device evidence.
    records = [r for r in report["cases"] if r["status"] == "bundled"]
    if len(records) != 1:
        raise ValueError("W8A8 execution requires one case per shard")
    original = directory / "model.before_w8a8.mlir"
    original.write_bytes((directory / "model.mlir").read_bytes())
    operations = list(module.walk())
    request = dict(
        selected_ordinals=[operations.index(op) for op in selected],
        quant_min=SPEC.quant_min,
        quant_max=SPEC.quant_max,
        eps=SPEC.eps,
    )
    (directory / "w8a8-request.json").write_text(json.dumps(request, indent=2) + "\n")
    record = records[0]
    record["fp32_case"] = copy.deepcopy(record["case"])
    record["case"] = copy.deepcopy(record["case"])
    record["w8a8_policy"] = policy_record()
    accumulators = []
    count = quantize_module(module, selected, accumulators=accumulators)
    record["w8a8_accumulators"] = accumulators
    record["w8a8_selected_ordinals"] = request["selected_ordinals"]
    report["output_count"] += len(accumulators)
    # Add observation-only boundary results without changing user result indices.
    semantic = directory / "semantic_io.json"
    if semantic.exists():
        data = json.loads(semantic.read_text())
        data["outputs"].extend([None] * len(accumulators))
        semantic.write_text(json.dumps(data, indent=2) + "\n")
    (directory / "model.mlir").write_text(str(module) + "\n")
    return dict(quantized_contractions=count, scales=[], policy=record["w8a8_policy"])


def complete_w8a8_reference(directory, report, dispatch):
    """Qualify only after source-bound provider selection exposes actual primitives."""
    from merlin.common.paths import python_import_roots
    from merlin.llvmlower.toolchain import m2m_python
    from merlin.targetgen import core_aten_w8a8_reference
    from merlin.targetgen.core_aten_batch import _tensor_documents

    request_path = directory / "w8a8-request.json"
    request = json.loads(request_path.read_text())
    request["dispatch"] = dispatch
    request_path.write_text(json.dumps(request, indent=2) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in python_import_roots())
    process = subprocess.run(
        [str(m2m_python()), "-m", core_aten_w8a8_reference.__name__, str(directory.resolve())],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (directory / "w8a8-reference.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise ValueError(f"independent W8A8 reference failed: {process.stderr[-2000:]}")
    reference = json.loads((directory / "w8a8-reference.json").read_text())
    record = next(r for r in report["cases"] if r["status"] == "bundled")
    record["scales"] = reference["scales"]
    record["reference_engine"] = reference["engine"]
    record["w8a8_reference_bytes"] = reference["output_bytes"]
    record["w8a8_dispatch"] = dispatch
    expected = _tensor_documents(record["case"]["expected"])
    if len(expected) != len(record["output_indices"]):
        raise ValueError("W8A8 oracle needs tensor output documents")

    def reference_values(document, index):
        values = reference["outputs"][index]
        if document["dtype"] in {"complex64", "complex128"}:
            packed = np.asarray(values)
            if packed.shape != (*document["shape"], 2):
                raise ValueError("complex oracle output does not match the physical ABI")

            def encode(value):
                if len(value) == 2 and all(isinstance(v, (float, int)) for v in value):
                    return dict(kind="complex", real=value[0], imag=value[1])
                return [encode(v) for v in value]

            values = encode(values)
        return values

    for document, index in zip(expected, record["output_indices"], strict=True):
        document["values"] = reference_values(document, index)
    if record.get("semantic_boundary"):
        posts = _tensor_documents(record["case"]["post_arguments"])
        for document, index in zip(posts, record["semantic_boundary"]["post_indices"], strict=True):
            document["values"] = reference_values(document, index)
    record["case"]["comparison_parameters"] = dict(rtol=0.0, atol=0.0, equal_nan=False, storage_bits=True)
    report["numeric_preparation"].update(
        scales=reference["scales"], reference_engine=reference["engine"], dispatch=dispatch
    )


def policy_record():
    from merlin.llvmlower.contraction_quantization import SPEC

    return dict(
        name="int8_w8a8",
        operand_range=[SPEC.quant_min, SPEC.quant_max],
        eps=SPEC.eps,
        granularity="dynamic_per_tensor",
        quantization="f32_reciprocal_first_roundeven_clamp",
        dequantization="f32_acc_cast_then_f64_scale_product_and_multiply_then_one_f32_round_then_storage_cast",
        nonfinite_inputs="sanitize_before_integer_conversion_and_propagate_canonical_nan_to_entire_contraction",
        nonrepresentable_finite_result="runtime_assert_and_oracle_refusal",
        subnormals="quantize_to_zero_via_eps_floor",
        signed_zero="distinct_output_bit_patterns",
        correctness_gate="raw_i32_accumulator_and_final_float_storage_bits",
        nan_comparison="canonical_nan_storage_bits_only",
    )


def validate_primitive_dispatch(numeric, *, full_k, primitive_ks):
    from merlin.llvmlower.contraction_quantization import SPEC
    from merlin.targetgen.operation_numerics import integer_split_k_limit

    bound = integer_split_k_limit(numeric, tile_dim=1, full_k=full_k, operand_range=(SPEC.quant_min, SPEC.quant_max))
    if bound is None or not primitive_ks:
        raise ValueError("W8A8 dispatch lacks selected internal arithmetic or observed primitive K")
    if any(type(k) is not int or not 0 < k <= bound["max_true_k"] for k in primitive_ks):
        raise ValueError("W8A8 dispatched primitive K exceeds internal MAC partial-sum bound")
    return dict(full_k=full_k, primitive_ks=primitive_ks, numerical_semantics=numeric["numerical_semantics"], **bound)


def checked_w8a8_builder(builder, *, target, package, directory, report):
    """Screen the actual emitted target IR before catalog selection and linking.

    Operation/extent names are target-owned observation data, never ISA guesses.
    The catalog's target IR hash binds the observations to the compiled object.
    """
    import hashlib

    from xdsl.dialects.llvm import LLVM
    from xdsl.parser import Parser

    from merlin.common.paths import repo_root
    from merlin.frontends.linalg_mlir import make_context
    from merlin.targetgen.software_spec import load_software_spec, software_spec_path_for_recipe
    from merlin.targetgen.target_experiment import descriptor_for, load_target_experiment

    te = load_target_experiment(descriptor_for(target))
    spec_path = software_spec_path_for_recipe(repo_root() / te.numeric_profile)
    spec = load_software_spec(spec_path, target=target)
    if spec.get("status") != "reviewed":
        raise ValueError("W8A8 requires a reviewed internal arithmetic declaration")
    observation = spec["numerical_semantics"]["internal_arithmetic"].get("primitive_k_observation")
    if not observation:
        raise ValueError("W8A8 target lacks primitive K observation contract")

    def checked(prepared, workdir):
        manifest_path, object_path = builder(prepared, workdir)
        manifest = json.loads(Path(manifest_path).read_text())
        digest = manifest["compilation"]["target_ir_sha256"]
        sources = [p for p in Path(workdir).glob("*.mlir") if hashlib.sha256(p.read_bytes()).hexdigest() == digest]
        if len(sources) != 1:
            raise ValueError("W8A8 catalog lacks a unique hash-bound target IR")
        ctx = make_context()
        ctx.load_dialect(LLVM)
        from merlin.targetgen.plugins import load_module

        identity = hashlib.sha256(str(Path(package).resolve()).encode()).hexdigest()
        dialect_module = load_module(
            package, observation["dialect_module"], package_name=f"core_aten_backend_{identity}"
        )
        ctx.load_dialect(getattr(dialect_module, observation["dialect_symbol"]))
        module = Parser(ctx, sources[0].read_text()).parse_module()
        primitives, callees = {}, {}
        for function in module.body.block.ops:
            if function.name != "llvm.func":
                continue
            ks, calls = [], []
            for op in function.walk():
                name = op.op_name.data if op.name == "builtin.unregistered" else op.name
                if op.name == "llvm.call":
                    calls.append(op.callee.root_reference.data)
                if name == observation["operation"]:
                    value = op.attributes.get(observation["extent_attribute"]) or op.properties.get(
                        observation["extent_attribute"]
                    )
                    if value is None or not hasattr(value, "value"):
                        raise ValueError("W8A8 primitive has no static declared K extent")
                    ks.append(value.value.data)
            primitives[function.sym_name.data] = ks
            callees[function.sym_name.data] = calls

        def reachable_primitives(symbol, visited=()):
            if symbol in visited or symbol not in primitives:
                raise ValueError("W8A8 catalog has recursive or unobserved primitive calls")
            return primitives[symbol] + [
                k for callee in callees[symbol] for k in reachable_primitives(callee, (*visited, symbol))
            ]

        kernels = {k["symbol"]: k for k in manifest["kernels"]}
        # Follow actual LLVM calls rather than assuming a batched wrapper layout.
        dispatch_by_region = {}
        for binding in manifest["bindings"]:
            kernel = kernels[binding["symbol"]]
            proof = validate_primitive_dispatch(
                spec, full_k=kernel["dimensions"]["k"], primitive_ks=reachable_primitives(binding["symbol"])
            )
            proof.update(
                target_ir_sha256=digest,
                software_spec_path=str(spec_path.resolve()),
                target_ir_path=str(sources[0].resolve()),
                object_path=str(Path(object_path).resolve()),
                software_spec_sha256=hashlib.sha256(spec_path.read_bytes()).hexdigest(),
                object_sha256=hashlib.sha256(Path(object_path).read_bytes()).hexdigest(),
                observation=observation,
            )
            dispatch_by_region[binding["region"]] = proof
        record = next(r for r in report["cases"] if r["status"] == "bundled")
        dispatch = [dispatch_by_region[f"w8a8_contraction_{ordinal}"] for ordinal in record["w8a8_selected_ordinals"]]
        complete_w8a8_reference(directory, report, dispatch)
        (directory / "w8a8-dispatch.json").write_text(json.dumps(dispatch, indent=2) + "\n")
        return manifest_path, object_path

    return checked


def exact_w8a8_gate(record, outputs):
    """Compare raw readout words and final storage bytes, retaining signed zero."""
    if len(record.get("w8a8_accumulators", [])) != len(record["scales"]):
        return dict(
            name="raw_i32_accumulator_and_final_float_storage_bits",
            passed=False,
            reason="raw accumulator observation roster differs from oracle",
            accumulators=[],
            final_float_storage=[],
        )
    accumulator_details = []
    for abi, site in zip(record.get("w8a8_accumulators", []), record["scales"], strict=True):
        expected = np.asarray(site["accumulator"], dtype=np.int32).reshape(abi["shape"]).tobytes()
        raw = outputs[abi["index"]] if abi["index"] < len(outputs) else None
        accumulator_details.append(dict(index=abi["index"], passed=raw == expected))
    final = []
    references = record.get("w8a8_reference_bytes", [])
    indices = [*record["output_indices"], *record.get("semantic_boundary", {}).get("post_indices", [])]
    for index in dict.fromkeys(indices):
        passed = index < len(references) and index < len(outputs) and outputs[index].hex() == references[index]
        final.append(dict(index=index, passed=passed))
    passed = bool(accumulator_details) and all(d["passed"] for d in [*accumulator_details, *final])
    return dict(
        name="raw_i32_accumulator_and_final_float_storage_bits",
        passed=passed,
        accumulators=accumulator_details,
        final_float_storage=final,
    )


def fp32_error(record, outputs) -> list[dict]:
    """Report compiled-output error vs original golden, without affecting correctness."""
    from merlin.targetgen.core_aten_batch_grade import _DTYPES, _complex_values, _leaves

    errors = []
    for index, abi, expected in zip(
        record["output_indices"], record["output_abi"], _leaves(record["fp32_case"]["expected"]), strict=True
    ):
        complex_type = {"complex<f32>": np.complex64, "complex<f64>": np.complex128}.get(abi["dtype"])
        compute_type = np.complex128 if complex_type else np.float64
        actual = np.frombuffer(outputs[index], complex_type or _DTYPES[abi["dtype"]]).astype(compute_type)
        golden = np.asarray(_complex_values(expected["values"]), dtype=compute_type).reshape(-1)
        difference = np.abs(actual - golden)
        relative = np.divide(difference, np.abs(golden), out=np.where(difference == 0, 0.0, np.inf), where=golden != 0)
        norm = float(np.linalg.norm(actual) * np.linalg.norm(golden))
        cosine = float(np.real(np.vdot(actual, golden)) / norm) if norm else float(np.array_equal(actual, golden))
        error_energy = float(np.sum(difference**2))
        golden_energy = float(np.sum(np.abs(golden) ** 2))
        rmse = float(np.sqrt(error_energy / difference.size)) if difference.size else 0.0
        sqnr = (
            float(10 * np.log10(golden_energy / error_energy))
            if golden_energy and error_energy
            else (float("inf") if error_energy == 0 else float("-inf"))
        )
        normalized_l2 = (
            float(np.sqrt(error_energy / golden_energy))
            if golden_energy
            else (0.0 if error_energy == 0 else float("inf"))
        )
        # An output epilogue can combine several contraction sites. Report each
        # site's LSB explicitly; no invented single scale for composed outputs.
        lsbs = [site.get("output_lsb", float(site["lhs"]) * float(site["rhs"])) for site in record["scales"]]
        errors.append(
            dict(
                max_abs=float(np.max(difference, initial=0)),
                max_rel=float(np.max(relative, initial=0)),
                relative_inf_at_zero_golden=int(np.sum((golden == 0) & (difference != 0))),
                rmse=rmse,
                sqnr_db=sqnr,
                cosine=cosine,
                normalized_l2=normalized_l2,
                error_in_output_lsbs=[
                    dict(site=i, lsb=lsb, max_abs=float(np.max(difference, initial=0) / lsb), rmse=rmse / lsb)
                    for i, lsb in enumerate(lsbs)
                ],
                contraction_diagnostics=[
                    dict(
                        operands=site.get("operands"),
                        accumulator=site.get("accumulator_metrics"),
                        k_split=site.get("k_split"),
                    )
                    for site in record["scales"]
                ],
            )
        )
    return errors
