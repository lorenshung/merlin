"""Opt-in Core ATen numeric policy preparation and separate approximation metrics."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def prepare_spec_int8(
    directory: Path,
    report: dict,
    *,
    target: str,
    facts: dict,
    software_spec: Path,
    host_capabilities: Path,
    provider=None,
    package=None,
):
    """Emit admitted sites and preserve independently refused source contractions."""
    import hashlib

    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.targetgen.core_aten_spec_int8 import admission_plan, select_spec

    selection = select_spec(target, facts, software_spec, host_capabilities)
    original = (directory / "model.mlir").read_bytes()
    module = parse_mlir_file(directory / "model.mlir")
    plan = admission_plan(module, selection)
    plan["source_sha256_before"] = hashlib.sha256(original).hexdigest()
    selected = [site for site in plan["sites"] if site["status"] == "admitted"]
    recipes = [
        r
        for r in selection["recipes"]
        if r["activation"]
        and r["activation"]["dtype"] == "int8"
        and r["activation"]["mode"] == "static"
        and r["activation"]["granularity"] == "tensor"
        and r["activation"]["symmetric"]
        and (r["activation"]["quant_min"], r["activation"]["quant_max"]) == (-128, 127)
    ]
    if selected and (len(recipes) != 1 or provider is None or not hasattr(provider, "admit_numeric_schedule")):
        for site in selected:
            site.update(
                status="refused",
                reasons=["static activation recipe or qualified provider schedule binding unavailable"],
            )
        selected = []
    reference = None
    if selected:
        (directory / "model.before_spec_int8.mlir").write_bytes(original)

        def declaration_keys(site):
            def key(step):
                row = next(row for row in site["steps"] if row["step"] == step)
                return json.dumps(
                    {
                        authority: row[authority]["declaration"]
                        for authority in ("software", "host")
                        if authority in row
                    },
                    sort_keys=True,
                )

            return dict(quantize=key(1), contraction=key(3), cast=key(5), mul=key(6))

        from merlin.targetgen.capsule_source import M2MUnavailable, PytorchRefSource
        from merlin.targetgen.core_aten_batch import build_core_aten_batch
        from merlin.targetgen.core_aten_capture import case_capture_name, loader_source
        from merlin.targetgen.quant_recipe import digest

        records = [r for r in report["cases"] if r["status"] == "bundled"]
        if len(records) != 1:
            raise ValueError("static capture needs single-case attribution")
        source_case = records[0]["case"]
        capture_root = directory / "capture-int8"
        capture_dir = capture_root / case_capture_name(source_case)
        capture_dir.mkdir(parents=True, exist_ok=True)
        loader = capture_dir / "capsule.pytorch.py"
        loader.write_text(loader_source(source_case))
        recipe = dict(
            recipes[0],
            accumulator_dtype=selection["readout_facets"][0]["accumulator_dtype"],
            software_admission=selection["spec"],
            software_numerical_engine="integer_reference",
            host_readout_declarations=declaration_keys(selected[0])["mul"],
        )
        recipe["recipe_sha256"] = digest(recipe)
        try:
            artifact = PytorchRefSource().capture_loader(loader, "int8", recipe=recipe, workdir=capture_dir / "work")
        except M2MUnavailable as exc:
            # Only the proved absence of an eligible frontend operation grants
            # unchanged host fallback. Other capture failures remain failures.
            if "the model has no operator the recipe's families cover" not in str(exc):
                raise
            for site in selected:
                site.update(
                    status="refused",
                    reasons=["no eligible contraction visible to PT2E; decomposition-only contractions stay on host"],
                )
            selected = []
        if selected:
            reference_path = Path(artifact.meta["weights_manifest"]).parent / "spec-int8-reference.json"
            reference = json.loads(reference_path.read_text())
            (directory / "spec-int8-reference.json").write_bytes(reference_path.read_bytes())
            (capture_dir / "capsule.linalg.mlir").write_text(artifact.linalg_mlir)
            (capture_dir / "inputs.json").write_text(json.dumps(artifact.inputs))
            (capture_dir / "golden.json").write_text(json.dumps(artifact.golden))
            (capture_dir / "capture.json").write_text(json.dumps({"capture_meta": artifact.meta}))
            prepared_dir = directory / "capture-bundle"
            prepared_report = build_core_aten_batch({"cases": [source_case]}, capture_root, prepared_dir)
            prepared_module = parse_mlir_file(prepared_dir / "model.mlir")
            captured_plan = admission_plan(prepared_module, selection, captured=True)
            if any(site["status"] != "admitted" for site in captured_plan["sites"]):
                # Refuse the complete capture before replacing the host source.
                plan = captured_plan
                selected = []
                plan["source_sha256_before"] = hashlib.sha256(original).hexdigest()
                plan["source_sha256_after"] = plan["source_sha256_before"]
                document = {"selection": selection, **plan}
                (directory / "spec-int8-admission.json").write_text(json.dumps(document, indent=2) + "\n")
                records[0].update(numeric_admission=plan, numeric_selection=selection["identities"])
                return document
            selected = captured_plan["sites"]
            if len(selected) != len(reference["scales"]):
                raise ValueError("captured contractions and independent PT2E reference census differ")
            module = prepared_module
            plan = captured_plan
            plan["source_sha256_before"] = hashlib.sha256(original).hexdigest()
            plan["capture_recipe"] = recipe
            plan["capture_meta"] = artifact.meta
            observations = []
            from xdsl.dialects.builtin import StringAttr

            from merlin.kernels.shapes import observe_contractions

            contractions = [op for op, _ in observe_contractions(module)]
            for site, contraction in zip(selected, contractions, strict=True):
                # Readout preserves two separately rounded frontend f32 multiplies.
                casts = [
                    u.operation
                    for u in contraction.results[0].uses
                    if any(o.name == "arith.sitofp" for o in u.operation.walk())
                ]
                if len(casts) != 1:
                    raise ValueError("captured contraction lacks a unique host i32 cast")
                cast = casts[0]
                readouts = [
                    u.operation for u in cast.results[0].uses if any(o.name == "arith.mulf" for o in u.operation.walk())
                ]
                if len(readouts) != 1:
                    raise ValueError("captured host cast lacks a unique scale multiply")
                first_readout = readouts[0]
                readouts = [
                    u.operation
                    for u in first_readout.results[0].uses
                    if any(o.name == "arith.mulf" for o in u.operation.walk())
                ]
                if len(readouts) != 1:
                    raise ValueError("captured readout lacks the second frontend scale multiply")
                readout = readouts[0]
                site["host_readout_operation_ordinals"] = [
                    list(module.walk()).index(o) for o in (first_readout, readout)
                ]
                keys = declaration_keys(site)
                for op, key in ((contraction, "contraction"), (cast, "cast"), (first_readout, "mul"), (readout, "mul")):
                    op.attributes["prov.declarations"] = StringAttr(keys[key])
                contraction.attributes["prov.region_id"] = StringAttr(
                    f"spec_int8_contraction_{site['source_operation_ordinal']}"
                )
                contraction.attributes["prov.operand_load"] = StringAttr("transfer_contracts.operand_load")
                contraction.attributes["prov.accumulator_readout"] = StringAttr(
                    "transfer_contracts.accumulator_readout"
                )
                readout.attributes["prov.role"] = StringAttr("spec_int8_host_readout")
                observations.append((contraction.results[0], readout.results[0]))
            operations = list(module.walk())
            for row in plan["quantization_sites"]:
                keys = {authority: row[authority]["declaration"] for authority in ("software", "host")}
                operations[row["source_operation_ordinal"]].attributes["prov.declarations"] = StringAttr(
                    json.dumps(keys, sort_keys=True)
                )
            proof = provider.admit_numeric_schedule(str(module), package=package, numeric_policy=selection["spec"])
            plan["schedule_proof"] = proof
            plan["quantized_contractions"] = len(selected)
            plan["calibration"].update(
                performed=True,
                recipe=recipe,
                graph="TorchAO PT2E prepared floating graph",
                scales_frozen_before_conversion=True,
            )
            # Keep the fresh capture's complete result ABI and boundary mapping.
            fresh = prepared_report["cases"][0]
            for index, value in zip(
                fresh.get("semantic_boundary", {}).get("post_indices", []), artifact.inputs, strict=True
            ):
                if index != len(reference["outputs"]):
                    raise ValueError("PT2E reference requires unmutated trailing input observations")
                reference["outputs"].append(value)
            records[0].update({k: v for k, v in fresh.items() if k not in ("routing", "capture_directory")})
            for name in ("inputs.npz", "input_order.json", "weights.safetensors.manifest.json", "semantic_io.json"):
                path = prepared_dir / name
                if path.exists():
                    (directory / name).write_bytes(path.read_bytes())
            report["output_count"] = prepared_report["output_count"]
        if not selected:
            plan["status"] = "refused"
        if selected:
            records = [r for r in report["cases"] if r["status"] == "bundled"]
            if len(records) != 1:
                raise ValueError("static int8 needs single-case attribution")
            record = records[0]
            record["fp32_case"] = copy.deepcopy(record["case"])
            record["reference_engine"] = reference["engine"]
            # Keep the full original result structure, including complex components.
            from merlin.targetgen.core_aten_batch import _tensor_documents

            record["case"] = copy.deepcopy(record["case"])

            def values(document, index):
                value = reference["outputs"][index]
                if document["dtype"] in {"complex64", "complex128"}:

                    def encode(v):
                        if len(v) == 2 and all(isinstance(x, (int, float)) for x in v):
                            return dict(kind="complex", real=v[0], imag=v[1])
                        return [encode(x) for x in v]

                    value = encode(value)
                document["values"] = value

            for document, index in zip(
                _tensor_documents(record["case"]["expected"]), record["output_indices"], strict=True
            ):
                values(document, index)
            if record.get("semantic_boundary"):
                for document, index in zip(
                    _tensor_documents(record["case"]["post_arguments"]),
                    record["semantic_boundary"]["post_indices"],
                    strict=True,
                ):
                    values(document, index)
            record["case"]["comparison_parameters"] = dict(rtol=0.0, atol=0.0, equal_nan=True)
            function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
            returned = next(op for op in function.body.block.ops if op.name == "func.return")
            originals = list(returned.operands)
            # Retain actual i32 and host readout buffers as additional grading outputs.
            returned.operands = [*originals, *(v for pair in observations for v in pair)]
            from xdsl.dialects.builtin import FunctionType

            function.properties["function_type"] = FunctionType.from_lists(
                [a.type for a in function.body.block.args], [v.type for v in returned.operands]
            )
            expected_bytes = {
                str(i): np.asarray(v, dtype=np.float32).tobytes().hex()
                for i, v in enumerate(reference["outputs"])
                if str(originals[i].type.element_type) == "f32"
            }
            for i, (site, scale) in enumerate(zip(selected, reference["scales"], strict=True)):
                acc_index = len(originals) + 2 * i
                expected_bytes[str(acc_index)] = np.asarray(scale["accumulator"], np.int32).tobytes().hex()
                expected_bytes[str(acc_index + 1)] = np.asarray(scale["readout"], np.float32).tobytes().hex()
                binding = next(
                    b
                    for b in proof["bindings"]
                    if b["region"] == f"spec_int8_contraction_{site['source_operation_ordinal']}"
                )
                schedule = next(k for k in proof["kernels"] if k["symbol"] == binding["symbol"])
                actual_bound = (
                    schedule["padded_primitive_k"]
                    * scale["operands"][0]["quantized_max_abs"]
                    * scale["operands"][1]["quantized_max_abs"]
                )
                site["accumulator_headroom"] = dict(
                    mesh_uniform_bound=schedule["max_partial_sum_bound"],
                    mesh_actual_operand_bound=actual_bound,
                    mesh_positive_limit=schedule["mac_positive_limit"],
                    uniform_headroom=schedule["headroom"],
                    actual_operand_headroom=schedule["mac_positive_limit"] - actual_bound,
                    full_i32_max_abs=scale["accumulator_max_abs"],
                    full_i32_headroom=(1 << 31) - 1 - scale["accumulator_max_abs"],
                    k_split=schedule["k_split"],
                    padded_primitive_k=schedule["padded_primitive_k"],
                )
                site.update(
                    accumulator_output_index=acc_index,
                    readout_output_index=acc_index + 1,
                    actual_dispatch="provider catalog with independently checked WS schedule",
                    scales={k: v for k, v in scale.items() if k not in {"accumulator", "readout"}},
                )
            plan["integer_reference"] = dict(
                status="prepared",
                engine=reference["engine"],
                expected_output_bytes=expected_bytes,
                final_f32_comparison="exact bit patterns; signed zero distinct",
                accumulator_comparison="exact i32",
            )
            plan["scales"] = [
                {k: v for k, v in scale.items() if k not in {"accumulator", "readout"}} for scale in reference["scales"]
            ]
            record["scales"] = plan["scales"]
            products = {scale["product"] for scale in plan["scales"]}
            if len(products) == 1:
                record["output_scale"] = next(iter(products))
            report["output_count"] += 2 * len(observations)
            from xdsl.dialects.builtin import StringAttr

            from merlin.llvmlower.result_layout import ATTRIBUTE

            if ATTRIBUTE in module.attributes:
                layout = json.loads(module.attributes[ATTRIBUTE].data)
                layout["results"].extend([None] * (2 * len(observations)))
                module.attributes[ATTRIBUTE] = StringAttr(json.dumps(layout, sort_keys=True))
            module.verify()
            (directory / "model.mlir").write_text(str(module) + "\n")
    plan["source_sha256_after"] = hashlib.sha256((directory / "model.mlir").read_bytes()).hexdigest()
    document = {"selection": selection, **plan}
    (directory / "spec-int8-admission.json").write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    for case in report["cases"]:
        if case["status"] == "bundled":
            case["numeric_admission"] = plan
            case["numeric_selection"] = selection["identities"]
    return document


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
        lsbs = [site.get("output_lsb", float(site["lhs"]) * float(site["rhs"])) for site in record.get("scales", [])]
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
                    for site in record.get("scales", [])
                ],
                output_lsb_error="not_applicable"
                if not record.get("output_scale")
                else float(np.max(difference, initial=0) / record["output_scale"]),
                output_lsb_unit="contraction readout scale product; list gives each stage when no single scale exists",
                contraction_output_lsb_errors=[
                    dict(scale=scale["product"], max_error=float(np.max(difference, initial=0) / scale["product"]))
                    for scale in record.get("scales", [])
                    if "product" in scale
                ],
                output_f32_ulp_error=float(
                    np.max(
                        np.divide(difference, np.abs(np.spacing(golden.astype(np.float32))).astype(np.float64)),
                        initial=0,
                    )
                )
                if not complex_type
                else "complex_components_reported_separately",
                relative_error_at_zero=int(np.count_nonzero((golden == 0) & (difference != 0))),
            )
        )
    return errors
