"""Quantized numeric fallback, selected only after a native exact path fails."""

import copy
import json


def prepare(directory, report, *, target, facts, software_spec, host_capabilities, provider=None, package=None, **_):
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.quantized_native import parameters
    from merlin.targetgen.core_aten_components import prepare_integer_components
    from merlin.targetgen.core_aten_spec_int8 import select_spec
    from merlin.targetgen.native_unit_policy import construction_evidence

    selection = select_spec(target, facts, software_spec, host_capabilities)
    rule = selection["spec"].get("numerical_semantics", {}).get("quantized_native_fallback")
    if not rule:
        return None
    from merlin.targetgen.operator_review import review_record

    review = review_record(rule)
    if not review["operator_accepted"]:
        return None
    contract = parameters(selection["spec"])
    if any(
        a["dtype"] not in {"f16", "bf16", "f32", "f64", "complex<f32>"}
        for r in report["cases"]
        if r.get("status") == "bundled"
        for a in r.get("output_abi", [])
    ):
        return None
    module = parse_mlir_file(directory / "model.mlir")
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    from merlin.kernels.shapes import observe_contractions

    complex_inputs = any(
        a["dtype"].startswith("complex")
        for r in report["cases"]
        if r.get("status") == "bundled"
        for a in r.get("input_abi", [])
    )
    if complex_inputs and list(observe_contractions(module)):
        (directory / "quantized-native-refusal.json").write_text(
            json.dumps(dict(reason="runtime complex products require a composition")) + "\n"
        )
        return None
    if not function.body.block.args or any(
        str(v.type.element_type) not in {"f16", "bf16", "f32"} for v in function.body.block.args
    ):
        return None
    from merlin.llvmlower.uniform_numeric_literals import compile_time_operand
    from merlin.targetgen.native_construction_patterns import Payloads

    policy = construction_evidence(
        module, excluded=selection["spec"]["numerical_semantics"]["native_unit_contract"]["excluded_constructions"]
    )
    # Keep source construction exclusions. IEEE-only offset objections and
    # opaque frozen-control helpers may disappear under the rational contract;
    # prove their normalized payload flow instead of granting an exemption.
    hard = [r for r in policy["evidence"] if r["reason"] not in {"readout_offset_rounding", "unproved_payload_carrier"}]
    if hard:
        return None
    from merlin.llvmlower.quantized_native import finite_source

    try:
        finite_source(module)
    except ValueError as exc:
        (directory / "quantized-native-refusal.json").write_text(json.dumps(dict(reason=str(exc))) + "\n")
        return None
    payloads = Payloads(module)
    for primitive in module.walk():
        if primitive.name not in {"arith.mulf", "math.fma"}:
            continue
        if any(compile_time_operand(v) or payloads.constant(v) is not None for v in primitive.operands[:2]):
            continue
        parent = primitive.parent_op()
        if (
            parent is not None
            and parent.name == "linalg.generic"
            and all(it.data.value == "parallel" for it in parent.get_iterator_types())
        ):
            # Reduction-dependent operands do not change a parallel product's
            # mathematical role: it is still a composition of native units.
            (directory / "quantized-native-refusal.json").write_text(
                json.dumps(dict(reason="parallel runtime product requires a composition")) + "\n"
            )
            return None
    normalized = construction_evidence(
        module, excluded=selection["spec"]["numerical_semantics"]["native_unit_contract"]["excluded_constructions"]
    )
    if any(r["reason"] != "readout_offset_rounding" for r in normalized["evidence"]):
        return None
    policy = dict(
        normalized,
        status="eligible",
        evidence=[r for r in normalized["evidence"] if r["reason"] != "readout_offset_rounding"],
        replaced_ieee_obligations=[r for r in normalized["evidence"] if r["reason"] == "readout_offset_rounding"],
        original_evidence=policy["evidence"],
        contract="quantized_rational",
    )
    source = (directory / "model.mlir").read_bytes()
    saved = copy.deepcopy(report)
    try:
        plan = prepare_integer_components(
            directory,
            report,
            selection,
            provider,
            package,
            module,
            limb_attribute=f"{target}.limb",
            quantized_contract=contract,
        )
        if plan is None:
            from merlin.llvmlower.entry_weight_projection import _pure_dce
            from merlin.targetgen.core_aten_exact_dtype import prepare_bitwise

            _pure_dce(function)
            plan = prepare_bitwise(directory, report, selection, provider, package, module)
            if plan is not None:
                plan.update(
                    quantized_contract=contract,
                    host_floor=dict(status="not_applicable", comparison="independent_quantized_exact_rational"),
                    quantized_non_contraction_sites=plan["exact_dtype_sites"],
                    source_normalization="complete_exact_rational_zero_polynomial_then_uniform_movement",
                )
        if plan is None:
            refusal = directory / "exact-dtype-refusal.json"
            (directory / "quantized-native-refusal.json").write_text(
                refusal.read_text()
                if refusal.exists()
                else json.dumps(dict(reason="no complete quantized source producer")) + "\n"
            )
            return None
        records = [r for r in report["cases"] if r.get("status") == "bundled"]
        request = dict(contract=contract, records=records, output_count=report["output_count"])
        (directory / "quantized-native-request.json").write_text(json.dumps(request) + "\n")
        from merlin.targetgen.core_aten_numeric import _source_reference

        process = _source_reference(directory, "--quantized-native")
        if process.returncode:
            raise ArithmeticError("independent quantized reference qualification failed: " + process.stderr[-3000:])
        reference = json.loads((directory / "quantized-native-reference.json").read_text())
        plan["precision_class"] = "b"
        plan["integer_reference"] = dict(
            status="prepared",
            engine="independent_source_quantized_exact_rational",
            expected_output_bytes=reference["expected_output_bytes"],
        )
        plan["native_admission"] = dict(
            status="device",
            reason_code="native_family_quantized_contract",
            native_unit_policy=policy,
            quantized_contract_operator_review=review,
        )
        plan["calibration"] = dict(source="declared_observer_domain", evaluation_inputs="never_read", performed=False)
        for record in records:
            record["numeric_admission"] = plan
        (directory / "spec-int8-admission.json").write_text(json.dumps(plan, indent=2) + "\n")
        return plan
    except (ValueError, RuntimeError) as exc:
        (directory / "quantized-native-refusal.json").write_text(json.dumps(dict(reason=str(exc))) + "\n")
        return None
    finally:
        # A candidate cannot strand a changed source or ABI when it declines.
        if "plan" not in locals() or plan is None or not plan.get("integer_reference"):
            (directory / "model.mlir").write_bytes(source)
            for record, original in zip(report["cases"], saved["cases"], strict=True):
                record.clear()
                record.update(original)
