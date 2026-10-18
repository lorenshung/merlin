"""Qualify exact native component sources against the selected provider."""

import hashlib
import json


def require_component_identity(directory, plan):
    """Data-specialized bounds apply only to the bound source and input bytes."""
    if not plan.get("exact_dtype_sites"):
        return
    source = (directory / "model.mlir").read_text().rstrip("\n")
    proof = plan.get("schedule_proof") or plan.get("provider_catalog") or {}
    if hashlib.sha256(source.encode()).hexdigest() != proof.get("source_sha256"):
        raise ValueError("exact dtype source changed after its provider proof")
    expected = plan.get("input_bound_proof", {}).get("input_sha256")
    if expected is not None and hashlib.sha256((directory / "inputs.npz").read_bytes()).hexdigest() != expected:
        raise ValueError("exact component input bytes changed after the bound proof")


def unlowered_data_reductions(module, limb_attribute):
    """A device prelude cannot stand in for any retained floating data reduction."""
    from xdsl.dialects.builtin import IndexType, IntegerType

    result = []
    for ordinal, op in enumerate(module.walk()):
        # Exact component readout uses integer arithmetic. A retained math
        # intrinsic is source computation, even when a conversion before it
        # happened to produce an eligible integer device region.
        if op.name.startswith("math."):
            result.append(ordinal)
            continue
        if op.name in {
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.negf",
            "arith.cmpf",
            "arith.cmpi",
            "arith.maximumf",
            "arith.minimumf",
            "arith.maxnumf",
            "arith.minnumf",
            "arith.maxsi",
            "arith.maxui",
            "arith.minsi",
            "arith.minui",
        }:
            # Index predicates select addresses or representation components;
            # they are not retained numerical comparisons of tensor values.
            if op.operands and all(isinstance(value.type, IndexType) for value in op.operands):
                continue
            parent = op
            while parent is not None and "merlin.exact_host_arithmetic" not in parent.attributes:
                parent = parent.parent_op()
            if parent is None:
                result.append(ordinal)
            continue
        if op.name not in {"linalg.generic", "linalg.reduce", "scf.for"} or limb_attribute in op.attributes:
            continue
        children = list(op.walk())[1:]
        reduction = (
            op.name in {"linalg.reduce", "scf.for"}
            or any(it.data.value == "reduction" for it in op.get_iterator_types())
            or any(child.name == "scf.for" for child in children)
        )
        if not reduction:
            continue
        for child in children:
            if child.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf"}:
                if any(getattr(value.owner, "name", None) != "arith.constant" for value in child.operands):
                    result.append(ordinal)
                    break
            if child.name != "arith.muli":
                continue
            typ = child.results[0].type
            if not isinstance(typ, IntegerType) and str(typ) not in {"f16", "bf16", "f32", "f64"}:
                continue
            if all(getattr(value.owner, "name", None) != "arith.constant" for value in child.operands):
                result.append(ordinal)
                break
    return result


def prepare_integer_components(
    directory, report, selection, provider, package, module, *, limb_attribute, native_f32=False
):
    import numpy as np

    from merlin.kernels.shapes import observe_contractions
    from merlin.llvmlower.integer_components import rewrite_integer_components
    from merlin.targetgen.software_spec import admit_operation
    from merlin.targetgen.transfer_contracts import screen_transfer_contract

    if not callable(getattr(provider, "admit_numeric_schedule", None)):
        return None
    if any(key.startswith("prov.quant.") for op in module.walk() for key in op.attributes):
        return None
    with np.load(directory / "inputs.npz") as inputs:
        bounds = []
        arrays = []
        for key in sorted(inputs.files, key=lambda s: int(s.removeprefix("in"))):
            value = inputs[key]
            arrays.append(value.copy())
            bounds.append(max((abs(int(v)) for v in value.flat), default=0) if value.dtype.kind in "biu" else None)
    recorded_integer_bounds = list(bounds)
    grids = []
    # Float components have a distinct representation and must never inherit an integer cast.
    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward"]
    if not functions:
        return None
    records = [record for record in report["cases"] if record.get("status") in {None, "bundled"}]
    native_types = [abi["dtype"] for record in records for abi in record.get("input_abi", [])]
    ieee_policy = selection.get("host", {}).get("native_ieee_special_values", {})
    ieee_enabled = (
        ieee_policy.get("schema") == "merlin.native_ieee_special_values.v1" and ieee_policy.get("enabled") is True
    )
    from merlin.llvmlower.integer_float_boundaries import has_literal_zero_divisor

    if not ieee_enabled and has_literal_zero_divisor(module):
        (directory / "exact-dtype-refusal.json").write_text(
            json.dumps(dict(family="F2", reason="native zero-divisor readout lacks a CPU NaN-payload policy")) + "\n"
        )
        return None

    if (
        not ieee_enabled
        and any(dtype == "bf16" and np.isnan(value).any() for dtype, value in zip(native_types, arrays))
        and any(op.name == "arith.negf" and str(op.results[0].type) == "bf16" for op in module.walk())
    ):
        (directory / "exact-dtype-refusal.json").write_text(
            json.dumps(
                dict(family="F2", reason="native bf16 negation lacks an explicit CPU NaN-canonicalization policy")
            )
            + "\n"
        )
        return None
    from merlin.llvmlower.boolean_components import rewrite_boolean_components

    native_arithmetic = any(
        op.name
        in {
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.negf",
            "math.fma",
            "linalg.matmul",
            "linalg.batch_matmul",
        }
        for op in module.walk()
    )
    order_dependent = bool(observe_contractions(module)) or any(
        op.name == "linalg.reduce"
        or (op.name == "scf.for" and any(str(v.type) in {"f16", "bf16", "f32", "f64"} for v in op.results))
        or (op.name == "linalg.generic" and any(it.data.value == "reduction" for it in op.get_iterator_types()))
        for op in module.walk()
    )
    from merlin.targetgen.core_aten_exact_dtype import native_dtype_request

    has_special_inputs = any(v.dtype.kind == "f" and not np.isfinite(v).all() for v in arrays)
    ieee_pointwise = (
        ieee_enabled
        and selection["host"]["native_ieee_special_values"].get("pointwise_nonfinite")
        == "affected_elements_source_ieee_only"
        and has_special_inputs
    )
    source_agreement_required = ieee_pointwise or (not order_dependent and (native_dtype_request(report) or native_f32))
    closure_required = (
        order_dependent
        and any(
            dtype in ({"f16", "bf16", "f64", "complex<f32>"} | ({"f32"} if native_f32 else set()))
            for dtype in native_types
        )
        and all(np.isfinite(value).all() for value in arrays)
    )
    if closure_required:
        from merlin.targetgen.core_aten_host_floor import native_boundary_has_nan

        if native_boundary_has_nan({**report, "cases": records}):
            (directory / "exact-dtype-refusal.json").write_text(
                json.dumps(
                    dict(
                        family="F2",
                        code="ieee_intermediate_nonfinite_policy_unproved",
                        reason=(
                            "finite native inputs produce CPU NaN; no rational error floor or closed overflow policy"
                        ),
                    )
                )
                + "\n"
            )
            return None
    if source_agreement_required or closure_required:
        from merlin.targetgen.core_aten_numeric import _source_reference

        observed = _source_reference(directory, "--native-source" if source_agreement_required else "--ieee-closure")
        closure_path = directory / "source-ieee-closure.json"
        closure = json.loads(closure_path.read_text()) if observed.returncode == 0 and closure_path.exists() else {}
        if source_agreement_required and "output_bytes" not in closure:
            (directory / "exact-dtype-refusal.json").write_text(
                json.dumps(
                    dict(
                        family="F2",
                        code="native_source_observation_unavailable",
                        reason="native component qualification requires complete independent source bit observations",
                    )
                )
                + "\n"
            )
            return None
        if closure and (
            closure.get("source_sha256") != hashlib.sha256((directory / "model.mlir").read_bytes()).hexdigest()
            or closure.get("input_sha256") != hashlib.sha256((directory / "inputs.npz").read_bytes()).hexdigest()
        ):
            raise ValueError("source IEEE closure observation does not bind the current source and inputs")
        if ieee_pointwise and (
            "output_bytes" not in closure or closure.get("ieee_special_audit", {}).get("ambiguous_nan_payload", True)
        ):
            (directory / "exact-dtype-refusal.json").write_text(
                json.dumps(
                    dict(
                        family="F2",
                        code="ieee_pointwise_source_policy_unproved",
                        reason="special propagation requires a complete source proof and unambiguous NaN payloads",
                        source_ieee_policy=closure,
                    )
                )
                + "\n"
            )
            return None
        if source_agreement_required and "output_bytes" in closure:
            from merlin.targetgen.core_aten_host_floor import grade_native_bits

            if order_dependent:
                from merlin.targetgen.core_aten_host_floor import grade_special_bits

                compare = grade_special_bits
            else:
                compare = grade_native_bits
            comparisons = [
                compare(record, [bytes.fromhex(raw) for raw in closure["output_bytes"]]) for record in records
            ]
            if any(row["status"] != "pass" for row in comparisons):
                (directory / "exact-dtype-refusal.json").write_text(
                    json.dumps(
                        dict(
                            family="F2",
                            code="native_source_eager_disagreement",
                            reason="captured typed source differs from the independent native CPU boundary",
                            source_agreement=comparisons,
                        )
                    )
                    + "\n"
                )
                return None
        if closure_required and closure.get("status") == "source_nan":
            (directory / "exact-dtype-refusal.json").write_text(
                json.dumps(
                    dict(
                        family="F2",
                        code="ieee_intermediate_nonfinite_policy_unproved",
                        reason=(
                            "finite source has IEEE overflow/invalid intermediates; "
                            "native NaN propagation lacks a closed proof"
                        ),
                    )
                )
                + "\n"
            )
            return None
    known_bounds, boolean_count = rewrite_boolean_components(module)
    from merlin.llvmlower.integer_float_boundaries import rewrite_integer_float_boundaries

    conversion_grids = {}
    conversion_bounds, conversion_count = rewrite_integer_float_boundaries(
        module, bounds, output_grids=conversion_grids
    )
    known_bounds.update(conversion_bounds)
    from merlin.llvmlower.ieee_zero_division import rewrite_zero_division

    zero_division_count = rewrite_zero_division(module) if ieee_enabled else 0
    if has_literal_zero_divisor(module):
        (directory / "exact-dtype-refusal.json").write_text(
            json.dumps(
                dict(
                    family="F2",
                    reason="zero-divisor source is outside the closed IEEE pointwise family"
                    if ieee_enabled
                    else "native zero-divisor readout lacks a CPU NaN-payload policy",
                )
            )
            + "\n"
        )
        return None
    sign_proofs = []
    unquantizable_float_words = any(
        dtype == "f32" and not np.isfinite(value).all() for dtype, value in zip(native_types, arrays, strict=True)
    )
    mixed_finite_arithmetic = not has_special_inputs and any(
        op.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "math.fma"} for op in module.walk()
    )
    if (
        unquantizable_float_words
        or (native_f32 and "f32" in native_types)
        or any(dtype in {"i8", "i16", "i32", "i64", "f16", "bf16", "f64", "complex<f32>"} for dtype in native_types)
    ):
        from merlin.llvmlower.native_sign_components import rewrite_native_sign_components

        sign_bounds, sign_proofs = rewrite_native_sign_components(
            module, canonical_bf16_nan=ieee_enabled, defer_floating_negation=mixed_finite_arithmetic
        )
        known_bounds.update(sign_bounds)
    floating_proofs = []
    zero_limbs = {}
    remaining_float_arithmetic = any(
        op.name
        in {
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.negf",
            "math.fma",
            "linalg.matmul",
            "linalg.batch_matmul",
        }
        for op in module.walk()
    )
    if any(b is None for b in bounds) and sign_proofs and not remaining_float_arithmetic:
        # Raw IEEE words need no finite numerical grid, even for signaling NaNs.
        bounds = [0 if b is None else b for b in bounds]
    elif any(b is None for b in bounds) or (conversion_count and native_arithmetic):
        widening = any(op.name == "arith.extf" and str(op.result.type) == "f64" for op in module.walk())
        if (
            not conversion_count
            and not native_f32
            and not ieee_pointwise
            and not widening
            and not any(dtype in {"f16", "bf16", "f64", "complex<f32>"} for dtype in native_types)
        ):
            return None
        from merlin.llvmlower.fixed_point_ir import finite_grid
        from merlin.llvmlower.floating_components import rewrite_floating_components

        try:
            grids = [
                finite_grid(np.where(np.isfinite(value), value, 0).flat if ieee_pointwise else value.flat)
                if value.dtype.kind == "f"
                else None
                for value in arrays
            ]
        except ValueError as exc:
            (directory / "exact-dtype-refusal.json").write_text(json.dumps(dict(family="F2", reason=str(exc))) + "\n")
            return None
        complex_indices = [i for i, dtype in enumerate(native_types) if dtype == "complex<f32>"]
        try:
            floating_bounds, floating_proofs = rewrite_floating_components(
                module,
                grids,
                complex_inputs=complex_indices,
                allow_reassociation=order_dependent,
                known_grids=conversion_grids,
                input_values=arrays,
                zero_limbs=zero_limbs,
                ieee_pointwise=ieee_pointwise,
            )
        except ValueError as exc:
            (directory / "exact-dtype-refusal.json").write_text(
                json.dumps(dict(family="F2", code="ieee_source_policy_unproved", reason=str(exc))) + "\n"
            )
            return None
        known_bounds.update(floating_bounds)
        if not floating_proofs and not boolean_count:
            if conversion_count:
                (directory / "exact-dtype-refusal.json").write_text(
                    json.dumps(dict(family="F2", reason="integer conversion alone is not a device reduction")) + "\n"
                )
            return None
        if (
            floating_proofs
            and native_arithmetic
            and not any(proof["source_data_arithmetic"] for proof in floating_proofs)
        ):
            (directory / "exact-dtype-refusal.json").write_text(
                json.dumps(
                    dict(
                        family="F2",
                        reason="no exact native arithmetic producer; conversion alone is not device placement",
                    )
                )
                + "\n"
            )
            return None
        bounds = [0 if b is None else b for b in bounds]
    if order_dependent and conversion_count and not (floating_proofs or boolean_count or sign_proofs):
        (directory / "exact-dtype-refusal.json").write_text(
            json.dumps(dict(family="F2", reason="integer conversion alone is not a device reduction")) + "\n"
        )
        return None
    if zero_division_count:
        from merlin.llvmlower.ieee_zero_division import retain_nan_narrowing

        retain_nan_narrowing(module)
    plans = rewrite_integer_components(
        module,
        limb_attribute=limb_attribute,
        input_bounds=bounds,
        known_bounds=known_bounds,
        known_zero_limbs=zero_limbs,
    )
    if not plans:
        return None
    from merlin.llvmlower.literal_zero_components import fold_literal_zero_components

    folded_zero_producers = fold_literal_zero_components(module, limb_attribute)
    from merlin.llvmlower.entry_weight_projection import _pure_dce

    for function in functions:
        _pure_dce(function)
    from merlin.llvmlower.component_nan_readout import qualify_nan_readout

    nan_readouts = qualify_nan_readout(module, floating_proofs)
    remaining_products = unlowered_data_reductions(module, limb_attribute)
    if remaining_products:
        (directory / "exact-dtype-refusal.json").write_text(
            json.dumps(
                dict(
                    family="F2",
                    reason=("native arithmetic remains in host source; partial preludes do not place the operation"),
                    source_operation_ordinals=remaining_products,
                )
            )
            + "\n"
        )
        return None
    if order_dependent and floating_proofs:
        from merlin.targetgen.native_rational_reference import rational_outputs

        try:
            for record in records:
                rational_outputs(record["case"], finite_subproblem=ieee_pointwise)
        except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
            (directory / "exact-dtype-refusal.json").write_text(
                json.dumps(dict(family="F2", reason="independent native reduction reference unavailable: " + str(exc)))
                + "\n"
            )
            return None
    for origin, destination, dtype in (("host", "accelerator", "i8"), ("accelerator", "host", "i32")):
        decision = screen_transfer_contract(
            selection["spec"],
            source_placement=origin,
            destination_placement=destination,
            operand_dtype=dtype,
            result_dtype=dtype,
            operand_layout="row_major_contiguous",
            result_layout="row_major_contiguous",
        )
        if decision["status"] != "admitted":
            return None
    try:
        proof = provider.admit_numeric_schedule(str(module), package=package, numeric_policy=selection["spec"])
    except ValueError as exc:
        (directory / "exact-dtype-refusal.json").write_text(
            json.dumps(
                dict(
                    family="F2",
                    reason="selected provider refuses exact components: " + str(exc),
                    source_sha256=hashlib.sha256(str(module).encode()).hexdigest(),
                )
            )
            + "\n"
        )
        return None
    if (
        proof.get("status") != "proven"
        or proof.get("source_sha256") != hashlib.sha256(str(module).encode()).hexdigest()
    ):
        reason = proof.get("reason", "component provider proof unavailable")
        (directory / "exact-dtype-refusal.json").write_text(
            json.dumps(dict(family="F2", reason=reason, provider_proof=proof)) + "\n"
        )
        return None
    sites = []
    for ordinal, op in enumerate(module.walk()):
        if limb_attribute not in op.attributes:
            continue
        binding = [
            b
            for b in proof.get("bindings", [])
            if b["source_operation_ordinal"] == ordinal and b.get("numeric", {}).get("family") in {"F2", "F4"}
        ]
        contraction = any(it.data.value == "reduction" for it in op.get_iterator_types())
        family = (
            "contraction"
            if binding and binding[0]["numeric"].get("form") == "bilinear"
            else "reduction"
            if contraction
            else "elementwise_map"
        )
        signature = dict(
            family=family,
            numerical_family="F5" if boolean_count else "F4" if any(p["family"] == "F4" for p in plans) else "F2",
            operand_dtype="i8",
            accum_dtype="i32",
            readout_dtype="i32",
            rank=len(op.results[0].type.get_shape()),
            layout="row_major_contiguous",
            aliasing="disjoint_inputs_outputs",
        )
        decision = admit_operation(selection["spec"], family, signature, "accelerator")
        host = admit_operation(selection["host"], family, signature, "host")
        if decision["status"] != "admitted" or host["status"] != "admitted" or len(binding) != 1:
            (directory / "exact-dtype-refusal.json").write_text(
                json.dumps(
                    dict(
                        family="F2",
                        reason="component software/host/source binding incomplete",
                        software=decision,
                        host=host,
                        bindings=binding,
                    )
                )
                + "\n"
            )
            return None
        numeric = binding[0]["numeric"]
        if numeric.get("readout") != "full_i32" or numeric.get("recombination") != "retained_source_host_exact_integer":
            raise ValueError("component producer would replace the exact host epilogue")
        sites.append(
            dict(source_operation_ordinal=ordinal, family=family, status="admitted", software=decision, numeric=numeric)
        )
    plan = dict(
        status="admitted",
        numerical_family="F5" if boolean_count else "F4" if any(p["family"] == "F4" for p in plans) else "F2",
        literal_zero_producers_folded=folded_zero_producers,
        quantized_contractions=0,
        native_contractions=0,
        quantized_non_contraction_sites=0,
        exact_dtype_sites=len(sites),
        sites=sites,
        schedule_proof=proof,
        limb_plans=plans,
        floating_representation_proofs=floating_proofs,
        ieee_classification_readouts=nan_readouts,
        zero_limb_omission_proof=dict(
            method="complete_recorded_input_dyadic_coefficients",
            producer_count=len(zero_limbs),
            omitted_digits=sum(len(v) for v in zero_limbs.values()),
        ),
        native_sign_proofs=sign_proofs,
        ieee_special_family=(
            dict(
                kind="literal_zero_division",
                producer="complete_integer_numerator",
                count=zero_division_count,
                invalid_nan="x86_negative_quiet_bfloat16_positive_canonical",
            )
            if zero_division_count
            else dict(
                kind="closed_source_index_dependence",
                producer="complete_exact_finite_integer_numerator",
                dependence_proofs=[p["ieee_dependence_proof"] for p in floating_proofs if "ieee_dependence_proof" in p],
                output_floor="native_eager_bits_including_nan_payload_and_signed_zero",
            )
            if ieee_pointwise and any("ieee_dependence_proof" in p for p in floating_proofs)
            else None
        ),
        boolean_semantics={"source": "closed_boolean_or_and", "integer_components": boolean_count},
        integer_float_boundaries={"count": conversion_count, "rounding": "source_sitofp_before_host_epilogue"},
        input_bound_proof=dict(
            method="complete_immutable_input_payload",
            absolute_bounds=recorded_integer_bounds,
            dyadic_grids=[
                None if value is None else dict(binary_exponent=value[0], integer_absolute_bound=value[1])
                for value in grids
            ],
            input_sha256=hashlib.sha256((directory / "inputs.npz").read_bytes()).hexdigest(),
        ),
        host_floor=dict(
            status="required",
            comparison="native_eager_bits_every_boundary_result",
            order_dependent_reduction=order_dependent and bool(floating_proofs),
            finite_subproblem_overlay=ieee_pointwise,
        ),
    )
    document = dict(selection=selection, **plan)
    (directory / "model.mlir").write_text(str(module) + "\n")
    (directory / "spec-int8-admission.json").write_text(json.dumps(document, indent=2) + "\n")
    for record in records:
        record.update(numeric_admission=plan, numeric_selection=selection["identities"])
    return document
