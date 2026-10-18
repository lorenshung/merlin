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


def unlowered_data_products(module, limb_attribute):
    """A device prelude cannot stand in for a source bilinear reduction."""
    from xdsl.dialects.builtin import IntegerType

    result = []
    for ordinal, op in enumerate(module.walk()):
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
            if child.name not in {"arith.mulf", "arith.muli"}:
                continue
            typ = child.results[0].type
            if not isinstance(typ, IntegerType) and str(typ) not in {"f16", "bf16", "f32", "f64"}:
                continue
            if all(getattr(value.owner, "name", None) != "arith.constant" for value in child.operands):
                result.append(ordinal)
                break
    return result


def prepare_integer_components(directory, report, selection, provider, package, module, *, limb_attribute):
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
    native_types = [abi["dtype"] for record in report["cases"] for abi in record.get("input_abi", [])]
    if any(dtype == "bf16" and np.isnan(value).any() for dtype, value in zip(native_types, arrays)) and any(
        op.name == "arith.negf" and str(op.results[0].type) == "bf16" for op in module.walk()
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
    known_bounds, boolean_count = rewrite_boolean_components(module)
    from merlin.llvmlower.integer_float_boundaries import has_literal_zero_divisor, rewrite_integer_float_boundaries

    if has_literal_zero_divisor(module):
        (directory / "exact-dtype-refusal.json").write_text(
            json.dumps(dict(family="F2", reason="native zero-divisor readout lacks a CPU NaN-payload policy")) + "\n"
        )
        return None

    conversion_bounds, conversion_count = rewrite_integer_float_boundaries(module, bounds)
    known_bounds.update(conversion_bounds)
    sign_proofs = []
    if any(dtype in {"i8", "i16", "i32", "i64", "f16", "bf16", "f64", "complex<f32>"} for dtype in native_types):
        from merlin.llvmlower.native_sign_components import rewrite_native_sign_components

        sign_bounds, sign_proofs = rewrite_native_sign_components(module)
        known_bounds.update(sign_bounds)
    floating_proofs = []
    remaining_float_arithmetic = any(
        op.name
        in {
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.negf",
            "linalg.matmul",
            "linalg.batch_matmul",
        }
        for op in module.walk()
    )
    if any(b is None for b in bounds) and sign_proofs and not remaining_float_arithmetic:
        # Raw IEEE words need no finite numerical grid, even for signaling NaNs.
        bounds = [0 if b is None else b for b in bounds]
    elif any(b is None for b in bounds):
        if not any(dtype in {"f16", "bf16", "f64", "complex<f32>"} for dtype in native_types):
            return None
        from merlin.llvmlower.fixed_point_ir import finite_grid
        from merlin.llvmlower.floating_components import rewrite_floating_components

        try:
            grids = [finite_grid(value.flat) if value.dtype.kind == "f" else None for value in arrays]
        except ValueError as exc:
            (directory / "exact-dtype-refusal.json").write_text(json.dumps(dict(family="F2", reason=str(exc))) + "\n")
            return None
        complex_indices = [i for i, dtype in enumerate(native_types) if dtype == "complex<f32>"]
        floating_bounds, floating_proofs = rewrite_floating_components(
            module, grids, complex_inputs=complex_indices, allow_reassociation=order_dependent
        )
        known_bounds.update(floating_bounds)
        if not floating_proofs and not boolean_count:
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
    plans = rewrite_integer_components(
        module, limb_attribute=limb_attribute, input_bounds=bounds, known_bounds=known_bounds
    )
    if not plans:
        return None
    from merlin.llvmlower.entry_weight_projection import _pure_dce

    for function in functions:
        _pure_dce(function)
    remaining_products = unlowered_data_products(module, limb_attribute)
    if remaining_products:
        (directory / "exact-dtype-refusal.json").write_text(
            json.dumps(
                dict(
                    family="F2",
                    reason=(
                        "native bilinear reduction remains in host source; "
                        "partial preludes do not place the operation"
                    ),
                    source_operation_ordinals=remaining_products,
                )
            )
            + "\n"
        )
        return None
    if order_dependent and floating_proofs:
        from merlin.targetgen.native_rational_reference import rational_outputs

        try:
            for record in report["cases"]:
                rational_outputs(record["case"])
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
        quantized_contractions=0,
        native_contractions=0,
        quantized_non_contraction_sites=0,
        exact_dtype_sites=len(sites),
        sites=sites,
        schedule_proof=proof,
        limb_plans=plans,
        floating_representation_proofs=floating_proofs,
        native_sign_proofs=sign_proofs,
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
        ),
    )
    document = dict(selection=selection, **plan)
    (directory / "model.mlir").write_text(str(module) + "\n")
    (directory / "spec-int8-admission.json").write_text(json.dumps(document, indent=2) + "\n")
    for record in report["cases"]:
        record.update(numeric_admission=plan, numeric_selection=selection["identities"])
    return document
