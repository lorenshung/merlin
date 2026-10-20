"""A native IEEE proof cannot replace a requested quantized-output bit contract."""


def requires_quantized_model(report):
    from merlin.targetgen.functional_truth import expectation

    for record in report.get("cases", ()):
        case = record.get("case") or {}
        if record.get("status") not in {None, "bundled"} or not {"schema", "arguments", "expected"} <= case.keys():
            continue
        floating = any(
            abi.get("dtype") in {"f16", "bf16", "f32", "f64"} or abi.get("dtype", "").startswith("complex<")
            for abi in record.get("input_abi", ())
        )
        if floating and expectation(case)["precision_class"] == "quantized_exact":
            return True
    return False


def proves_quantized_model(plan, software):
    from merlin.llvmlower.quantized_native import parameters

    engine = plan.get("integer_reference", {}).get("engine")
    if plan.get("quantized_contract") is not None:
        return (
            plan["quantized_contract"] == parameters(software)
            and engine == "independent_source_quantized_exact_rational"
        )
    # The legacy static recipe independently seals input scales and its
    # rational output reference. An eager/native component proof does neither.
    recipe = plan.get("capture_recipe") or {}
    numeric = software.get("numerical_semantics", {})
    domain = recipe.get("calibration_domain")
    return bool(
        domain
        and domain == numeric.get("calibration_domain")
        and domain.get("evaluation_inputs") == "never_read"
        and recipe.get("quantization_nonfinite") == numeric.get("quantization_nonfinite")
        and engine == "independent_pt2e_exact_rational"
    )
