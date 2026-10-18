"""Transactional host fallback for unprofitable exact numeric preparations."""

import copy
import json
import os


def prepare(directory, report, implementation, **kwargs):
    from merlin.common.yaml import load_yaml
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.placement_cost import estimate
    from merlin.targetgen.placement_reference import estimate_reference

    selected = kwargs.get("software_spec")
    declaration = load_yaml(selected).get("placement_cost_model") if selected is not None else None
    if declaration is None:
        return implementation(directory, report, **kwargs)
    # Preparation can change the whole exported ABI and its inputs. A rejected
    # placement must restore the original host program and all of its sidecars.
    original_files = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    original_report = copy.deepcopy(report)
    original_cases = list(report["cases"])
    policy = os.environ.get("MERLIN_PLACEMENT_POLICY", declaration.get("default_policy", "strict_per_shape"))
    if policy not in {"reference_scale", "strict_per_shape"}:
        raise ValueError(f"unknown placement profitability policy: {policy}")
    plan = implementation(directory, report, **kwargs)
    # One numeric candidate belongs only to bundled records. Some preparation
    # paths annotate all retained corpus rows, including not-in-shard metadata;
    # restore those rows before publishing this candidate's policy evidence.
    for record, saved in zip(report["cases"], original_report["cases"], strict=True):
        if saved.get("status") not in {None, "bundled"}:
            record.clear()
            record.update(saved)
    if not any(
        plan.get(key, 0)
        for key in (
            "exact_dtype_sites",
            "native_contractions",
            "quantized_contractions",
            "quantized_non_contraction_sites",
        )
    ):
        plan["placement_cost"] = dict(
            policy=policy,
            status="host",
            reason_code="no_admitted_device_candidate",
            strict_policy=dict(
                policy="strict_per_shape",
                status="not_applicable",
                reason_code=plan.get("refusal_code", "no_admitted_device_candidate"),
            ),
        )
        for record in report["cases"]:
            if record.get("status") in {None, "bundled"}:
                record["numeric_admission"] = plan
        (directory / "spec-int8-admission.json").write_text(json.dumps(plan, indent=2) + "\n")
        return plan
    strict = estimate(
        parse_mlir_text(original_files["model.mlir"].decode()),
        parse_mlir_text((directory / "model.mlir").read_text()),
        plan,
        declaration,
        kwargs["facts"],
    )
    strict["policy"] = "strict_per_shape"
    cost = (
        estimate_reference(parse_mlir_text(original_files["model.mlir"].decode()), plan, declaration, kwargs["facts"])
        if policy == "reference_scale"
        else dict(strict)
    )
    cost["strict_policy"] = strict
    plan["placement_cost"] = cost
    if cost["status"] != "device":
        (directory / "unprofitable-device-source.mlir").write_bytes((directory / "model.mlir").read_bytes())
        (directory / "unprofitable-device-admission.json").write_text(json.dumps(plan, indent=2) + "\n")
        for name, data in original_files.items():
            (directory / name).write_bytes(data)
        # The runner retains references to these records across preparation.
        # Preserve their identities so later routing/build updates reach the
        # restored report rather than abandoned candidate dictionaries.
        for record, saved in zip(original_cases, original_report["cases"], strict=True):
            record.clear()
            record.update(saved)
        original_report["cases"] = original_cases
        report.clear()
        report.update(original_report)
        plan = dict(
            selection=plan["selection"],
            status="host",
            sites=[],
            quantized_contractions=0,
            native_contractions=0,
            quantized_non_contraction_sites=0,
            exact_dtype_sites=0,
            refusal_code=cost["reason_code"],
            host_reason=cost["reason_code"],
            placement_cost=cost,
        )
    for record in report["cases"]:
        if record.get("status") in {None, "bundled"}:
            record.update(numeric_admission=plan, numeric_selection=plan["selection"]["identities"])
    (directory / "spec-int8-admission.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan
