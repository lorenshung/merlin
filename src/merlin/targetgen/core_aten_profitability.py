"""Transactional host fallback for unprofitable exact numeric preparations."""

import copy
import json


def prepare(directory, report, implementation, **kwargs):
    from merlin.common.yaml import load_yaml
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.placement_cost import estimate

    selected = kwargs.get("software_spec")
    declaration = load_yaml(selected).get("placement_cost_model") if selected is not None else None
    if declaration is None:
        return implementation(directory, report, **kwargs)
    # Preparation can change the whole exported ABI and its inputs. A rejected
    # placement must restore the original host program and all of its sidecars.
    original_files = {p.name: p.read_bytes() for p in directory.iterdir() if p.is_file()}
    original_report = copy.deepcopy(report)
    plan = implementation(directory, report, **kwargs)
    if not any(
        plan.get(key, 0)
        for key in (
            "exact_dtype_sites",
            "native_contractions",
            "quantized_contractions",
            "quantized_non_contraction_sites",
        )
    ):
        return plan
    cost = estimate(
        parse_mlir_text(original_files["model.mlir"].decode()),
        parse_mlir_text((directory / "model.mlir").read_text()),
        plan,
        declaration,
        kwargs["facts"],
    )
    plan["placement_cost"] = cost
    if cost["status"] != "device":
        (directory / "unprofitable-device-source.mlir").write_bytes((directory / "model.mlir").read_bytes())
        (directory / "unprofitable-device-admission.json").write_text(json.dumps(plan, indent=2) + "\n")
        for name, data in original_files.items():
            (directory / name).write_bytes(data)
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
        record.update(numeric_admission=plan, numeric_selection=plan["selection"]["identities"])
    (directory / "spec-int8-admission.json").write_text(json.dumps(plan, indent=2) + "\n")
    return plan
