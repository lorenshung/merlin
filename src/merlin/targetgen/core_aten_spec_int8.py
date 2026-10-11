"""Fail-closed admission for a declared static int8 full-call pipeline.

A scale's storage type is not its product arithmetic. Phase-0 form resolution
must not fill numerical semantics from a header or a target name. In particular,
an admitted MAC does not grant a narrowing readout or a host transfer.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from merlin.targetgen.software_spec import admit_operation, load_software_spec

QUANTIZE = "quantized_decomposed.quantize_per_tensor.default"
DEQUANTIZE = "quantized_decomposed.dequantize_per_tensor.default"


def host_signature(op: str, rank: int, parameters: dict) -> dict:
    return {
        "family": "elementwise_map",
        "ordered_operand_dtypes": ["f32" if op == QUANTIZE else "i8", "f32", "i64"],
        "ordered_result_dtypes": ["i8" if op == QUANTIZE else "f32"],
        "operand_dtype": "f32" if op == QUANTIZE else "i8",
        "rank": rank,
        "quantization_parameters": parameters,
    }


def select_spec(target: str, facts: dict, spec_path: Path, host_path: Path) -> dict:
    """Resolve authored forms through the existing Phase-0 mechanism, retaining evidence."""
    from merlin.common.yaml import load_yaml
    from merlin.targetgen.host_capabilities import validate_host_capabilities
    from merlin.targetgen.quant_recipe import derive_candidates
    from merlin.targetgen.readout_facet import for_target
    from merlin.targetgen.spec_fact_drift import fact_capabilities, resolve_spec
    from merlin.targetgen.target_registry import load_contract

    spec = load_software_spec(spec_path, target=target)
    host = validate_host_capabilities(load_yaml(host_path))
    contract = load_contract(target)
    facets = for_target(target, facts=facts, contract=contract)
    candidates = [candidate.to_dict() for candidate in derive_candidates(contract, facets)]
    recipes = [candidate["recipe"] for candidate in candidates if candidate["recipe"] is not None]
    capabilities = fact_capabilities(
        target=target,
        contract=contract,
        raw_facts=facts,
        readout_facets=[facet.to_dict() for facet in facets],
        quantization_candidates=candidates,
    )
    resolved, resolution = resolve_spec(spec, capabilities)
    return {
        "spec": resolved,
        "host": host,
        "resolution": resolution,
        "recipes": recipes,
        "quantization_candidates": candidates,
        "readout_facets": [facet.to_dict() for facet in facets],
        "identities": {
            name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (("software_spec", spec_path), ("host_capabilities", host_path))
        },
        "product_derivation": {
            "status": "declared"
            if (spec["numerical_semantics"].get("readout") or {}).get("acc_scale_product_dtype")
            else "unknown",
            "reason": (
                "resolve_spec derives hardware forms, not product arithmetic; scale carrier/ABI dtype is insufficient"
            ),
            "selected_source": facts.get("facts", {}).get("source"),
            "selected_unknowns": facts.get("facts", {}).get("datapaths_undeterminable", []),
        },
    }


def pipeline_refusals(selection: dict) -> list[str]:
    """All composition blockers, independently of any workload's dtype/geometry."""
    spec = selection["spec"]
    semantics = spec.get("numerical_semantics") or {}
    readout = semantics.get("readout") or {}
    reasons = []
    if spec.get("status") != "reviewed" or selection["host"].get("status") != "reviewed":
        reasons.append("software and host declarations must both be reviewed")
    if readout.get("acc_scale_product_dtype") != "f32":
        reasons.append("acc_scale_product_dtype is not declared or established by selected RTL facts/resolve_spec")
    if readout.get("acc_scale_rounding") != "half_even":
        reasons.append("declared acc_scale rounding is not supported half_even")
    if readout.get("narrowing") != "saturate_to_declared_dtype":
        reasons.append("declared readout does not establish saturating narrowing")
    transfer = (spec.get("transfer_contracts") or {}).get("accumulator_readout") or {}
    if (
        transfer.get("from") != "accelerator"
        or transfer.get("to") != "host"
        or (transfer.get("copy") or {}).get("dtype") not in {"int8", "i8"}
    ):
        reasons.append("no declared i8 accelerator-to-host readout transfer (selected accumulator_readout is i32)")
    return reasons


def admission_plan(module, selection: dict) -> dict:
    """Inventory proposed typed operations; no source mutation or offload grant.

    The proposed integer signatures are explicitly hypothetical until a complete
    pipeline is admitted. They never route an f32 source op on hardware recognition.
    """
    from merlin.kernels.shapes import observe_contractions, zero_initialised
    from merlin.llvmlower.contraction_quantization import closed_float_mac
    from merlin.targetgen.operation_numerics import integer_split_k_limit

    spec, host = selection["spec"], selection["host"]
    parameters = {"zero_point": 0, "quant_min": -128, "quant_max": 127}
    sites = []
    ordinals = {op: index for index, op in enumerate(module.walk())}
    for op, shape in observe_contractions(module):
        rank = len(op.results[0].type.get_shape())
        signature = {
            "family": "contraction",
            "ordered_operand_dtypes": ["i8", "i8", "i32"],
            "ordered_result_dtypes": ["i32"],
            "operand_dtype": "i8",
            "accum_dtype": "i32",
            "readout_dtype": "i32",
            "rank": rank,
            "layout": "row_major_contiguous",
            "tails": "zero_pad_valid_window",
            "broadcasting": "independent_batches" if rank > 2 else "none",
            "aliasing": "disjoint_inputs_outputs",
        }
        contraction = admit_operation(spec, "contraction", signature, "accelerator")
        host_ops = []
        for src in op.operands[:2]:
            observed = host_signature(QUANTIZE, len(src.type.get_shape()), parameters)
            host_ops.append(
                {
                    "op": QUANTIZE,
                    "signature": observed,
                    "software": admit_operation(spec, QUANTIZE, observed, "host"),
                    "host": admit_operation(host, QUANTIZE, observed, "host"),
                }
            )
        observed = host_signature(DEQUANTIZE, rank, {"zero_point": 0})
        host_ops.append(
            {
                "op": DEQUANTIZE,
                "signature": observed,
                "software": admit_operation(spec, DEQUANTIZE, observed, "host"),
                "host": admit_operation(host, DEQUANTIZE, observed, "host"),
            }
        )
        reasons = []
        if shape.dtypes != ("f32", "f32", "f32"):
            reasons.append(f"host quantize requires exact f32 operands/results; source is {shape.dtypes}")
        if not closed_float_mac(op):
            reasons.append("source is not a closed unflagged floating multiply/add contraction")
        if not zero_initialised(op):
            reasons.append("source contraction initializer is not proven zero")
        if contraction["status"] != "admitted":
            reasons.append("post-quantization contraction signature refused: " + contraction["reason"])
        for row in host_ops:
            for authority in ("software", "host"):
                if row[authority]["status"] != "admitted":
                    reasons.append(
                        f"{row['op']} rank {row['signature']['rank']} {authority}: {row[authority]['reason']}"
                    )
        try:
            # A one-element numerical bound, not a claim about the provider's
            # actual dispatch geometry. No primitive executes under a refusal.
            bound = integer_split_k_limit(spec, tile_dim=1, full_k=shape.reduction[0])
            if bound is None:
                reasons.append("no declared internal partial-sum arithmetic")
        except (IndexError, ValueError) as exc:
            bound = None
            reasons.append(str(exc))
        reasons.extend(pipeline_refusals(selection))
        # The selected provider currently offers only a raw-i32 catalog. Even a
        # future spec declaration cannot silently switch that ABI to an i8 one.
        reasons.append("no qualified full-call static-i8 readout provider/dispatch binding selected")
        sites.append(
            {
                "source_operation_ordinal": ordinals[op],
                "source_operation": op.name,
                "source_dtypes": list(shape.dtypes),
                "result_rank": rank,
                "proposed_signature": signature,
                "signature_stage": "proposed_after_quantization",
                "contraction_admission": contraction,
                "host_operations": host_ops,
                "partial_sum_obligation": bound,
                "actual_dispatch": "none; refused before mutation",
                "status": "refused",
                "reasons": reasons,
            }
        )
    return {
        "sites": sites,
        "quantized_contractions": 0,
        "scales": [],
        "status": "refused" if sites else "not_applicable",
        "fallback": "original_host_source",
        "calibration": {
            "mode": "static_per_tensor",
            "source": "calibration-on-input",
            "disclosure": "If admitted, calibration uses the evaluation case's own inputs, not held-out data.",
            "performed": False,
            "reason": "no complete declared int8 pipeline is admissible",
        },
        "integer_reference": {"status": "not_applicable", "reason": "no int8 device readout executes"},
    }
