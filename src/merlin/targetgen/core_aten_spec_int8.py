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
        "host_scale_product": {
            "arithmetic": "exact f64 product of f32 scales, rounded once to f32",
            "admission": "finite normal f32 constant required",
            "authority": "explicit static numeric policy; host multiply declaration independently screened",
        },
    }


def declaration_location(selection, authority, key):
    """Bind a decision to the exact authored YAML node, including rank restrictions."""
    import yaml

    identity = selection.get("identities", {}).get("software_spec" if authority == "software" else "host_capabilities")
    if not identity:
        return {"declaration": key}
    root = yaml.compose(Path(identity["path"]).read_text())

    def field(node, name):
        return next((v for k, v in node.value if k.value == name), None)

    operations = field(root, "operations")
    if authority == "software":
        node = field(operations, key)
    else:
        node = next(n for n in operations.value if field(n, "id").value == key)
    signature = field(node, "signature")
    rank_node = field(signature or node, "ranks")
    return {
        **identity,
        "declaration": key,
        "line": node.start_mark.line + 1,
        "restriction_line": (rank_node or signature or node).start_mark.line + 1,
    }


def pipeline_refusals(selection):
    return (
        []
        if selection["spec"].get("status") == selection["host"].get("status") == "reviewed"
        else ["software and host declarations must both be reviewed"]
    )


def admission_plan(module, selection, *, captured=False):
    """Independently screen all six steps at each source contraction's actual ranks."""
    from merlin.kernels.shapes import observe_contractions, zero_initialised
    from merlin.llvmlower.contraction_quantization import closed_float_mac
    from merlin.targetgen.operation_numerics import integer_split_k_limit

    spec, host = selection["spec"], selection["host"]
    ordinals = {op: i for i, op in enumerate(module.walk())}
    sites = []
    for op, shape in observe_contractions(module):
        rank = len(op.results[0].type.get_shape())
        steps, reasons = [], pipeline_refusals(selection)

        def screen(step, name, signature, placement="host"):
            row = {"step": step, "op": name, "signature": signature}
            for authority, document in (("software", spec), ("host", host)):
                if placement == "accelerator" and authority == "host":
                    continue
                decision = admit_operation(document, name, signature, placement)
                if decision.get("declaration"):
                    decision["source"] = declaration_location(selection, authority, decision["declaration"])
                row[authority] = decision
                if decision["status"] != "admitted":
                    where = decision.get("source", {})
                    reasons.append(
                        f"step {step} {name} rank {signature['rank']} {authority}: "
                        f"{where.get('path', '')}:{where.get('restriction_line', '')} "
                        f"{decision.get('declaration', '')}: {decision['reason']}"
                    )
            steps.append(row)

        for src in op.operands[:2]:
            screen(
                1,
                QUANTIZE,
                host_signature(
                    QUANTIZE, len(src.type.get_shape()), {"zero_point": 0, "quant_min": -128, "quant_max": 127}
                ),
            )
        transfers = spec.get("transfer_contracts", {})
        for step, key, source, destination, dtype, ranks in (
            (2, "operand_load", "host", "accelerator", "int8", [len(v.type.get_shape()) for v in op.operands[:2]]),
            (4, "accumulator_readout", "accelerator", "host", "i32", [rank]),
        ):
            from merlin.targetgen.transfer_contracts import screen_transfer_contract

            transfer = next((r for r in transfers.get("declarations", []) if r["id"] == key), {})
            decision = screen_transfer_contract(
                spec,
                source_placement=source,
                destination_placement=destination,
                operand_dtype=dtype,
                result_dtype=dtype,
                operand_layout="row_major_contiguous",
                result_layout="row_major_contiguous",
            )
            admitted = decision["status"] == "admitted" and key in decision["matching_declarations"]
            if step == 4:
                admitted &= transfer.get("semantics", {}).get("valid_window") == "required"
            row = {
                "step": step,
                "declaration": f"transfer_contracts.{key}",
                "status": "admitted" if admitted else "refused",
                "ranks": ranks,
                "constraints": transfer,
            }
            identity = selection.get("identities", {}).get("software_spec")
            if identity:
                import yaml

                root = yaml.compose(Path(identity["path"]).read_text())
                node = next(v for k, v in root.value if k.value == "transfer_contracts")
                node = next(v for k, v in node.value if k.value == key)
                row["source"] = {**identity, "line": node.start_mark.line + 1}
            steps.append(row)
            if not admitted:
                reasons.append(f"step {step} {row['declaration']} refused at {row.get('source')}")
        signature = dict(
            family="contraction",
            ordered_operand_dtypes=["i8", "i8", "i32"],
            ordered_result_dtypes=["i32"],
            operand_dtype="i8",
            accum_dtype="i32",
            readout_dtype="i32",
            rank=rank,
            layout="row_major_contiguous",
            tails="zero_pad_valid_window",
            broadcasting="independent_batches" if rank > 2 else "none",
            aliasing="disjoint_inputs_outputs",
        )
        screen(3, "contraction", signature, "accelerator")
        screen(
            5,
            "aten._to_copy.default",
            dict(
                family="elementwise_map",
                ordered_operand_dtypes=["i32", "f32"],
                ordered_result_dtypes=["f32"],
                operand_dtype="i32",
                rank=rank,
            ),
        )
        screen(
            6,
            "aten.mul.Tensor",
            dict(
                family="elementwise_map",
                ordered_operand_dtypes=["f32", "f32", "f32"],
                ordered_result_dtypes=["f32"],
                operand_dtype="f32",
                rank=rank,
            ),
        )
        expected = ("i8", "i8", "i32") if captured else ("f32", "f32", "f32")
        if shape.dtypes != expected:
            reasons.append(f"source dtype {shape.dtypes} is not {expected}")
        closed = closed_integer_mac(op) if captured else closed_float_mac(op)
        if not closed or not zero_initialised(op):
            reasons.append("source is not a closed unflagged from-zero float MAC")
        try:
            bound = integer_split_k_limit(spec, tile_dim=1, full_k=shape.reduction[0])
            if bound is None:
                reasons.append("no declared internal partial-sum arithmetic")
        except (ValueError, IndexError) as exc:
            bound = None
            reasons.append(str(exc))
        sites.append(
            dict(
                source_operation_ordinal=ordinals[op],
                source_operation=op.name,
                source_dtypes=list(shape.dtypes),
                result_rank=rank,
                steps=sorted(steps, key=lambda s: s["step"]),
                partial_sum_obligation=bound,
                status="refused" if reasons else "admitted",
                reasons=reasons,
            )
        )
    quantizers = []
    if captured:
        from merlin.common import mlir_query as mq

        for op in module.walk():
            if mq.op_name(op) != "quant_ext.quantize_per_tensor":
                continue
            from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

            parameters = {
                "zero_point": _constant_of(op.operands[2]),
                "quant_min": op.properties["quant_min"].value.data,
                "quant_max": op.properties["quant_max"].value.data,
            }
            signature = host_signature(QUANTIZE, len(op.operands[0].type.get_shape()), parameters)
            signature["ordered_operand_dtypes"] = [str(v.type.element_type) for v in op.operands]
            signature["ordered_result_dtypes"] = [str(op.results[0].type.element_type)]
            row = {"source_operation_ordinal": ordinals[op], "source_operation": mq.op_name(op), "signature": signature}
            for authority, document in (("software", spec), ("host", host)):
                decision = admit_operation(document, QUANTIZE, signature, "host")
                if decision.get("declaration"):
                    decision["source"] = declaration_location(selection, authority, decision["declaration"])
                row[authority] = decision
                if decision["status"] != "admitted":
                    location = decision.get("source", {})
                    for site in sites:
                        site["status"] = "refused"
                        site["reasons"].append(
                            f"step 1 captured quantizer {authority}: {location.get('path', '')}:"
                            f"{location.get('restriction_line', '')} {decision['reason']}"
                        )
            quantizers.append(row)
    return dict(
        sites=sites,
        quantization_sites=quantizers,
        quantized_contractions=0,
        scales=[],
        status="admitted"
        if any(s["status"] == "admitted" for s in sites)
        else "refused"
        if sites
        else "not_applicable",
        calibration=dict(
            mode="static_per_tensor",
            source="calibration-on-evaluation-input",
            disclosure="Scales are observed on this evaluation case's own inputs, not held-out calibration data.",
            performed=False,
        ),
        integer_reference=dict(status="not_run"),
    )


def closed_integer_mac(op):
    """Only signed i8 extensions and a closed, unflagged i32 multiply/add."""
    from xdsl.dialects import arith
    from xdsl.dialects.linalg.ops import YieldOp

    if len(op.operands) != 3 or len(op.results) != 1 or not hasattr(op, "body"):
        return False
    block = op.body.block
    if len(block.args) != 3 or len(block.ops) != 5:
        return False
    a, b, mul, add, yielded = list(block.ops)
    return (
        isinstance(a, arith.ExtSIOp)
        and isinstance(b, arith.ExtSIOp)
        and tuple(a.operands) == (block.args[0],)
        and tuple(b.operands) == (block.args[1],)
        and isinstance(mul, arith.MuliOp)
        and not mul.overflow_flags.data
        and tuple(mul.operands) == (a.result, b.result)
        and isinstance(add, arith.AddiOp)
        and not add.overflow_flags.data
        and tuple(add.operands) in ((mul.result, block.args[2]), (block.args[2], mul.result))
        and isinstance(yielded, YieldOp)
        and tuple(yielded.operands) == (add.result,)
    )
