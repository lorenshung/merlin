"""Fail-closed admission for a declared static int8 full-call pipeline.

A scale's storage type is not its product arithmetic. Phase-0 form resolution
must not fill numerical semantics from a header or a target name. In particular,
an admitted MAC does not grant a narrowing readout or a host transfer.
"""

from __future__ import annotations

import hashlib
import math
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
        # The selected hardware capability declaration, for the HW screen of non-contraction sites.
        "capability_contract": contract,
        "resolution": resolution,
        "recipes": recipes,
        "quantization_candidates": candidates,
        "readout_facets": [facet.to_dict() for facet in facets],
        "identities": {
            name: {"path": str(path.resolve()), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for name, path in (("software_spec", spec_path), ("host_capabilities", host_path))
        },
    }


def declaration_location(selection, authority, key, fields=()):
    """Bind a decision to the exact authored YAML node, including rank restrictions."""
    import yaml

    identity = selection.get("identities", {}).get("software_spec" if authority == "software" else "host_capabilities")
    if not identity:
        return {"declaration": key}
    root = yaml.compose(Path(identity["path"]).read_text())

    def field(node, name):
        if not isinstance(node, yaml.MappingNode):
            return None
        return next((v for k, v in node.value if k.value == name), None)

    operations = field(root, "operations")
    if authority == "software":
        node = field(operations, key) or operations
    else:
        node = next((n for n in operations.value if field(n, "id").value == key), operations)
    if fields:
        for name in fields:
            node = field(node, name) or node
        return {**identity, "declaration": key, "line": node.start_mark.line + 1}
    signature = field(node, "signature") if isinstance(node, yaml.MappingNode) else None
    rank_node = field(signature or node, "ranks") if isinstance(signature or node, yaml.MappingNode) else None
    return {
        **identity,
        "declaration": key,
        "line": node.start_mark.line + 1,
        "restriction_line": (rank_node or signature or node).start_mark.line + 1,
    }


def pipeline_refusals(selection):
    reasons = []
    if selection["spec"].get("status") != "reviewed" or selection["host"].get("status") != "reviewed":
        reasons.append("software and host declarations must both be reviewed")
    return reasons


def readout_refusals(selection, decision):
    """Check the numerical contract of the declaration actually admitted at step 6."""
    reasons = []
    for authority, document in (("software", selection["spec"]), ("host", selection["host"])):
        key = decision[authority].get("declaration")
        row = next((r for r in document["operations"] if r["id"] == key), {})
        location = declaration_location(selection, authority, key)
        if "aten.mul.Tensor" not in row.get("ops", []) or row.get("placement") != "host":
            reasons.append(
                f"{location.get('path', '')}:{location.get('line', '')} {key}: "
                "required host aten.mul.Tensor declaration absent"
            )
        if authority == "host":
            contract = row.get("numerical_contract", {})
            semantics = contract.get("semantics", {})
            for field, expected in (
                ("arithmetic", "ieee754_binary32"),
                ("rounding", "round_to_nearest_even"),
                ("per_element", "correctly_rounded_single_operation"),
            ):
                if contract.get("status") != "reviewed" or semantics.get(field) != expected:
                    where = declaration_location(selection, authority, key, ("numerical_contract", "semantics", field))
                    reasons.append(
                        f"{where.get('path', '')}:{where.get('line', '')} {key}: "
                        f"required {field}={expected} declaration absent"
                    )
    return reasons


def admission_plan(module, selection, *, captured=False, native=False):
    """Independently screen all six steps at each source contraction's actual ranks."""
    from merlin.kernels.shapes import accumulator_init, observe_contractions, zero_initialised
    from merlin.llvmlower.contraction_quantization import closed_float_mac
    from merlin.targetgen.core_aten_readout import host_readout_bridge, planned_readout_geometry
    from merlin.targetgen.operation_numerics import integer_split_k_limit

    spec, host = selection["spec"], selection["host"]
    ordinals = {op: i for i, op in enumerate(module.walk())}
    sites = []
    for op, shape in observe_contractions(module):
        rank = len(op.results[0].type.get_shape())
        steps, reasons = [], pipeline_refusals(selection)
        # A capture in the integer_readout numerics reads the accumulator out ON THE DEVICE: one fused
        # requantization to int8, the bias an int32 accumulator seed. Then there is no host cast, no
        # host multiply and no cast exactness bound on K; the readout is a fused contraction stage.
        readout = device_readout(op) if (captured or native) else None
        seeded = accumulator_init(op) == "int32_seed"
        geometry = planned_readout_geometry(op) if not native and readout is None else None
        readout_proof = None
        readout_rank = rank
        if geometry is not None:
            if captured:
                try:
                    _, _, readout_proof = host_readout_bridge(op)
                    readout_rank = readout_proof["cast_rank"]
                except ValueError as exc:
                    reasons.append(str(exc))
            else:
                readout_rank = len(geometry["flat_shape"])
                readout_proof = {
                    "status": "planned_static_geometry_requires_capture_proof",
                    "cast_rank": readout_rank,
                    **geometry,
                }

        def screen(step, name, signature, placement="host"):
            row = {"step": step, "op": name, "signature": signature}
            for authority, document in (("software", spec), ("host", host)):
                if placement in {"accelerator", "fused_accelerator"} and authority == "host":
                    continue
                if name == "contraction":
                    from merlin.targetgen.software_spec import admit_contraction_operation

                    source = getattr(op.attributes.get("prov.aten"), "data", name)
                    decision = admit_contraction_operation(document, source, signature, placement)
                    row["source_operator"] = source
                else:
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

        if not native:
            for src in op.operands[:2]:
                screen(
                    1,
                    QUANTIZE,
                    host_signature(
                        QUANTIZE, len(src.type.get_shape()), {"zero_point": 0, "quant_min": -128, "quant_max": 127}
                    ),
                )
        transfers = spec.get("transfer_contracts", {})
        crossings = [
            (2, "operand_load", "host", "accelerator", "int8", [len(v.type.get_shape()) for v in op.operands[:2]]),
            (4, "int8_readout", "accelerator", "host", "int8", [rank])
            if readout is not None
            else (4, "accumulator_readout", "accelerator", "host", "i32", [rank]),
        ]
        if seeded:
            crossings.insert(1, (2, "accumulator_seed_load", "host", "accelerator", "i32", [rank]))
        for step, key, source, destination, dtype, ranks in crossings:
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
        cast_obligation = None
        if readout is not None:
            # The fused readout stage: acc_scale (and a ReLU before it) on the contraction's accumulator.
            stages = ["relu", "acc_scale"] if readout["relu"] else ["acc_scale"]
            screen(
                4,
                "elementwise_map",
                dict(
                    family="elementwise_map",
                    operand_dtype="i32",
                    readout_dtype="i8",
                    rank=rank,
                    epilogues=stages,
                    composed_with=["contraction"],
                    scale_granularity=readout["granularity"],
                ),
                "fused_accelerator",
            )
            steps[-1]["device_readout"] = {key: value for key, value in readout.items() if key != "operation"}
            if readout["relu"] and any(scale <= 0 for scale in readout["scales"]):
                reasons.append(
                    "step 4 a readout ReLU runs before the scale, so it cannot pair with a nonpositive scale"
                )
        if not native and readout is None:
            screen(
                5,
                "aten._to_copy.default",
                dict(
                    family="elementwise_map",
                    ordered_operand_dtypes=["i32", "f32"],
                    ordered_result_dtypes=["f32"],
                    operand_dtype="i32",
                    rank=readout_rank,
                ),
            )
            cast_decision = steps[-1]["host"]
            cast_key = cast_decision.get("declaration", "host_i32_to_f32_cast")
            cast_declaration = next((r for r in host["operations"] if r["id"] == cast_key), {})
            contract = cast_declaration.get("numerical_contract", {})
            exact_range = contract.get("semantics", {}).get("exact_range", "")
            location = declaration_location(
                selection, "host", cast_key, ("numerical_contract", "semantics", "exact_range")
            )
            prefix, separator, exponent = exact_range.partition("|x| <= 2^")
            exact_limit = 2 ** int(exponent) if separator and not prefix and exponent.isdecimal() else None
            operand_ranges = [row["signature"]["quantization_parameters"] for row in steps if row["step"] == 1]
            cast_bound = math.prod(shape.reduction) * math.prod(
                max(abs(r["quant_min"]), abs(r["quant_max"])) for r in operand_ranges
            )
            cast_obligation = dict(
                full_k=math.prod(shape.reduction),
                operand_ranges=operand_ranges,
                max_abs_bound=cast_bound,
                exact_limit=exact_limit,
                source=location,
            )
            if contract.get("status") != "reviewed" or exact_limit is None or cast_bound > exact_limit:
                reasons.append(
                    f"step 5 {location.get('path', '')}:{location.get('line', '')} {cast_key}: "
                    f"accumulator bound {cast_bound} exceeds or lacks declared exact_range {exact_range!r}"
                )
            screen(
                6,
                "aten.mul.Tensor",
                dict(
                    family="elementwise_map",
                    ordered_operand_dtypes=["f32", "f32", "f32"],
                    ordered_result_dtypes=["f32"],
                    operand_dtype="f32",
                    rank=readout_rank,
                ),
            )
            steps[-1]["readout_multiply"] = 1
            if all(steps[-1][a]["status"] == "admitted" for a in ("software", "host")):
                reasons.extend(readout_refusals(selection, steps[-1]))
            screen(
                6,
                "aten.mul.Tensor",
                dict(
                    family="elementwise_map",
                    ordered_operand_dtypes=["f32", "f32", "f32"],
                    ordered_result_dtypes=["f32"],
                    operand_dtype="f32",
                    rank=readout_rank,
                ),
            )
            steps[-1]["readout_multiply"] = 2
            if all(steps[-1][a]["status"] == "admitted" for a in ("software", "host")):
                reasons.extend(readout_refusals(selection, steps[-1]))
        expected = ("i8", "i8", "i32") if captured or native else ("f32", "f32", "f32")
        if shape.dtypes != expected:
            reasons.append(f"source dtype {shape.dtypes} is not {expected}")
        closed = closed_integer_mac(op) if captured or native else closed_float_mac(op)
        if not closed or not (zero_initialised(op) or (seeded and readout is not None)):
            reasons.append(
                "source is not a closed unflagged MAC from zero (or from an int32 seed it reads out on device)"
            )
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
                host_readout_bridge=readout_proof,
                readout="device_int8" if readout is not None else "host",
                accumulator_init="int32_seed" if seeded else "zero",
                steps=sorted(steps, key=lambda s: s["step"]),
                partial_sum_obligation=bound,
                cast_exact_range_obligation=cast_obligation,
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
        # Elementwise, reduction and movement sites, screened by the same authorities; informational
        # for the contraction pipeline above, whose status they do not change.
        non_contraction_sites=non_contraction_sites(module, selection, contract=selection.get("capability_contract")),
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


def host_readout_chain(contraction, *, with_bridge=False):
    """Require the exact captured cast/readout chain, including any proved views."""
    from merlin.targetgen.core_aten_readout import host_readout_bridge

    bridge = host_readout_bridge(contraction)
    return bridge if with_bridge else bridge[0]


def source_declaration_continuity(initial, captured):
    """Refuse loss of exact frontend obligations through an overload rewrite."""
    from collections import Counter

    initial_steps = [step for site in initial["sites"] for step in site["steps"] if step["step"] == 3]
    required = Counter(
        step["source_operator"] for step in initial_steps if step["software"].get("applicable_exact_source_declaration")
    )
    observed = Counter(
        step.get("source_operator") for site in captured["sites"] for step in site["steps"] if step["step"] == 3
    )
    missing = required - observed
    return {
        "status": "refused" if missing else "preserved",
        "initial_source_admissions": initial_steps,
        "captured_source_operator_counts": dict(observed),
        "unpreserved_exact_source_operators": dict(missing),
    }


def device_readout(op) -> dict | None:
    """The fused int8 readout that consumes ``op``'s accumulator, or None.

    The frontend's ``integer_readout`` numerics emit it as the accumulator's ONLY user: a parallel
    ``linalg.generic`` tagged ``prov.role = "readout"`` with an i8 result, its scale, zero point, range
    and ReLU carried as ``prov.quant.*`` attributes. Read from those attributes, never from a name; an
    absent or malformed one means there is no readout this plan can state.
    """
    from merlin.common import mlir_query as mq

    result = op.results[0] if len(op.results) == 1 else None
    users = list(result.uses) if result is not None else []
    if len(users) != 1:
        return None
    reader = users[0].operation
    if mq.op_name(reader) != "linalg.generic" or mq.attr_str(reader, "prov.role") != "readout":
        return None
    if str(getattr(reader.results[0].type, "element_type", "")) != "i8":
        return None  # an f32 acc_readout leaves the multiply to the host
    attributes = reader.attributes

    def number(key):
        value = attributes.get(key)
        data = getattr(getattr(value, "value", None), "data", None)
        return data if isinstance(data, (int, float)) and not isinstance(data, bool) else None

    scale = attributes.get("prov.quant.scale")
    if scale is not None and hasattr(scale, "data") and isinstance(scale.data, tuple):
        scales = [getattr(getattr(item, "value", None), "data", None) for item in scale.data]
    else:
        scales = [number("prov.quant.scale")]
    parameters = {key: number(f"prov.quant.{key}") for key in ("zero_point", "quant_min", "quant_max")}
    relu = mq.attr_str(reader, "prov.quant.relu")
    if (
        any(not isinstance(v, (int, float)) for v in scales)
        or None in parameters.values()
        or relu not in {"true", "false"}
    ):
        return None
    return {
        "scales": [float(v) for v in scales],
        "granularity": "tensor" if len(scales) == 1 else "channel",
        **{key: int(value) for key, value in parameters.items()},
        "relu": relu == "true",
        "operation": reader,
    }


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


#: Carriers that are views or scalar plumbing, not independent device work (C52: views stay host
#: metadata and are never materialised), so they are not screened as sites.
_NON_SITE_CARRIERS = frozenset(
    {
        "tensor.empty",
        "tensor.extract_slice",
        "tensor.collapse_shape",
        "tensor.expand_shape",
        "tensor.reshape",
        "tensor.cast",
        "tensor.dim",
        "tensor.extract",
        "tensor.yield",
        "linalg.yield",
        "linalg.index",
    }
)
#: Tensor-dialect carriers whose work is data motion.
_TENSOR_MOVEMENT = frozenset({"tensor.concat", "tensor.insert_slice", "tensor.pad", "tensor.splat"})


def _site_family(op) -> str | None:
    """The semantic family of a top-level non-contraction site, structurally; None when not a site."""
    from merlin.common import mlir_query as mq
    from merlin.targetgen import semantic_families as sf
    from merlin.xdsl_dialects.lowering import contraction_coverage as cc

    name = mq.op_name(op)
    if name in _NON_SITE_CARRIERS:
        return None
    if name == "linalg.generic":
        return {
            "sum-reduction": "reduction",
            "max-reduction": "reduction",
            "absmax": "reduction",
            "other-reduction": "reduction",
            "movement": "movement",
            "elementwise": "elementwise_map",
        }.get(cc.classify_generic(op))
    if name == "linalg.reduce":
        return "reduction"
    if name in _TENSOR_MOVEMENT:
        return "movement"
    if name.startswith("linalg."):
        family = sf.from_op(name.rpartition(".")[2])
        return family if family in {"movement", "elementwise_map", "reduction"} else None
    return None


_DEQUANTIZE_OPS = frozenset({"quant_ext.dequantize_per_tensor"})
_QUANTIZE_OPS = frozenset({"quant_ext.quantize_per_tensor"})
#: Value producers that carry no data from a host tensor: a constant, a splat of one, an empty init.
_CONSTANT_PRODUCERS = frozenset({"arith.constant", "tensor.empty", "tensor.splat", "linalg.fill"})


def quantized_islands(module) -> set[int]:
    """``id``s of the floating-point operations that compute only between int8 quantization edges.

    PT2E leaves a quantized non-contraction operation as ``dequantize -> f32 op -> quantize``. Such an
    op computes on int8 codes and its result is read only as int8: it is int8 work whose floating
    spelling is the frontend's, exactly as a backend that matches the sandwich treats it. An op belongs
    to an island when every tensor operand comes from a dequantize, a constant producer or another
    island op, and every user is a quantize or another island op (a fixpoint, so chains -- a window
    sum followed by its division, the slices of a pad -- form one island).
    """
    from merlin.common import mlir_query as mq

    operations = [op for op in module.walk() if mq.op_name(op) not in {"builtin.module", "func.func", "func.return"}]

    def floating(op) -> bool:
        return any("f32" in str(getattr(result.type, "element_type", "")) for result in op.results)

    def ranked(op) -> bool:
        return any(list(getattr(result.type, "get_shape", lambda: [])()) for result in op.results)

    candidates = {
        id(op): op
        for op in operations
        if floating(op)
        and mq.op_name(op) not in _DEQUANTIZE_OPS | _QUANTIZE_OPS | {"tensor.empty", "arith.constant"}
        # A rank > 0 fill is the work of fill/full/full_like; a scalar one is a quantization parameter.
        and (mq.op_name(op) not in _CONSTANT_PRODUCERS or ranked(op))
        and not _nested_in_linalg(op)
    }
    changed = True
    while changed:
        changed = False
        for key, op in list(candidates.items()):
            sourced = all(
                (owner := getattr(value, "owner", None)) is not None
                and hasattr(owner, "results")
                and (
                    mq.op_name(owner) in _DEQUANTIZE_OPS
                    or mq.op_name(owner) in _CONSTANT_PRODUCERS
                    or id(owner) in candidates
                )
                for value in op.operands
                if hasattr(value.type, "get_shape")
            )
            consumed = all(
                mq.op_name(use.operation) in _QUANTIZE_OPS or id(use.operation) in candidates
                for result in op.results
                for use in result.uses
            ) and any(result.uses for result in op.results)
            if not (sourced and consumed):
                del candidates[key]
                changed = True
    return set(candidates)


def _dense_tensor(tensor_type) -> bool:
    """A ranked tensor type with no layout encoding (the default dense row-major layout)."""
    from xdsl.dialects.builtin import NoneAttr

    encoding = getattr(tensor_type, "encoding", None)
    return encoding is None or isinstance(encoding, NoneAttr)


def _nested_in_linalg(op) -> bool:
    parent = op.parent_op()
    while parent is not None:
        if getattr(parent, "name", "").startswith("linalg."):
            return True
        parent = parent.parent_op()
    return False


def non_contraction_sites(module, selection, *, contract=None, prospective=False):
    """Screen every top-level non-contraction site of an integer program for device placement.

    A site is one elementwise, reduction or movement operation outside the contractions
    :func:`admission_plan` screens. Each passes through the same authorities a contraction does: the
    operand-load and readout transfer contracts, SW :func:`admit_operation` (standalone, and fused
    behind a contraction producer when one feeds it) and, given the selected capability contract,
    HW :func:`eligibility.is_eligible`. Nothing here selects a backend or a lowering: an admitted site
    is a placement candidate only. A float site is screened like any other and refused by precision,
    which is the honest answer until a recipe quantizes it.

    ``prospective`` screens a FLOATING source program as the int8 program its static recipe would
    make of it (every floating site as int8 work), the way a contraction is screened prospectively
    before its recapture; it decides whether a recapture is worth making, never what is placed.
    """
    from merlin.common import mlir_query as mq
    from merlin.kernels.shapes import observe_contractions
    from merlin.targetgen import access_observations
    from merlin.targetgen import semantic_families as sf
    from merlin.targetgen.eligibility import (
        RegionDescriptor,
        capability_map_from_contract,
        is_eligible,
        undetermined_families_from_contract,
    )
    from merlin.targetgen.transfer_contracts import screen_transfer_contract
    from merlin.xdsl_dialects.lowering import contraction_coverage as cc

    spec = selection["spec"]
    capability_map = capability_map_from_contract(contract) if contract is not None else None
    undetermined = undetermined_families_from_contract(contract) if contract is not None else None
    contractions = {id(op) for op, _shape in observe_contractions(module)}
    ordinals = {id(op): i for i, op in enumerate(module.walk())}
    islands = quantized_islands(module)

    def precision(value) -> str | None:
        dtype = mq.type_shape_dtype(value.type)[1]
        return dtype or None

    def nested(op) -> bool:
        parent = op.parent_op()
        while parent is not None:
            if mq.op_name(parent).startswith("linalg."):
                return True
            parent = parent.parent_op()
        return False

    def produced_by_contraction(op) -> bool:
        return any(id(getattr(value, "owner", None)) in contractions for value in op.operands)

    sites = []

    def quantization_parameter(op) -> bool:
        """A scalar or a value read only by quantize/dequantize ops: a scale or zero point, not work."""
        results = list(op.results)
        if results and all(not list(getattr(r.type, "get_shape", lambda: [0])()) for r in results):
            return True
        users = [use.operation for result in results for use in result.uses]
        return (
            bool(users)
            and all(mq.op_name(user) in _DEQUANTIZE_OPS | _QUANTIZE_OPS for user in users)
            and all(use.index != 0 for result in results for use in result.uses)
        )

    for op in module.walk():
        if id(op) in contractions or nested(op) or quantization_parameter(op):
            continue
        family = _site_family(op)
        if family is None:
            continue
        name = mq.op_name(op)
        tensors = [value for value in op.operands if hasattr(value.type, "get_shape")]
        results = list(op.results)
        operand_dtypes = [precision(value) for value in op.operands]
        result_dtypes = [precision(value) for value in results]
        rank = len(results[0].type.get_shape()) if results and hasattr(results[0].type, "get_shape") else None
        source = mq.attr_str(op, "prov.aten") or name
        observations = access_observations.observe(op, family)
        if name.startswith("tensor.") and all(
            _dense_tensor(value.type) for value in [*op.operands, *op.results] if hasattr(value.type, "get_shape")
        ):
            # Tensor-dialect values are immutable SSA with the default dense row-major layout.
            observations = {**observations, "layout": "row_major_contiguous", "aliasing": "disjoint_inputs_outputs"}
        form = sf.carrier_form(op) if family == "movement" else None
        kind = cc.reduction_kind(op) if family == "reduction" else None
        in_dtype = precision(tensors[0]) if tensors else (operand_dtypes[0] if operand_dtypes else None)
        out_dtype = result_dtypes[0] if result_dtypes else None
        quantized = id(op) in islands or (prospective and "f32" in result_dtypes)
        if quantized:
            # Int8 work in the frontend's floating spelling (dequantize -> op -> quantize): its codes are
            # int8 on both edges, so it is screened as the int8 operation the island computes.
            operand_dtypes = ["i8" if dtype == "f32" else dtype for dtype in operand_dtypes]
            result_dtypes = ["i8" if dtype == "f32" else dtype for dtype in result_dtypes]
            in_dtype = "i8" if in_dtype in (None, "f32") else in_dtype
            out_dtype = "i8" if out_dtype == "f32" else out_dtype
        signature = {
            "family": family,
            "operand_dtype": in_dtype,
            "ordered_operand_dtypes": operand_dtypes,
            "ordered_result_dtypes": result_dtypes,
            "rank": rank,
            # The pipeline's own calibration policy is static per-tensor (admission_plan).
            "scale_granularity": "tensor",
            **{key: value for key, value in observations.items() if value is not None},
        }
        if form is not None:
            signature["form"] = form
        if kind is not None:
            signature["kind"] = kind
        steps, reasons = [], list(pipeline_refusals(selection))
        for step, key, origin, destination, dtype in (
            (2, "operand_load", "host", "accelerator", in_dtype),
            (4, "readout", "accelerator", "host", out_dtype),
        ):
            decision = screen_transfer_contract(
                spec,
                source_placement=origin,
                destination_placement=destination,
                operand_dtype=dtype,
                result_dtype=dtype,
                operand_layout="row_major_contiguous",
                result_layout="row_major_contiguous",
            )
            steps.append({"step": step, "transfer": key, "dtype": dtype, **decision})
        placements = [("accelerator", signature)]
        if produced_by_contraction(op) and family == "elementwise_map":
            placements.append(("fused_accelerator", {**signature, "composed_with": ["contraction"]}))
        software = [admit_operation(spec, source, observed, placement) for placement, observed in placements]
        index = next(
            (i for i, d in enumerate(software) if d["status"] == "admitted"),
            next((i for i, d in enumerate(software) if d["status"] == "unknown"), 0),
        )
        chosen = software[index]
        fused = ("contraction",) if placements[index][0] == "fused_accelerator" else None
        if chosen.get("declaration"):
            chosen = {**chosen, "source": declaration_location(selection, "software", chosen["declaration"])}
        steps.append({"step": 3, "software": chosen, "placement": placements[index][0]})
        hardware = None
        if capability_map is not None:
            verdict = is_eligible(
                RegionDescriptor(
                    op=name.rpartition(".")[2],
                    family=family,
                    in_dtype=in_dtype,
                    out_dtype=out_dtype,
                    rank=rank,
                    layout=signature.get("layout"),
                    form=form,
                    kind=kind,
                    scale_granularity=None,
                ),
                capability_map,
                undetermined=undetermined,
                fused_with=fused,
            )
            hardware = {
                "status": "admitted" if verdict.eligible else "unknown" if verdict.undetermined else "unsupported",
                "reason": verdict.reason,
                "refusal": verdict.refusal,
            }
            steps.append({"step": 3, "hardware": hardware})
        statuses = [row["status"] for row in steps if "transfer" in row] + [chosen["status"]]
        statuses.append(hardware["status"] if hardware is not None else "unknown")
        for row in steps:
            if "transfer" in row and row["status"] not in {"admitted", "not_applicable"}:
                reasons.append(
                    f"step {row['step']} {row['transfer']} {row['dtype']}: {row.get('reason', row['status'])}"
                )
        if chosen["status"] != "admitted":
            reasons.append(f"step 3 {source} software {chosen.get('declaration', '')}: {chosen['reason']}")
        if hardware is None:
            reasons.append("step 3 hardware: no selected capability contract")
        elif hardware["status"] != "admitted":
            reasons.append(f"step 3 hardware: {hardware['reason']}")
        status = (
            "admitted"
            if not reasons
            else "refused"
            if any(value == "unsupported" for value in statuses) or pipeline_refusals(selection)
            else "unknown"
        )
        sites.append(
            {
                "source_operation_ordinal": ordinals[id(op)],
                "source_operation": name,
                "source_operator": source,
                "family": family,
                "quantized_island": quantized,
                "signature": signature,
                "steps": steps,
                "status": status,
                "reasons": reasons,
                "scope": "placement screen only; no backend, lowering or numerics is selected here",
            }
        )
    return sites
