"""Static, source-derived profitability estimates, separate from numeric legality.

All rates are selected declaration data. Geometry comes from selected RTL facts.
These estimates are neither measured latency nor hardware certification. Unknown
iteration counts, geometry or scalar costs refuse profitable placement.
"""

import hashlib
import math


def tensor_bytes(typ):
    if not hasattr(typ, "get_shape"):
        return 0
    shape = typ.get_shape()
    if any(d < 0 for d in shape):
        raise ValueError("dynamic tensor cost unavailable")
    element = typ.element_type
    if str(element).startswith("complex<"):
        return 2 * math.prod(shape) * tensor_bytes_scalar(element.element_type)
    return math.prod(shape) * tensor_bytes_scalar(element)


def tensor_bytes_scalar(typ):
    bits = getattr(getattr(typ, "width", None), "data", None)
    if bits is None:
        bits = getattr(typ, "bitwidth", None)
    if type(bits) is not int:
        raise ValueError(f"unknown element width in cost model: {typ}")
    return max(1, (bits + 7) // 8)


def iterations(op):
    if op.name.startswith("linalg.") and hasattr(op, "get_static_loop_ranges"):
        sizes = op.get_static_loop_ranges()
        if any(size < 0 for size in sizes):
            raise ValueError("dynamic iteration cost unavailable")
        return math.prod(sizes)
    if op.name == "linalg.reduce":
        shape = op.operands[0].type.get_shape()
        if any(size < 0 for size in shape):
            raise ValueError("dynamic reduction cost unavailable")
        return math.prod(shape)
    if op.name == "scf.for":

        def literal(value):
            owner = value.owner
            if getattr(owner, "name", None) != "arith.constant":
                raise ValueError("nonliteral loop cost unavailable")
            return owner.value.value.data

        lo, hi, step = (literal(v) for v in op.operands[:3])
        if step <= 0:
            raise ValueError("nonpositive loop cost unavailable")
        return max(0, (hi - lo + step - 1) // step)
    return 1


def scalar_cost(op, rates):
    name = op.name
    if name == "builtin.unregistered":
        name = op.op_name.data
    if name in {"arith.constant", "func.return", "linalg.yield", "scf.yield", "tensor.empty", "linalg.index"}:
        return 0
    if name.startswith("arith."):
        types = [str(v.type) for v in op.results]
        wide = max((int(t[1:]) for t in types if t.startswith("i") and t[1:].isdigit()), default=0)
        units = max(1, math.ceil(wide / rates["host_word_bits"]))
        return units * rates["host_arithmetic_cycles"]
    if name.startswith("math."):
        return rates["host_math_cycles"]
    if name in {"tensor.extract", "tensor.insert", "memref.load", "memref.store"}:
        return rates["host_index_cycles"]
    if name in {"tensor.splat", "linalg.fill", "linalg.copy"}:
        return sum(tensor_bytes(v.type) for v in op.results) * rates["host_copy_cycles_per_byte"]
    if name.startswith(("tensor.", "memref.", "bufferization.")):
        # Shape/descriptor work is still charged even for metadata-only sources.
        return rates["host_index_cycles"]
    if name.startswith("quant_ext."):
        return sum(tensor_bytes(v.type) for v in op.results) * rates["host_quantize_cycles_per_byte"]
    if op.regions or name in {"builtin.module", "func.func"}:
        return 0
    raise ValueError(f"unknown scalar operation cost: {name}")


def host_work(module, rates, excluded=()):
    excluded = set(excluded)

    def visit(op, multiplier):
        if op in excluded:
            return 0
        count = iterations(op)
        named_compute = hasattr(op, "get_static_loop_ranges") and not op.regions
        own = count * rates["host_arithmetic_cycles"] if named_compute else scalar_cost(op, rates)
        if op.name.startswith("linalg.") and op.name not in {
            "linalg.copy",
            "linalg.fill",
            "linalg.yield",
            "linalg.index",
        }:
            # A yield-only materialization still reads/writes storage. Index
            # descriptors and empty allocations alone do not pay that traffic.
            own += transfer_bytes(op) * rates["host_copy_cycles_per_byte"]
        return multiplier * own + sum(
            visit(child, multiplier * count) for region in op.regions for block in region.blocks for child in block.ops
        )

    return visit(module, 1)


def transfer_bytes(op):
    operands = op.operands
    if op.name == "linalg.generic" and len(op.body.block.args) == len(operands):
        operands = [v for v, arg in zip(operands, op.body.block.args, strict=True) if arg.uses]
    return sum(tensor_bytes(v.type) for v in (*operands, *op.results))


def estimate(original, prepared, plan, declaration, facts):
    """Compare the original host program with the complete proposed placement."""
    result = dict(schema="merlin.placement_cost.v1", status="host", declaration=declaration)
    try:
        if declaration.get("schema") != "merlin.placement_cost.v1":
            raise ValueError("unrecognized placement cost declaration")
        rates = declaration["rates"]
        if any(type(v) not in {int, float} or v < 0 or not math.isfinite(v) for v in rates.values()):
            raise ValueError("invalid placement cost rates")
        arrays = facts.get("facts", facts).get("arrays", [])
        geometries = [(a["rows"], a["cols"]) for a in arrays if a.get("corroborated")]
        if len(geometries) != 1 or min(geometries[0]) <= 0:
            raise ValueError("unique corroborated device geometry unavailable")
        rows, cols = geometries[0]
        operations = list(prepared.walk())
        sites = [s for s in plan["sites"] if s.get("status") == "admitted"]
        sites += [
            s
            for s in plan.get("non_contraction_sites", [])
            if s.get("status") == "admitted" and s.get("quantized_island")
        ]
        roots = {operations[s["source_operation_ordinal"]] for s in sites}
        if not roots:
            raise ValueError("no complete proposed device producer")
        traffic = sum(transfer_bytes(op) for op in roots)
        instructions = sum(math.ceil(transfer_bytes(op) / cols) for op in roots)
        compute = sum(math.ceil(iterations(op) / (rows * cols)) for op in roots)
        fences = len(roots)
        count_basis = "logical_transfer_rows_and_selected_RTL_array"
        catalog = plan.get("provider_catalog")
        if catalog:
            if catalog.get("source_sha256") != hashlib.sha256(str(prepared).encode()).hexdigest():
                raise ValueError("provider cost catalog has a different source identity")
            kernels = {kernel["symbol"]: kernel for kernel in catalog["kernels"]}
            invocations = [kernels[b["symbol"]] for b in catalog["bindings"]]
            if not invocations:
                raise ValueError("provider cost catalog has no invoked kernels")
            instructions = sum(kernel["instructions"] for kernel in invocations)
            fences = sum(kernel["fences"] for kernel in invocations)
            if plan.get("numerical_family") == "F1":
                # The closed F1 contract does no MAC work. Count actual emitted
                # commands and bound every byte DMA command by a full row.
                compute = 0
                memories = [
                    m
                    for m in facts.get("facts", facts).get("memories", [])
                    if m.get("name") == declaration["dma_memory"]
                ]
                if len(memories) != 1:
                    raise ValueError("selected DMA memory row fact unavailable")
                row_bytes = memories[0]["row_elems"] * memories[0]["elem_bits"] // 8
                if row_bytes <= 0:
                    raise ValueError("invalid selected DMA memory row width")
                roles = declaration["dma_count_roles"]
                if not isinstance(roles, list) or not roles or not all(isinstance(role, str) for role in roles):
                    raise ValueError("explicit provider DMA count roles unavailable")
                commands = sum(kernel["counts"].get(role, 0) for kernel in invocations for role in roles)
                traffic = max(traffic, commands * row_bytes)
            count_basis = "source_bound_provider_emitted_instruction_and_fence_counts"
        before_after = host_work(prepared, rates, roots)
        host = host_work(original, rates)
        device_parts = dict(
            dma=traffic * rates["dma_cycles_per_byte"],
            instructions=instructions * rates["instruction_issue_cycles"],
            compute=compute * rates["compute_tile_cycles"],
            fences=fences * rates["fence_cycles"],
            setup=rates["dispatch_cycles"],
            host_pre_post=before_after,
        )
        device = sum(device_parts.values())
        result.update(
            status="device" if device < host else "host",
            reason_code="profitable_device_placement" if device < host else "device_cost_not_profitable",
            host_cycles=host,
            device_cycles=device,
            device_parts=device_parts,
            transfer_bytes=traffic,
            instructions=instructions,
            fences=fences,
            instruction_count_basis=count_basis,
            producers=len(roots),
            geometry=dict(rows=rows, cols=cols, authority="selected_corroborated_RTL_array"),
        )
    except (ValueError, KeyError, TypeError, NotImplementedError, AssertionError) as exc:
        result.update(reason_code="placement_cost_unproved", reason=str(exc))
    return result
