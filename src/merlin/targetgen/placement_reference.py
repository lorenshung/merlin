"""Family throughput at a selected, declared reference workload.

This policy qualifies a semantic family independently of its test dimensions.
It does not grant numerical legality, producer coverage, or a runtime speedup.
The per-shape estimator remains a separate observation.
"""

import math


def _outputs(module):
    return [v for op in module.walk() if op.name == "func.return" for v in op.operands if hasattr(v.owner, "name")]


def movement_pattern(module):
    """Distinguish row selectors from element gathers using typed index maps."""
    evidence = []
    index_operations = 0

    def expression_cost(expr):
        return (1 + expression_cost(expr.lhs) + expression_cost(expr.rhs)) if hasattr(expr, "lhs") else 0

    for op in module.walk():
        if op.name == "linalg.generic":
            if any(child.name == "tensor.extract" for child in op.walk()):
                evidence.append(dict(operation=op.name, reason="explicit_element_index_lookup"))
            maps = list(op.indexing_maps)
            for mapping, operand in zip(maps, op.operands, strict=True):
                shape = operand.type.get_shape()
                if not shape or not mapping.data.results:
                    continue
                inner = mapping.data.results[-1]
                output_inner = maps[-1].data.results[-1]
                if inner != output_inner:
                    index_operations += expression_cost(inner)
                    evidence.append(dict(operation=op.name, input_map=str(mapping), reason="nonunit_inner_selector"))
        elif op.name == "linalg.transpose":
            permutation = list(op.permutation.get_values())
            if permutation and permutation[-1] != len(permutation) - 1:
                evidence.append(dict(operation=op.name, reason="permuted_innermost_axis"))
        elif op.name == "linalg.broadcast":
            dimensions = list(op.dimensions.get_values())
            if len(op.results[0].type.get_shape()) - 1 in dimensions:
                evidence.append(dict(operation=op.name, reason="broadcast_innermost_axis"))
        elif op.name in {"tensor.extract_slice", "tensor.insert_slice"}:
            strides = op.properties.get("static_strides")
            if strides is None or any(s != 1 for s in strides.get_values()):
                if strides is not None:
                    values = list(strides.get_values())
                    # A strided insert selects base vs update: two bounds per
                    # dimension, congruence for each strided axis, and select.
                    index_operations += (
                        2 * len(values) + sum(s != 1 for s in values) + 1
                        if op.name == "tensor.insert_slice"
                        else sum(s != 1 for s in values)
                    )
                evidence.append(dict(operation=op.name, reason="nonunit_or_dynamic_slice_stride"))
    return ("element_gather" if evidence else "row_contiguous"), evidence, index_operations


def estimate_reference(original, plan, declaration, facts):
    """Compare declared family reference work; never use an overload roster."""
    result = dict(schema="merlin.placement_reference.v1", status="host", policy="reference_scale")
    try:
        reference = declaration["reference_scale"]
        rates = reference["rates"]
        if any(type(v) not in {int, float} or not math.isfinite(v) or v <= 0 for v in rates.values()):
            raise ValueError("reference throughput rates must be finite and positive")
        outputs = _outputs(original)
        if not outputs:
            result.update(reason_code="no_materialized_output_work")
            return result
        shapes = [v.type.get_shape() for v in outputs if hasattr(v.type, "get_shape")]
        if not shapes or any(any(d < 0 for d in shape) for shape in shapes):
            raise ValueError("static materialized output extent unavailable")
        if all(math.prod(shape) <= reference["scalar_max_elements"] for shape in shapes):
            result.update(reason_code="scalar_work_retained_host")
            return result
        sites = [
            s for key in ("sites", "non_contraction_sites") for s in plan.get(key, []) if s.get("status") == "admitted"
        ]
        families = {s.get("family") for s in sites}
        if not families:
            raise ValueError("no admitted semantic producer family")
        if (
            plan.get("quantized_contractions")
            or plan.get("native_contractions")
            or any(op.name in {"linalg.matmul", "linalg.batch_matmul"} for op in original.walk())
        ):
            families.add("contraction")
        if any(op.name == "linalg.reduce" for op in original.walk()):
            families.add("reduction")
        family = (
            "movement"
            if plan.get("numerical_family") == "F1"
            else "contraction"
            if "contraction" in families
            else "reduction"
            if families & {"reduction", "window_mean"}
            else "elementwise_map"
        )
        if (
            family == "movement"
            and any(op.name == "linalg.index" for op in original.walk())
            and not any(op.name == "tensor.extract" for op in original.walk())
        ):
            # Static index-generated values were evaluated into a literal table
            # by the byte-selector provider. Moving that table does none of the
            # source arithmetic, so it is not an accelerated computation.
            result.update(reason_code="index_generated_constant_host", family=family)
            return result
        workload = reference["workloads"][family]
        if any(type(v) is not int or v <= 0 for v in workload.values()):
            raise ValueError("reference workload extents must be positive integers")
        setup = declaration["rates"]["dispatch_cycles"] + declaration["rates"]["fence_cycles"]
        pattern, pattern_evidence, index_operations = (
            movement_pattern(original) if family == "movement" else (None, [], 0)
        )
        if family == "movement":
            size = workload["bytes"]
            # Both input and output traffic is paid, including literal bytes
            # materialized by the host for fills. Reference rows are workload
            # dimensions, not hardware geometry.
            traffic = 2 * size
            host = traffic / rates["host_memcpy_bytes_per_cycle"]
            literal = any(
                op.name in {"tensor.splat", "linalg.fill", "arith.constant"}
                and any(hasattr(v.type, "get_shape") for v in op.results)
                for op in original.walk()
            )
            if literal:
                host += size / workload["element_bytes"] / rates["host_scalar_ops_per_cycle"]
            commands = math.ceil(traffic / workload["row_bytes"])
            if pattern == "element_gather":
                host += size / workload["element_bytes"] * index_operations / rates["host_scalar_ops_per_cycle"]
                memories = [
                    m
                    for m in facts.get("facts", facts).get("memories", [])
                    if m.get("name") == declaration["dma_memory"]
                ]
                if len(memories) != 1:
                    raise ValueError("selected DMA row width unavailable")
                row_bytes = memories[0]["row_elems"] * memories[0]["elem_bits"] // 8
                commands = math.ceil(traffic / workload["element_bytes"])
                traffic = commands * row_bytes
            device_parts = dict(
                dma=traffic / rates["dma_bytes_per_cycle"],
                instructions=commands / rates["device_commands_per_cycle"],
                setup=setup,
            )
        else:
            size = workload["elements"]
            work = size * workload["operations_per_element"]
            traffic = size * workload["bytes_per_element"]
            host = work / rates["host_scalar_ops_per_cycle"] + traffic / rates["host_memcpy_bytes_per_cycle"]
            device_parts = dict(
                dma=traffic / rates["dma_bytes_per_cycle"],
                compute=work / rates["device_mac_per_cycle"],
                host_pre_post=size / rates["host_pre_post_elements_per_cycle"],
                instructions=math.ceil(traffic / workload["row_bytes"]) / rates["device_commands_per_cycle"],
                setup=setup,
            )
        device = sum(device_parts.values())
        result.update(
            status="device" if device < host else "host",
            reason_code="profitable_reference_family" if device < host else "reference_family_not_profitable",
            family=family,
            pattern=pattern,
            pattern_evidence=pattern_evidence,
            selector_scalar_operations=index_operations,
            workload=workload,
            rates=rates,
            host_cycles=host,
            device_cycles=device,
            device_parts=device_parts,
            transfer_bytes=traffic,
            evidence=reference["evidence"],
        )
    except (ValueError, KeyError, TypeError, AttributeError, NotImplementedError) as exc:
        result.update(reason_code="reference_family_cost_unproved", reason=str(exc))
    return result
