"""Expose pure tuple regions with at most one reduction contribution per result.

Snapshot every original initializer as an explicit input. This retains coupled
tuple arithmetic and each IEEE boundary while giving each result a closed body.
Repeated reductions and unknown/effectful region bodies remain untouched.
"""


def expose_singleton_tuple_regions(module):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IndexType, IntegerAttr
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr
    from xdsl.rewriter import Rewriter
    from xdsl.traits import Pure

    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) < 2 or len(op.body.blocks) != 1:
            continue
        body = op.body.block
        if body.last_op.name != "linalg.yield" or len(body.last_op.operands) != len(op.results):
            continue
        if any(
            c.regions or c.name == "builtin.unregistered" or not (c.has_trait(Pure) or isinstance(c, L.IndexOp))
            for c in list(body.ops)[:-1]
        ):
            continue
        try:
            ranges = op.get_static_loop_ranges()
        except (AttributeError, ValueError):
            continue
        removed = {i for i, it in enumerate(op.get_iterator_types()) if it.data.value == "reduction"}
        if any(ranges[i] != 1 for i in removed) or any(n <= 0 for n in ranges):
            continue
        kept = [i for i in range(len(ranges)) if i not in removed]
        replacements = [
            AffineConstantExpr(0) if i in removed else AffineDimExpr(kept.index(i)) for i in range(len(ranges))
        ]
        maps = [
            AffineMapAttr(m.data.replace_dims_and_symbols(replacements, [], len(kept), 0))
            for m in op.get_indexing_maps()
        ]
        inputs = list(op.inputs) + list(op.outputs)
        operations, results = [], []
        for index, typ in enumerate(v.type for v in op.results):
            empty = tensor.EmptyOp([], typ)
            scalar = Block(arg_types=[v.type.element_type for v in inputs] + [typ.element_type])
            mapping = dict(zip(body.args, scalar.args[:-1], strict=True))
            for child in list(body.ops)[:-1]:
                if child.name == "linalg.index":
                    axis = child.dim.value.data
                    if not 0 <= axis < len(ranges):
                        break
                    clone = (
                        arith.ConstantOp(IntegerAttr(0, IndexType()))
                        if axis in removed
                        else L.IndexOp(kept.index(axis))
                    )
                else:
                    clone = child.clone(value_mapper=mapping)
                scalar.add_op(clone)
                mapping.update(zip(child.results, clone.results, strict=True))
            else:
                scalar.add_op(L.YieldOp(mapping[body.last_op.operands[index]]))
                result = L.GenericOp(
                    inputs=inputs,
                    outputs=[empty.tensor],
                    body=Region(scalar),
                    indexing_maps=ArrayAttr([*maps, maps[len(op.inputs) + index]]),
                    iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(kept)),
                    result_types=[typ],
                )
                result.attributes.update(op.attributes)
                operations.extend([empty, result])
                results.append(result.results[0])
                continue
            break
        if len(results) != len(op.results):
            continue
        Rewriter.replace_op(op, operations, results)
        changed += 1
    if changed:
        module.verify()
    return changed
