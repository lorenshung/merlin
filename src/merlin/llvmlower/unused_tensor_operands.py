"""Remove tensor operands whose scalar argument is unused by a generic region."""


def prune_unused_tensor_inputs(module):
    from xdsl.dialects.builtin import ArrayAttr, DenseArrayBase, i32
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.body.blocks) != 1:
            continue
        n = len(op.inputs)
        if len(op.body.block.args) != n + len(op.outputs):
            continue
        unused = [i for i in range(n) if not op.body.block.args[i].uses]
        if not unused:
            continue
        kept = [i for i in range(n) if i not in unused]
        maps = list(op.get_indexing_maps())
        retained_maps = [maps[i].data for i in kept] + [m.data for m in maps[n:]]
        if not retained_maps or any(m.num_symbols for m in retained_maps):
            continue
        combined = AffineMap(retained_maps[0].num_dims, 0, tuple(e for m in retained_maps for e in m.results))
        if combined.inverse_permutation() is None:
            continue  # An unused scalar may still define a reduction's loop domain.
        extents = {}
        consistent = True
        for i in kept + list(range(n, len(op.operands))):
            for extent, expression in zip(op.operands[i].type.get_shape(), maps[i].data.results, strict=True):
                if isinstance(expression, AffineDimExpr):
                    if extent < 0 or extents.get(expression.position, extent) != extent:
                        consistent = False
                    extents[expression.position] = extent
        for i in unused:
            for extent, expression in zip(op.operands[i].type.get_shape(), maps[i].data.results, strict=True):
                if isinstance(expression, AffineDimExpr):
                    consistent &= extent >= 0 and extents.get(expression.position) == extent
                elif isinstance(expression, AffineConstantExpr):
                    consistent &= 0 <= expression.value < extent
                else:
                    consistent = False  # No proof that the eliminated load was in bounds.
        if not consistent:
            continue
        clone = op.clone()
        for index in reversed(unused):
            clone.body.block.erase_arg(clone.body.block.args[index])
        clone.operands = [op.inputs[i] for i in kept] + list(op.outputs)
        maps = list(op.get_indexing_maps())
        clone.properties["indexing_maps"] = ArrayAttr([maps[i] for i in kept] + maps[n:])
        clone.properties["operandSegmentSizes"] = DenseArrayBase.from_list(i32, [len(kept), len(op.outputs)])
        Rewriter.replace_op(op, clone)
        changed += 1
    module.verify()
    return changed
