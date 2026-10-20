"""Expose statically proved rank changes as materialized element selectors."""


def expose_static_reshape_movement(module):
    from math import prod

    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, DenseIntOrFPElementsAttr, TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineConstantExpr, AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    changed = 0
    for op in list(module.walk()):
        if op.name not in {"tensor.reshape", "tensor.collapse_shape", "tensor.expand_shape", "tensor.cast"}:
            continue
        if len(op.results) != 1 or not isinstance(op.results[0].type, TensorType):
            continue
        source, result = op.operands[0], op.results[0]
        if not isinstance(source.type, TensorType) or source.type.element_type != result.type.element_type:
            continue
        before, after = source.type.get_shape(), result.type.get_shape()
        if any(d < 0 for d in (*before, *after)) or not prod(before) or prod(before) != prod(after):
            continue
        if op.name == "tensor.reshape":
            shape = op.operands[1].owner
            attr = shape.properties.get("value") if getattr(shape, "name", None) == "arith.constant" else None
            if not isinstance(attr, DenseIntOrFPElementsAttr) or tuple(attr.get_values()) != after:
                continue
        elif len(op.operands) != 1:
            continue
        # Row-major logical flattening is independent of the caller's physical
        # stride. Bufferization carries the real source descriptor into this map.
        flat = AffineConstantExpr(0)
        for axis in range(len(after)):
            flat = flat + AffineDimExpr(axis) * prod(after[axis + 1 :])
        indices = tuple((flat // prod(before[axis + 1 :])) % extent for axis, extent in enumerate(before))
        destination = tensor.EmptyOp([], result.type)
        block = Block(arg_types=[source.type.element_type, result.type.element_type])
        block.add_op(L.YieldOp(block.args[0]))
        selected = L.GenericOp(
            inputs=[source],
            outputs=[destination.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr(
                [AffineMapAttr(AffineMap(len(after), 0, indices)), AffineMapAttr(AffineMap.identity(len(after)))]
            ),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(after)),
            result_types=[result.type],
        )
        selected.attributes.update(op.attributes)
        Rewriter.replace_op(op, [destination, selected], [selected.results[0]])
        changed += 1
    return changed
