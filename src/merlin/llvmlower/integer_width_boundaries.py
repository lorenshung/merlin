"""Retain integer storage normalization before every width-changing boundary.

An extension is not a homomorphism between two modular rings. Compute its
operand in the original ring, then execute the source extension on the exact
recombined storage word. This also preserves unsigned interpretation of i8.
"""


def preserve_integer_width_boundaries(module):
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import ArrayAttr, TensorType, UnitAttr
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, BlockArgument, Region
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.pointwise_primitives import expose_pointwise_primitives

    expose_pointwise_primitives(module, integer_boundaries=True)
    count = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1:
            continue
        if "merlin.exact_host_arithmetic" in op.attributes:
            continue
        body = list(op.body.block.ops)
        if len(body) != 2 or body[0].name not in {"arith.extsi", "arith.extui", "arith.trunci"}:
            continue
        cast = body[0]
        if cast.operands[0] not in op.body.block.args[:len(op.inputs)] or list(body[1].operands) != list(cast.results):
            continue
        source_index = list(op.body.block.args).index(cast.operands[0])
        if any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        source = op.inputs[source_index]
        if not isinstance(source, BlockArgument) or source.owner.parent_op().name != "func.func":
            # Intermediate source-width producers are lowered independently.
            # Their already recombined storage requires only the original host
            # cast, never another device identity producer for each bit cast.
            op.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
            count += 1
            continue
        source_type = op.inputs[source_index].type.element_type
        typ = TensorType(source_type, op.results[0].type.get_shape())
        empty = tensor.EmptyOp([], typ)
        block = Block(arg_types=[source_type, source_type])
        block.add_op(L.YieldOp(block.args[0]))
        carrier = L.GenericOp(
            inputs=[op.inputs[source_index]], outputs=[empty.tensor], body=Region(block),
            indexing_maps=ArrayAttr([op.get_indexing_maps().data[source_index],op.get_indexing_maps().data[-1]]),
            iterator_types=op.get_iterator_types(), result_types=[typ],
        )
        # The component carrier returns source-width storage bits. The original
        # cast consumes those bits after full-i32 limb recombination.
        from xdsl.dialects.builtin import AffineMapAttr
        from xdsl.ir.affine import AffineMap

        identity = AffineMapAttr(AffineMap.identity(len(typ.get_shape())))
        host_block = Block(arg_types=[source_type, op.results[0].type.element_type])
        host_cast = cast.clone(value_mapper={cast.operands[0]:host_block.args[0]})
        host_block.add_ops([host_cast,L.YieldOp(host_cast.results[0])])
        host = L.GenericOp(inputs=[carrier.results[0]],outputs=list(op.outputs),body=Region(host_block),
                           indexing_maps=ArrayAttr([identity,identity]),iterator_types=op.get_iterator_types(),
                           result_types=[op.results[0].type])
        host.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
        Rewriter.replace_op(op, [empty, carrier, host], list(host.results))
        count += 1
    return count
