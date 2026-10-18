"""IEEE sign negation as a modular integer sign-bit addition.

Every payload bit is preserved, including signed zeros and NaN payloads. The
linear data term is an exact byte component producer; the sign-bit offset is
explicit modular host recombination. No floating conversion is involved.
"""


def rewrite_native_sign_components(module):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntegerAttr, IntegerType, TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    widths = {"f16": 16, "bf16": 16, "f32": 32, "f64": 64}
    bounds = {}
    proofs = []
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 1 or len(op.results) != 1:
            continue
        if any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        body = list(op.body.block.ops)
        if len(body) != 2 or body[0].name != "arith.negf" or body[0].operands[0] is not op.body.block.args[0]:
            continue
        source = op.inputs[0]
        bits = widths.get(str(source.type.element_type))
        if bits is None or source.type.element_type != op.results[0].type.element_type:
            continue
        element = IntegerType(bits)
        shape = op.results[0].type.get_shape()
        identity = AffineMapAttr(AffineMap.identity(len(source.type.get_shape())))
        iterators = ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(source.type.get_shape()))
        empty = tensor.EmptyOp([], TensorType(element, source.type.get_shape()))
        block = Block(arg_types=[source.type.element_type, element])
        cast = arith.BitcastOp(block.args[0], element)
        block.add_ops([cast, L.YieldOp(cast.result)])
        packed = L.GenericOp(
            inputs=[source],
            outputs=[empty.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([identity, identity]),
            iterator_types=iterators,
            result_types=[empty.tensor.type],
        )
        bounds[packed.results[0]] = 1 << (bits - 1)
        destination = tensor.EmptyOp([], TensorType(element, shape))
        block = Block(arg_types=[element, element])
        offset = arith.ConstantOp(IntegerAttr(-(1 << (bits - 1)), element))
        addition = arith.AddiOp(block.args[0], offset.result)
        block.add_ops([offset, addition, L.YieldOp(addition.result)])
        changed = L.GenericOp(
            inputs=[packed.results[0]],
            outputs=[destination.tensor],
            body=Region(block),
            indexing_maps=op.get_indexing_maps(),
            iterator_types=op.get_iterator_types(),
            result_types=[destination.tensor.type],
        )
        bounds[changed.results[0]] = 1 << (bits - 1)
        out = tensor.EmptyOp([], op.results[0].type)
        block = Block(arg_types=[element, source.type.element_type])
        cast = arith.BitcastOp(block.args[0], source.type.element_type)
        block.add_ops([cast, L.YieldOp(cast.result)])
        out_map = AffineMapAttr(AffineMap.identity(len(shape)))
        result = L.GenericOp(
            inputs=[changed.results[0]],
            outputs=[out.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([out_map, out_map]),
            iterator_types=op.get_iterator_types(),
            result_types=[out.tensor.type],
        )
        Rewriter.replace_op(op, [empty, packed, destination, changed, out, result], [result.results[0]])
        proofs.append(
            dict(
                source="arith.negf",
                representation="ieee_storage_bits",
                width=bits,
                arithmetic="sign_bit_offset_modulo_storage_width",
                nan_payload="preserved",
            )
        )
    module.verify()
    return bounds, proofs
