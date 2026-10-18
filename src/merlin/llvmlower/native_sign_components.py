"""Native sign and closed compare/select epilogues over exact integer words.

Every payload bit is preserved, including signed zeros and NaN payloads. The
linear data term is an exact byte component producer; the sign-bit offset is
explicit modular host recombination. No floating conversion is involved.
"""


def rewrite_native_sign_components(module, *, canonical_bf16_nan=False):
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
        argument = op.body.block.args[0]
        negate = len(body) == 2 and body[0].name == "arith.negf" and body[0].operands[0] is argument
        clamp = (
            len(body) == 4
            and body[0].name == "arith.constant"
            and body[1].name in {"arith.cmpi", "arith.cmpf"}
            and body[2].name == "arith.select"
            and body[3].name == "linalg.yield"
            and set(body[1].operands) == {argument, body[0].results[0]}
            and body[2].operands[0] is body[1].results[0]
            and set(body[2].operands[1:]) == {argument, body[0].results[0]}
            and list(body[3].operands) == list(body[2].results)
        )
        if not (negate or clamp):
            continue
        source = op.inputs[0]
        bits = widths.get(str(source.type.element_type))
        floating = bits is not None
        if isinstance(source.type.element_type, IntegerType) and source.type.element_type.width.data in {8, 16, 32, 64}:
            bits = source.type.element_type.width.data
        if bits is None or source.type.element_type != op.results[0].type.element_type:
            continue
        element = IntegerType(bits)
        shape = source.type.get_shape() if clamp else op.results[0].type.get_shape()
        identity = AffineMapAttr(AffineMap.identity(len(source.type.get_shape())))
        iterators = ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(source.type.get_shape()))
        empty = tensor.EmptyOp([], TensorType(element, source.type.get_shape()))
        block = Block(arg_types=[source.type.element_type, element])
        if floating:
            cast = arith.BitcastOp(block.args[0], element)
            block.add_ops([cast, L.YieldOp(cast.result)])
        else:
            block.add_op(L.YieldOp(block.args[0]))
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
        offset = arith.ConstantOp(IntegerAttr(-(1 << (bits - 1)) if negate else 0, element))
        addition = arith.AddiOp(block.args[0], offset.result)
        block.add_ops([offset, addition, L.YieldOp(addition.result)])
        changed = L.GenericOp(
            inputs=[packed.results[0]],
            outputs=[destination.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([identity, identity]) if clamp else op.get_indexing_maps(),
            iterator_types=iterators if clamp else op.get_iterator_types(),
            result_types=[destination.tensor.type],
        )
        bounds[changed.results[0]] = 1 << (bits - 1)
        out = tensor.EmptyOp([], TensorType(source.type.element_type, shape))
        block = Block(arg_types=[element, source.type.element_type])
        if floating:
            raw = block.args[0]
            if negate and canonical_bf16_nan and str(source.type.element_type) == "bf16":
                mask = arith.ConstantOp(IntegerAttr(0x7FFF, element))
                infinity = arith.ConstantOp(IntegerAttr(0x7F80, element))
                canonical = arith.ConstantOp(IntegerAttr(0x7FC0, element))
                magnitude = arith.AndIOp(raw, mask.result)
                nan = arith.CmpiOp(magnitude.result, infinity.result, "ugt")
                selected = arith.SelectOp(nan.result, canonical.result, raw)
                block.add_ops([mask, infinity, canonical, magnitude, nan, selected])
                raw = selected.result
            cast = arith.BitcastOp(raw, source.type.element_type)
            block.add_ops([cast, L.YieldOp(cast.result)])
        else:
            block.add_op(L.YieldOp(block.args[0]))
        out_map = AffineMapAttr(AffineMap.identity(len(shape)))
        result = L.GenericOp(
            inputs=[changed.results[0]],
            outputs=[out.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([out_map, out_map]),
            iterator_types=iterators if clamp else op.get_iterator_types(),
            result_types=[out.tensor.type],
        )
        operations = [empty, packed, destination, changed, out, result]
        if clamp:
            retained = op.clone(value_mapper={source: result.results[0]})
            operations.append(retained)
            result = retained
        Rewriter.replace_op(op, operations, [result.results[0]])
        proofs.append(
            dict(
                source="arith.negf" if negate else "closed_constant_compare_select",
                representation="ieee_storage_bits" if floating else "native_signed_integer",
                width=bits,
                arithmetic="sign_bit_offset_modulo_storage_width" if negate else "source_compare_select_retained_host",
                nan_payload=(
                    "canonical_positive_quiet"
                    if negate and canonical_bf16_nan and str(source.type.element_type) == "bf16"
                    else "preserved"
                    if negate
                    else "source_predicate_semantics"
                ),
            )
        )
    module.verify()
    return bounds, proofs
