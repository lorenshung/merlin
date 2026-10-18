"""Separable native IEEE zero-divisor readout over a complete device numerator.

This policy selects the recorded x86 CPU invalid-result convention: negative
quiet default NaN for binary16/32/64, positive canonical quiet NaN for BF16.
It uses the computed numerator's bits; no eager outputs or original floating
arithmetic are replayed. Finite-grid admission remains a separate obligation.
"""


def rewrite_zero_division(module):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, FloatAttr, IntegerAttr, IntegerType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.fixed_point_ir import FORMAT
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    count = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 2 or len(op.results) != 1:
            continue
        body = list(op.body.block.ops)
        if (
            len(body) != 2
            or body[0].name != "arith.divf"
            or any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types())
        ):
            continue
        if list(body[0].operands) != list(op.body.block.args[:2]) or _constant_of(op.inputs[1]) != 0:
            continue
        typ = op.results[0].type
        token = str(typ.element_type)
        if token not in FORMAT:
            continue
        one = arith.ConstantOp(FloatAttr(1.0, typ.element_type))
        ones = tensor.SplatOp(one.result, [], op.inputs[1].type)
        empty = tensor.EmptyOp([], typ)
        block = Block(arg_types=[typ.element_type] * 3)
        product = arith.MulfOp(block.args[0], block.args[1])
        block.add_ops([product, L.YieldOp(product.result)])
        numerator = L.GenericOp(
            inputs=[op.inputs[0], ones.result],
            outputs=[empty.tensor],
            body=Region(block),
            indexing_maps=op.get_indexing_maps(),
            iterator_types=op.get_iterator_types(),
            result_types=[typ],
        )
        destination = tensor.EmptyOp([], typ)
        block = Block(arg_types=[typ.element_type] * 3)

        def emit(child):
            block.add_op(child)
            return child.results[0]

        fraction, exponent, _ = FORMAT[token]
        storage = IntegerType(1 + fraction + exponent)

        def constant(value):
            return emit(arith.ConstantOp(IntegerAttr(value, storage)))

        raw = emit(arith.BitcastOp(block.args[0], storage))
        divisor_raw = emit(arith.BitcastOp(block.args[1], storage))
        sign_mask = constant(1 << (fraction + exponent))
        sign = emit(arith.AndIOp(emit(arith.XOrIOp(raw, divisor_raw)), sign_mask))
        magnitude = emit(arith.AndIOp(raw, constant((1 << (fraction + exponent)) - 1)))
        is_zero = emit(arith.CmpiOp(magnitude, constant(0), "eq"))
        infinity = constant(((1 << exponent) - 1) << fraction)
        signed_infinity = emit(arith.OrIOp(infinity, sign))
        invalid_nan = ((1 << exponent) - 1) << fraction | 1 << (fraction - 1)
        if token != "bf16":
            invalid_nan |= 1 << (fraction + exponent)
        result = emit(arith.SelectOp(is_zero, constant(invalid_nan), signed_infinity))
        result = emit(arith.BitcastOp(result, typ.element_type))
        block.add_op(L.YieldOp(result))
        identity = AffineMapAttr(AffineMap.identity(len(typ.get_shape())))
        overlay = L.GenericOp(
            inputs=[numerator.results[0], op.inputs[1]],
            outputs=[destination.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([identity, list(op.get_indexing_maps())[1], identity]),
            iterator_types=op.get_iterator_types(),
            result_types=[typ],
        )
        Rewriter.replace_op(op, [one, ones, empty, numerator, destination, overlay], [overlay.results[0]])
        count += 1
    module.verify()
    return count


def retain_nan_narrowing(module):
    """Make native CPU NaN narrowing bits explicit; finite conversions stay intact."""
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, IntegerType

    from merlin.llvmlower.fixed_point_ir import FORMAT

    for op in list(module.walk()):
        if op.name != "arith.truncf" or str(op.operands[0].type) not in FORMAT or str(op.result.type) not in FORMAT:
            continue
        source, target = op.operands[0], op.result.type
        sf, se, _ = FORMAT[str(source.type)]
        tf, te, _ = FORMAT[str(target)]
        if sf < tf:
            continue
        block = op.parent_block()
        operations = []

        def emit(child):
            operations.append(child)
            return child.results[0]

        wide, narrow = IntegerType(1 + sf + se), IntegerType(1 + tf + te)

        def constant(value, typ=wide):
            return emit(arith.ConstantOp(IntegerAttr(value, typ)))

        raw = emit(arith.BitcastOp(source, wide))
        magnitude = emit(arith.AndIOp(raw, constant((1 << (sf + se)) - 1)))
        nan = emit(arith.CmpiOp(magnitude, constant(((1 << se) - 1) << sf), "ugt"))
        if str(target) == "bf16":
            bits = constant(0x7FC0, narrow)
        else:
            # Keep sign and high payload bits, quiet signaling NaNs.
            sign = emit(arith.ShRUIOp(emit(arith.AndIOp(raw, constant(1 << (sf + se)))), constant(sf + se - tf - te)))
            payload = emit(arith.ShRUIOp(emit(arith.AndIOp(raw, constant((1 << sf) - 1))), constant(sf - tf)))
            bits = emit(
                arith.OrIOp(sign, emit(arith.OrIOp(payload, constant((((1 << te) - 1) << tf) | 1 << (tf - 1)))))
            )
            bits = emit(arith.TruncIOp(bits, narrow))
        decoded = emit(arith.BitcastOp(bits, target))
        selected = emit(arith.SelectOp(nan, decoded, op.result))
        # Replace only old users; the select deliberately observes the finite
        # source conversion and must not acquire a self reference.
        users = list(op.result.uses)
        for child in operations:
            block.insert_op_after(child, op if child is operations[0] else operations[operations.index(child) - 1])
        for use in users:
            if use.operation is not selected.owner:
                use.operation.operands[use.index] = selected
    module.verify()
