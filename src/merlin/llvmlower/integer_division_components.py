"""Full-domain literal integer quotients via exact reciprocal products.

For 0<=n<2**w and d>0, M=ceil(2**s/d), s=w+ceil(log2(d)),
gives floor(n*M/2**s) in {floor(n/d), floor(n/d)+1}. One exact
multiply/compare correction recovers the quotient. Zero and CPU trapping
signed overflow domains are not licensed by this value-producing theorem.
"""


def literal_divisor(op):
    from xdsl.ir import BlockArgument

    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    value = op.operands[1]
    if isinstance(value, BlockArgument):
        parent = value.owner.parent_op()
        if parent is None or parent.name != "linalg.generic" or value.index >= len(parent.inputs):
            return None
        value = parent.inputs[value.index]
    literal = _constant_of(value)
    return literal if type(literal) is int else None


def reciprocal(width, divisor):
    if width not in {8, 16, 32, 64} or not divisor:
        raise ValueError("unproved literal division domain")
    d = abs(divisor)
    shift = width + (d - 1).bit_length()
    multiplier = ((1 << shift) + d - 1) // d
    # Exact universal inequalities; these are not sampled numerator checks.
    assert 0 <= multiplier * d - (1 << shift) < d
    assert ((1 << width) - 1) * (multiplier * d - (1 << shift)) < d * (1 << shift)
    return multiplier, shift


def rewrite_integer_divisions(module):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, IntegerType, TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.floating_component_ir import point

    bounds, proofs = {}, []
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1:
            continue
        body = list(op.body.block.ops)
        if len(body) != 2 or body[0].name not in {"arith.divsi", "arith.remsi", "arith.floordivsi"}:
            continue
        scalar = body[0]
        dtype = op.results[0].type.element_type
        if not isinstance(dtype, IntegerType) or dtype.width.data not in {8, 16, 32, 64}:
            continue
        divisor = literal_divisor(scalar)
        if divisor is None or not divisor or (divisor == -1 and dtype.width.data >= 32):
            continue
        if any(it.data.value != "parallel" for it in op.get_iterator_types()):
            continue
        arg = scalar.operands[0]
        if arg not in op.body.block.args[: len(op.inputs)]:
            continue
        index = op.body.block.args.index(arg)
        source, amap = op.inputs[index], list(op.get_indexing_maps())[index]
        wide = IntegerType(256)
        multiplier, shift = reciprocal(dtype.width.data, divisor)
        operations = []

        def magnitude(args, emit):
            value = emit(arith.ExtSIOp(args[0], wide))
            zero = emit(arith.ConstantOp(IntegerAttr(0, wide)))
            return emit(arith.SelectOp(emit(arith.CmpiOp(value, zero, "slt")), emit(arith.SubiOp(zero, value)), value))

        packed = point([source], TensorType(wide, source.type.get_shape()), magnitude, operations)
        bounds[packed] = 1 << (dtype.width.data - 1)
        empty = tensor.EmptyOp([], TensorType(wide, op.results[0].type.get_shape()))
        block = Block(arg_types=[wide, wide])
        constant = arith.ConstantOp(IntegerAttr(multiplier, wide))
        product = arith.MuliOp(block.args[0], constant.result)
        block.add_ops([constant, product, L.YieldOp(product.result)])
        integer = L.GenericOp(
            inputs=[packed],
            outputs=[empty.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([amap, list(op.get_indexing_maps())[-1]]),
            iterator_types=op.get_iterator_types(),
            result_types=[empty.tensor.type],
        )
        operations.extend([empty, integer])
        bounds[integer.results[0]] = (1 << (dtype.width.data - 1)) * multiplier
        output = tensor.EmptyOp([], op.results[0].type)
        block = Block(arg_types=[wide, dtype, dtype])

        def emit(child):
            block.add_op(child)
            return child.results[0]

        def c(value):
            return emit(arith.ConstantOp(IntegerAttr(value, wide)))

        original = emit(arith.ExtSIOp(block.args[1], wide))
        negative = emit(arith.CmpiOp(original, c(0), "slt"))
        n = emit(arith.SelectOp(negative, emit(arith.SubiOp(c(0), original)), original))
        q = emit(arith.ShRUIOp(block.args[0], c(shift)))
        excess = emit(arith.CmpiOp(emit(arith.MuliOp(q, c(abs(divisor)))), n, "ugt"))
        q = emit(arith.SubiOp(q, emit(arith.SelectOp(excess, c(1), c(0)))))
        signed = emit(arith.SelectOp(negative, emit(arith.SubiOp(c(0), q)), q))
        if divisor < 0:
            signed = emit(arith.SubiOp(c(0), signed))
        remainder = emit(arith.SubiOp(original, emit(arith.MuliOp(signed, c(divisor)))))
        result = signed
        if scalar.name == "arith.remsi":
            result = remainder
        elif scalar.name == "arith.floordivsi":
            residual = emit(arith.CmpiOp(remainder, c(0), "ne"))
            sign_differs = emit(arith.CmpiOp(original, c(0), "sge" if divisor < 0 else "slt"))
            adjust = emit(arith.AndIOp(residual, sign_differs))
            result = emit(arith.SubiOp(signed, emit(arith.SelectOp(adjust, c(1), c(0)))))
        result = emit(arith.TruncIOp(result, dtype))
        block.add_op(L.YieldOp(result))
        readout = L.GenericOp(
            inputs=[integer.results[0], source],
            outputs=[output.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([list(op.get_indexing_maps())[-1], amap, list(op.get_indexing_maps())[-1]]),
            iterator_types=op.get_iterator_types(),
            result_types=[output.tensor.type],
        )
        from xdsl.dialects.builtin import UnitAttr

        readout.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
        operations.extend([output, readout])
        Rewriter.replace_op(op, operations, [readout.results[0]])
        proofs.append(
            dict(
                form="full_dtype_literal_quotient",
                width=dtype.width.data,
                divisor=divisor,
                multiplier=str(multiplier),
                shift=shift,
                rounding=scalar.name,
                proof="universal_reciprocal_interval_and_one_correction",
                source_data_arithmetic=True,
            )
        )
    module.verify()
    return bounds, proofs
