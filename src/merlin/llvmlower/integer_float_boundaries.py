"""Keep native signed-integer conversion before true-division host epilogues.

The byte producers compute the complete integer numerator. One explicit integer
RNE conversion reproduces the source's sitofp, including any rounding before
subsequent floating operations. No division is moved across that boundary.
"""


def has_literal_zero_divisor(module):
    """Portable divf does not specify the CPU's invalid-result NaN bits.

    Follow only uniform literals and generic input arguments. Loop-carried
    values, arbitrary expressions and tensor data are deliberately not inferred.
    """
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr
    from xdsl.ir import BlockArgument

    def zero(value, seen):
        if value in seen:
            return False
        seen = seen | {value}
        owner = value.owner
        if isinstance(value, BlockArgument):
            parent = owner.parent_op()
            if parent is not None and parent.name == "linalg.generic" and value.index < len(parent.inputs):
                return zero(parent.inputs[value.index], seen)
            return False
        if owner.name == "arith.constant":
            literal = owner.properties.get("value", owner.attributes.get("value"))
            return isinstance(literal, (FloatAttr, IntegerAttr)) and literal.value.data == 0
        if owner.name in {"tensor.splat", "linalg.fill", "arith.extf", "arith.truncf", "arith.sitofp"}:
            return zero(owner.operands[0], seen)
        return False

    return any(op.name == "arith.divf" and zero(op.operands[1], set()) for op in module.walk())


def rewrite_integer_float_boundaries(module, input_bounds, *, output_grids=None):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import ArrayAttr, IntegerAttr, TensorType, UnitAttr, i128
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.fixed_point_ir import FORMAT, encode_fixed_integer
    from merlin.llvmlower.pointwise_primitives import expose_pointwise_primitives

    expose_pointwise_primitives(module)
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    bounds = dict(zip(function.body.block.args, input_bounds))
    added = {}
    count = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 1 or len(op.results) != 1:
            continue
        if any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        body = list(op.body.block.ops)
        if (
            len(body) != 2
            or body[0].name not in {"arith.sitofp", "arith.uitofp"}
            or body[0].operands[0] is not op.body.block.args[0]
        ):
            continue
        source = op.inputs[0]
        if str(source.type.element_type) not in {"i1", "i8", "i16", "i32", "i64"}:
            continue
        if str(op.results[0].type.element_type) not in FORMAT:
            continue
        magnitude = bounds.get(source)
        if magnitude is None:
            from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

            literal = _constant_of(source)
            if type(literal) is int:
                magnitude = abs(literal)
        if magnitude is None:
            magnitude = 1 << (source.type.element_type.width.data - 1)
        shape = op.results[0].type.get_shape()
        integer_type = TensorType(i128, shape)
        constant = arith.ConstantOp(IntegerAttr(0, i128))
        seed = tensor.SplatOp(constant.result, [], integer_type)
        block = Block(arg_types=[source.type.element_type, i128])
        unsigned = body[0].name == "arith.uitofp"
        if unsigned and str(source.type.element_type) != "i1":
            continue
        extend = (arith.ExtUIOp if unsigned else arith.ExtSIOp)(block.args[0], i128)
        block.add_ops([extend, L.YieldOp(extend.result)])
        integer = L.GenericOp(
            inputs=[source],
            outputs=[seed.result],
            body=Region(block),
            indexing_maps=op.get_indexing_maps(),
            iterator_types=op.get_iterator_types(),
            result_types=[integer_type],
        )
        added[integer.results[0]] = magnitude
        empty = tensor.EmptyOp([], op.results[0].type)
        block = Block(arg_types=[i128, op.results[0].type.element_type])

        def emit(child):
            block.add_op(child)
            return child.results[0]

        result = encode_fixed_integer(block.args[0], 0, op.results[0].type.element_type, emit)
        block.add_op(L.YieldOp(result))
        identity = list(op.get_indexing_maps())[-1]
        converted = L.GenericOp(
            inputs=[integer.results[0]],
            outputs=[empty.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([identity, identity]),
            iterator_types=op.get_iterator_types(),
            result_types=[op.results[0].type],
        )
        converted.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
        if output_grids is not None:
            # Integer-to-float rounding preserves integrality. The rounded
            # magnitude is bounded by twice the original integer magnitude.
            output_grids[converted.results[0]] = (0, 2 * magnitude)
        Rewriter.replace_op(op, [constant, seed, integer, empty, converted], [converted.results[0]])
        count += 1
    module.verify()
    return added, count
