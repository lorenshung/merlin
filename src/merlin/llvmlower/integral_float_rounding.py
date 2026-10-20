"""Integral IEEE images are fixed points of floor/ceil/trunc/RNE."""


def fold_integral_float_rounding(module):
    from fractions import Fraction

    from xdsl.dialects.builtin import IntegerType
    from xdsl.ir import BlockArgument
    from xdsl.rewriter import Rewriter

    from merlin.targetgen.native_payload_carriers import UNKNOWN, LiteralPayloads
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    payloads = LiteralPayloads()
    memo = {}

    def integral(value):
        if value in memo:
            return memo[value]
        memo[value] = False
        if payloads.classify(value) & UNKNOWN:
            return False
        owner = value.owner
        name = getattr(owner, "name", None)
        result = False
        if isinstance(value, BlockArgument):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                result = integral(parent.inputs[value.index])
        elif name in {"arith.sitofp", "arith.uitofp"}:
            typ = owner.operands[0].type
            result = isinstance(typ, IntegerType) and typ.width.data <= 64
        elif name in {"arith.extf", "arith.truncf"}:
            result = integral(owner.operands[0])
        elif name == "arith.divf":
            divisor = _constant_of(owner.operands[1])
            if isinstance(owner.operands[1], BlockArgument):
                argument = owner.operands[1]
                parent = argument.owner.parent_op()
                if getattr(parent, "name", None) == "linalg.generic" and argument.index < len(parent.inputs):
                    divisor = _constant_of(parent.inputs[argument.index])
            if isinstance(divisor, (int, float)) and divisor:
                from math import isfinite

                if isfinite(divisor):
                    reciprocal = 1 / abs(Fraction(divisor))
                    # Power-of-two reciprocal has no fractional image or
                    # intermediate rounding other than IEEE overflow to inf.
                    n = reciprocal.numerator
                    result = reciprocal.denominator == 1 and n & (n - 1) == 0 and integral(owner.operands[0])
        elif name == "linalg.generic" and len(owner.results) == 1:
            if all(it.data.value == "parallel" for it in owner.get_iterator_types()):
                result = integral(owner.body.block.last_op.operands[0])
        memo[value] = result
        return result

    def finite_integer_cast(value):
        """A <=64-bit integer conversion to binary32/64 cannot overflow."""
        owner = value.owner
        name = getattr(owner, "name", None)
        if name in {"arith.sitofp", "arith.uitofp"} and str(value.type) in {"f32", "f64", "bf16"}:
            typ = owner.operands[0].type
            return isinstance(typ, IntegerType) and typ.width.data <= 64
        if isinstance(value, BlockArgument):
            parent = owner.parent_op()
            return bool(
                getattr(parent, "name", None) == "linalg.generic"
                and value.index < len(parent.inputs)
                and finite_integer_cast(parent.inputs[value.index])
            )
        if name == "linalg.generic" and len(owner.results) == 1:
            return all(it.data.value == "parallel" for it in owner.get_iterator_types()) and finite_integer_cast(
                owner.body.block.last_op.operands[0]
            )
        return False

    def literal(value):
        if isinstance(value, BlockArgument):
            parent = value.owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                value = parent.inputs[value.index]
        return _constant_of(value)

    count = 0
    # Only finite integer images license these identities. Infinities from an
    # overflowing narrow cast would make remainder NaN and must stay intact.
    for op in list(module.walk()):
        name = getattr(getattr(op, "op_name", None), "data", op.name)
        if name not in {"arith.divf", "arith.remf"} or literal(op.operands[1]) != 1:
            continue
        flags = op.properties.get("fastmath")
        if flags is not None and flags.data:
            continue
        data = op.operands[0]
        if not finite_integer_cast(data) or payloads.classify(data) & UNKNOWN:
            continue
        if name == "arith.divf":
            Rewriter.replace_op(op, [], [data])
            count += 1
            continue
        result = op.results[0]
        safe = True
        subtractions = []
        for use in list(result.uses):
            child = use.operation
            if child.name == "arith.subf" and use.index == 1 and child.operands[0] is data and not child.fastmath.data:
                subtractions.append(child)
            elif child.name == "arith.cmpf" and literal(child.operands[1 - use.index]) == 0 and not child.fastmath.data:
                continue
            else:
                safe = False
                break
        if not safe:
            continue
        for child in subtractions:
            Rewriter.replace_op(child, [], [data])
        # Remaining comparisons cannot observe the sign of zero.
        from xdsl.dialects import arith
        from xdsl.dialects.builtin import FloatAttr

        zero = arith.ConstantOp(FloatAttr(0.0, result.type))
        Rewriter.replace_op(op, [zero], [zero.result])
        count += 1
    for op in list(module.walk()):
        name = getattr(getattr(op, "op_name", None), "data", op.name)
        if name not in {"math.floor", "math.ceil", "math.trunc", "math.round", "math.roundeven"}:
            continue
        flags = op.properties.get("fastmath")
        if flags is not None and flags.data:
            continue
        if (
            len(op.operands) == len(op.results) == 1
            and op.operands[0].type == op.results[0].type
            and str(op.results[0].type) in {"f16", "bf16", "f32", "f64"}
            and integral(op.operands[0])
        ):
            Rewriter.replace_op(op, [], [op.operands[0]])
            count += 1
    for op in list(module.walk()):
        if (
            op.name == "arith.subf"
            and not op.fastmath.data
            and op.operands[0] is op.operands[1]
            and finite_integer_cast(op.operands[0])
        ):
            from xdsl.dialects import arith
            from xdsl.dialects.builtin import FloatAttr

            zero = arith.ConstantOp(FloatAttr(0.0, op.results[0].type))
            Rewriter.replace_op(op, [zero], [zero.result])
            count += 1
    return count


def fold_integral_image_identities(module):
    """Close nested guards after each preceding identity exposes their image."""
    from merlin.llvmlower.boolean_unit_image import fold_boolean_unit_image
    from merlin.llvmlower.integer_image_predicates import fold_integer_image_predicates

    total = 0
    while True:
        changed = (
            fold_integral_float_rounding(module)
            + fold_boolean_unit_image(module)
            + fold_integer_image_predicates(module)
        )
        total += changed
        if not changed:
            return total
