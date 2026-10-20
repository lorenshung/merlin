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

    count = 0
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
    return count
