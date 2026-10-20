"""Erase neutral IEEE operations only behind a complete NaN-word guard.

The primitive and its input agree bitwise outside NaNs and have the same NaN
class. Every use must inspect only that class or select a separate NaN word.
An escaping raw result, payload bitcast or relaxed operation refuses folding.
"""

import math


def fold_nan_guarded_neutral(module):
    from xdsl.ir import BlockArgument
    from xdsl.rewriter import Rewriter

    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    def literal(value):
        if isinstance(value, BlockArgument):
            parent = value.owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                value = parent.inputs[value.index]
        return _constant_of(value)

    def classification(op, value):
        return (
            getattr(op, "name", None) == "arith.cmpf"
            and op.operands[0] is value
            and op.operands[1] is value
            and op.predicate.value.data in {7, 14}
            and not op.fastmath.data
        )

    changed = 0
    for op in list(module.walk()):
        if op.name not in {"arith.maximumf", "arith.minimumf", "arith.mulf", "arith.divf"}:
            continue
        if op.fastmath.data:
            continue
        replacement = None
        for slot in (0,) if op.name == "arith.divf" else (0, 1):
            value = literal(op.operands[1 - slot])
            if (
                op.name in {"arith.mulf", "arith.divf"}
                and value == 1
                or op.name == "arith.maximumf"
                and value == -math.inf
                or op.name == "arith.minimumf"
                and value == math.inf
            ):
                replacement = op.operands[slot]
                break
        if replacement is None:
            continue
        result = op.results[0]
        for use in result.uses:
            consumer = use.operation
            if classification(consumer, result):
                continue
            if consumer.name != "arith.select" or use.index == 0:
                break
            condition = consumer.operands[0].owner
            if not classification(condition, result):
                break
            nonnan_slot = 2 if condition.predicate.value.data == 14 else 1
            if use.index != nonnan_slot or consumer.operands[3 - nonnan_slot] is result:
                break
        else:
            Rewriter.replace_op(op, [], [replacement])
            changed += 1
    return changed
