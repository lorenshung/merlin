"""Erase arithmetic identities proved over the complete unsigned Boolean image."""

import math
from fractions import Fraction


def fold_boolean_unit_image(module):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, IntegerAttr, i1
    from xdsl.ir import BlockArgument
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.singleton_selectors import _relation
    from merlin.targetgen.native_payload_carriers import UNKNOWN, LiteralPayloads
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    payloads = LiteralPayloads()
    memo = {}

    def literal(value):
        if isinstance(value, BlockArgument):
            parent = value.owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                value = parent.inputs[value.index]
        owner = value.owner
        if getattr(owner, "name", None) == "arith.bitcast":
            bits = _constant_of(owner.operands[0])
            width = getattr(getattr(owner.operands[0].type, "width", None), "data", None)
            if isinstance(bits, int) and width in {16, 32, 64} and str(value.type) in {"f16", "bf16", "f32", "f64"}:
                raw = (bits % (1 << width)).to_bytes(width // 8, "little")
                if len(value.type.pack([0.0])) == len(raw):
                    return value.type.unpack(raw, 1)[0]
        return _constant_of(value)

    def unit(value):
        if value in memo:
            return memo[value]
        memo[value] = False
        owner = value.owner
        name = getattr(owner, "name", None)
        answer = False
        if name == "arith.uitofp" and str(owner.operands[0].type) == "i1":
            answer = True
        elif name in {"arith.extf", "arith.truncf"}:
            answer = unit(owner.operands[0])
        elif isinstance(value, BlockArgument):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                answer = unit(parent.inputs[value.index])
        elif name == "linalg.generic":
            operations = list(owner.body.block.ops)
            if (
                len(owner.results) == 1
                and operations[-1].name == "linalg.yield"
                and all(it.data.value == "parallel" for it in owner.get_iterator_types())
            ):
                answer = unit(operations[-1].operands[0])
        if answer and payloads.classify(value) & UNKNOWN:
            answer = False
        memo[value] = answer
        return answer

    changed = 0
    for op in list(module.walk()):
        if any(payloads.classify(value) & UNKNOWN for value in op.operands):
            continue
        if op.name == "cf.assert" and literal(op.operands[0]):
            Rewriter.erase_op(op)
            changed += 1
            continue
        if len(op.results) != 1:
            continue
        flags = op.properties.get("fastmath")
        if flags is not None and flags.data:
            continue
        name = getattr(getattr(op, "op_name", None), "data", op.name)
        replacement = None
        zero = False
        integer_zero = False
        predicate_literal = None
        if name == "arith.cmpf":
            a, b = [literal(v) for v in op.operands]
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                predicate_literal = _relation(
                    op.predicate.value.data, (a > b) - (a < b), math.isnan(a) or math.isnan(b), floating=True
                )
            else:
                for slot in (0, 1):
                    threshold = literal(op.operands[1 - slot])
                    if not unit(op.operands[slot]) or not isinstance(threshold, (int, float)):
                        continue
                    answers = []
                    for bit in (0.0, 1.0):
                        left, right = (bit, threshold) if slot == 0 else (threshold, bit)
                        answers.append(
                            _relation(
                                op.predicate.value.data,
                                (left > right) - (left < right),
                                math.isnan(threshold),
                                floating=True,
                            )
                        )
                    if answers[0] == answers[1]:
                        predicate_literal = answers[0]
                        break
        elif name in {"arith.andi", "arith.ori", "arith.xori"} and str(op.results[0].type) == "i1":
            a, b = [literal(v) for v in op.operands]
            if a is not None and b is not None:
                predicate_literal = {
                    "arith.andi": bool(a) and bool(b),
                    "arith.ori": bool(a) or bool(b),
                    "arith.xori": bool(a) != bool(b),
                }[name]
            else:
                for slot, bound in enumerate((a, b)):
                    if bound is None:
                        continue
                    if name == "arith.andi" and not bound:
                        predicate_literal = False
                    elif name == "arith.ori" and bound:
                        predicate_literal = True
                    elif (name == "arith.andi" and bound) or (name in {"arith.ori", "arith.xori"} and not bound):
                        replacement = op.operands[1 - slot]
                    break
        elif name == "arith.select":
            condition = literal(op.operands[0])
            if condition is not None:
                replacement = op.operands[1 if condition else 2]
        if name == "math.copysign" and len(op.operands) == 2:
            magnitude = literal(op.operands[0])
            if isinstance(magnitude, (int, float)) and math.isfinite(magnitude) and unit(op.operands[1]):
                constant = arith.ConstantOp(FloatAttr(abs(magnitude), op.results[0].type))
                Rewriter.replace_op(op, [constant], [constant.result])
                changed += 1
                memo.clear()
                continue
        if name == "arith.subf" and op.operands[0] is op.operands[1] and unit(op.operands[0]):
            zero = True  # Both Boolean floating words are finite and nonnegative.
        if name == "arith.andi":
            for slot in (0, 1):
                cast = op.operands[slot].owner
                mask = literal(op.operands[1 - slot])
                if getattr(cast, "name", None) == "arith.bitcast" and isinstance(mask, int) and unit(cast.operands[0]):
                    typ = cast.operands[0].type
                    one = int.from_bytes(typ.pack([1.0]), "little")
                    if not mask & one:
                        integer_zero = True
        if name in {"math.sqrt", "math.floor", "math.ceil", "math.trunc", "math.round", "math.roundeven"}:
            if len(op.operands) == 1 and unit(op.operands[0]):
                replacement = op.operands[0]
        elif name in {"arith.divf", "arith.remf", "arith.subf", "arith.addf", "math.powf"} and len(op.operands) == 2:
            data, constant = op.operands
            bound = literal(constant)
            if isinstance(bound, (int, float)) and math.isfinite(bound) and unit(data):
                if name == "arith.divf" and bound == 1:
                    replacement = data
                elif name == "math.powf" and bound > 0:
                    replacement = data
                elif name in {"arith.addf", "arith.subf"} and bound == 0 and math.copysign(1.0, bound) > 0:
                    replacement = data
                elif name == "arith.remf" and abs(bound) > 1:
                    replacement = data
                elif name == "arith.remf" and bound and (1 / abs(Fraction(bound))).denominator == 1:
                    zero = True
        if name == "arith.select" and replacement is None:
            predicate = op.operands[0].owner
            if getattr(predicate, "name", None) == "arith.cmpf" and not predicate.fastmath.data:
                for reverse in (False, True):
                    data = predicate.operands[int(reverse)]
                    threshold = literal(predicate.operands[int(not reverse)])
                    if not unit(data) or not isinstance(threshold, (int, float)):
                        continue
                    answers = []
                    for bit in (0.0, 1.0):
                        a, b = (threshold, bit) if reverse else (bit, threshold)
                        yes = _relation(
                            predicate.predicate.value.data, (a > b) - (a < b), math.isnan(threshold), floating=True
                        )
                        chosen = op.operands[1 if yes else 2]
                        value = bit if chosen is data else literal(chosen)
                        if not isinstance(value, (int, float)):
                            break
                        answers.append(value)
                    if len(answers) == 2 and data.type == op.results[0].type:
                        if all(
                            data.type.pack([value]) == data.type.pack([bit])
                            for value, bit in zip(answers, (0.0, 1.0), strict=True)
                        ):
                            replacement = data
                            break
        if integer_zero:
            constant = arith.ConstantOp(IntegerAttr(0, op.results[0].type))
            Rewriter.replace_op(op, [constant], [constant.result])
            changed += 1
        elif predicate_literal is not None:
            constant = arith.ConstantOp(IntegerAttr(int(predicate_literal), i1))
            Rewriter.replace_op(op, [constant], [constant.result])
            changed += 1
        elif replacement is not None and replacement.type == op.results[0].type:
            Rewriter.replace_op(op, [], [replacement])
            changed += 1
        elif zero:
            constant = arith.ConstantOp(FloatAttr(0.0, op.results[0].type))
            Rewriter.replace_op(op, [constant], [constant.result])
            changed += 1
        if replacement is not None or zero:
            memo.clear()
    if changed:
        from xdsl.transforms.dead_code_elimination import region_dce

        region_dce(module.body)
    return changed
