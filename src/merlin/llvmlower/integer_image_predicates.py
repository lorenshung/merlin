"""Fold comparisons proved uniform over an integer-to-float image.

IEEE RNE of an integer is an integral finite value or infinity, never NaN.
Monotonic rounded dtype endpoints and that discrete-image fact are independent
of tensor payloads; arbitrary floating inputs retain their predicates.
"""

import math
from fractions import Fraction


def fold_integer_image_predicates(module):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, IntegerAttr, IntegerType, i1
    from xdsl.ir import BlockArgument
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import input_literal as literal
    from merlin.targetgen.closed_scalar import predicate_holds
    from merlin.targetgen.exact_arithmetic import round_binary
    from merlin.targetgen.native_payload_carriers import UNKNOWN, LiteralPayloads

    memo = {}
    payloads = LiteralPayloads()

    def rounded(value, typ):
        if not math.isfinite(value):
            return value
        token = str(typ)
        if token not in {"f16", "bf16", "f32", "f64"}:
            raise ValueError("unproved floating format")
        try:
            return round_binary(
                Fraction(value),
                16 if token in {"f16", "bf16"} else 64 if token == "f64" else 32,
                bfloat16=token == "bf16",
                overflow="infinity",
            )
        except OverflowError:
            return math.copysign(math.inf, value)

    def domain(value):
        if value in memo:
            return memo[value]
        memo[value] = None
        owner = value.owner
        name = getattr(owner, "name", None)
        result = None
        if name in {"arith.sitofp", "arith.uitofp"}:
            typ = owner.operands[0].type
            if isinstance(typ, IntegerType):
                width = typ.width.data
                lo, hi = (
                    (-(1 << (width - 1)), (1 << (width - 1)) - 1) if name == "arith.sitofp" else (0, (1 << width) - 1)
                )
                result = rounded(lo, value.type), rounded(hi, value.type)
        elif name in {"arith.extf", "arith.truncf"}:
            original = domain(owner.operands[0])
            if original is not None:
                result = tuple(rounded(v, value.type) for v in original)
        elif name is None and isinstance(value, BlockArgument):
            parent = owner.parent_op()
            if getattr(parent, "name", None) == "linalg.generic" and value.index < len(parent.inputs):
                result = domain(parent.inputs[value.index])
        elif name == "linalg.generic":
            ops = list(owner.body.block.ops)
            if (
                len(owner.results) == 1
                and len(ops) == 2
                and ops[-1].name == "linalg.yield"
                and list(ops[-1].operands) == list(ops[0].results)
            ):
                result = domain(ops[0].results[0])
        elif name in {
            "tensor.reshape",
            "tensor.cast",
            "tensor.expand_shape",
            "tensor.collapse_shape",
            "tensor.extract_slice",
            "tensor.extract",
            "linalg.transpose",
            "linalg.copy",
            "linalg.broadcast",
        }:
            result = domain(owner.operands[0])
        if result is not None and payloads.classify(value) & UNKNOWN:
            result = None
        memo[value] = result
        return result

    count = 0
    for op in list(module.walk()):
        if op.name != "arith.cmpf" or op.fastmath.data:
            continue
        for reverse in (False, True):
            data = op.operands[int(reverse)]
            constant = literal(op.operands[int(not reverse)])
            if not isinstance(constant, (int, float)):
                continue
            try:
                bounds = domain(data)
            except (TypeError, ValueError, OverflowError):
                bounds = None
            if bounds is None:
                continue
            pred = op.predicate.value.data
            lo, hi = bounds
            answer = None
            if math.isnan(constant):
                answer = predicate_holds(pred, 0, True, floating=True)
            elif pred in {1, 6, 8, 13}:
                unattainable = (
                    (math.isfinite(constant) and not float(constant).is_integer()) or constant < lo or constant > hi
                )
                if unattainable:
                    answer = pred in {6, 13}
            elif pred in {2, 3, 4, 5, 9, 10, 11, 12}:
                relations = [
                    ((constant > x) - (constant < x)) if reverse else ((x > constant) - (x < constant)) for x in bounds
                ]
                answers = {predicate_holds(pred, relation, floating=True) for relation in relations}
                if len(answers) == 1:
                    answer = answers.pop()
            if answer is not None:
                Rewriter.replace_op(op, arith.ConstantOp(IntegerAttr(int(answer), i1)))
                count += 1
                break
    if count:
        for op in list(module.walk()):
            if op.name != "linalg.generic" or len(op.results) != 1:
                continue
            rank = len(op.results[0].type.get_shape())
            if any(it.data.value != "parallel" for it in op.get_iterator_types()) or list(op.get_indexing_maps())[
                -1
            ] != AffineMapAttr(AffineMap.identity(rank)):
                continue
            body = list(op.body.block.ops)
            if (
                len(body) == 2
                and body[0].name == "arith.constant"
                and body[1].name == "linalg.yield"
                and list(body[1].operands) == list(body[0].results)
            ):
                constant = body[0].clone()
                splat = tensor.SplatOp(constant.results[0], [], op.results[0].type)
                Rewriter.replace_op(op, [constant, splat], [splat.result])
    return count
