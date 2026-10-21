"""Prove storage selectors over the complete image of one Boolean conversion."""


def fold_boolean_image_predicates(module):
    from xdsl.dialects.builtin import AffineMapAttr, i1
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower._ir import point, splat
    from merlin.llvmlower.singleton_selectors import _relation
    from merlin.targetgen.native_payload_carriers import UNKNOWN, LiteralPayloads
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    payloads = LiteralPayloads()
    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 2 or len(op.results) != 1:
            continue
        if str(op.results[0].type.element_type) != "i1":
            continue
        body = list(op.body.block.ops)
        if len(body) != 2 or body[0].name != "arith.cmpf" or body[1].name != "linalg.yield" or body[0].fastmath.data:
            continue
        if list(body[0].operands) != list(op.body.block.args[:2]) or list(body[1].operands) != list(body[0].results):
            continue
        if any(it.data.value != "parallel" for it in op.get_iterator_types()):
            continue
        for slot in (0, 1):
            conversion = op.inputs[slot].owner
            constant = _constant_of(op.inputs[1 - slot])
            if not isinstance(constant, (int, float)) or getattr(conversion, "name", None) != "linalg.generic":
                continue
            if len(conversion.inputs) != 1 or len(conversion.results) != 1:
                continue
            source = conversion.inputs[0]
            if str(source.type.element_type) != "i1" or payloads.classify(source) & UNKNOWN:
                continue
            scalar = list(conversion.body.block.ops)
            if (
                len(scalar) != 2
                or scalar[0].name not in {"arith.uitofp", "arith.sitofp"}
                or scalar[1].name != "linalg.yield"
            ):
                continue
            if list(scalar[0].operands) != [conversion.body.block.args[0]]:
                continue
            if list(scalar[1].operands) != list(scalar[0].results):
                continue
            rank = len(conversion.results[0].type.get_shape())
            maps = list(conversion.get_indexing_maps())
            if any(it.data.value != "parallel" for it in conversion.get_iterator_types()):
                continue
            if maps[-1].data != AffineMap.identity(rank):
                continue
            import math

            answers = []
            for bit in (0, 1):
                value = -bit if scalar[0].name == "arith.sitofp" else bit
                a, b = (value, constant) if slot == 0 else (constant, value)
                answers.append(
                    _relation(body[0].predicate.value.data, (a > b) - (a < b), math.isnan(constant), floating=True)
                )
            replacement = []
            if answers[0] == answers[1]:
                result = splat(int(answers[0]), i1, op.results[0].type.get_shape(), replacement)
            elif answers == [False, True]:
                indexing = maps[0].data.compose(list(op.get_indexing_maps())[slot].data)
                result = point(
                    [source],
                    op.results[0].type,
                    lambda args, emit: args[0],
                    replacement,
                    maps=[AffineMapAttr(indexing), list(op.get_indexing_maps())[-1]],
                    iterators=op.get_iterator_types(),
                )
            else:
                continue
            Rewriter.replace_op(op, replacement, [result])
            changed += 1
            break
    if changed:
        module.verify()
    return changed
