"""Fold closed one-Boolean integer programs only to identity or uniform fill.

Both possible storage words are evaluated exactly, with every original integer
width boundary. An arbitrary two-entry result table is never emitted.
"""


def fold_boolean_domain(module):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntegerAttr, IntegerType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.targetgen.native_payload_carriers import UNKNOWN, LiteralPayloads
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    payloads = LiteralPayloads()
    count = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1:
            continue
        if any(it.data.value != "parallel" for it in op.get_iterator_types()):
            continue
        element = op.results[0].type.element_type
        if not isinstance(element, IntegerType):
            continue
        runtime = [i for i, value in enumerate(op.inputs) if _constant_of(value) is None]
        if len(runtime) != 1 or str(op.inputs[runtime[0]].type.element_type) != "i1":
            continue
        maps = list(op.get_indexing_maps())
        rank = len(op.results[0].type.get_shape())
        if maps[-1] != AffineMapAttr(AffineMap.identity(rank)):
            continue
        if payloads.classify(op.inputs[runtime[0]]) & UNKNOWN:
            continue
        body = op.body.block
        operations = list(body.ops)
        if len(op.inputs) == 1 and (
            (
                len(operations) == 1
                and operations[0].name == "linalg.yield"
                and operations[0].operands[0] is body.args[0]
            )
            or (
                len(operations) == 2
                and operations[0].name == "arith.extui"
                and operations[0].operands[0] is body.args[0]
                and list(operations[1].operands) == list(operations[0].results)
            )
        ):
            continue
        answers = []
        evaluations = []
        valid = True
        for bit in (0, 1):
            env = {body.args[runtime[0]]: bit}
            for i, value in enumerate(op.inputs):
                if i != runtime[0]:
                    literal = _constant_of(value)
                    if not isinstance(literal, int) or not isinstance(value.type.element_type, IntegerType):
                        valid = False
                        break
                    env[body.args[i]] = literal % (1 << value.type.element_type.width.data)
            if not valid:
                break
            for child in body.ops:
                if child.name == "linalg.yield":
                    if child.operands[0] not in env:
                        valid = False
                        break
                    answers.append(env[child.operands[0]])
                    break
                if child.name == "cf.assert":
                    if len(child.operands) != 1 or child.operands[0] not in env or not env[child.operands[0]]:
                        valid = False
                        break
                    continue
                if len(child.results) != 1 or not isinstance(child.results[0].type, IntegerType) or child.regions:
                    valid = False
                    break
                flags = child.properties.get("overflowFlags")
                if flags is not None and str(flags).split("<")[-1] != "none>":
                    valid = False
                    break
                width = child.results[0].type.width.data
                if child.name == "arith.constant" and isinstance(child.value, IntegerAttr):
                    value = child.value.value.data
                elif any(v not in env for v in child.operands):
                    valid = False
                    break
                else:
                    xs = [env[v] for v in child.operands]
                    if child.name == "arith.select":
                        value = xs[1 if xs[0] else 2]
                    elif child.name == "arith.cmpi":
                        a, b = xs
                        predicate = child.predicate.value.data
                        if predicate in {2, 3, 4, 5}:
                            bits = child.operands[0].type.width.data
                            a = a - (1 << bits) if a >= 1 << (bits - 1) else a
                            b = b - (1 << bits) if b >= 1 << (bits - 1) else b
                        value = (a == b, a != b, a < b, a <= b, a > b, a >= b, a < b, a <= b, a > b, a >= b)[predicate]
                    elif child.name in {"arith.divsi", "arith.floordivsi", "arith.remsi", "arith.divui", "arith.remui"}:
                        a, b = xs
                        bits = child.operands[0].type.width.data
                        signed = child.name not in {"arith.divui", "arith.remui"}
                        if signed:
                            a = a - (1 << bits) if a >= 1 << (bits - 1) else a
                            b = b - (1 << bits) if b >= 1 << (bits - 1) else b
                        if not b or (signed and a == -(1 << (bits - 1)) and b == -1):
                            valid = False
                            break
                        quotient = abs(a) // abs(b)
                        if (a < 0) != (b < 0):
                            quotient = -quotient
                        value = (
                            a // b
                            if child.name == "arith.floordivsi"
                            else a - quotient * b
                            if child.name in {"arith.remsi", "arith.remui"}
                            else quotient
                        )
                    elif child.name in {"arith.extui", "arith.trunci"}:
                        value = xs[0]
                    elif child.name == "arith.extsi":
                        bits = child.operands[0].type.width.data
                        value = xs[0] - (1 << bits) if xs[0] >= 1 << (bits - 1) else xs[0]
                    elif child.name in {
                        "arith.andi",
                        "arith.ori",
                        "arith.xori",
                        "arith.addi",
                        "arith.subi",
                        "arith.muli",
                    }:
                        a, b = xs
                        value = {
                            "arith.andi": lambda: a & b,
                            "arith.ori": lambda: a | b,
                            "arith.xori": lambda: a ^ b,
                            "arith.addi": lambda: a + b,
                            "arith.subi": lambda: a - b,
                            "arith.muli": lambda: a * b,
                        }[child.name]()
                    else:
                        valid = False
                        break
                env[child.results[0]] = value % (1 << width)
            if not valid:
                break
            evaluations.append(env)
        if valid and len(evaluations) == 2:
            # A two-word evaluation must not erase a runtime product/OR or
            # runtime quotient construction merely because its outputs coincide.
            def static(value):
                return (
                    value in evaluations[0]
                    and value in evaluations[1]
                    and evaluations[0][value] == evaluations[1][value]
                )

            for child in body.ops:
                if child.name in {"arith.andi", "arith.ori", "arith.xori", "arith.addi", "arith.subi", "arith.muli"}:
                    if not any(static(v) for v in child.operands):
                        valid = False
                        break
                elif child.name in {"arith.divsi", "arith.floordivsi", "arith.remsi", "arith.divui", "arith.remui"}:
                    if not static(child.operands[1]):
                        valid = False
                        break
        if not valid or len(answers) != 2 or not (answers[0] == answers[1] or answers == [0, 1]):
            continue
        if answers[0] == answers[1]:
            constant = arith.ConstantOp(IntegerAttr(answers[0], element))
            filled = tensor.SplatOp(constant.result, [], op.results[0].type)
            Rewriter.replace_op(op, [constant, filled], [filled.result])
        else:
            # Keep the original selector map; only the proved scalar body changes.
            empty = tensor.EmptyOp([], op.results[0].type)
            source = op.inputs[runtime[0]]
            block = Block(arg_types=[source.type.element_type, element])
            if element.width.data == 1:
                value = block.args[0]
            else:
                extended = arith.ExtUIOp(block.args[0], element)
                block.add_op(extended)
                value = extended.result
            block.add_op(L.YieldOp(value))
            maps = list(op.get_indexing_maps())
            selected = L.GenericOp(
                inputs=[source],
                outputs=[empty.tensor],
                body=Region(block),
                indexing_maps=ArrayAttr([maps[runtime[0]], maps[-1]]),
                iterator_types=op.get_iterator_types(),
                result_types=[op.results[0].type],
            )
            Rewriter.replace_op(op, [empty, selected], [selected.results[0]])
        count += 1
    return count
