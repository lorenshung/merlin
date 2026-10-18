"""Expose scalar IEEE operation boundaries as closed tensor pointwise nodes."""


def expose_pointwise_primitives(module):
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    allowed = {
        "arith.constant",
        "arith.addf",
        "arith.subf",
        "arith.mulf",
        "arith.divf",
        "arith.negf",
        "arith.extf",
        "arith.truncf",
        "arith.sitofp",
        "arith.uitofp",
        "arith.bitcast",
        "arith.cmpf",
        "arith.cmpi",
        "arith.select",
        "arith.andi",
        "arith.ori",
        "arith.xori",
        "arith.shli",
        "arith.shrui",
        "arith.extui",
        "arith.trunci",
        "linalg.yield",
    }
    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1:
            continue
        if any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        body = op.body.block
        if any(child.name not in allowed for child in body.ops):
            continue
        arithmetic = [child for child in body.ops if child.name not in {"arith.constant", "linalg.yield"}]
        if len(arithmetic) <= 1:
            continue
        # A closed word predicate has no floating arithmetic boundary to expose.
        # In particular, literal casts belong to that predicate, not its data path.
        dependent = set(body.args)
        numeric = {
            "arith.addf",
            "arith.subf",
            "arith.mulf",
            "arith.divf",
            "arith.negf",
            "arith.extf",
            "arith.truncf",
            "arith.sitofp",
            "arith.uitofp",
        }
        has_numeric_data = False
        for child in body.ops:
            if any(value in dependent for value in child.operands):
                dependent.update(child.results)
                has_numeric_data |= child.name in numeric
        if not has_numeric_data:
            continue
        shape = op.results[0].type.get_shape()
        identity = AffineMapAttr(AffineMap.identity(len(shape)))
        maps = list(op.get_indexing_maps())
        if maps[-1] != identity:
            continue
        values = {arg: (value, maps[i]) for i, (arg, value) in enumerate(zip(body.args, op.inputs))}
        operations = []
        valid = True
        for child in body.ops:
            if child.name == "linalg.yield":
                continue
            if child.name == "arith.constant":
                literal = child.clone()
                splat = tensor.SplatOp(literal.results[0], [], TensorType(literal.results[0].type, shape))
                operations.extend([literal, splat])
                values[child.results[0]] = splat.result, identity
                continue
            if any(value not in values for value in child.operands):
                valid = False
                break
            inputs = [values[value][0] for value in child.operands]
            typ = TensorType(child.results[0].type, shape)
            empty = tensor.EmptyOp([], typ)
            block = Block(arg_types=[value.type.element_type for value in inputs] + [typ.element_type])
            scalar = child.clone(value_mapper=dict(zip(child.operands, block.args)))
            block.add_ops([scalar, L.YieldOp(scalar.results[0])])
            generic = L.GenericOp(
                inputs=inputs,
                outputs=[empty.tensor],
                body=Region(block),
                indexing_maps=ArrayAttr([values[value][1] for value in child.operands] + [identity]),
                iterator_types=op.get_iterator_types(),
                result_types=[typ],
            )
            generic.attributes.update(op.attributes)
            operations.extend([empty, generic])
            values[child.results[0]] = generic.results[0], identity
        if not valid or body.last_op.operands[0] not in values:
            continue
        result, result_map = values[body.last_op.operands[0]]
        if result_map != identity or result.type != op.results[0].type:
            continue
        Rewriter.replace_op(op, operations, [result])
        changed += 1
    module.verify()
    return changed
