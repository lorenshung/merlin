"""Canonicalize one-contribution integer monoid reductions to storage movement."""


def fold_singleton_reductions(module):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerType, TensorType, i64
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.additive_regions import positive_zero_seed

    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.reduce" or len(op.operands) != 2 or len(op.results) != 1:
            continue
        source, seed = op.operands
        typ = source.type.element_type
        if not isinstance(typ, IntegerType) or typ != seed.type.element_type:
            continue
        shape = source.type.get_shape()
        if any(n < 0 for n in shape):
            continue
        dimensions = tuple(op.dimensions.get_values())
        if any(d < 0 or d >= len(shape) or shape[d] != 1 for d in dimensions):
            continue
        if len(set(dimensions)) != len(dimensions) or not positive_zero_seed(seed):
            continue
        body = op.regions[0].block
        ops = list(body.ops)
        if len(ops) != 2 or ops[0].name not in {"arith.addi", "arith.ori"} or ops[1].name != "linalg.yield":
            continue
        if set(ops[0].operands) != set(body.args) or list(ops[1].operands) != list(ops[0].results):
            continue
        flags = ops[0].properties.get("overflowFlags")
        if flags is not None and str(flags) != "#arith.overflow<none>":
            continue
        result_type = op.results[0].type
        expected = tuple(n for d, n in enumerate(shape) if d not in dimensions)
        if result_type.element_type != typ or result_type.get_shape() != expected:
            continue
        sizes = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [len(expected)]), list(expected)))
        movement = tensor.ReshapeOp(source, sizes.result, result_type)
        Rewriter.replace_op(op, [sizes, movement], [movement.result])
        changed += 1
    module.verify()
    return changed
