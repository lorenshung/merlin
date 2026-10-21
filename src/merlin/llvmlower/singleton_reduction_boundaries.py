"""A one-iteration reduction is the same scalar body with parallel iterators.

Keep every original floating operation and initialization word. This does not
replace a floating monoid by a byte copy or change its NaN/zero behavior.
"""


def expose_singleton_reduction_boundaries(module):
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import TensorType
    from xdsl.ir import Block
    from xdsl.rewriter import Rewriter
    from xdsl.traits import Pure

    from merlin.llvmlower._ir import PARALLEL, generic, identity, reshape

    changed = 0
    for op in list(module.walk()):
        if op.name != "linalg.reduce" or len(op.operands) != 2 or len(op.results) != 1:
            continue
        source, seed = op.operands
        if not isinstance(source.type, TensorType) or not isinstance(seed.type, TensorType):
            continue
        shape = source.type.get_shape()
        dimensions = tuple(op.dimensions.get_values())
        if any(n < 0 for n in shape) or len(set(dimensions)) != len(dimensions):
            continue
        if any(d < 0 or d >= len(shape) or shape[d] != 1 for d in dimensions):
            continue
        expected = tuple(n for i, n in enumerate(shape) if i not in dimensions)
        result_type = op.results[0].type
        if (
            result_type.get_shape() != expected
            or seed.type != result_type
            or source.type.element_type != result_type.element_type
        ):
            continue
        body = op.regions[0].block
        if len(body.args) != 2 or body.last_op.name != "linalg.yield":
            continue
        if any(
            child.regions or child.name == "builtin.unregistered" or not child.has_trait(Pure)
            for child in list(body.ops)[:-1]
        ):
            continue
        operations = []
        reshaped = reshape(source, expected, operations)
        empty = tensor.EmptyOp([], result_type)
        parallel_body = Block(arg_types=[result_type.element_type] * 3)
        mapping = dict(zip(body.args, parallel_body.args[:2], strict=True))
        for child in body.ops:
            cloned = child.clone(value_mapper=mapping)
            parallel_body.add_op(cloned)
            mapping.update(zip(child.results, cloned.results, strict=True))
        rank = len(expected)
        result = generic([reshaped, seed], empty.tensor, parallel_body, [identity(rank)] * 3, [PARALLEL] * rank)
        result.attributes.update(op.attributes)
        Rewriter.replace_op(op, [*operations, empty, result], [result.results[0]])
        changed += 1
    if changed:
        module.verify()
    return changed
