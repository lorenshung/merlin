"""Shape-only canonicalization of zero-iteration additive reductions."""

from math import prod


def fold_empty_reductions(module):
    from xdsl.rewriter import Rewriter

    count = 0
    for op in list(module.walk()):
        if op.name not in {"linalg.generic", "linalg.reduce"} or not op.results:
            continue
        if op.name == "linalg.reduce":
            seeds = [op.init]
            empty = prod(op.input.type.get_shape()) == 0
        else:
            seeds = list(op.outputs)
            try:
                empty = any(
                    iterator.data.value == "reduction" and extent == 0
                    for iterator, extent in zip(op.get_iterator_types(), op.get_static_loop_ranges(), strict=True)
                )
            except (AttributeError, ValueError):
                continue
        if not empty or len(seeds) != len(op.results):
            continue
        if any(a.type != b.type for a, b in zip(seeds, op.results)):
            continue
        # No body iteration executes: the result is precisely the original
        # seed, regardless of its dtype. No tensor payload is inspected.
        Rewriter.replace_op(op, [], seeds)
        count += 1
    if count:
        from xdsl.dialects import tensor

        for op in list(module.walk()):
            if op.name == "tensor.extract" and not op.indices:
                source = op.tensor.owner
                if getattr(source, "name", None) == "tensor.splat":
                    Rewriter.replace_op(op, [], [source.operands[0]])
            elif (
                op.name == "tensor.from_elements"
                and op.operands
                and all(value is op.operands[0] for value in op.operands)
            ):
                filled = tensor.SplatOp(op.operands[0], [], op.results[0].type)
                Rewriter.replace_op(op, [filled], [filled.result])
    return count
