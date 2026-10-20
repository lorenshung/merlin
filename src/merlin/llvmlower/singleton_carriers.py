"""Reassemble one extracted storage word through a shape-only tensor view."""


def fold_singleton_carriers(module):
    from math import prod

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, IntegerAttr, TensorType, i64
    from xdsl.rewriter import Rewriter

    count = 0
    for op in list(module.walk()):
        if op.name != "tensor.from_elements" or len(op.operands) != 1:
            continue
        extracted = op.operands[0].owner
        if getattr(extracted, "name", None) != "tensor.extract":
            continue
        source = extracted.tensor
        if (
            prod(source.type.get_shape()) != 1
            or prod(op.results[0].type.get_shape()) != 1
            or source.type.element_type != op.results[0].type.element_type
        ):
            continue
        indices = list(extracted.indices)
        if any(
            getattr(v.owner, "name", None) != "arith.constant"
            or not isinstance(v.owner.value, IntegerAttr)
            or v.owner.value.value.data != 0
            for v in indices
        ):
            continue
        shape = list(op.results[0].type.get_shape())
        sizes = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [len(shape)]), shape))
        result = tensor.ReshapeOp(source, sizes.result, op.results[0].type)
        Rewriter.replace_op(op, [sizes, result], [result.result])
        count += 1
    return count
