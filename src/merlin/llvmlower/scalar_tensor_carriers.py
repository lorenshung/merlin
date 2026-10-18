"""Fold private scalar tensor packaging into a bit-preserving reshape."""


def fold_scalar_tensor_roundtrips(module):
    """Keep rank-zero data in tensor form across singleton packaging.

    A single-use private linalg result is extracted without indices and packed
    into one singleton element. It has no independently observable alias; the
    rewrite preserves its element bits, result type, and boundary metadata.
    """
    from math import prod

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType, i64
    from xdsl.rewriter import Rewriter

    count = 0
    for op in list(module.walk()):
        if op.name != "tensor.from_elements" or len(op.operands) != 1:
            continue
        extracted = op.operands[0]
        extract = extracted.owner
        if getattr(extract, "name", None) != "tensor.extract" or len(extract.operands) != 1:
            continue
        source = extract.operands[0]
        producer = source.owner
        if getattr(producer, "name", None) not in {"linalg.generic", "linalg.reduce"}:
            continue
        if source.type.get_shape() != () or not source.has_one_use() or not extracted.has_one_use():
            continue
        result_type = op.results[0].type
        shape = result_type.get_shape()
        if prod(shape) != 1 or source.type.element_type != result_type.element_type:
            continue
        dimensions = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [len(shape)]), list(shape)))
        reshape = tensor.ReshapeOp(source, dimensions.result, result_type)
        reshape.attributes.update(op.attributes)
        Rewriter.replace_op(op, [dimensions, reshape], [reshape.result])
        Rewriter.erase_op(extract)
        count += 1
    module.verify()
    return count
