"""Expose shape-only carriers of uniform literal storage as uniform fills."""


def expose_uniform_literal_carriers(module):
    from math import prod

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, FloatAttr, IntegerAttr, TensorType
    from xdsl.rewriter import Rewriter

    def scalar(value):
        owner = value.owner
        if getattr(owner, "name", None) == "tensor.splat":
            literal = owner.operands[0].owner
            if getattr(literal, "name", None) == "arith.constant":
                attr = literal.properties.get("value")
                return attr if isinstance(attr, (FloatAttr, IntegerAttr)) else None
        if getattr(owner, "name", None) == "arith.constant":
            attr = owner.properties.get("value")
            if isinstance(attr, DenseIntOrFPElementsAttr) and attr.is_splat():
                return attr.get_attrs()[0]
        return None

    changed = 0
    for op in list(module.walk()):
        if op.name not in {"tensor.reshape", "tensor.cast"} or len(op.results) != 1:
            continue
        source, result = op.operands[0], op.results[0]
        if not isinstance(source.type, TensorType) or not isinstance(result.type, TensorType):
            continue
        shapes = source.type.get_shape(), result.type.get_shape()
        if any(n < 0 for shape in shapes for n in shape) or prod(shapes[0]) != prod(shapes[1]):
            continue
        if source.type.element_type != result.type.element_type:
            continue
        attr = scalar(source)
        if attr is None:
            continue
        constant = arith.ConstantOp(attr)
        filled = tensor.SplatOp(constant.result, [], result.type)
        Rewriter.replace_op(op, [constant, filled], [filled.result])
        changed += 1
    module.verify()
    return changed
