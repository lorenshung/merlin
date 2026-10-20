"""Make direct user tensor results explicit device movement candidates."""


def expose_argument_results(module, report):
    from xdsl.dialects import tensor
    from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, TensorType
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap

    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward"]
    if len(functions) != 1:
        return 0
    function = functions[0]
    returned = function.body.block.last_op
    if returned.name != "func.return":
        return 0
    indices = {
        i for row in report.get("cases", ()) if row.get("status") == "bundled" for i in row.get("output_indices", ())
    }
    values = list(returned.operands)
    changed = 0
    for index in sorted(indices):
        if not 0 <= index < len(values):
            continue
        value = values[index]
        if value not in function.body.block.args or not isinstance(value.type, TensorType):
            continue
        shape = value.type.get_shape()
        if any(n <= 0 for n in shape):
            continue
        empty = tensor.EmptyOp([], value.type)
        scalar = Block(arg_types=[value.type.element_type] * 2)
        scalar.add_op(L.YieldOp(scalar.args[0]))
        identity = AffineMapAttr(AffineMap.identity(len(shape)))
        copied = L.GenericOp(
            inputs=[value],
            outputs=[empty.tensor],
            body=Region(scalar),
            indexing_maps=ArrayAttr([identity, identity]),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(shape)),
            result_types=[value.type],
        )
        function.body.block.insert_ops_before([empty, copied], returned)
        values[index] = copied.results[0]
        changed += 1
    returned.operands = values
    return changed
