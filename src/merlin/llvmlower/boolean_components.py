"""Boolean OR/AND as integer numerators followed by an exact truth predicate."""


def rewrite_boolean_components(module):
    from math import prod

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        DenseIntOrFPElementsAttr,
        IntegerAttr,
        TensorType,
        i8,
        i32,
        i64,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    bounds = {}
    changed = 0

    def point(inputs, typ, fn, maps, iterators, operations):
        empty = tensor.EmptyOp([], typ)
        body = Block(arg_types=[value.type.element_type for value in inputs] + [typ.element_type])
        result = fn(body)
        body.add_op(L.YieldOp(result))
        generic = L.GenericOp(
            inputs=inputs,
            outputs=[empty.tensor],
            body=Region(body),
            indexing_maps=maps,
            iterator_types=iterators,
            result_types=[typ],
        )
        operations.extend([empty, generic])
        return generic.results[0]

    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.results) != 1 or str(op.results[0].type.element_type) != "i1":
            continue
        body = list(op.body.block.ops)
        if len(body) != 2 or body[0].name not in {"arith.ori", "arith.andi"}:
            continue
        if len(op.inputs) != 2 or list(body[0].operands) != list(op.body.block.args[:2]):
            continue
        if any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        shape = op.results[0].type.get_shape()
        identity = AffineMapAttr(AffineMap.identity(len(shape)))
        operations = []
        inputs = []
        for i, source in enumerate(op.inputs):

            def extend(block):
                value = arith.ExtUIOp(block.args[0], i8)
                block.add_op(value)
                return value.result

            value = point(
                [source],
                TensorType(i8, shape),
                extend,
                ArrayAttr([list(op.get_indexing_maps())[i], identity]),
                op.get_iterator_types(),
                operations,
            )
            bounds[value] = 1
            inputs.append(value)
        zero = arith.ConstantOp(IntegerAttr(0, i32))
        seed = tensor.SplatOp(zero.result, [], TensorType(i32, shape))
        operations.extend([zero, seed])
        if body[0].name == "arith.ori":

            def numerator(block):
                left, right = arith.ExtSIOp(block.args[0], i32), arith.ExtSIOp(block.args[1], i32)
                addition = arith.AddiOp(left.result, right.result)
                block.add_ops([left, right, addition])
                return addition.result

            result = point(
                inputs, seed.result.type, numerator, ArrayAttr([identity] * 3), op.get_iterator_types(), operations
            )
        else:
            flat = []
            for value in inputs:
                sizes = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [3]), [prod(shape), 1, 1]))
                reshaped = tensor.ReshapeOp(value, sizes.result, TensorType(i8, [prod(shape), 1, 1]))
                operations.extend([sizes, reshaped])
                bounds[reshaped.result] = 1
                flat.append(reshaped.result)
            batch_seed = tensor.SplatOp(zero.result, [], TensorType(i32, [prod(shape), 1, 1]))
            operations.append(batch_seed)
            b = Block(arg_types=[i8, i8, i32])
            a, w = arith.ExtSIOp(b.args[0], i32), arith.ExtSIOp(b.args[1], i32)
            product = arith.MuliOp(a.result, w.result)
            addition = arith.AddiOp(product.result, b.args[2])
            b.add_ops([a, w, product, addition, L.YieldOp(addition.result)])
            maps = ArrayAttr(
                [
                    AffineMapAttr(AffineMap(4, 0, tuple(AffineDimExpr(i) for i in axes)))
                    for axes in [(0, 1, 3), (0, 3, 2), (0, 1, 2)]
                ]
            )
            contraction = L.GenericOp(
                inputs=flat,
                outputs=[batch_seed.result],
                body=Region(b),
                indexing_maps=maps,
                iterator_types=ArrayAttr(
                    [L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * 3 + [L.IteratorTypeAttr(L.IteratorType.REDUCTION)]
                ),
                result_types=[batch_seed.result.type],
            )
            sizes = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [len(shape)]), list(shape)))
            reshaped = tensor.ReshapeOp(contraction.results[0], sizes.result, seed.result.type)
            operations.extend([contraction, sizes, reshaped])
            result = reshaped.result
        bounds[result] = 2

        def truth(block):
            constant = arith.ConstantOp(IntegerAttr(0, i32))
            predicate = arith.CmpiOp(block.args[0], constant.result, "ne")
            block.add_ops([constant, predicate])
            return predicate.result

        final = point(
            [result], op.results[0].type, truth, ArrayAttr([identity] * 2), op.get_iterator_types(), operations
        )
        Rewriter.replace_op(op, operations, [final])
        changed += 1
    module.verify()
    return bounds, changed
