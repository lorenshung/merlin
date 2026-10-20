"""Expose closed integer sums with an index-only static mask as contractions.

The mask is a coefficient tensor, not a precomputed result. Integer modular
addition permits reassociation; floating accumulation boundaries are retained.
"""


def expose_static_masked_sums(module):
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        DenseIntOrFPElementsAttr,
        IntegerAttr,
        IntegerType,
        TensorType,
        i8,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.additive_regions import positive_zero_seed

    count = 0
    for op in list(module.walk()):
        if op.name != "linalg.generic" or len(op.inputs) != 1 or len(op.outputs) != 1 or len(op.results) != 1:
            continue
        element = op.results[0].type.element_type
        if not isinstance(element, IntegerType) or element.width.data < 8 or op.inputs[0].type.element_type != element:
            continue
        if not positive_zero_seed(op.outputs[0]):
            continue
        body = op.body.block
        children = list(body.ops)
        if any(
            child.name
            not in {"linalg.index", "arith.constant", "arith.cmpi", "arith.select", "arith.addi", "linalg.yield"}
            for child in children
        ):
            continue
        if len(children) < 3 or children[-2].name != "arith.addi" or children[-1].name != "linalg.yield":
            continue
        addition = children[-2]
        flags = addition.properties.get("overflowFlags")
        if flags is not None and str(flags) != "#arith.overflow<none>":
            continue
        if list(children[-1].operands) != list(addition.results) or body.args[-1] not in addition.operands:
            continue
        value = next((v for v in addition.operands if v is not body.args[-1]), None)
        selected = getattr(value, "owner", None)
        if getattr(selected, "name", None) != "arith.select":
            continue
        predicate = selected.operands[0].owner
        if getattr(predicate, "name", None) != "arith.cmpi":
            continue

        def zero(value):
            owner = value.owner
            return (
                getattr(owner, "name", None) == "arith.constant"
                and isinstance(owner.value, IntegerAttr)
                and owner.value.value.data == 0
            )

        if selected.operands[1] is body.args[0] and zero(selected.operands[2]):
            inverted = False
        elif selected.operands[2] is body.args[0] and zero(selected.operands[1]):
            inverted = True
        else:
            continue
        indices = [v.owner for v in predicate.operands]
        if any(getattr(owner, "name", None) != "linalg.index" for owner in indices):
            continue
        axes = [owner.dim.value.data for owner in indices]
        iterators = list(op.get_iterator_types())
        reductions = [i for i, it in enumerate(iterators) if it.data.value == "reduction"]
        if len(set(axes)) != 2 or len(reductions) != 1 or reductions[0] not in axes:
            continue
        # The checked complete body cannot hide another data computation.
        expected = {*indices, predicate, selected, addition, children[-1]}
        zero_owner = selected.operands[2 if not inverted else 1].owner
        if zero_owner in children:
            expected.add(zero_owner)
        if set(children) != expected:
            continue
        try:
            ranges = op.get_static_loop_ranges()
        except (ValueError, AttributeError):
            continue
        if any(n <= 0 for n in ranges):
            continue
        from math import prod

        rank = len(op.results[0].type.get_shape())
        if reductions != [rank] or len(iterators) != rank + 1:
            continue
        scan_axis = next(i for i in axes if i != rank)
        original_shape = op.results[0].type.get_shape()
        if op.inputs[0].type.get_shape() != original_shape or ranges[rank] != original_shape[scan_axis]:
            continue
        maps = list(op.get_indexing_maps())
        source_axes = tuple(AffineDimExpr(rank if d == scan_axis else d) for d in range(rank))
        if maps[0].data != AffineMap(rank + 1, 0, source_axes) or maps[-1].data != AffineMap(
            rank + 1, 0, tuple(AffineDimExpr(d) for d in range(rank))
        ):
            continue
        permutation = [d for d in range(rank) if d != scan_axis] + [scan_axis]
        operations = []

        def transpose(value, order):
            if order == list(range(len(order))):
                return value
            shape = value.type.get_shape()
            typ = TensorType(value.type.element_type, [shape[d] for d in order])
            empty = tensor.EmptyOp([], typ)
            block = Block(arg_types=[typ.element_type, typ.element_type])
            block.add_op(L.YieldOp(block.args[0]))
            indexing = AffineMap(len(order), 0, tuple(AffineDimExpr(order.index(d)) for d in range(len(order))))
            copied = L.GenericOp(
                inputs=[value],
                outputs=[empty.tensor],
                body=Region(block),
                indexing_maps=ArrayAttr([AffineMapAttr(indexing), AffineMapAttr(AffineMap.identity(len(order)))]),
                iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(order)),
                result_types=[typ],
            )
            operations.extend([empty, copied])
            return copied.results[0]

        def reshape(value, shape):
            if value.type.get_shape() == tuple(shape):
                return value
            from xdsl.dialects.builtin import i64

            sizes = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [len(shape)]), list(shape)))
            result = tensor.ReshapeOp(value, sizes.result, TensorType(value.type.element_type, shape))
            operations.extend([sizes, result])
            return result.result

        k = original_shape[scan_axis]
        n = prod(original_shape) // k
        transposed = transpose(op.inputs[0], permutation)
        matrix = reshape(transposed, [n, k])

        def compare(a, b):
            return (a == b, a != b, a < b, a <= b, a > b, a >= b, a < b, a <= b, a > b, a >= b)[
                predicate.predicate.value.data
            ]

        weights = []
        for source in range(k):
            for destination in range(k):
                coords = {rank: source, scan_axis: destination}
                weights.append(int(compare(coords[axes[0]], coords[axes[1]]) != inverted))
        constant = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i8, [k, k]), weights))
        zero_value = arith.ConstantOp(IntegerAttr(0, element))
        seed = tensor.SplatOp(zero_value.result, [], TensorType(element, [n, k]))
        scalar = Block(arg_types=[element, i8, element])
        coefficient = scalar.args[1]
        if element.width.data > 8:
            extended = arith.ExtSIOp(coefficient, element)
            scalar.add_op(extended)
            coefficient = extended.result
        product = arith.MuliOp(scalar.args[0], coefficient)
        total = arith.AddiOp(product.result, scalar.args[2])
        scalar.add_ops([product, total, L.YieldOp(total.result)])
        contraction = L.GenericOp(
            inputs=[matrix, constant.result],
            outputs=[seed.result],
            body=Region(scalar),
            indexing_maps=ArrayAttr(
                [AffineMapAttr(AffineMap(3, 0, tuple(AffineDimExpr(i) for i in ds))) for ds in [(0, 2), (2, 1), (0, 1)]]
            ),
            iterator_types=ArrayAttr(
                [L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * 2 + [L.IteratorTypeAttr(L.IteratorType.REDUCTION)]
            ),
            result_types=[seed.result.type],
        )
        contraction.attributes.update(op.attributes)
        operations.extend([constant, zero_value, seed, contraction])
        restored = reshape(contraction.results[0], transposed.type.get_shape())
        restored = transpose(restored, [permutation.index(d) for d in range(rank)])
        Rewriter.replace_op(op, operations, [restored])
        count += 1
    return count
