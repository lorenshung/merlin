"""Exact radix-256 integer producers with explicit modular host recombination.

Balanced digits are computed from the immutable source tensors at runtime.
Only marked signed-byte/i32 producers are device work. The original storage
width determines each host shift/add, so wraparound is retained at every source
operation boundary. No floating conversion or output-dependent specialization
is used. Smaller input limb counts require a bound on the complete input bytes.
"""

from __future__ import annotations

from math import prod


def rewrite_integer_components(module, *, limb_attribute, input_bounds=(), known_bounds=None, known_zero_limbs=None):
    import json

    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        IntegerAttr,
        IntegerType,
        StringAttr,
        TensorType,
        i8,
        i32,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineMap
    from xdsl.rewriter import Rewriter

    functions = [op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward"]
    if len(functions) != 1:
        raise ValueError("integer components require one forward function")
    bounds = dict(zip(functions[0].body.block.args, input_bounds))
    bounds.update(known_bounds or {})
    plans = []

    def indices(value, digits):
        omitted = (known_zero_limbs or {}).get(value, set())
        selected = [j for j in range(digits) if j not in omitted]
        return selected or [0]

    def width(value):
        typ = value.type
        return (
            typ.element_type.width.data
            if isinstance(typ, TensorType) and isinstance(typ.element_type, IntegerType)
            else 0
        )

    def absolute(value):
        if value in bounds:
            return bounds[value]
        op = value.owner
        if getattr(op, "name", None) == "arith.constant":
            attr = op.properties.get("value", op.attributes.get("value"))
            if hasattr(attr, "get_values"):
                return max((abs(int(v)) for v in attr.get_values()), default=0)
        return 1 << (width(value) - 1) if width(value) else None

    def count(value, *, mathematical=False):
        bits = width(value)
        magnitude = absolute(value)
        digits, capacity = 1, 127
        while magnitude > capacity:
            digits += 1
            capacity = 256 * capacity + 127
        return digits if mathematical else min((bits + 7) // 8, digits)

    def init(shape, element, value=0):
        constant = arith.ConstantOp(IntegerAttr(value, element))
        splat = tensor.SplatOp(constant.result, [], TensorType(element, shape))
        return [constant, splat], splat.result

    def point(inputs, result_type, fn):
        empty = tensor.EmptyOp([], result_type)
        block = Block(arg_types=[v.type.element_type for v in inputs] + [result_type.element_type])

        def emit(op):
            block.add_op(op)
            return op.results[0]

        result = fn(block.args[:-1], emit)
        block.add_op(L.YieldOp(result))
        rank = len(result_type.get_shape())
        identity = AffineMapAttr(AffineMap.identity(rank))
        generic = L.GenericOp(
            inputs=inputs,
            outputs=[empty.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([identity] * (len(inputs) + 1)),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * rank),
            result_types=[result_type],
        )
        return [empty, generic], generic.results[0]

    def digit(source, index):
        if width(source) == 8 and index == 0:
            return [], source
        wide = IntegerType(max(16, 2 * width(source)))

        def scalar(args, emit):
            remaining = emit((arith.ExtUIOp if width(source) == 1 else arith.ExtSIOp)(args[0], wide))
            if index:
                # Balanced radix carries below this digit are the geometric
                # offset 128*(1+256+...+256**(index-1)). Widening before the
                # addition preserves the carry above the source storage width.
                offset = emit(arith.ConstantOp(IntegerAttr(128 * ((1 << (8 * index)) - 1) // 255, wide)))
                remaining = emit(arith.AddiOp(remaining, offset))
                shift = emit(arith.ConstantOp(IntegerAttr(8 * index, wide)))
                remaining = emit(arith.ShRSIOp(remaining, shift))
            return emit(arith.TruncIOp(remaining, i8))

        return point([source], TensorType(i8, source.type.get_shape()), scalar)

    def mark(producer, entries, component=None):
        from merlin.llvmlower.additive_regions import positive_zero_seed

        entries = list(entries)
        covered = {entry["operand"] for entry in entries}
        zero_exponent = 8 * entries[0]["index"] + entries[0]["exponent"] if entries else 0
        for index, value in enumerate(producer.inputs):
            if (
                index not in covered
                and width(value) == 8
                and not producer.body.block.args[index].uses
                and positive_zero_seed(value)
            ):
                # An unread literal byte tensor still supplies its static loop
                # domain. The plan must cover that operand as the exact one-byte
                # zero representation at the data plane's binary exponent.
                # Scaling zero by that exponent is exact for every limb index.
                entries.append(dict(operand=index, index=0, count=1, exponent=zero_exponent))
        plan = dict(version=1, family="F2", radix=256, inputs=entries)
        if component is not None:
            plan.update(family="F4", component=component)
        producer.attributes[limb_attribute] = StringAttr(json.dumps(plan, sort_keys=True))
        plans.append(plan)

    def weighted(result, component, exponent, operations):
        element = result.type.element_type

        def scalar(args, emit):
            value = args[1]
            if element.width.data > 32:
                value = emit(arith.ExtSIOp(value, element))
            elif element.width.data < 32:
                value = emit(arith.TruncIOp(value, element))
            if exponent:
                shift = emit(arith.ConstantOp(IntegerAttr(exponent, element)))
                value = emit(arith.ShLIOp(value, shift))
            return emit(arith.AddiOp(args[0], value))

        ops, result = point([result, component], result.type, scalar)
        operations.extend(ops)
        return result

    def affine(op):
        if op.name != "linalg.generic" or any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            return None
        block = op.body.block
        expressions = {}
        for i, arg in enumerate(block.args[: len(op.inputs)]):
            source = op.inputs[i].owner
            while getattr(source, "name", None) in {"tensor.splat", "linalg.fill"}:
                source = source.operands[0].owner
            attr = (
                (source.properties.get("value") or source.attributes.get("value"))
                if getattr(source, "name", None) == "arith.constant"
                else None
            )
            expressions[arg] = ({}, attr.value.data) if isinstance(attr, IntegerAttr) else ({i: 1}, 0)
        for child in block.ops:
            if child.name == "linalg.yield":
                return expressions.get(child.operands[0])
            if child.name == "arith.constant":
                attr = child.properties.get("value")
                if not isinstance(attr, IntegerAttr):
                    return None
                expressions[child.results[0]] = ({}, attr.value.data)
                continue
            if child.name in {"arith.extsi", "arith.extui", "arith.trunci"}:
                # Intermediate narrowing is allowed only at the final source width.
                if child.name == "arith.trunci" and child.results[0].type != op.results[0].type.element_type:
                    return None
                if child.operands[0] not in expressions:
                    return None
                expressions[child.results[0]] = expressions[child.operands[0]]
                continue
            if child.name not in {"arith.addi", "arith.subi", "arith.muli"}:
                return None
            if any(v not in expressions for v in child.operands):
                return None
            (a, ac), (b, bc) = [expressions[v] for v in child.operands]
            if child.name == "arith.muli":
                if a and b:
                    return None
                coefficient, constant = (bc, ac * bc) if a else (ac, ac * bc)
                terms = {i: n * coefficient for i, n in (a or b).items()}
            else:
                sign = -1 if child.name == "arith.subi" else 1
                terms = a.copy()
                for i, n in b.items():
                    terms[i] = terms.get(i, 0) + sign * n
                constant = ac + sign * bc
            expressions[child.results[0]] = ({i: n for i, n in terms.items() if n}, constant)
        return None

    from merlin.kernels.shapes import observe_contractions

    contractions = dict(observe_contractions(module))

    def bilinear(op):
        if op not in contractions or len(op.operands) != 3:
            return False
        if any(width(value) not in (1, 8, 16, 32, 64, 128, 512, 1024) for value in op.operands[:2]):
            return False
        body = op.body.block
        if any(child.name not in {"arith.extsi", "arith.muli", "arith.addi", "linalg.yield"} for child in body.ops):
            return False
        yielded = body.last_op.operands[0].owner
        if getattr(yielded, "name", None) != "arith.addi" or body.args[2] not in yielded.operands:
            return False
        product_value = next((v for v in yielded.operands if v is not body.args[2]), None)
        product = getattr(product_value, "owner", None)
        if getattr(product, "name", None) != "arith.muli":
            return False
        leaves = []
        for value in product.operands:
            while getattr(value.owner, "name", None) == "arith.extsi":
                value = value.owner.operands[0]
            leaves.append(value)
        if set(leaves) != set(body.args[:2]):
            return False
        typ = op.results[0].type
        bits = width(op.results[0])
        operations, result = [], op.operands[2]
        counts = [count(v, mathematical=bits > width(v)) for v in op.operands[:2]]
        digits = []
        for source, n in zip(op.operands[:2], counts, strict=True):
            values = {}
            for index in indices(source, n):
                split_ops, limb = digit(source, index)
                operations.extend(split_ops)
                values[index] = limb
            digits.append(values)
        maps = list(op.get_indexing_maps())
        for j, lhs in digits[0].items():
            for k, rhs in digits[1].items():
                exponent = 8 * (j + k)
                if exponent >= bits:
                    continue
                seed_ops, seed = init(typ.get_shape(), i32)
                operations.extend(seed_ops)
                block = Block(arg_types=[i8, i8, i32])
                a = arith.ExtSIOp(block.args[0], i32)
                b = arith.ExtSIOp(block.args[1], i32)
                product = arith.MuliOp(a.result, b.result)
                addition = arith.AddiOp(product.result, block.args[2])
                block.add_ops([a, b, product, addition, L.YieldOp(addition.result)])
                producer = L.GenericOp(
                    inputs=[lhs, rhs],
                    outputs=[seed],
                    body=Region(block),
                    indexing_maps=ArrayAttr(maps),
                    iterator_types=op.get_iterator_types(),
                    result_types=[seed.type],
                )
                mark(
                    producer,
                    [dict(operand=i, index=index, count=counts[i], exponent=0) for i, index in enumerate((j, k))],
                    getattr(op.attributes.get("merlin.exact_component"), "data", None),
                )
                operations.append(producer)
                result = weighted(result, producer.results[0], exponent, operations)
        bound = prod(contractions[op].reduction) * absolute(op.operands[0]) * absolute(op.operands[1])
        bounds[result] = min(bound + absolute(op.operands[2]), 1 << (bits - 1))
        Rewriter.replace_op(op, operations, [result])
        return True

    def sum_reduction(op):
        if op.name != "linalg.generic" or len(op.results) != 1 or len(op.outputs) != 1:
            return False
        if not any(it.data == L.IteratorType.REDUCTION for it in op.get_iterator_types()):
            return False
        block = op.body.block
        if len(list(block.ops)) != 2 or block.first_op.name != "arith.addi":
            return False
        if list(block.last_op.operands) != list(block.first_op.results):
            return False
        accumulator = block.args[-1]
        used = list(block.first_op.operands)
        if accumulator not in used:
            return False
        data = used[1] if used[0] is accumulator else used[0]
        if data not in block.args[:-1]:
            return False
        source_index = list(block.args).index(data)
        # Window dimensions may be supplied by unused shape-only operands.
        # Their maps determine the iteration domain but their values are unread.
        source, seed = op.inputs[source_index], op.outputs[0]
        if width(source) not in (1, 8, 16, 32, 64, 128, 512, 1024):
            return False
        bits = width(op.results[0])
        digits = count(source, mathematical=bits > width(source))
        operations, result = [], seed
        for index in indices(source, digits):
            if 8 * index >= bits:
                continue
            split_ops, limb = digit(source, index)
            operations.extend(split_ops)
            seed_ops, zero = init(result.type.get_shape(), i32)
            operations.extend(seed_ops)
            inputs = []
            for j, original_input in enumerate(op.inputs):
                if j == source_index:
                    inputs.append(limb)
                else:
                    shape_ops, shape_only = init(original_input.type.get_shape(), i8)
                    operations.extend(shape_ops)
                    inputs.append(shape_only)
            body = Block(arg_types=[i8] * len(inputs) + [i32])
            extended = arith.ExtSIOp(body.args[source_index], i32)
            addition = arith.AddiOp(extended.result, body.args[-1])
            body.add_ops([extended, addition, L.YieldOp(addition.result)])
            producer = L.GenericOp(
                inputs=inputs,
                outputs=[zero],
                body=Region(body),
                indexing_maps=op.get_indexing_maps(),
                iterator_types=op.get_iterator_types(),
                result_types=[zero.type],
            )
            mark(
                producer,
                [dict(operand=source_index, index=index, count=digits, exponent=0)],
                getattr(op.attributes.get("merlin.exact_component"), "data", None),
            )
            operations.append(producer)
            result = weighted(result, producer.results[0], 8 * index, operations)
        extent = prod(source.type.get_shape()) // max(1, prod(result.type.get_shape()))
        bounds[result] = min(extent * absolute(source) + absolute(seed), 1 << (bits - 1))
        Rewriter.replace_op(op, operations, [result])
        return True

    original = list(module.walk())
    for op in original:
        if len(op.results) != 1 or width(op.results[0]) not in (1, 8, 16, 32, 64, 128, 512, 1024):
            continue
        if "merlin.exact_host_arithmetic" in op.attributes or any(
            k.startswith("prov.quant.") or k == limb_attribute for k in op.attributes
        ):
            continue
        if "merlin.exact_additive" in op.attributes:
            from merlin.llvmlower.additive_regions import data_sources, retype_additive

            sources = data_sources(op, lambda typ: str(typ) in {"i128", "i1024"})
            if not sources:
                continue
            digits = max(count(value, mathematical=True) for value in sources)
            operations, result = init(op.results[0].type.get_shape(), op.results[0].type.element_type)
            for index in range(digits):
                replacements, carriers = {}, []
                for source in sources:
                    split_ops, limb = digit(source, index)
                    operations.extend(split_ops)

                    def extend(args, emit):
                        return emit(arith.ExtSIOp(args[0], i32))

                    carrier_ops, carrier = point([limb], TensorType(i32, source.type.get_shape()), extend)
                    operations.extend(carrier_ops)
                    replacements[source] = carrier
                    carriers.append(carrier)
                lifted = retype_additive(op, replacements, i32, floating=False)
                if lifted is None:
                    break
                producer_ops, producer = lifted
                operations.extend(producer_ops)
                mark(
                    producer,
                    [
                        dict(operand=list(producer.inputs).index(v), index=index, count=digits, exponent=0)
                        for v in carriers
                    ],
                )
                result = weighted(result, producer.results[0], 8 * index, operations)
            else:
                bounds[result] = absolute(op.results[0])
                Rewriter.replace_op(op, operations, [result])
                continue
        if bilinear(op) or sum_reduction(op):
            continue
        expression = affine(op)
        if expression is None:
            continue
        terms, bias = expression
        if not terms or any(width(op.inputs[i]) not in (1, 8, 16, 32, 64, 128, 512, 1024) for i in terms):
            continue
        typ = op.results[0].type
        bits = width(op.results[0])
        # Provider's full-domain i32 bound is authoritative; avoid unrepresentable literals.
        if any(abs(c) * 128 >= (1 << 31) for c in terms.values()) or not -(1 << (bits - 1)) <= bias < (1 << (bits - 1)):
            continue
        operations, result = init(typ.get_shape(), typ.element_type, bias)
        for i, coefficient in terms.items():
            source = op.inputs[i]
            digits = count(source, mathematical=bits > width(source))
            for index in indices(source, digits):
                if 8 * index >= bits:
                    continue
                split_ops, limb = digit(source, index)
                operations.extend(split_ops)
                seed_ops, seed = init(typ.get_shape(), i32)
                operations.extend(seed_ops)
                body = Block(arg_types=[i8, i32])
                extended = arith.ExtSIOp(body.args[0], i32)
                multiplier = arith.ConstantOp(IntegerAttr(coefficient, i32))
                product = arith.MuliOp(extended.result, multiplier.result)
                body.add_ops([extended, multiplier, product, L.YieldOp(product.result)])
                maps = list(op.get_indexing_maps())
                producer = L.GenericOp(
                    inputs=[limb],
                    outputs=[seed],
                    body=Region(body),
                    indexing_maps=ArrayAttr([maps[i], maps[-1]]),
                    iterator_types=op.get_iterator_types(),
                    result_types=[seed.type],
                )
                mark(
                    producer,
                    [dict(operand=0, index=index, count=digits, exponent=0)],
                    getattr(op.attributes.get("merlin.exact_component"), "data", None),
                )
                operations.append(producer)
                result = weighted(result, producer.results[0], 8 * index, operations)
        bound = abs(bias) + sum(abs(c) * absolute(op.inputs[i]) for i, c in terms.items())
        bounds[result] = min(bound, 1 << (bits - 1))
        Rewriter.replace_op(op, operations, [result])
    module.verify()
    return plans
