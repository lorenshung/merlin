"""Finite native floating sources as bounded exact integer component programs."""

from fractions import Fraction
from math import prod

from merlin.llvmlower.fixed_point_ir import FORMAT, encode_fixed_integer, finite_grid


def rewrite_floating_components(module, input_grids, *, complex_inputs=()):
    """Return new integer bounds and closed finite representation proofs.

    Source packing uses exact power-of-two scaling in binary64, bounded to
    52 significant integer bits. Integer recombination fits signed i128. IEEE
    output conversion uses integer RNE. Source sign bits determine the sign of
    an exact zero without replaying the original floating computation.
    """
    from xdsl.dialects import arith, tensor
    from xdsl.dialects.builtin import (
        AffineMapAttr,
        ArrayAttr,
        DenseIntOrFPElementsAttr,
        FloatAttr,
        IntegerAttr,
        TensorType,
        f64,
        i64,
        i128,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.pointwise_primitives import expose_pointwise_primitives

    expose_pointwise_primitives(module)
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    grids = dict(zip(function.body.block.args, input_grids))
    complex_sources = {function.body.block.args[i] for i in complex_inputs}
    components = {}
    bounds = {}
    proofs = []

    def floating(value):
        return hasattr(value.type, "element_type") and str(value.type.element_type) in FORMAT

    def combine(items):
        items = [item for item in items if item is not None]
        if not items:
            return None
        exponent = min((e for e, b in items if b), default=0)
        bound = max((b * (1 << (e - exponent)) for e, b in items if b), default=0)
        return (exponent, bound) if bound.bit_length() <= 52 else None

    def component(value):
        if value in components:
            return components[value]
        op = value.owner
        if getattr(op, "name", None) == "linalg.generic" and len(op.inputs) == 1:
            body = list(op.body.block.ops)
            if len(body) == 1 and body[0].name == "linalg.yield" and body[0].operands[0] is op.body.block.args[0]:
                source = op.inputs[0]
                if source in complex_sources:
                    from xdsl.ir.affine import AffineConstantExpr

                    results = list(op.get_indexing_maps())[0].data.results
                    if results and isinstance(results[-1], AffineConstantExpr) and results[-1].value in (0, 1):
                        return "real" if results[-1].value == 0 else "imag"
                return component(source)
        return None

    def product_component(values):
        tags = [component(value) for value in values]
        if not any(tags):
            return None
        if all(tag in (None, "real", "imag") for tag in tags):
            return "".join("i" if tag == "imag" else "r" for tag in tags)
        return None

    def linear_component(values, operation):
        tags = [component(value) for value in values]
        if tags and all(tag == tags[0] for tag in tags):
            return tags[0]
        if tags == ["rr", "ii"] and operation == "arith.subf":
            return "real"
        if set(tags) == {"ri", "ir"} and operation == "arith.addf":
            return "imag"
        return None

    def grid(value):
        if value in grids:
            return grids[value]
        op = value.owner
        if getattr(op, "name", None) == "arith.constant":
            attr = op.properties.get("value") or op.attributes.get("value")
            values = list(attr.get_values()) if hasattr(attr, "get_values") else [attr.value.data]
            return finite_grid(values)
        carriers = {
            "tensor.splat",
            "tensor.concat",
            "tensor.extract_slice",
            "tensor.insert_slice",
            "tensor.collapse_shape",
            "tensor.expand_shape",
            "tensor.cast",
            "linalg.copy",
            "linalg.broadcast",
            "linalg.transpose",
        }
        if getattr(op, "name", None) in {"tensor.splat", "linalg.fill"}:
            return grid(op.operands[0])
        if getattr(op, "name", None) in carriers:
            children = [v for v in op.operands if floating(v)]
            values = [grid(v) for v in children]
            return combine(values) if values and all(v is not None for v in values) else None
        if getattr(op, "name", None) == "linalg.generic" and len(op.inputs) == 1:
            if all(child.name in {"arith.extf", "linalg.yield"} for child in op.body.block.ops):
                return grid(op.inputs[0])
        return None

    def point(inputs, typ, fn, operations):
        empty = tensor.EmptyOp([], typ)
        block = Block(arg_types=[v.type.element_type for v in inputs] + [typ.element_type])

        def emit(op):
            block.add_op(op)
            return op.results[0]

        value = fn(block.args[:-1], emit)
        block.add_op(L.YieldOp(value))
        identity = AffineMapAttr(AffineMap.identity(len(typ.get_shape())))
        generic = L.GenericOp(
            inputs=inputs,
            outputs=[empty.tensor],
            body=Region(block),
            indexing_maps=ArrayAttr([identity] * (len(inputs) + 1)),
            iterator_types=ArrayAttr([L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * len(typ.get_shape())),
            result_types=[typ],
        )
        operations.extend([empty, generic])
        return generic.results[0]

    def pack(value, exponent, operations):
        source_grid = grid(value)
        if source_grid is None:
            raise ValueError("floating source lacks a complete finite dyadic input proof")
        e, bound = source_grid
        if bound and e < exponent:
            raise ValueError("packing would round a source dyadic coefficient")
        aligned = bound * (1 << (e - exponent)) if bound else 0
        if aligned.bit_length() > 52:
            raise ValueError("aligned fixed-point coefficient exceeds the exact binary64 packing bound")

        def scalar(args, emit):
            value = args[0]
            if value.type != f64:
                value = emit(arith.ExtFOp(value, f64))
            scale = emit(arith.ConstantOp(FloatAttr(float(Fraction(2) ** (-exponent)), f64)))
            scaled = emit(arith.MulfOp(value, scale))
            return emit(arith.FPToSIOp(scaled, i64))

        result = point([value], TensorType(i64, value.type.get_shape()), scalar, operations)
        bounds[result] = aligned
        return result, aligned

    def zero(shape, operations):
        constant = arith.ConstantOp(IntegerAttr(0, i128))
        seed = tensor.SplatOp(constant.result, [], TensorType(i128, shape))
        operations.extend([constant, seed])
        bounds[seed.result] = 0
        return seed.result

    def reshape(value, shape, operations):
        literal = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(TensorType(i64, [len(shape)]), list(shape)))
        changed = tensor.ReshapeOp(value, literal.result, TensorType(value.type.element_type, shape))
        operations.extend([literal, changed])
        if value in bounds:
            bounds[changed.result] = bounds[value]
        return changed.result

    def publish(op, integer, exponent, integer_bound, operations, compute_type=None):
        compute_type = compute_type or op.results[0].type.element_type
        output_type = op.results[0].type.element_type

        def convert(values, emit):
            rounded = encode_fixed_integer(values[0], exponent, compute_type, emit)
            if output_type != compute_type:
                rounded = emit(arith.TruncFOp(rounded, output_type))
            return rounded

        converted = point([integer], op.results[0].type, convert, operations)
        # Exact zero has a source-defined sign, independent of the discarded
        # floating computation. Underflow of a nonzero integer keeps the RNE sign.
        identity = AffineMapAttr(AffineMap.identity(len(op.results[0].type.get_shape())))
        maps = list(op.get_indexing_maps())
        empty = tensor.EmptyOp([], op.results[0].type)
        block = Block(arg_types=[output_type, i128] + [v.type.element_type for v in op.inputs] + [output_type])

        def emit(operation):
            block.add_op(operation)
            return operation.results[0]

        def sign(value):
            from xdsl.dialects.builtin import IntegerType

            f, e, _ = FORMAT[str(value.type)]
            typ = IntegerType(f + e + 1)
            raw = emit(arith.BitcastOp(value, typ))
            mask = emit(arith.ConstantOp(IntegerAttr(1 << (f + e), typ)))
            masked = emit(arith.AndIOp(raw, mask))
            zero = emit(arith.ConstantOp(IntegerAttr(0, typ)))
            return emit(arith.CmpiOp(masked, zero, "ne"))

        scalar_ops = [child for child in op.body.block.ops if child.name != "linalg.yield"]
        negative = None
        if len(scalar_ops) == 1:
            scalar = scalar_ops[0]
            if all(value in op.body.block.args[: len(op.inputs)] for value in scalar.operands):
                operands = [block.args[2 + op.body.block.args.index(value)] for value in scalar.operands]
                if scalar.name in {"arith.extf", "arith.truncf", "arith.negf"}:
                    negative = sign(operands[0])
                    if scalar.name == "arith.negf":
                        from xdsl.dialects.builtin import i1

                        one = emit(arith.ConstantOp(IntegerAttr(1, i1)))
                        negative = emit(arith.XOrIOp(negative, one))
                elif scalar.name in {"arith.mulf", "arith.divf"}:
                    negative = emit(arith.XOrIOp(sign(operands[0]), sign(operands[1])))
                elif scalar.name in {"arith.addf", "arith.subf"}:
                    signs = [sign(value) for value in operands]
                    if scalar.name == "arith.subf":
                        from xdsl.dialects.builtin import i1

                        one = emit(arith.ConstantOp(IntegerAttr(1, i1)))
                        signs[1] = emit(arith.XOrIOp(signs[1], one))
                    negative = emit(arith.AndIOp(*signs))
                    for value in operands:
                        zero = emit(arith.ConstantOp(FloatAttr(0.0, value.type)))
                        is_zero = emit(arith.CmpfOp(value, zero, "oeq"))
                        negative = emit(arith.AndIOp(negative, is_zero))
        positive_zero = emit(arith.ConstantOp(FloatAttr(0.0, output_type)))
        signed_zero = positive_zero
        if negative is not None:
            negative_zero = emit(arith.ConstantOp(FloatAttr(-0.0, output_type)))
            signed_zero = emit(arith.SelectOp(negative, negative_zero, positive_zero))
        integer_zero = emit(arith.ConstantOp(IntegerAttr(0, i128)))
        exact_zero = emit(arith.CmpiOp(block.args[1], integer_zero, "eq"))
        selected = emit(arith.SelectOp(exact_zero, signed_zero, block.args[0]))
        block.add_op(L.YieldOp(selected))
        if op in contractions:
            # A positive-zero seeded contraction rounds exact cancellation to +0.
            result = converted
        else:
            final = L.GenericOp(
                inputs=[converted, integer, *op.inputs],
                outputs=[empty.tensor],
                body=Region(block),
                indexing_maps=ArrayAttr([identity, identity, *maps[:-1], identity]),
                iterator_types=op.get_iterator_types(),
                result_types=[op.results[0].type],
            )
            operations.extend([empty, final])
            result = final.results[0]
        max_exponent = integer_bound.bit_length() - 1 + exponent if integer_bound else 0
        grids[result] = (
            (exponent, 2 * integer_bound)
            if (2 * integer_bound).bit_length() <= 52 and max_exponent <= FORMAT[str(output_type)][2]
            else None
        )
        tag = getattr(integer.owner.attributes.get("merlin.exact_component"), "data", None)
        if tag:
            components[result] = tag
        proofs.append(
            dict(
                source_operation=op.name,
                common_exponent=exponent,
                integer_absolute_bound=integer_bound,
                reconstruction="bounded_i128",
                packing="exact_binary64_power_of_two",
                rounding="integer_quotient_remainder_rne",
                exact_zero_sign="source_ieee_sign_bits",
            )
        )
        Rewriter.replace_op(op, operations, [result])

    from merlin.kernels.shapes import observe_contractions

    contractions = dict(observe_contractions(module))
    for op in list(function.body.block.ops):
        if op in contractions and len(op.operands) == 3 and all(floating(v) for v in op.operands):
            operands = op.operands[:2]
            source_grids = [grid(v) for v in operands]
            seed_grid = grid(op.operands[2])
            if any(g is None for g in source_grids) or seed_grid is None or seed_grid[1] != 0:
                continue
            exponent = sum(e for e, b in source_grids)
            operations = []
            try:
                packed = [pack(value, e, operations) for value, (e, b) in zip(operands, source_grids, strict=True)]
            except ValueError:
                continue
            bound = prod(contractions[op].reduction) * prod(b for value, b in packed)
            if bound.bit_length() >= 120:
                continue
            seed = zero(op.results[0].type.get_shape(), operations)
            body = Block(arg_types=[i64, i64, i128])
            a, b = arith.ExtSIOp(body.args[0], i128), arith.ExtSIOp(body.args[1], i128)
            product = arith.MuliOp(a.result, b.result)
            addition = arith.AddiOp(product.result, body.args[2])
            body.add_ops([a, b, product, addition, L.YieldOp(addition.result)])
            integer = L.GenericOp(
                inputs=[v for v, b in packed],
                outputs=[seed],
                body=Region(body),
                indexing_maps=op.get_indexing_maps(),
                iterator_types=op.get_iterator_types(),
                result_types=[seed.type],
            )
            from xdsl.dialects.builtin import StringAttr

            tag = product_component(operands)
            if tag:
                integer.attributes["merlin.exact_component"] = StringAttr(tag)
            operations.append(integer)
            bounds[integer.results[0]] = bound
            publish(op, integer.results[0], exponent, bound, operations)
            continue
        if op.name != "linalg.generic" or len(op.results) != 1 or not floating(op.results[0]):
            continue
        original = op.body.block
        arithmetic = [
            child
            for child in original.ops
            if child.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf"}
        ]
        if not arithmetic:
            arithmetic = [child for child in original.ops if child.name in {"arith.extf", "arith.truncf"}]
        if len(arithmetic) != 1 or any(it.data != L.IteratorType.PARALLEL for it in op.get_iterator_types()):
            continue
        arithmetic = arithmetic[0]
        if any(
            child.name
            not in {
                "arith.addf",
                "arith.subf",
                "arith.mulf",
                "arith.divf",
                "arith.negf",
                "arith.extf",
                "arith.truncf",
                "linalg.yield",
            }
            for child in original.ops
        ):
            continue
        args = []
        for operand in arithmetic.operands:
            while getattr(operand.owner, "name", None) == "arith.extf":
                operand = operand.owner.operands[0]
            if operand not in original.args[: len(op.inputs)]:
                break
            args.append(original.args.index(operand))
        if len(args) != len(arithmetic.operands) or any(grid(op.inputs[i]) is None for i in args):
            continue
        if arithmetic.name == "arith.divf":
            # General rational divisors need a separately proved literal source.
            continue
        input_values = [op.inputs[i] for i in args]
        input_grids_here = [grid(v) for v in input_values]
        exponent = (
            sum(e for e, b in input_grids_here)
            if arithmetic.name == "arith.mulf"
            else min((e for e, b in input_grids_here if b), default=0)
        )
        packing_exponents = (
            [e for e, b in input_grids_here] if arithmetic.name == "arith.mulf" else [exponent] * len(args)
        )
        operations = []
        try:
            packed = [pack(value, e, operations) for value, e in zip(input_values, packing_exponents, strict=True)]
        except ValueError:
            continue
        shape = op.results[0].type.get_shape()
        integer_bound = (
            prod(b for value, b in packed) if arithmetic.name == "arith.mulf" else sum(b for value, b in packed)
        )
        if integer_bound.bit_length() >= 120:
            continue
        seed = zero(shape, operations)
        if arithmetic.name == "arith.mulf":
            # A K=1 batched contraction represents general pointwise products.
            # Broadcast the source operands into the original output map first.
            expanded = []
            for (value, bound), arg in zip(packed, args, strict=True):
                empty = tensor.EmptyOp([], TensorType(i64, shape))
                body = Block(arg_types=[i64, i64])
                body.add_op(L.YieldOp(body.args[0]))
                carrier = L.GenericOp(
                    inputs=[value],
                    outputs=[empty.tensor],
                    body=Region(body),
                    indexing_maps=ArrayAttr([list(op.get_indexing_maps())[arg], list(op.get_indexing_maps())[-1]]),
                    iterator_types=op.get_iterator_types(),
                    result_types=[empty.tensor.type],
                )
                operations.extend([empty, carrier])
                expanded.append(carrier.results[0])
                bounds[carrier.results[0]] = bound
            batch_shape = [prod(shape), 1, 1]
            expanded = [reshape(v, batch_shape, operations) for v in expanded]
            batch_seed = zero(batch_shape, operations)
            body = Block(arg_types=[i64, i64, i128])
            a, b = arith.ExtSIOp(body.args[0], i128), arith.ExtSIOp(body.args[1], i128)
            product = arith.MuliOp(a.result, b.result)
            addition = arith.AddiOp(product.result, body.args[2])
            body.add_ops([a, b, product, addition, L.YieldOp(addition.result)])
            maps = ArrayAttr(
                [
                    AffineMapAttr(AffineMap(4, 0, tuple(AffineDimExpr(i) for i in axes)))
                    for axes in [(0, 1, 3), (0, 3, 2), (0, 1, 2)]
                ]
            )
            integer = L.GenericOp(
                inputs=expanded,
                outputs=[batch_seed],
                body=Region(body),
                indexing_maps=maps,
                iterator_types=ArrayAttr(
                    [L.IteratorTypeAttr(L.IteratorType.PARALLEL)] * 3 + [L.IteratorTypeAttr(L.IteratorType.REDUCTION)]
                ),
                result_types=[batch_seed.type],
            )
            from xdsl.dialects.builtin import StringAttr

            tag = product_component(input_values)
            if tag:
                integer.attributes["merlin.exact_component"] = StringAttr(tag)
            operations.append(integer)
            bounds[integer.results[0]] = integer_bound
            result = reshape(integer.results[0], shape, operations)
            publish(op, result, exponent, integer_bound, operations, arithmetic.results[0].type)
            continue
        body = Block(arg_types=[i64] * len(packed) + [i128])
        values = []
        for value in body.args[:-1]:
            extended = arith.ExtSIOp(value, i128)
            body.add_op(extended)
            values.append(extended.result)
        if arithmetic.name == "arith.negf":
            constant = arith.ConstantOp(IntegerAttr(0, i128))
            body.add_op(constant)
            computed = arith.SubiOp(constant.result, values[0])
        elif arithmetic.name in {"arith.extf", "arith.truncf"}:
            computed = None
        else:
            cls = arith.AddiOp if arithmetic.name == "arith.addf" else arith.SubiOp
            computed = cls(*values)
        if computed is not None:
            body.add_op(computed)
        body.add_op(L.YieldOp(computed.result if computed is not None else values[0]))
        maps = list(op.get_indexing_maps())
        integer = L.GenericOp(
            inputs=[value for value, bound in packed],
            outputs=[seed],
            body=Region(body),
            indexing_maps=ArrayAttr([maps[i] for i in args] + [maps[-1]]),
            iterator_types=op.get_iterator_types(),
            result_types=[seed.type],
        )
        from xdsl.dialects.builtin import StringAttr

        tag = linear_component(input_values, arithmetic.name)
        if tag:
            integer.attributes["merlin.exact_component"] = StringAttr(tag)
        operations.append(integer)
        bounds[integer.results[0]] = integer_bound
        publish(op, integer.results[0], exponent, integer_bound, operations, arithmetic.results[0].type)
    module.verify()
    return bounds, proofs
