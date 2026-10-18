"""Finite native floating sources as bounded exact integer component programs."""

from fractions import Fraction
from math import prod

from merlin.llvmlower.fixed_point_ir import FORMAT, encode_fixed_integer, finite_grid


def rewrite_floating_components(
    module,
    input_grids,
    *,
    complex_inputs=(),
    allow_reassociation=False,
    known_grids=None,
    input_values=None,
    zero_limbs=None,
):
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
        IntegerType,
        TensorType,
        UnitAttr,
        f64,
        i64,
        i128,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap
    from xdsl.rewriter import Rewriter

    from merlin.llvmlower.additive_regions import expose_reduce_domains
    from merlin.llvmlower.pointwise_primitives import expose_pointwise_primitives

    expose_reduce_domains(module)
    expose_pointwise_primitives(module)
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    grids = dict(zip(function.body.block.args, input_grids))
    grids.update(known_grids or {})
    wide_packing = any(g is not None and g[1].bit_length() > 52 for g in input_grids)
    if wide_packing:
        i64, i128 = IntegerType(512), IntegerType(1024)
    packing_limit = 511 if wide_packing else 52
    reconstruction_limit = i128.width.data - 8

    complex_sources = {function.body.block.args[i] for i in complex_inputs}
    components = {}
    bounds = {}
    proofs = []
    raw_forms = {}

    payloads = dict(zip(function.body.block.args, input_values or ()))

    def payload(value):
        import numpy as np

        if value in payloads:
            return payloads[value]
        owner = value.owner
        if getattr(owner, "name", None) == "linalg.generic" and len(owner.inputs) == 1:
            body = list(owner.body.block.ops)
            if all(child.name in {"arith.extf", "linalg.yield"} for child in body):
                yielded = body[-1].operands[0]
                while getattr(yielded.owner, "name", None) == "arith.extf":
                    yielded = yielded.owner.operands[0]
                if yielded is not owner.body.block.args[0]:
                    return None
                data = payload(owner.inputs[0])
                if data is None:
                    return None
                amap, outmap = [a.data for a in owner.get_indexing_maps()]
                shape = value.type.get_shape()
                if outmap != AffineMap.identity(len(shape)):
                    return None
                result = np.empty(shape, dtype=object)
                for index in np.ndindex(shape):
                    result[index] = data[amap.eval(index, ())]
                payloads[value] = result
                return result
        if getattr(owner, "name", None) in {"tensor.reshape", "tensor.expand_shape", "tensor.collapse_shape"}:
            data = payload(owner.operands[0])
            if data is not None:
                return data.reshape(value.type.get_shape())
        return None

    def record_zero_limbs(result, source, exponent):
        if not wide_packing or zero_limbs is None:
            return
        data = payload(source)
        if data is None:
            return
        coefficients = [Fraction.from_float(float(v)) / Fraction(2) ** exponent for v in data.flat]
        if any(v.denominator != 1 for v in coefficients):
            raise ValueError("recorded zero-limb premise is not an exact dyadic integer")
        remaining = [v.numerator for v in coefficients]
        omitted = set()
        for j in range(i64.width.data // 8):
            digits = [((v + 128) % 256) - 128 for v in remaining]
            if not any(digits):
                omitted.add(j)
            remaining = [(v - d) // 256 for v, d in zip(remaining, digits, strict=True)]
        zero_limbs[result] = omitted

    def floating(value):
        return hasattr(value.type, "element_type") and str(value.type.element_type) in FORMAT

    def combine(items):
        items = [item for item in items if item is not None]
        if not items:
            return None
        exponent = min((e for e, b in items if b), default=0)
        bound = max((b * (1 << (e - exponent)) for e, b in items if b), default=0)
        return (exponent, bound) if bound.bit_length() <= packing_limit else None

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
        if allow_reassociation and value in raw_forms and raw_forms[value][3] == 1:
            return raw_forms[value][1:3]
        if value in grids:
            return grids[value]
        op = value.owner
        if getattr(op, "name", None) == "arith.constant":
            attr = op.properties.get("value", op.attributes.get("value"))
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
                yielded = op.body.block.last_op.operands[0]
                while getattr(yielded.owner, "name", None) == "arith.extf":
                    yielded = yielded.owner.operands[0]
                if yielded is op.body.block.args[0]:
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
        generic.attributes["merlin.exact_host_arithmetic"] = UnitAttr()
        operations.extend([empty, generic])
        return generic.results[0]

    def pack(value, exponent, operations):
        if allow_reassociation and value in raw_forms:
            integer, source_exponent, bound, divisor = raw_forms[value]
            if divisor != 1 or exponent > source_exponent:
                raise ValueError("raw rational component cannot be packed as an integer")
            aligned = bound << (source_exponent - exponent)
            if aligned.bit_length() >= i64.width.data - 1:
                raise ValueError("raw exact component exceeds the signed-i64 operand representation")

            def scalar(values, emit):
                result = values[0]
                if source_exponent != exponent:
                    shift = emit(arith.ConstantOp(IntegerAttr(source_exponent - exponent, i128)))
                    result = emit(arith.ShLIOp(result, shift))
                return emit(arith.TruncIOp(result, i64))

            packed = point([integer], TensorType(i64, value.type.get_shape()), scalar, operations)
            bounds[packed] = aligned
            return packed, aligned
        source_grid = grid(value)
        if source_grid is None:
            raise ValueError("floating source lacks a complete finite dyadic input proof")
        e, bound = source_grid
        if bound and e < exponent:
            raise ValueError("packing would round a source dyadic coefficient")
        aligned = bound * (1 << (e - exponent)) if bound else 0
        if aligned.bit_length() > packing_limit:
            raise ValueError("aligned fixed-point coefficient exceeds the exact binary64 packing bound")
        if not wide_packing and not -1023 <= exponent <= 1074:
            raise ValueError("exact binary64 packing scale is outside its representable exponent range")

        def scalar(args, emit):
            value = args[0]
            if not wide_packing:
                if value.type != f64:
                    value = emit(arith.ExtFOp(value, f64))
                scale = emit(arith.ConstantOp(FloatAttr(float(Fraction(2) ** (-exponent)), f64)))
                scaled = emit(arith.MulfOp(value, scale))
                return emit(arith.FPToSIOp(scaled, i64))
            fraction_bits, exponent_bits, bias = FORMAT[str(value.type)]
            storage = IntegerType(1 + fraction_bits + exponent_bits)
            raw = emit(arith.BitcastOp(value, storage))
            raw = emit(arith.ExtUIOp(raw, i64))

            def constant(v):
                return emit(arith.ConstantOp(IntegerAttr(v, i64)))

            def binary(cls, a, b):
                return emit(cls(a, b))

            field = binary(
                arith.AndIOp, binary(arith.ShRUIOp, raw, constant(fraction_bits)), constant((1 << exponent_bits) - 1)
            )
            normal = emit(arith.CmpiOp(field, constant(0), "ne"))
            significand = binary(arith.AndIOp, raw, constant((1 << fraction_bits) - 1))
            implicit = emit(arith.SelectOp(normal, constant(1 << fraction_bits), constant(0)))
            significand = binary(arith.OrIOp, significand, implicit)
            effective = emit(arith.SelectOp(normal, field, constant(1)))
            shift = binary(arith.SubiOp, effective, constant(bias + fraction_bits + exponent))
            nonnegative = emit(arith.CmpiOp(shift, constant(0), "sge"))
            left = emit(arith.SelectOp(nonnegative, shift, constant(0)))
            right = emit(arith.SelectOp(nonnegative, constant(0), binary(arith.SubiOp, constant(0), shift)))
            magnitude = binary(arith.ShRUIOp, binary(arith.ShLIOp, significand, left), right)
            negative = emit(
                arith.CmpiOp(
                    binary(arith.AndIOp, raw, constant(1 << (fraction_bits + exponent_bits))), constant(0), "ne"
                )
            )
            return emit(arith.SelectOp(negative, binary(arith.SubiOp, constant(0), magnitude), magnitude))

        result = point([value], TensorType(i64, value.type.get_shape()), scalar, operations)
        bounds[result] = aligned
        record_zero_limbs(result, value, exponent)
        return result, aligned

    def zero(shape, operations):
        constant = arith.ConstantOp(IntegerAttr(0, i128))
        seed = tensor.SplatOp(constant.result, [], TensorType(i128, shape))
        operations.extend([constant, seed])
        bounds[seed.result] = 0
        return seed.result

    def reshape(value, shape, operations):
        literal = arith.ConstantOp(
            DenseIntOrFPElementsAttr.from_list(TensorType(IntegerType(64), [len(shape)]), list(shape))
        )
        changed = tensor.ReshapeOp(value, literal.result, TensorType(value.type.element_type, shape))
        operations.extend([literal, changed])
        if value in bounds:
            bounds[changed.result] = bounds[value]
        if zero_limbs is not None and value in zero_limbs:
            zero_limbs[changed.result] = zero_limbs[value]
        return changed.result

    def publish(op, integer, exponent, integer_bound, operations, compute_type=None, denominator=1):
        compute_type = compute_type or op.results[0].type.element_type
        output_type = op.results[0].type.element_type

        def convert(values, emit):
            rounded = encode_fixed_integer(values[0], exponent, compute_type, emit, denominator=denominator)
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
        if (
            op in contractions
            or any(it.data == L.IteratorType.REDUCTION for it in op.get_iterator_types())
            or any(child.name == "scf.for" for child in op.body.walk())
        ):
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
        max_exponent = (2 * integer_bound).bit_length() - 1 + exponent if integer_bound else 0
        grids[result] = (
            (exponent, 2 * integer_bound)
            if denominator == 1
            and (2 * integer_bound).bit_length() <= packing_limit
            and max_exponent < FORMAT[str(output_type)][2]
            else None
        )
        raw_forms[result] = (integer, exponent, integer_bound, denominator)
        tag = getattr(integer.owner.attributes.get("merlin.exact_component"), "data", None)
        if tag:
            components[result] = tag
        proofs.append(
            dict(
                source_operation=op.name,
                source_data_arithmetic=any(
                    child.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf"}
                    for child in op.body.walk()
                ),
                common_exponent=exponent,
                integer_absolute_bound=integer_bound,
                reconstruction=f"bounded_i{i128.width.data}",
                packing="exact_ieee_bit_fields" if wide_packing else "exact_binary64_power_of_two",
                rounding="integer_quotient_remainder_rne",
                exact_zero_sign="source_ieee_sign_bits",
            )
        )
        Rewriter.replace_op(op, operations, [result])

    from merlin.kernels.shapes import observe_contractions

    contractions = dict(observe_contractions(module))
    for op in list(function.body.block.ops):
        if op.name == "linalg.generic" and len(op.results) == 1 and floating(op.results[0]):
            from xdsl.dialects.builtin import StringAttr

            from merlin.llvmlower.additive_regions import data_sources, retype_additive

            sources = data_sources(op, lambda typ: str(typ) in FORMAT)
            has_reduction = any(it.data == L.IteratorType.REDUCTION for it in op.get_iterator_types()) or any(
                child.name == "scf.for" for child in op.body.walk()
            )
            source_grids = [grid(v) for v in sources]
            seed_grid = grid(op.outputs[0]) if op.body.block.args[-1].uses else (0, 0)
            if has_reduction and sources and all(g is not None for g in source_grids) and seed_grid == (0, 0):
                exponent = min((e for e, b in source_grids if b), default=0)
                operations = []
                try:
                    packed = [pack(value, exponent, operations) for value in sources]
                except ValueError:
                    packed = []
                if packed:
                    wide_packed = []
                    for value, magnitude in packed:

                        def extend(values, emit):
                            return emit(arith.ExtSIOp(values[0], i128))

                        wide = point([value], TensorType(i128, value.type.get_shape()), extend, operations)
                        bounds[wide] = magnitude
                        wide_packed.append(wide)
                    lifted = retype_additive(op, dict(zip(sources, wide_packed)), i128, floating=True)
                    if lifted is not None:
                        additive_ops, integer = lifted
                        # Qualification requires a full-domain signed-i32 proof
                        # for each emitted digit producer. That bounds this exact
                        # weighted host sum without assumptions about repeated
                        # gathers, loop counts, or cancellation in the source.
                        magnitude = max(b for v, b in packed)
                        digits, capacity = 1, 127
                        while magnitude > capacity:
                            digits += 1
                            capacity = 256 * capacity + 127
                        bound = (1 << 31) * sum(1 << (8 * j) for j in range(digits))
                        if bound.bit_length() < reconstruction_limit:
                            integer.attributes["merlin.exact_additive"] = StringAttr("closed_zero_seed_sum")
                            operations.extend(additive_ops)
                            bounds[integer.results[0]] = bound
                            publish(op, integer.results[0], exponent, bound, operations)
                            continue
        if (
            allow_reassociation
            and not wide_packing
            and op.name == "linalg.generic"
            and len(op.results) == 1
            and floating(op.results[0])
            and any(child.name == "scf.for" for child in op.body.walk())
        ):
            from merlin.llvmlower.static_polynomial_components import lift_static_polynomial
            from merlin.llvmlower.static_polynomials import PolynomialRefusal

            try:
                operations, integer, exponent, bound = lift_static_polynomial(op, grid, pack, bounds)
            except (PolynomialRefusal, ValueError):
                pass
            else:
                result = reshape(integer, op.results[0].type.get_shape(), operations)
                publish(op, result, exponent, bound, operations)
                continue
        if op in contractions and len(op.operands) == 3 and all(floating(v) for v in op.operands):
            from merlin.llvmlower.additive_regions import positive_zero_seed

            operands = op.operands[:2]
            source_grids = [grid(v) for v in operands]
            seed_grid = grid(op.operands[2])
            if (
                any(g is None for g in source_grids)
                or seed_grid is None
                or seed_grid[1] != 0
                or not positive_zero_seed(op.operands[2])
            ):
                continue
            exponent = sum(e for e, b in source_grids)
            operations = []
            try:
                packed = [pack(value, e, operations) for value, (e, b) in zip(operands, source_grids, strict=True)]
            except ValueError:
                continue
            bound = prod(contractions[op].reduction) * prod(b for value, b in packed)
            if bound.bit_length() >= reconstruction_limit:
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
        denominator, divisor_exponent, divisor_negative = 1, 0, False
        if arithmetic.name == "arith.divf":
            from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

            literal = _constant_of(op.inputs[args[1]])
            if literal is None or not literal:
                continue
            divisor = Fraction.from_float(float(literal))
            divisor_negative = divisor < 0
            divisor = abs(divisor)
            divisor_exponent = -(divisor.denominator.bit_length() - 1)
            denominator = divisor.numerator
            while denominator % 2 == 0:
                divisor_exponent += 1
                denominator //= 2
            if denominator >= 1 << 31:
                continue
            args = args[:1]
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
        if arithmetic.name == "arith.divf":
            packing_exponents = [input_grids_here[0][0]]
            exponent = packing_exponents[0] - divisor_exponent
        operations = []
        try:
            packed = [pack(value, e, operations) for value, e in zip(input_values, packing_exponents, strict=True)]
        except ValueError:
            continue
        shape = op.results[0].type.get_shape()
        integer_bound = (
            prod(b for value, b in packed) if arithmetic.name == "arith.mulf" else sum(b for value, b in packed)
        )
        if integer_bound.bit_length() >= reconstruction_limit:
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
                if zero_limbs is not None and value in zero_limbs:
                    zero_limbs[carrier.results[0]] = zero_limbs[value]
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
        if arithmetic.name == "arith.negf" or (arithmetic.name == "arith.divf" and divisor_negative):
            constant = arith.ConstantOp(IntegerAttr(0, i128))
            body.add_op(constant)
            computed = arith.SubiOp(constant.result, values[0])
        elif arithmetic.name in {"arith.extf", "arith.truncf", "arith.divf"}:
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
        publish(
            op,
            integer.results[0],
            exponent,
            integer_bound,
            operations,
            arithmetic.results[0].type,
            denominator=denominator,
        )
    module.verify()
    return bounds, proofs
