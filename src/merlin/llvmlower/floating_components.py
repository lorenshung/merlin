"""Finite native floating sources as bounded exact integer component programs."""

from fractions import Fraction
from math import prod

from merlin.llvmlower.fixed_point_ir import FORMAT, finite_grid
from merlin.llvmlower.floating_component_ir import pack_ieee_integer, point
from merlin.llvmlower.floating_component_readout import component_readout
from merlin.llvmlower.ieee_pointwise import finite_or_zero, finite_values


def rewrite_floating_components(
    module,
    input_grids,
    *,
    complex_inputs=(),
    allow_reassociation=False,
    known_grids=None,
    input_values=None,
    zero_limbs=None,
    ieee_pointwise=False,
    quantized_contract=None,
):
    """Return new integer bounds and closed finite representation proofs.

    Packing uses a proved binary64 product or integer IEEE bit fields.
    Host recombination and RNE retain the complete exact integer numerator.
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
        f64,
        i64,
        i128,
    )
    from xdsl.dialects.linalg import ops as L
    from xdsl.ir import Block, Region
    from xdsl.ir.affine import AffineDimExpr, AffineMap

    from merlin.llvmlower.additive_regions import expose_reduce_domains
    from merlin.llvmlower.pointwise_primitives import expose_pointwise_primitives

    expose_reduce_domains(module)
    expose_pointwise_primitives(module)
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    grids = dict(zip(function.body.block.args, input_grids))
    grids.update(known_grids or {})
    packing_grids = [g for g in grids.values() if g is not None and g[1]]
    for op in function.walk():
        if op.name != "arith.constant":
            continue
        typ = op.results[0].type
        if str(getattr(typ, "element_type", typ)) not in FORMAT:
            continue
        attr = op.properties.get("value", op.attributes.get("value"))
        values = list(attr.get_values()) if hasattr(attr, "get_values") else [attr.value.data]
        constant_grid = finite_grid(finite_values(values) if ieee_pointwise else values)
        if constant_grid[1]:
            packing_grids.append(constant_grid)
    common_exponent = min((e for e, b in packing_grids), default=0)
    # A division can introduce the output format's smallest dyadic quantum,
    # even when the input itself has a short integral representation.
    aligned_bits = max(((b << (e - common_exponent)).bit_length() for e, b in packing_grids), default=0)
    wide_packing = aligned_bits > 52 or any(
        op.name == "arith.divf" and str(op.results[0].type) in {"bf16", "f32", "f64"} for op in function.walk()
    )
    if wide_packing:
        # A bilinear result may become a later linear producer's operand.
        # Size that operand representation for both significands and carries,
        # rather than proving the first GEMM and stranding its epilogue.
        required_bits = aligned_bits + 2
        if any(op.name in {"arith.mulf", "math.fma"} for op in function.walk()):
            required_bits = max(required_bits, 2 * aligned_bits + 32)
        packing_width = 1 << (max(512, required_bits) - 1).bit_length()
        i64, i128 = IntegerType(packing_width), IntegerType(2 * packing_width)
    packing_limit = i64.width.data - 1 if wide_packing else 52
    reconstruction_limit = i128.width.data - 8

    complex_sources = {function.body.block.args[i] for i in complex_inputs}
    components = {}
    bounds = {}
    proofs = []
    raw_forms = {}

    from merlin.llvmlower.floating_component_ir import payload_zero_limb_recorder

    record_zero_limbs = payload_zero_limb_recorder(
        function.body.block.args, input_values or (), zero_limbs, i64, wide_packing
    )

    def floating(value):
        return hasattr(value.type, "element_type") and str(value.type.element_type) in FORMAT

    def combine(items):
        items = [item for item in items if item is not None]
        if not items:
            return None
        exponent = min((e for e, b in items if b), default=0)
        bound = max((b * (1 << (e - exponent)) for e, b in items if b), default=0)
        return (exponent, bound) if bound.bit_length() <= packing_limit else None

    from merlin.llvmlower.floating_component_ir import component_taggers

    product_component, linear_component = component_taggers(components, complex_sources)

    def scalar_constant(value):
        """Interpret closed scalar coefficient conversions at their typed RNE boundaries."""
        from merlin.targetgen.exact_arithmetic import round_binary

        owner = value.owner
        name = getattr(owner, "name", None)
        if name == "arith.constant":
            attribute = owner.properties.get("value", owner.attributes.get("value"))
            return attribute.value.data if isinstance(attribute, (FloatAttr, IntegerAttr)) else None
        if name == "arith.select":
            condition = scalar_constant(owner.operands[0])
            return scalar_constant(owner.operands[1 if condition else 2]) if condition is not None else None
        if name == "arith.cmpf" and owner.predicate.value.data in (7, 14):
            from math import isnan

            values = [scalar_constant(operand) for operand in owner.operands]
            if any(value is None for value in values):
                return None
            unordered = any(isnan(value) for value in values)
            return unordered if owner.predicate.value.data == 14 else not unordered
        if name not in {"arith.extf", "arith.truncf", "arith.sitofp", "arith.uitofp"}:
            return None
        original = scalar_constant(owner.operands[0])
        if original is None or str(value.type) not in FORMAT:
            return None
        if name == "arith.uitofp":
            original = int(original) & ((1 << owner.operands[0].type.width.data) - 1)
        if str(value.type) == "bf16":
            return round_binary(original, 16, bfloat16=True)
        return round_binary(original, {"f16": 16, "f32": 32, "f64": 64}[str(value.type)])

    def grid(value):
        if allow_reassociation and value in raw_forms and raw_forms[value][3] == 1:
            return raw_forms[value][1:3]
        if value in grids:
            return grids[value]
        op = value.owner
        if getattr(op, "name", None) == "arith.constant":
            attr = op.properties.get("value", op.attributes.get("value"))
            values = list(attr.get_values()) if hasattr(attr, "get_values") else [attr.value.data]
            try:
                return finite_grid(finite_values(values) if ieee_pointwise else values)
            except (ValueError, OverflowError):
                return None
        if not hasattr(value.type, "element_type"):
            try:
                literal = scalar_constant(value)
                return finite_grid([literal]) if literal is not None else None
            except (ValueError, OverflowError):
                return None
        carriers = {
            "tensor.splat",
            "tensor.concat",
            "tensor.extract_slice",
            "tensor.insert_slice",
            "tensor.collapse_shape",
            "tensor.expand_shape",
            "tensor.cast",
            "tensor.reshape",
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
        if getattr(op, "name", None) == "linalg.generic":
            children = list(op.body.block.ops)
            if len(children) == 2 and children[0].name == "arith.select":
                select = children[0]
                args = op.body.block.args
                if all(value in args[: len(op.inputs)] for value in select.operands):
                    condition, left, right = [op.inputs[args.index(value)] for value in select.operands]
                    predicate = condition.owner
                    if getattr(predicate, "name", None) == "linalg.generic":
                        body = list(predicate.body.block.ops)
                        if (
                            len(body) == 2
                            and body[0].name == "arith.cmpf"
                            and body[0].predicate.value.data == 14
                            and all(grid(value) is not None for value in predicate.inputs)
                        ):
                            return grid(right)
                    candidates = [grid(left), grid(right)]
                    return combine(candidates) if all(g is not None for g in candidates) else None
        return None

    def pack(value, exponent, operations):
        if allow_reassociation and value in raw_forms and raw_forms[value][3] == 1:
            integer, source_exponent, bound, divisor = raw_forms[value]
            if divisor != 1 or (bound and exponent > source_exponent):
                raise ValueError("raw rational component cannot be packed as an integer")
            aligned = bound << (source_exponent - exponent) if bound else 0
            if aligned.bit_length() >= i64.width.data - 1:
                raise ValueError("raw exact component exceeds the signed-i64 operand representation")

            def scalar(values, emit):
                result = values[0]
                if bound and source_exponent != exponent:
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

        from merlin.llvmlower.component_carriers import carry, data_operand_indices

        def contains_raw(source):
            if source in raw_forms:
                return raw_forms[source][3] == 1
            owner = source.owner
            indices = data_operand_indices(owner) if hasattr(owner, "name") else []
            return any(contains_raw(owner.operands[i]) for i in indices)

        indices = data_operand_indices(value.owner) if hasattr(value.owner, "name") else []
        if allow_reassociation and indices and contains_raw(value):
            packed = [pack(value.owner.operands[i], exponent, operations)[0] for i in indices]
            result = carry(value, packed, i64, operations)
            bounds[result] = aligned
            return result, aligned

        def scalar(args, emit):
            value = args[0]
            if ieee_pointwise:
                value = finite_or_zero(value, emit)
            if not wide_packing:
                if value.type != f64:
                    value = emit(arith.ExtFOp(value, f64))
                scale = emit(arith.ConstantOp(FloatAttr(float(Fraction(2) ** (-exponent)), f64)))
                scaled = emit(arith.MulfOp(value, scale))
                return emit(arith.FPToSIOp(scaled, i64))
            return pack_ieee_integer(value, exponent, i64, emit)

        result = point([value], TensorType(i64, value.type.get_shape()), scalar, operations)
        bounds[result] = aligned
        if not ieee_pointwise:
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

    from merlin.kernels.shapes import observe_contractions

    contractions = dict(observe_contractions(module))
    publish = component_readout(
        point=point,
        integer_type=i128,
        contractions=contractions,
        grids=grids,
        raw_forms=raw_forms,
        components=components,
        proofs=proofs,
        packing_limit=packing_limit,
        wide_packing=wide_packing,
        source_grid=grid,
        ieee_pointwise=ieee_pointwise,
    )
    if quantized_contract is not None:

        def publish(op, integer, exponent, integer_bound, operations, compute_type=None, denominator=1):
            from xdsl.rewriter import Rewriter

            from merlin.llvmlower.quantized_native import output_boundary

            result = output_boundary(
                integer, exponent, integer_bound, denominator, op.results[0].type, operations, quantized_contract
            )
            raw_forms[result] = (integer, exponent, integer_bound, denominator)
            grids[result] = (quantized_contract["exponent"], 128)
            proofs.append(
                dict(
                    source_operation=op.name,
                    rounding="output_quantize_rne",
                    integer_absolute_bound=integer_bound,
                    exponent=exponent,
                    denominator=denominator,
                )
            )
            Rewriter.replace_op(op, operations, [result])

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
                        # Each complete digit producer has a signed-i32 proof;
                        # its bound covers repeated gathers and every loop term.
                        magnitude = max(b for v, b in packed)
                        digits, capacity = 1, 127
                        while magnitude > capacity:
                            digits += 1
                            capacity = 256 * capacity + 127
                        bound = (1 << 31) * sum(1 << (8 * j) for j in range(digits))
                        from merlin.llvmlower.static_polynomials import PolynomialRefusal, analyze_static_polynomial

                        try:
                            atoms, rows = analyze_static_polynomial(op, require_bilinear=False)
                            magnitudes = dict(zip(sources, (b for _, b in packed), strict=True))
                            indexed_bound = max(
                                (
                                    sum(abs(c) * prod(magnitudes[atoms[i][0]] for i in key) for key, c in row.items())
                                    for row in rows
                                ),
                                default=0,
                            )
                            if Fraction(indexed_bound).denominator != 1:
                                raise PolynomialRefusal("additive component bound is not integral")
                            bound = min(bound, int(indexed_bound))
                        except (PolynomialRefusal, KeyError):
                            pass
                        if bound.bit_length() < reconstruction_limit:
                            integer.attributes["merlin.exact_additive"] = StringAttr("closed_zero_seed_sum")
                            operations.extend(additive_ops)
                            bounds[integer.results[0]] = bound
                            publish(op, integer.results[0], exponent, bound, operations)
                            continue
        if (
            allow_reassociation
            and op.name == "linalg.generic"
            and len(op.results) == 1
            and floating(op.results[0])
            and op not in contractions
            and (
                any(child.name == "scf.for" for child in op.body.walk())
                or any(it.data == L.IteratorType.REDUCTION for it in op.get_iterator_types())
            )
        ):
            from merlin.llvmlower.static_polynomial_components import lift_static_polynomial
            from merlin.llvmlower.static_polynomials import PolynomialRefusal

            try:
                operations, integer, exponent, bound = lift_static_polynomial(
                    op, grid, pack, bounds, operand_type=i64, result_type=i128
                )
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
            if child.name in {"arith.addf", "arith.subf", "arith.mulf", "arith.divf", "arith.negf", "math.fma"}
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
                "math.fma",
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
        if arithmetic.name == "math.fma":
            product_exponent = sum(e for e, _ in input_grids_here[:2])
            exponent = min(product_exponent, input_grids_here[2][0])
            packing_exponents = [e for e, _ in input_grids_here[:2]] + [exponent]
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
        if arithmetic.name == "math.fma":
            integer_bound = ((packed[0][1] * packed[1][1]) << (product_exponent - exponent)) + packed[2][1]
        if integer_bound.bit_length() >= reconstruction_limit:
            continue
        seed = zero(shape, operations)
        if arithmetic.name in {"arith.mulf", "math.fma"}:
            # A K=1 batched contraction represents general pointwise products.
            # Broadcast the source operands into the original output map first.
            expanded = []
            for (value, bound), arg in zip(packed[:2], args[:2], strict=True):
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
            if arithmetic.name == "math.fma":
                empty = tensor.EmptyOp([], TensorType(i128, shape))
                body = Block(arg_types=[i128, i64, i128])
                product = body.args[0]
                if product_exponent != exponent:
                    shift = arith.ConstantOp(IntegerAttr(product_exponent - exponent, i128))
                    shifted = arith.ShLIOp(product, shift.result)
                    body.add_ops([shift, shifted])
                    product = shifted.result
                extended = arith.ExtSIOp(body.args[1], i128)
                added = arith.AddiOp(product, extended.result)
                body.add_ops([extended, added, L.YieldOp(added.result)])
                maps = list(op.get_indexing_maps())
                final = L.GenericOp(
                    inputs=[result, packed[2][0]],
                    outputs=[empty.tensor],
                    body=Region(body),
                    indexing_maps=ArrayAttr([maps[-1], maps[args[2]], maps[-1]]),
                    iterator_types=op.get_iterator_types(),
                    result_types=[empty.tensor.type],
                )
                operations.extend([empty, final])
                result = final.results[0]
                bounds[result] = integer_bound
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
