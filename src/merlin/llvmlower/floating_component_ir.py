"""Exact host preparation for finite IEEE component producers."""

from fractions import Fraction

from xdsl.dialects import arith, tensor
from xdsl.dialects.builtin import AffineMapAttr, ArrayAttr, IntegerAttr, IntegerType, UnitAttr
from xdsl.dialects.linalg import ops as L
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from merlin.llvmlower.fixed_point_ir import FORMAT


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


def payload_zero_limb_recorder(arguments, input_values, zero_limbs, integer_type, wide_packing):
    """Prove zero balanced digits using complete immutable source payloads only."""
    i64 = integer_type
    payloads = dict(zip(arguments, input_values))

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

    return record_zero_limbs


def pack_ieee_integer(value, exponent, integer_type, emit):
    """Decode finite IEEE bits to a proved integral common-exponent coefficient."""
    i64 = integer_type
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
    # Zero has no grid constraint. Its decoded exponent can be far outside
    # the chosen coefficient width, where even shifting zero is poison.
    is_zero = emit(arith.CmpiOp(significand, constant(0), "eq"))
    left = emit(arith.SelectOp(is_zero, constant(0), left))
    right = emit(arith.SelectOp(is_zero, constant(0), right))
    magnitude = binary(arith.ShRUIOp, binary(arith.ShLIOp, significand, left), right)
    negative = emit(
        arith.CmpiOp(binary(arith.AndIOp, raw, constant(1 << (fraction_bits + exponent_bits))), constant(0), "ne")
    )
    return emit(arith.SelectOp(negative, binary(arith.SubiOp, constant(0), magnitude), magnitude))


def component_taggers(components, complex_sources):
    """Follow closed real/imaginary selectors and Gaussian arithmetic components."""

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

    return product_component, linear_component
