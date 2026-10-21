"""Exact host preparation for finite IEEE component producers."""

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
