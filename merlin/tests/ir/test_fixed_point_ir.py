"""Compare emitted integer rounding IR with independent nearest-IEEE search."""

import random
from fractions import Fraction

import pytest
from xdsl.dialects.builtin import IntegerType, bf16, f16, f32, i128
from xdsl.ir import Block

from merlin.llvmlower.fixed_point_ir import FORMAT, encode_fixed_integer, finite_grid


def reference_bits(numerator, exponent, denominator, token):
    fraction_bits, exponent_bits, bias = FORMAT[token]
    sign = int(numerator < 0) << (fraction_bits + exponent_bits)
    value = abs(Fraction(numerator, denominator) * Fraction(2) ** exponent)
    maximum = (((1 << exponent_bits) - 1) << fraction_bits) - 1

    def decode(raw):
        e, f = raw >> fraction_bits, raw & ((1 << fraction_bits) - 1)
        return Fraction(f if e == 0 else (1 << fraction_bits) + f) * Fraction(2) ** (
            (1 - bias if e == 0 else e - bias) - fraction_bits
        )

    threshold = Fraction(2) ** bias * (2 - Fraction(1, 1 << (fraction_bits + 1)))
    if value >= threshold:
        return sign | (((1 << exponent_bits) - 1) << fraction_bits)
    low, high = 0, maximum
    while low < high:
        mid = (low + high) // 2
        if decode(mid) < value:
            low = mid + 1
        else:
            high = mid
    candidate = min({low, max(0, low - 1)}, key=lambda raw: (abs(decode(raw) - value), raw & 1))
    return sign | candidate


def emitted_bits(numerator, exponent, denominator, dtype):
    block = Block(arg_types=[i128])
    operations = []

    def emit(op):
        operations.append(op)
        return op.results[0]

    result = encode_fixed_integer(block.args[0], exponent, dtype, emit, denominator=denominator)
    values = {block.args[0]: numerator & ((1 << 128) - 1)}

    def signed(value, typ):
        width = typ.width.data
        return value - (1 << width) if value & (1 << (width - 1)) else value

    for op in operations:
        args = [values[v] for v in op.operands]
        name = op.name
        if name == "arith.constant":
            value = op.value.value.data
        elif name == "arith.select":
            value = args[1] if args[0] else args[2]
        elif name == "arith.cmpi":
            predicate = op.predicate.value.data
            left, right = args
            if predicate in (2, 3, 4, 5):
                left, right = [signed(v, operand.type) for v, operand in zip(args, op.operands)]
            value = int(
                [
                    left == right,
                    left != right,
                    left < right,
                    left <= right,
                    left > right,
                    left >= right,
                    left < right,
                    left <= right,
                    left > right,
                    left >= right,
                ][predicate]
            )
        elif name in {"arith.trunci", "arith.bitcast"}:
            value = args[0]
        else:
            left, right = args
            if name in {"arith.shli", "arith.shrui"}:
                assert 0 <= right < op.operands[0].type.width.data, (name, right)
            value = {
                "arith.addi": lambda: left + right,
                "arith.subi": lambda: left - right,
                "arith.andi": lambda: left & right,
                "arith.ori": lambda: left | right,
                "arith.shli": lambda: left << right,
                "arith.shrui": lambda: left >> right,
                "arith.divui": lambda: left // right,
                "arith.remui": lambda: left % right,
            }[name]()
        typ = op.results[0].type
        if isinstance(typ, IntegerType):
            value &= (1 << typ.width.data) - 1
        values[op.results[0]] = value
    return values[result]


@pytest.mark.parametrize("dtype", [f16, bf16, f32])
def test_emitted_rne_matches_nearest_ieee_search(dtype):
    rng = random.Random(1767)
    samples = [
        (n, e, d)
        for n in (0, 1, -1, 3, -3, (1 << 119) - 1, -(1 << 119) + 1)
        for e in (-300, -149, -24, -1, 0, 128)
        for d in (1, 3, 7)
    ]
    samples += [
        (rng.randrange(-(1 << 119), 1 << 119), rng.randrange(-300, 130), rng.choice([1, 3, 17, 65537]))
        for _ in range(200)
    ]
    for numerator, exponent, denominator in samples:
        assert emitted_bits(numerator, exponent, denominator, dtype) == reference_bits(
            numerator, exponent, denominator, str(dtype)
        ), (numerator, exponent, denominator, dtype)


def test_complete_dyadic_grid_refuses_nonfinite_or_unbounded_alignment():
    assert finite_grid([0.0, -0.0, 0.5, 1.25]) == (-2, 5)
    assert finite_grid([256.0, -512.0]) == (8, 2)
    with pytest.raises(ValueError, match="nonfinite"):
        finite_grid([1.0, float("inf")])
    with pytest.raises(ValueError, match="52-bit"):
        finite_grid([1.0, 2.0**-100])
