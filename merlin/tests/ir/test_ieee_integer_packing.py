"""Execute every packing shift, including zeros unconstrained by the grid."""

import struct
import sys
from fractions import Fraction

import pytest
from xdsl.dialects.builtin import IntegerType

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.floating_component_ir import pack_ieee_integer


@pytest.mark.parametrize(
    "dtype,width,exponent,values",
    [
        ("f64", 512, 0, [0.0, -0.0, 1.0, -1.0, 2.0**63, -(2.0**400)]),
        ("f64", 512, 400, [0.0, -0.0, 2.0**400, -(2.0**800)]),
        ("f64", 512, 971, [0.0, -0.0, sys.float_info.max, -sys.float_info.max]),
        ("f64", 1024, 0, [0.0, -0.0]),
        ("f64", 2048, 0, [0.0, -0.0, sys.float_info.max]),
        ("f32", 512, -149, [0.0, -0.0, 2.0**-149, (2.0 - 2.0**-23) * 2.0**127]),
    ],
)
def test_exact_packing_has_no_out_of_range_shift(dtype, width, exponent, values):
    module = parse_mlir_text(f"module {{ func.func @f(%x: {dtype}) {{ return }} }}")
    argument = next(op for op in module.walk() if op.name == "func.func").body.block.args[0]
    operations = []

    def emit(op):
        operations.append(op)
        return op.results[0]

    result = pack_ieee_integer(argument, exponent, IntegerType(width), emit)
    for value in values:
        raw = int.from_bytes(struct.pack("<d" if dtype == "f64" else "<f", value), "little")
        data = {argument: raw}
        for op in operations:
            args = [data[v] for v in op.operands]
            bits = op.results[0].type.width.data
            mask = (1 << bits) - 1
            if op.name == "arith.constant":
                answer = op.value.value.data
            elif op.name in {"arith.bitcast", "arith.extui"}:
                answer = args[0]
            elif op.name == "arith.cmpi":
                predicate = op.predicate.value.data
                operand_width = op.operands[0].type.width.data
                signed = [a - (1 << operand_width) if a >> (operand_width - 1) else a for a in args]
                assert predicate in {0, 1, 5}
                answer = (
                    args[0] == args[1]
                    if predicate == 0
                    else args[0] != args[1]
                    if predicate == 1
                    else signed[0] >= signed[1]
                )
            elif op.name == "arith.select":
                answer = args[1] if args[0] else args[2]
            elif op.name in {"arith.shli", "arith.shrui"}:
                assert 0 <= args[1] < bits, (op.name, args[1], bits, value)
                answer = args[0] << args[1] if op.name == "arith.shli" else args[0] >> args[1]
            elif op.name == "arith.andi":
                answer = args[0] & args[1]
            elif op.name == "arith.ori":
                answer = args[0] | args[1]
            elif op.name == "arith.subi":
                answer = args[0] - args[1]
            else:
                raise AssertionError(op.name)
            data[op.results[0]] = int(answer) & mask
        actual = data[result]
        if actual >> (width - 1):
            actual -= 1 << width
        assert actual == Fraction.from_float(value) / Fraction(2) ** exponent
