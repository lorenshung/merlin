"""Full-domain certificates, semantic refusals and executable readout arithmetic."""

import json
import random
from fractions import Fraction

import numpy as np
import pytest

from merlin.llvmlower.exact_readout import lower_exact_readouts
from merlin.llvmlower.exact_readout_source import Scalar, build_readout


def fixture(scales, *, size=513, floating=False, bias=False, **kwargs):
    import math

    from xdsl.dialects import func
    from xdsl.dialects.builtin import (
        ArrayAttr,
        IntegerAttr,
        ModuleOp,
        StringAttr,
        TensorType,
        UnitAttr,
        f32,
        i32,
        i64,
    )
    from xdsl.ir import Block, Region

    denominator = math.lcm(*(s.denominator // (s.denominator & -s.denominator) for s in scales))
    output_scale = None if floating else float(denominator)
    block = Block(arg_types=[TensorType(i32, [size, len(scales)]), *([TensorType(f32, [len(scales)])] if bias else [])])
    ops, result = build_readout(
        block.args[0],
        [s * Fraction(output_scale or 1) for s in scales],
        bias=block.args[1] if bias else None,
        output_scale=output_scale,
        **kwargs,
    )
    reader = result.owner
    reader.attributes.update(
        {
            "prov.role": StringAttr("readout"),
            "prov.quant.arithmetic": StringAttr("exact"),
            "prov.quant.scale_numerator": ArrayAttr([StringAttr(str(s.numerator)) for s in scales]),
            "prov.quant.scale_denominator": ArrayAttr([StringAttr(str(s.denominator)) for s in scales]),
        }
    )
    if not floating:
        from xdsl.dialects.builtin import FloatAttr, f64

        reader.attributes["prov.quant.output_scale"] = FloatAttr(output_scale, f64)
        for k, default in (("zero_point", 0), ("quant_min", -128), ("quant_max", 127)):
            reader.attributes[f"prov.quant.{k}"] = IntegerAttr(kwargs.get(k, default), i64)
        reader.attributes["prov.quant.relu"] = StringAttr(str(kwargs.get("relu", False)).lower())
        reader.attributes["prov.quant.storage_dtype"] = StringAttr("uint8" if kwargs.get("unsigned") else "int8")
    block.add_ops([*ops, func.ReturnOp(result)])
    fn = func.FuncOp("forward", ([v.type for v in block.args], [result.type]), Region(block))
    fn.attributes["llvm.emit_c_interface"] = UnitAttr()
    return ModuleOp([fn]), reader


def scalar(block, arguments, external=None):
    values = {**(external or {}), **dict(zip(block.args, arguments))}

    def signed(v, typ):
        w = typ.width.data
        return v - (1 << w) if v & (1 << (w - 1)) else v

    for op in block.ops:
        a = [values[v] for v in op.operands]
        name = op.name
        if name == "builtin.unregistered":
            name = op.op_name.data
        if name in {"linalg.yield", "scf.yield"}:
            return a[0]
        if name == "arith.constant" and hasattr(op.value, "get_values"):
            v = np.array(list(op.value.get_values()), dtype=object).reshape(op.results[0].type.get_shape())
        elif name == "tensor.extract":
            v = a[0][tuple(a[1:])]
        elif name == "arith.index_cast":
            v = a[0]
        elif name == "scf.if":
            v = scalar(op.regions[0 if a[0] else 1].block, [], values)
        elif name == "arith.constant":
            v = op.value.value.data
            if str(op.results[0].type) == "f32":
                import struct

                v = int.from_bytes(struct.pack("<f", v), "little")
        elif name == "arith.select":
            v = a[1] if a[0] else a[2]
        elif name in {"arith.trunci", "arith.extui", "arith.bitcast"}:
            v = a[0]
        elif name == "arith.uitofp":
            import struct

            from merlin.targetgen.exact_arithmetic import round_binary

            v = int.from_bytes(struct.pack("<f", round_binary(Fraction(a[0]))), "little")
        elif name == "arith.mulf":
            import struct

            from merlin.targetgen.exact_arithmetic import round_binary

            floats = [struct.unpack("<f", int(bits).to_bytes(4, "little"))[0] for bits in a]
            v = int.from_bytes(struct.pack("<f", round_binary(Fraction(floats[0]) * Fraction(floats[1]))), "little")
        elif name == "arith.extsi":
            v = signed(a[0], op.operands[0].type)
        elif name in {"llvm.call_intrinsic", "llvm.intr.ctlz"}:
            v = op.operands[0].type.width.data - a[0].bit_length()
        elif name == "cf.assert":
            assert a[0]
            continue
        elif name == "arith.cmpi":
            p = op.predicate.value.data
            x, y = a
            if 2 <= p <= 5:
                x, y = [signed(v, o.type) for v, o in zip(a, op.operands)]
            v = int([x == y, x != y, x < y, x <= y, x > y, x >= y, x < y, x <= y, x > y, x >= y][p])
        else:
            x, y = a
            if name in {"arith.shli", "arith.shrui"}:
                assert 0 <= y < op.operands[0].type.width.data
            v = {
                "arith.addi": lambda: x + y,
                "arith.subi": lambda: x - y,
                "arith.muli": lambda: x * y,
                "arith.xori": lambda: x ^ y,
                "arith.andi": lambda: x & y,
                "arith.ori": lambda: x | y,
                "arith.shrui": lambda: x >> y,
                "arith.shrsi": lambda: signed(x, op.operands[0].type) >> y,
                "arith.shli": lambda: x << y,
                "arith.maxsi": lambda: max(signed(x, op.operands[0].type), signed(y, op.operands[1].type)),
                "arith.minsi": lambda: min(signed(x, op.operands[0].type), signed(y, op.operands[1].type)),
            }[name]()
        if hasattr(op.results[0].type, "width"):
            v &= (1 << op.results[0].type.width.data) - 1
        values[op.results[0]] = v
    raise AssertionError("no yield")


@pytest.mark.parametrize(
    "scale",
    [
        Fraction(1, 2),
        Fraction(-1, 2),
        Fraction(1, 3),
        Fraction(123456789, 987654321),
        Fraction(0.03) * Fraction(0.02) / Fraction(0.01),
        Fraction(2) ** -100,
        Fraction(2) ** 60,
    ],
)
@pytest.mark.parametrize("zp,lo,hi,relu", [(0, -128, 127, False), (7, -127, 127, True), (131, 0, 255, False)])
def test_emitted_requantizer_exact_at_extremes_ties_and_random(scale, zp, lo, hi, relu):
    m, r = fixture([scale], zero_point=zp, quant_min=lo, quant_max=hi, relu=relu, unsigned=lo == 0)
    changed = lower_exact_readouts(m)
    p = json.loads(r.attributes["merlin.exact_readout_proof"].data)
    if p["method"] == "source_exact":
        assert scale in (Fraction(2) ** -100, Fraction(2) ** 60)
        return
    assert changed == 1
    m.verify()
    rng = random.Random(421)
    points = {-(1 << 31), (1 << 31) - 1, 0, -1, 1}
    points.update(rng.randrange(-(1 << 31), 1 << 31) for _ in range(200))
    for k in range(lo - zp - 1, hi - zp + 2):
        x = int(Fraction(2 * k + 1, 2) / scale)
        points.update(range(x - 2, x + 3))
    for v in points:
        if not -(1 << 31) <= v < 1 << 31:
            continue
        code = round(v * scale) + zp
        if relu:
            code = max(zp, code)
        assert scalar(r.body.block, [v & 0xFFFFFFFF, 0]) == min(hi, max(lo, code)) & 255


def test_complete_small_numerator_domain():
    scale = Fraction(23, 98)
    from merlin.llvmlower.requant_a import records

    params = dict(zero_point=0, quant_min=-128, quant_max=127, relu=False)
    p = records([scale], [Fraction(0)], params)
    assert p[0]["schema"] == "requant-A-v1"
    b = Scalar()
    from xdsl.dialects.builtin import i8, i32
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.ir import Block

    from merlin.llvmlower.requant_a import emit

    block = Block(arg_types=[i32, i8])
    result = emit(b, block.args[0], p)
    block.add_ops([*b.ops, YieldOp(result)])
    for v in range(-32768, 32769):
        assert scalar(block, [v & 0xFFFFFFFF, 0]) == min(127, max(-128, round(v * scale))) & 255


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize(
    "scale",
    [
        Fraction(1, 2),
        Fraction(-0.03) * Fraction(0.02),
        Fraction(float(np.float32(-0.03))) * Fraction(float(np.float32(0.02))),
        Fraction(2) ** -149,
        Fraction(2) ** 110,
    ],
)
def test_float_readout_bits(scale, bias):
    import struct

    from merlin.tests.ir.test_fixed_point_ir import reference_bits

    m, r = fixture([scale], floating=True, bias=bias)
    assert lower_exact_readouts(m) == 1
    m.verify()
    rng = random.Random(731)
    inputs = [-(1 << 31), (1 << 31) - 1, 0, -1, 1] + [rng.randrange(-(1 << 31), 1 << 31) for _ in range(100)]
    for v in inputs:
        bs = (
            rng.choice([0, 1, 0x7F7FFFFF, 0x80000000, 0x00800000, 0x80000001, 0xBF800000, rng.randrange(0x7F800000)])
            if bias
            else 0
        )
        bv = struct.unpack("<f", bs.to_bytes(4, "little"))[0]
        exact = v * scale + Fraction(bv)
        expected = reference_bits(exact.numerator, 0, exact.denominator, "f32")
        args = [v & 0xFFFFFFFF, *([bs] if bias else []), 0]
        assert scalar(r.body.block, args) == expected, (v, bs, scale)


def test_source_mutation_keeps_original_exact_body():
    from xdsl.dialects.builtin import IntegerAttr

    m, r = fixture([Fraction(1, 2)])
    first = next(o for o in r.body.block.ops if o.name == "arith.constant")
    first.properties["value"] = IntegerAttr(3, first.result.type)
    original = tuple(r.body.block.ops)
    assert lower_exact_readouts(m) == 0
    assert tuple(r.body.block.ops) == original
    assert "differs" in json.loads(r.attributes["merlin.exact_readout_proof"].data)["reason"]


@pytest.mark.parametrize("floating,bias", [(False, False), (True, False), (True, True)])
def test_native_readout_against_rational_reference(tmp_path, floating, bias):
    import hashlib
    import subprocess

    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang
    from merlin.targetgen.exact_arithmetic import round_binary
    from merlin.xdsl_dialects._common import text

    scales = [
        Fraction(float(np.float32(0.03))) * Fraction(float(np.float32(0.02))),
        Fraction(float(np.float32(-0.017))),
        Fraction(1, 8),
    ]
    if not floating:
        scales = [s / Fraction(0.01) for s in scales]
    module, reader = fixture(scales, floating=floating, bias=bias)
    assert lower_exact_readouts(module) == 1
    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path / "lower", vectorize=False)
    src = tmp_path / "readout.ll"
    src.write_text(llvm)
    obj = tmp_path / "readout.o"
    lib = tmp_path / f"readout_{hashlib.sha256(llvm.encode()).hexdigest()}.so"
    subprocess.run([str(clang()), "-O2", "-fPIC", "-c", str(src), "-o", str(obj)], check=True, capture_output=True)
    subprocess.run(
        ["cc", "-fPIC", "-shared", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(lib)],
        check=True,
        capture_output=True,
    )
    rng = np.random.default_rng(141)
    data = rng.integers(-(1 << 31), 1 << 31, (513, 3), dtype=np.int32)
    data[:7] = np.array([-(1 << 31), (1 << 31) - 1, -1, 0, 1, 2, 3], dtype=np.int32)[:, None]
    biases = np.array([0.1, 1e-40, -1e20], dtype=np.float32)
    expected = np.empty(data.shape, dtype=np.float32 if floating else np.int8)
    for ix in np.ndindex(data.shape):
        exact = int(data[ix]) * scales[ix[-1]] + (Fraction(float(biases[ix[-1]])) if bias else 0)
        expected[ix] = round_binary(exact) if floating else min(127, max(-128, round(exact)))
    actual = np.zeros_like(expected)
    arrays = [data, *([biases] if bias else []), actual]
    HostModel.load(str(lib))([(x.ctypes.data, x.shape) for x in arrays])
    assert actual.tobytes() == expected.tobytes()


@pytest.mark.parametrize("floating", [False, True])
def test_captured_trace_annotations_do_not_change_scalar_semantics(floating):
    from xdsl.dialects.builtin import ArrayAttr, StringAttr

    module, reader = fixture([Fraction(1, 2)], floating=floating)
    for inner in reader.body.walk():
        inner.attributes.update(
            {
                "prov.source_node_ids": ArrayAttr([StringAttr("source:5")]),
                "prov.origin_node_ids": ArrayAttr([StringAttr("origin:2")]),
                "prov.trace_role": StringAttr("lowering"),
            }
        )
    assert lower_exact_readouts(module) == 1
    assert json.loads(reader.attributes["merlin.exact_readout_proof"].data)["method"] != "source_exact"
    module.verify()


def test_unknown_scalar_attribute_is_not_ignored():
    from xdsl.dialects.builtin import UnitAttr

    module, reader = fixture([Fraction(1, 2)])
    reader.body.block.first_op.attributes["unknown.arithmetic_contract"] = UnitAttr()
    assert lower_exact_readouts(module) == 0


def test_normal_dyadic_certificate_refuses_a_coefficient_above_uint64():
    from merlin.llvmlower.exact_readout import normal_dyadic_plan

    assert normal_dyadic_plan([Fraction((1 << 64) + 1, 1 << 64)]) is None
