"""Independent rational/domain, source ownership, and compiled scan checks."""

import ctypes
import struct
import subprocess
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.scaled_integer_finite import (
    ScaledIntegerFiniteDomain,
    emit_finite_scale_scan,
    prove_broadcast_scaled_integer_finite,
    validate_broadcast_scaled_integer_finite,
)
from merlin.llvmlower.source_expression_interval import IntervalEffectContract
from merlin.llvmlower.toolchain import clang

EFFECTS = IntervalEffectContract(True, True, True, True, True)


def bits(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def exact(word):
    return Fraction.from_float(struct.unpack("<f", struct.pack("<I", word))[0])


@pytest.mark.parametrize(
    "width,constants",
    [
        (8, (0.5,)),
        (16, (-3.125,)),
        (32, (0.03125,)),
        (32, (0.01999999955,)),
        (64, (0.125, 0.5)),
        (32, (2**-149,)),
        (32, (0.0,)),
        (32, (-0.0,)),
    ],
)
def test_full_integer_domain_original_rounded_bound_and_threshold(width, constants):
    plan = ScaledIntegerFiniteDomain(width, tuple(bits(c) for c in constants))
    threshold = plan.scale_limit_word()
    # Independent native binary32 rounding upper bound; at each multiply the
    # upward neighbour covers the rounded source intermediate.
    bound = np.float32(2 ** (width - 1))
    for constant in constants:
        mathematical = exact(bits(float(bound))) * abs(exact(bits(constant)))
        bound = np.float32(float(mathematical))
        if exact(bits(float(bound))) < mathematical:
            bound = np.nextafter(bound, np.float32(np.inf))
    maximum = exact(0x7F7FFFFF)
    assert exact(threshold) * exact(bits(float(bound))) <= maximum
    if threshold < 0x7F7FFFFF:
        assert exact(threshold + 1) * exact(bits(float(bound))) > maximum
    for word in (0, 0x80000000, threshold, threshold | 0x80000000):
        assert plan.admits_scale_word(word)
    for word in (0x7F800000, 0xFF800000, 0x7FC00001, 0x7F800001, 0xFF800001):
        assert not plan.admits_scale_word(word)


@pytest.mark.parametrize(
    "plan",
    [
        ScaledIntegerFiniteDomain(1, (bits(1),)),
        ScaledIntegerFiniteDomain(128, (bits(1),)),
        ScaledIntegerFiniteDomain(32, (0x7F800000,)),
        ScaledIntegerFiniteDomain(32, (0x7FC00000,)),
        ScaledIntegerFiniteDomain(32, (0x7F7FFFFF,)),
        ScaledIntegerFiniteDomain(32, ()),
    ],
)
def test_unsupported_or_pre_scale_overflow_refuses(plan):
    with pytest.raises(ValueError):
        plan.scale_limit_word()


def source(rows=7, channels=13, *, constant="0.03125", scale_map="(d1)", extra="", live=False):
    return f"""module {{func.func @unrelated(%a:tensor<{rows}x{channels}xi32>,%s:tensor<{channels}xf32>)
      ->tensor<{rows}x{channels}xf32> {{
      %constant=arith.constant {constant}:f32
      %e=tensor.empty():tensor<{rows}x{channels}xf32>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0,d1)->(d0,d1)>,
           affine_map<(d0,d1)->{scale_map}>,affine_map<(d0,d1)->(d0,d1)>],
           iterator_types=["parallel","parallel"]}}
        ins(%a,%s:tensor<{rows}x{channels}xi32>,tensor<{channels}xf32>)
        outs(%e:tensor<{rows}x{channels}xf32>) {{
        ^bb0(%i:i32,%scale:f32,%old:f32):
          %f=arith.sitofp %i:i32 to f32
          %scaled_constant=arith.mulf %f,%constant:f32
          %b=arith.mulf %scaled_constant,%scale:f32
          {"%live=arith.addf %b,%scaled_constant:f32" if live else ""}
          {extra}
          linalg.yield %b:f32
        }}->tensor<{rows}x{channels}xf32>
      return %r:tensor<{rows}x{channels}xf32>
    }} }}"""


def proof(text):
    module = parse_mlir_text(text)
    producer = next(op for op in module.walk() if op.name == "linalg.generic")
    value = [op.results[0] for op in producer.body.block.ops if op.name == "arith.mulf"][-1]
    return module, prove_broadcast_scaled_integer_finite(value, effects=EFFECTS)


@pytest.mark.parametrize("rows,channels", [(7, 13), (3, 17), (8, 1)])
def test_typed_projection_full_integer_domain_and_live_original_values(rows, channels):
    _, candidate = proof(source(rows, channels, live=True))
    assert candidate.repetition == rows and candidate.scale_axes == (1,)
    assert candidate.domain_extents == (rows, channels)
    assert candidate.integer_operand == 0 and candidate.scale_operand == 1
    assert candidate.domain.constant_words == (bits(0.03125),)
    validate_broadcast_scaled_integer_finite(candidate)


@pytest.mark.parametrize(
    "text",
    [
        source(0, 13),
        source(1, 13),
        source(7, 13, scale_map="(d0 + d1)"),
        source(extra="%u=func.call @opaque():()->f32"),
        source().replace("arith.sitofp", "arith.uitofp"),
    ],
)
def test_empty_no_reuse_unknown_call_and_unsupported_coordinate_refuse(text):
    with pytest.raises(ValueError):
        proof(text)


def test_witness_binds_constants_maps_ancestor_context_and_source_placement():
    from xdsl.dialects.builtin import FloatAttr, StringAttr, f32

    for mutation in ("constant", "ancestor", "placement"):
        module, candidate = proof(source())
        if mutation == "constant":
            constant = next(op for op in module.walk() if op.name == "arith.constant")
            constant.properties["value"] = FloatAttr(0.0625, f32)
        elif mutation == "ancestor":
            module.body.block.first_op.attributes["strictfp"] = StringAttr("strictfp")
        else:
            candidate.value.owner.detach()
        with pytest.raises(ValueError):
            validate_broadcast_scaled_integer_finite(candidate)


def test_explicit_effect_contract_required():
    module = parse_mlir_text(source())
    producer = next(op for op in module.walk() if op.name == "linalg.generic")
    value = [op.results[0] for op in producer.body.block.ops if op.name == "arith.mulf"][-1]
    with pytest.raises(ValueError):
        prove_broadcast_scaled_integer_finite(value, effects=IntervalEffectContract())


def test_replaced_domain_or_projection_cannot_reuse_an_unchanged_source_witness():
    _, candidate = proof(source())
    for altered in (
        replace(candidate, domain=ScaledIntegerFiniteDomain(8, (bits(0.03125),))),
        replace(candidate, scale_axes=(0,)),
        replace(candidate, repetition=99),
        replace(candidate, _effects=IntervalEffectContract()),
    ):
        with pytest.raises(ValueError):
            validate_broadcast_scaled_integer_finite(altered)


def test_compiled_strided_scan_zero_tails_unknown_values_and_all_fenv_flags(tmp_path):
    plan = ScaledIntegerFiniteDomain(32, (bits(0.03125),))
    src = tmp_path / "scanner.c"
    src.write_text(
        emit_finite_scale_scan(plan, symbol="check")
        + """
float source_value(int i,float scale){volatile float a=(float)i*0.03125f;return a*scale;}
"""
    )
    library = tmp_path / "scanner.so"
    subprocess.run(
        [str(clang()), "-O2", "-ffp-contract=off", "-frounding-math", "-fPIC", "-shared", str(src), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    compiled = ctypes.CDLL(str(library))
    scan = compiled.check
    scan.argtypes = (ctypes.c_void_p, ctypes.c_size_t, ctypes.c_ssize_t)
    scan.restype = ctypes.c_int
    scale = np.array(
        [0, 0x80000000, 1, 0x80000001, plan.scale_limit_word(), plan.scale_limit_word() | 0x80000000, 0x00800000],
        np.uint32,
    )
    data = np.full(3 * len(scale) + 2, 0x7F800001, np.uint32)
    data[1 : 1 + 3 * len(scale) : 3] = scale
    original = data.copy()
    environment = ctypes.CDLL(None)
    old_mode = environment.fegetround()
    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            environment.fesetround(mode)
            for sticky in (0, 1, 4, 8, 16, 32, 61):
                environment.feclearexcept(61)
                environment.feraiseexcept(sticky)
                before = environment.fetestexcept(61)
                assert scan(data.ctypes.data + 4, len(scale), 3) == 1
                assert environment.fetestexcept(61) == before
                assert scan(None, 0, -1) == 1
                assert scan(None, 1, 1) == 0
                assert scan(data.ctypes.data, 1, 1) == 0
                assert scan(data.ctypes.data + 4, len(scale), -1) == 0
                assert scan(data.ctypes.data + 4, 2**63, 3) == 0
        environment.fesetround(0)
        compiled.source_value.argtypes = ctypes.c_int, ctypes.c_float
        compiled.source_value.restype = ctypes.c_float
        for integer in (-(2**31), -32769, -1, 0, 1, 32769, 2**31 - 1):
            for word in scale:
                value = struct.unpack("<f", struct.pack("<I", int(word)))[0]
                assert np.isfinite(compiled.source_value(integer, value))
    finally:
        environment.fesetround(old_mode)
    np.testing.assert_array_equal(data, original)
