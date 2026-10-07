"""Independent exact-rational boundaries and typed observer membership gates."""

from __future__ import annotations

import bisect
import ctypes
import struct
import subprocess
from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
from test_scaled_integer_finite_llvm import binding
from test_source_expression_interval import EFFECTS, analyze, source

from merlin.llvmlower.bounded_rne_word_cells import (
    binary32_word_in_rne_cell,
    prepare_bounded_rne_word_cells,
    validate_bounded_rne_word_cells,
)
from merlin.llvmlower.source_expression_interval import (
    build_source_interval_table,
    emit_source_interval_i8_lookup,
)
from merlin.llvmlower.toolchain import clang


def word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def ordered(raw):
    return (~raw & 0xFFFFFFFF) if raw >> 31 else raw ^ 0x80000000


def unorder(key):
    return key ^ 0x80000000 if key >> 31 else ~key & 0xFFFFFFFF


def oracle(raw):
    """Exact rational RNE independently of the key-interval construction."""
    value = struct.unpack("<f", struct.pack("<I", raw))[0]
    if np.isnan(value):
        return None
    if value <= -128:
        return -128
    if value >= 127:
        return 127
    return round(Fraction(value))


def cells():
    _, proof = analyze(source())
    return prepare_bounded_rne_word_cells(proof, effects=EFFECTS)


def test_total_non_nan_word_partition_and_all_boundary_neighbors():
    plan = cells()
    validate_bounded_rne_word_cells(plan)
    assert len(plan.ranges) == 256
    assert plan.ranges[0][0] == ordered(word(float("-inf")))
    assert plan.ranges[-1][1] == ordered(word(float("inf")))
    assert all(a[1] + 1 == b[0] for a, b in zip(plan.ranges, plan.ranges[1:]))
    for integer, (lower, upper) in zip(range(-128, 128), plan.ranges):
        for key in (lower - 1, lower, lower + 1, upper - 1, upper, upper + 1):
            raw = unorder(key)
            assert (lower <= key <= upper) == (oracle(raw) == integer)
    for raw in (0, 0x80000000, 1, 0x80000001):
        assert binary32_word_in_rne_cell(raw, 0, plan)
    for raw in (0x7F800001, 0xFF800001, 0x7FFFFFFF, 0xFFFFFFFF):
        assert not any(lo <= ordered(raw) <= hi for lo, hi in plan.ranges)


def test_unrelated_random_words_match_fraction_observer():
    plan = cells()
    ends = [hi for _, hi in plan.ranges]
    rng = np.random.default_rng(3921)
    for raw in rng.integers(0, 2**32, 8193, dtype=np.uint32).tolist():
        key = ordered(raw)
        index = bisect.bisect_left(ends, key)
        result = index - 128 if index < 256 and key >= plan.ranges[index][0] else None
        assert result == oracle(raw)


@pytest.mark.parametrize("mutation", ["cells", "expression", "factor", "context", "effects"])
def test_retained_source_and_numeric_plan_mutations_refuse(mutation):
    plan = cells()
    if mutation == "cells":
        plan = replace(plan, ranges=((0, 1), *plan.ranges[1:]))
    elif mutation == "expression":
        plan = replace(plan, expression_sha256="0" * 64)
    elif mutation == "factor":
        plan = replace(plan, quant_factor_word=word(4))
    elif mutation == "context":
        from xdsl.dialects.builtin import UnitAttr

        plan._observer.endpoint.owner.attributes["strictfp"] = UnitAttr()
    else:
        plan = replace(plan, _effects=replace(EFFECTS, flags_unobserved=False))
    with pytest.raises(ValueError):
        validate_bounded_rne_word_cells(plan)


def test_forged_numeric_observer_fields_are_rederived_from_typed_source():
    _, original = analyze(source())
    _, different = analyze(source(constant="3.0"))
    for forged in (replace(original, expression=different.expression), replace(original, quant_factor_bits=word(4))):
        with pytest.raises(ValueError, match="actual typed source"):
            prepare_bounded_rne_word_cells(forged, effects=EFFECTS)


def test_explicit_matching_actual_table_and_finite_observer_admission():
    bound = binding()
    observer = bound._observers[0]
    plan = prepare_bounded_rne_word_cells(observer, effects=EFFECTS)
    table = build_source_interval_table(observer.expression, effects=EFFECTS, leading_bits=9, max_table_bytes=4096)
    kwargs = dict(
        table_name="table", activation_name="activation", quantizer_name="quant", lookup_name="lookup", leading_bits=9
    )
    ordinary = emit_source_interval_i8_lookup(**kwargs)
    assert ordinary == emit_source_interval_i8_lookup(**kwargs, observer_word_cells=())
    assert "high_word=quant" in ordinary and "rne_cells" not in ordinary
    candidate = emit_source_interval_i8_lookup(
        **kwargs, finite_inputs=(bound,), finite_table=table, observer_word_cells=(plan,)
    )
    assert "lookup_rne_cells[256][2]" in candidate and "high_word=quant" not in candidate
    assert "low_product=lo*up,high_product=hi*up" in candidate
    assert "low_scaled=low_product*scale,high_scaled=high_product*scale" in candidate
    with pytest.raises(ValueError, match="matching source interval table"):
        emit_source_interval_i8_lookup(**kwargs, observer_word_cells=(plan,))
    _, other = analyze(source(constant="3.0"))
    unrelated = prepare_bounded_rne_word_cells(other, effects=EFFECTS)
    with pytest.raises(ValueError, match="actual lookup table"):
        emit_source_interval_i8_lookup(**kwargs, finite_table=table, observer_word_cells=(unrelated,))
    _, different_factor = analyze(source().replace("3.0:f32", "4.0:f32"))
    different = prepare_bounded_rne_word_cells(different_factor, effects=EFFECTS)
    with pytest.raises(ValueError, match="cover the selected finite"):
        emit_source_interval_i8_lookup(
            **kwargs, finite_inputs=(bound,), finite_table=table, observer_word_cells=(different,)
        )


def test_zero_sufficient_predicate_includes_half_ties_and_refuses_special_values():
    plan = cells()
    upper = plan.ranges[128][1] ^ 0x80000000
    assert upper == word(0.5)
    words = [word(x) for x in (-float("inf"), -0.50000006, -0.5, -0.25, -0.0, 0.0, 0.25, 0.5, 0.50000006, float("inf"))]
    words += [0x7F800001, 0xFF800001, 1, 0x80000001]
    words += np.random.default_rng(611).integers(0, 2**32, 79, dtype=np.uint32).tolist()
    for first in words:
        for second in words:
            if ((first | second) & 0x7FFFFFFF) <= upper:
                assert oracle(first) == oracle(second) == 0
    _, proof = analyze(source())
    table = build_source_interval_table(proof.expression, effects=EFFECTS, leading_bits=9, max_table_bytes=4096)
    kwargs = dict(
        table_name="table",
        activation_name="activation",
        quantizer_name="quant",
        lookup_name="lookup",
        leading_bits=9,
        finite_table=table,
    )
    emitted = emit_source_interval_i8_lookup(**kwargs, zero_observer_cells=(plan,))
    assert "_rne_cells[" not in emitted and "return 0;" in emitted and "high_word=quant" in emitted
    with pytest.raises(ValueError, match="choose one"):
        emit_source_interval_i8_lookup(**kwargs, zero_observer_cells=(plan,), observer_word_cells=(plan,))


@pytest.mark.parametrize("representation", ["observer_word_cells", "zero_observer_cells"])
def test_compiled_cells_and_source_finish_on_unrelated_tail_inputs(tmp_path, representation):
    _, proof = analyze(source())
    plan = prepare_bounded_rne_word_cells(proof, effects=EFFECTS)
    table = build_source_interval_table(proof.expression, effects=EFFECTS, leading_bits=9, max_table_bytes=4096)
    code = emit_source_interval_i8_lookup(
        table_name="table",
        activation_name="activation",
        quantizer_name="quant",
        lookup_name="lookup",
        leading_bits=9,
        finite_table=table,
        **{representation: (plan,)},
    )
    # Source rational multiplication and observer are compiled independently.
    # NaNs are outside the defined source fptosi observation domain.
    code += """
#include <math.h>
float activation(float x){return x*2.0f;}
signed char quant(float x){if(x<-128.0f)x=-128.0f;if(x>127.0f)x=127.0f;
 int n=(int)x;float d=x-(float)n;float a=fabsf(d);
 if(a>0.5f||(a==0.5f&&(n&1)))n+=x<0?-1:1;return (signed char)n;}
signed char original(float x,float up,float scale){float a=activation(x);float p=a*up;float s=p*scale;return quant(s);}
"""
    if representation == "observer_word_cells":
        code += "int member(uint32_t raw,int q){float x;__builtin_memcpy(&x,&raw,4);uint32_t w=bits(x);uint32_t k=w>>31?~w:w^0x80000000u;const uint32_t*r=lookup_rne_cells[q+128];return k>=r[0]&&k<=r[1];}\n"
    rows = np.frombuffer(table.data, "<f4").reshape(-1, 2)

    def literal(value):
        return ("-INFINITY" if value < 0 else "INFINITY") if np.isinf(value) else float(value).hex() + "f"

    literals = ",".join("{" + literal(lo) + "," + literal(hi) + "}" for lo, hi in rows)
    code += "const float table[512][2]={" + literals + "};\n"
    c, so = tmp_path / "cells.c", tmp_path / "cells.so"
    c.write_text(code)
    subprocess.run(
        [str(clang()), "-O3", "-ffp-contract=off", "-fPIC", "-shared", str(c), "-o", str(so)],
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(so))
    if representation == "observer_word_cells":
        library.member.argtypes = [ctypes.c_uint32, ctypes.c_int]
        for q in range(-128, 128):
            lo, hi = plan.ranges[q + 128]
            for key in (lo - 1, lo, hi, hi + 1):
                assert bool(library.member(unorder(key), q)) == (oracle(unorder(key)) == q)
    for name in ("lookup", "original"):
        fn = getattr(library, name)
        fn.argtypes = [ctypes.c_float] * 3
        fn.restype = ctypes.c_int8
    rng = np.random.default_rng(7881)
    for x, up in zip(rng.uniform(-61, 61, 203), rng.uniform(-17, 17, 203)):
        assert library.lookup(x, up, 3) == library.original(x, up, 3)
    for x in (-0.0, 0.0, np.nextafter(np.float32(0), np.float32(1)), -0.5, 0.5, -127.5, 126.5):
        for up in (-1.0, 1.0):
            assert library.lookup(x, up, 3) == library.original(x, up, 3)
