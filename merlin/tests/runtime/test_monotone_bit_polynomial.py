"""Exact source-word enclosure, floor boundaries and independent rational errors."""

import ctypes
import math
import random
import shutil
import struct
import subprocess
from fractions import Fraction

import pytest

from merlin.common.paths import merlin_dir

HEADERS = merlin_dir() / "runtime/c"
SOURCE = r"""
#include "monotone_bit_polynomial.h"
static merlin_bit_polynomial_plan plan(const float *a){
 return (merlin_bit_polynomial_plan){a[0],a[1],{a[2],a[3],a[4],a[5]},a[6],a[7]};
}
void bound(const float *a,float upper,float lo,float hi,uint32_t *out,double *error){
 merlin_bit_polynomial_plan s=plan(a);merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&env,&s,upper);
 merlin_f32_interval y=merlin_monotone_bit_polynomial_apply(merlin_interval(lo,hi),&p);
 out[0]=merlin_interval_bits(y.lo);out[1]=merlin_interval_bits(y.hi);
 out[2]=y.valid;out[3]=p.fast_valid;out[4]=p.word_budget;*error=p.rounding_error;
}
void checked(const float *a,float lo,float hi,uint32_t *out){
 merlin_bit_polynomial_plan s=plan(a);merlin_prepared_bit_polynomial p=merlin_bit_polynomial_prepare(&s);
 merlin_f32_interval y=merlin_prepared_bit_polynomial_apply(merlin_interval(lo,hi),&p);
 out[0]=merlin_interval_bits(y.lo);out[1]=merlin_interval_bits(y.hi);out[2]=y.valid;
}
void bound_words(const float *a,float upper,float lo,float hi,uint32_t *out){
 merlin_bit_polynomial_plan s=plan(a);merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&env,&s,upper);
 merlin_f32_interval y=merlin_monotone_bit_polynomial_apply_words(merlin_interval(lo,hi),&p);
 out[0]=merlin_interval_bits(y.lo);out[1]=merlin_interval_bits(y.hi);
 out[2]=y.valid;out[3]=p.fast_valid;
}
uint32_t original(const float *a,float x){
 merlin_bit_polynomial_plan p=plan(a);if(x<p.cutoff)return 0;
 float s=x*p.scale,f=s-floorf(s),v=p.coefficients[0];
 for(int i=1;i<4;i++)v=fmaf(f,v,p.coefficients[i]);
 return (uint32_t)(int32_t)fmaf(p.bit_multiplier,s-v,p.bit_bias);
}
float scaled(const float *a,float x){return x*a[1];}
float source_q(const float *a,float x,double *error){
 merlin_bit_polynomial_plan s=plan(a);merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&env,&s,8);
 float scaled=x*s.scale,magnitude=fabsf(scaled);
 int index=magnitude<=1?0:(int)((merlin_interval_bits(magnitude)>>23)&255)-126;
 *error=p.fast_valid&&index>=0&&index<26?p.source_error[index]:-1;
 return merlin_monotone_polynomial_source_q(scaled,&s);
}
int paired(const float *a,float low,float high,uint32_t *out){
 merlin_bit_polynomial_plan p=plan(a);
 merlin_polynomial_source_pair pair=merlin_monotone_polynomial_source_pair(low,high,&p);
 out[0]=merlin_interval_bits(pair.low);out[1]=merlin_interval_bits(pair.high);
 out[2]=merlin_interval_bits(merlin_monotone_polynomial_source_q(low,&p));
 out[3]=merlin_interval_bits(merlin_monotone_polynomial_source_q(high,&p));
 return 1;
}
int upward(void){return FE_UPWARD;}
int towardzero(void){return FE_TOWARDZERO;}
int rounding(int mode){return fesetround(mode);}
int nearest(void){return FE_TONEAREST;}
int downward(void){return FE_DOWNWARD;}
"""

PLANS = [
    [
        -87.3365478515625,
        1.4426950216293335,
        -0.079204238951206207,
        -0.22433836758136749,
        0.3035426139831543,
        0.00010703434963943437,
        8388608.0,
        1065353216.0,
    ],
    [-40.0, 0.5, 0.02, -0.1, 0.2, 0.004, 8388608.0, 1065353216.0],
    [-32.0, 0.75, -0.01, -0.02, 0.1, 0.001, 4194304.0, 1065353216.0],
]


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    w = tmp_path_factory.mktemp("monotone-polynomial")
    (w / "test.c").write_text(SOURCE)
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-frounding-math",
            "-shared",
            "-fPIC",
            "-I",
            str(HEADERS),
            str(w / "test.c"),
            "-lm",
            "-o",
            str(w / "test.so"),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(w / "test.so"))
    f = ctypes.c_float
    p = ctypes.POINTER(f)
    u = ctypes.POINTER(ctypes.c_uint32)
    lib.bound.argtypes = [p, f, f, f, u, ctypes.POINTER(ctypes.c_double)]
    lib.paired.argtypes = [p, f, f, u]
    lib.checked.argtypes = [p, f, f, u]
    lib.bound_words.argtypes = [p, f, f, f, u]
    lib.original.argtypes = [p, f]
    lib.original.restype = ctypes.c_uint32
    lib.scaled.argtypes = [p, f]
    lib.scaled.restype = f
    lib.source_q.argtypes = [p, f, ctypes.POINTER(ctypes.c_double)]
    lib.source_q.restype = f
    return lib


def bound(native, plan, lo, hi, upper=0):
    a = (ctypes.c_float * 8)(*plan)
    out = (ctypes.c_uint32 * 5)()
    error = ctypes.c_double()
    native.bound(a, upper, lo, hi, out, ctypes.byref(error))
    return a, list(out), error.value


@pytest.mark.parametrize("plan", PLANS)
def test_independent_original_words_across_actual_intervals(native, plan):
    rng = random.Random(231)
    for _ in range(500):
        left = rng.uniform(plan[0] - 1, 0)
        width = rng.choice([0, 1e-6, 1e-4, 0.01, 1, 8])
        right = min(0, left + width)
        a, out, _ = bound(native, plan, left, right)
        assert out[2:4] == [1, 1]
        left, right = ctypes.c_float(left).value, ctypes.c_float(right).value
        for x in [left, right] + [rng.uniform(left, right) for _ in range(16)]:
            word = native.original(a, x)
            assert out[0] <= word <= out[1], (left, right, x, out, word)


@pytest.mark.parametrize("plan", PLANS)
def test_domain_budget_encloses_exact_rational_source_transform(native, plan):
    rng = random.Random(89)
    a, out, error = bound(native, plan, plan[0], 0)
    assert out[3] and error > 0 and out[4] >= 2 * error
    coeff = list(map(Fraction, a))[2:6]
    for _ in range(300):
        x = ctypes.c_float(rng.uniform(a[0], 0)).value
        s = Fraction(native.scaled(a, x))
        f = s - math.floor(s)
        poly = coeff[0]
        for c in coeff[1:]:
            poly = f * poly + c
        real = Fraction(a[6]) * (s - poly) + Fraction(a[7])
        # Integer conversion contributes less than one extra encoded word.
        assert abs(Fraction(native.original(a, x)) - real) <= Fraction(error) + 1


@pytest.mark.parametrize("plan", PLANS)
def test_local_source_q_budget_with_independent_exact_rationals(native, plan):
    rng = random.Random(903)
    a = (ctypes.c_float * 8)(*plan)
    coeff = list(map(Fraction, a))[2:6]
    points = [0.0, -(2.0**-149), -(2.0**-126), -(2.0**-24), 2.0**-24, 1.0, 2.0, 4.0, 8.0, a[0]]
    points.extend(rng.uniform(a[0], 8) for _ in range(500))
    for x in points:
        x = ctypes.c_float(x).value
        s = Fraction(native.scaled(a, x))
        f = s - math.floor(s)
        real = coeff[0]
        for c in coeff[1:]:
            real = f * real + c
        real = s - real
        error = ctypes.c_double()
        q = native.source_q(a, x, ctypes.byref(error))
        assert error.value >= 0
        assert abs(Fraction(q) - real) <= Fraction(error.value)


def test_floor_cutoff_zero_and_subnormal_boundaries(native):
    p = PLANS[0]
    points = [p[0], 0.0, -(2.0**-149), -(2.0**-126), -(2.0**-24)]
    points.extend(k / p[1] for k in range(-125, 1))
    for x in points:
        for radius in [0, 1e-7, 1e-5]:
            lo = max(p[0] - 1, x - radius)
            hi = min(0, x + radius)
            a, out, _ = bound(native, p, lo, hi)
            assert out[2:4] == [1, 1]
            for sample in [lo, hi, x]:
                if lo <= sample <= hi:
                    assert out[0] <= native.original(a, sample) <= out[1]


@pytest.mark.parametrize("change", [(4, 1.25), (2, -0.5), (6, -1.0), (1, 0.0)])
def test_unproved_coefficients_retain_checked_implementation(native, change):
    plan = PLANS[0].copy()
    plan[change[0]] = change[1]
    a, out, _ = bound(native, plan, -1.0, -0.5)
    old = (ctypes.c_uint32 * 3)()
    native.checked(a, -1.0, -0.5, old)
    assert not out[3] and out[:3] == list(old)


def test_out_of_prepared_domain_retains_checked_implementation(native):
    a, out, _ = bound(native, PLANS[0], -0.5, 0.1)
    old = (ctypes.c_uint32 * 3)()
    native.checked(a, -0.5, 0.1, old)
    assert out[3] and out[:3] == list(old)


def test_unsupported_rounding_refuses_without_changing_mode(native):
    try:
        assert native.rounding(native.downward()) == 0
        _, out, _ = bound(native, PLANS[0], -1.0, -0.5)
        assert not out[2] and not out[3]
    finally:
        assert native.rounding(native.nearest()) == 0


@pytest.mark.parametrize("plan", PLANS)
def test_word_budget_encloses_independent_source_at_floor_and_cutoff(native, plan):
    rng = random.Random(1007)
    a = (ctypes.c_float * 8)(*plan)
    points = [a[0], 0.0, -(2.0**-149), -(2.0**-126)]
    points.extend(k / a[1] for k in range(math.ceil(a[0] * a[1]), 1))
    points.extend(rng.uniform(a[0], 0) for _ in range(400))
    for point in points:
        rounded = ctypes.c_float(point).value
        word = struct.unpack("<I", struct.pack("<f", rounded))[0]
        neighbors = [rounded]
        if rounded < 0:
            neighbors.extend(struct.unpack("<f", struct.pack("<I", word + d))[0] for d in (-1, 1))
        for width in (0.0, 1e-6, 0.01, 4.0):
            lo = ctypes.c_float(min(neighbors) - width).value
            hi = ctypes.c_float(min(0.0, max(neighbors) + width)).value
            out = (ctypes.c_uint32 * 4)()
            native.bound_words(a, 0.0, lo, hi, out)
            assert list(out)[2:] == [1, 1]
            samples = [lo, hi, *neighbors]
            samples.extend(rng.uniform(lo, hi) for _ in range(12))
            for value in samples:
                if lo <= value <= hi:
                    original = native.original(a, value)
                    assert out[0] <= original <= out[1], (plan, lo, hi, value, list(out), original)
            if lo == hi:
                assert out[0] == out[1] == native.original(a, lo)


@pytest.mark.parametrize("change", [(4, 1.25), (2, -0.5), (6, -1.0), (1, 0.0)])
def test_word_budget_unproved_plan_retains_checked_fallback(native, change):
    plan = PLANS[0].copy()
    plan[change[0]] = change[1]
    a = (ctypes.c_float * 8)(*plan)
    out = (ctypes.c_uint32 * 4)()
    old = (ctypes.c_uint32 * 3)()
    native.bound_words(a, 0.0, -1.0, -0.5, out)
    native.checked(a, -1.0, -0.5, old)
    assert not out[3] and list(out)[:3] == list(old)


def test_word_budget_out_of_domain_and_rounding_fallback(native):
    a = (ctypes.c_float * 8)(*PLANS[0])
    out = (ctypes.c_uint32 * 4)()
    old = (ctypes.c_uint32 * 3)()
    native.bound_words(a, 0.0, -0.5, 0.1, out)
    native.checked(a, -0.5, 0.1, old)
    assert out[3] and list(out)[:3] == list(old)
    try:
        assert native.rounding(native.downward()) == 0
        native.bound_words(a, 0.0, -1.0, -0.5, out)
        assert not out[2] and not out[3]
    finally:
        assert native.rounding(native.nearest()) == 0


@pytest.mark.parametrize("plan", PLANS)
def test_source_endpoint_pair_retains_every_source_result(native, plan):
    rng = random.Random(7531)
    a = (ctypes.c_float * 8)(*plan)
    out = (ctypes.c_uint32 * 4)()
    values = [
        -0.0,
        0.0,
        -(2.0**-149),
        2.0**-149,
        -(2.0**-126),
        2.0**-126,
        plan[0],
        -1.0,
        -1.5,
        -2.0,
        -128.0,
        -16777216.0,
    ]
    values += [ctypes.c_float(rng.uniform(-16777216, 0)).value for _ in range(2000)]
    for mode in [native.nearest(), native.downward(), native.upward(), native.towardzero()]:
        native.rounding(mode)
        try:
            for low, high in zip(values, values[::-1]):
                native.paired(a, low, high, out)
                assert list(out[:2]) == list(out[2:])
        finally:
            native.rounding(native.nearest())
