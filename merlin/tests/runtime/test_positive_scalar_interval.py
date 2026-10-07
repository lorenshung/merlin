"""Source scalar operations: independent samples, signed zero and refusals."""

import ctypes as C
import math
import random
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native compiler required")
    work = tmp_path_factory.mktemp("scalar-interval")
    header = merlin_dir() / "runtime/c"
    source = r"""
#include "positive_scalar_interval.h"
void scale(float lo,float hi,float factor,float *out,int *valid){
 merlin_f32_interval r=merlin_interval_positive_scale(merlin_interval(lo,hi),factor);
 out[0]=r.lo;out[1]=r.hi;*valid=r.valid;
}
void nonnegative(float lo,float hi,float factor,float *out,int *valid){
 merlin_f32_interval r=merlin_interval_nonnegative_scale(merlin_interval(lo,hi),factor);
 out[0]=r.lo;out[1]=r.hi;*valid=r.valid;
}
void product(float lo,float hi,float flo,float fhi,float *out,int *valid){
 merlin_f32_interval r=merlin_interval_positive_rhs_product(merlin_interval(lo,hi),merlin_interval(flo,fhi));
 out[0]=r.lo;out[1]=r.hi;*valid=r.valid;
}
void fused(float lo,float hi,float clo,float chi,float factor,float *out,int *valid){
 merlin_f32_interval r=merlin_interval_scalar_fma(factor,merlin_interval(lo,hi),merlin_interval(clo,chi));
 out[0]=r.lo;out[1]=r.hi;*valid=r.valid;
}
float source_scale(float x,float factor){return x*factor;}
float source_fma(float x,float c,float factor){return fmaf(factor,x,c);}
"""
    (work / "test.c").write_text(source)
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-I",
            str(header),
            str(work / "test.c"),
            "-lm",
            "-o",
            str(work / "test.so"),
        ],
        check=True,
    )
    lib = C.CDLL(str(work / "test.so"))
    f = C.c_float
    fp = C.POINTER(f)
    lib.scale.argtypes = [f] * 3 + [fp, C.POINTER(C.c_int)]
    lib.nonnegative.argtypes = [f] * 3 + [fp, C.POINTER(C.c_int)]
    lib.product.argtypes = [f] * 4 + [fp, C.POINTER(C.c_int)]
    lib.fused.argtypes = [f] * 5 + [fp, C.POINTER(C.c_int)]
    lib.source_scale.argtypes = [f] * 2
    lib.source_scale.restype = f
    lib.source_fma.argtypes = [f] * 3
    lib.source_fma.restype = f
    return lib


def evaluate(function, *args):
    out = (C.c_float * 2)()
    valid = C.c_int()
    function(*args, out, C.byref(valid))
    return list(out), valid.value


@pytest.mark.parametrize("factor", [2.0**-149, 2.0**-24, 0.125, 1.0, 1.5, 2.0**64])
def test_original_multiply_endpoints_and_interior(native, factor):
    rng = random.Random(10)
    points = [-0.0, 0.0, -(2.0**-149), 2.0**-149, -(2.0**126), 2.0**126]
    points += [struct.unpack("f", struct.pack("I", rng.getrandbits(32)))[0] for _ in range(400)]
    points = sorted(x for x in points if math.isfinite(x))
    for lo, hi in zip(points, points[1:]):
        out, valid = evaluate(native.scale, lo, hi, factor)
        expected = [native.source_scale(x, factor) for x in (lo, hi)]
        if all(math.isfinite(x) for x in expected):
            assert valid and out == expected
            mid = C.c_float((float(lo) + hi) / 2).value
            assert out[0] <= native.source_scale(mid, factor) <= out[1]
        else:
            assert not valid


@pytest.mark.parametrize("factor", [0.125, 1.0, 1.5, 0.0, -1.25])
def test_fused_source_samples_and_checked_fallback(native, factor):
    rng = random.Random(123)
    for _ in range(300):
        lo, hi = sorted(C.c_float(rng.uniform(-100, 100)).value for _ in range(2))
        clo, chi = sorted(C.c_float(rng.uniform(-10, 10)).value for _ in range(2))
        out, valid = evaluate(native.fused, lo, hi, clo, chi, factor)
        assert valid
        for x in (lo, hi, C.c_float((lo + hi) / 2).value):
            for c in (clo, chi, C.c_float((clo + chi) / 2).value):
                assert out[0] <= native.source_fma(x, c, factor) <= out[1]


def test_original_point_signed_zero_preserved(native):
    for x in (-0.0, 0.0):
        out, valid = evaluate(native.scale, x, x, 0.125)
        assert valid
        assert all(struct.pack("f", y) == struct.pack("f", x) for y in out)


@pytest.mark.parametrize("factor", [0.0, -0.0, 0.125, 1.0])
def test_nonnegative_point_source_zero_sign_and_finite_values(native, factor):
    for value in [-(2.0**126), -1.0, -(2.0**-149), -0.0, 0.0, 2.0**-149, 1.0, 2.0**126]:
        out, valid = evaluate(native.nonnegative, value, value, factor)
        expected = native.source_scale(value, factor)
        assert valid
        assert all(struct.pack("f", x) == struct.pack("f", expected) for x in out)


@pytest.mark.parametrize(
    "lo,hi,factor",
    [
        (-1.0, 2.0, 0.0),
        (-1.0, 2.0, -0.0),
        (-1.0, 2.0, 0.125),
        (1.0, 2.0, -1.0),
        (0.0, 1.0, math.nan),
        (0.0, 1.0, math.inf),
    ],
)
def test_nonnegative_source_interval_and_refusal(native, lo, hi, factor):
    out, valid = evaluate(native.nonnegative, lo, hi, factor)
    if math.isfinite(factor) and factor >= 0:
        assert valid
        for sample in [lo, hi, 0.0, (lo + hi) / 2]:
            if lo <= sample <= hi:
                assert out[0] <= native.source_scale(sample, factor) <= out[1]
    else:
        assert not valid


def test_positive_rhs_two_corners_enclose_independent_source_samples(native):
    rng = random.Random(3192)
    for _ in range(1000):
        lo, hi = sorted(C.c_float(rng.uniform(-100, 100)).value for _ in range(2))
        flo, fhi = sorted(C.c_float(2.0 ** rng.uniform(-130, 100)).value for _ in range(2))
        out, valid = evaluate(native.product, lo, hi, flo, fhi)
        assert valid
        for x in [lo, hi, C.c_float((lo + hi) / 2).value]:
            for y in [flo, fhi, C.c_float((flo + fhi) / 2).value]:
                assert out[0] <= native.source_scale(x, y) <= out[1]


def test_positive_rhs_preserves_point_signed_zero(native):
    for x in [-0.0, 0.0]:
        out, valid = evaluate(native.product, x, x, 2.0**-149, 2.0**100)
        assert valid
        assert all(struct.pack("f", y) == struct.pack("f", x) for y in out)


@pytest.mark.parametrize(
    "args",
    [
        (1.0, 2.0, 0.0, 1.0),
        (1.0, 2.0, -1.0, 1.0),
        (1.0, 2.0, 1.0, math.inf),
        (1.0, 2.0, math.nan, 1.0),
        (1.0, math.inf, 1.0, 2.0),
        (2.0, 1.0, 1.0, 2.0),
        (1.0, 2.0, 2.0, 1.0),
        (2.0**127, 2.0**127, 2.0, 2.0),
    ],
)
def test_positive_rhs_refuses_unknown_or_overflow(native, args):
    assert not evaluate(native.product, *args)[1]


@pytest.mark.parametrize(
    "lo,hi,factor",
    [(2.0, 1.0, 1.0), (0.0, 1.0, float("nan")), (0.0, 1.0, float("inf")), (0.0, 1.0, 0.0), (0.0, 1.0, -1.0)],
)
def test_unsupported_scale_refuses(native, lo, hi, factor):
    assert not evaluate(native.scale, lo, hi, factor)[1]
