"""Source arithmetic, polynomial boundaries, and final BF16 refusal behavior."""

from __future__ import annotations

import ctypes
import math
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

HEADER = Path(__file__).parents[2] / "runtime/c"
WRAPPER = r"""
#include "f32_interval_endpoint.h"
int polynomial(float lo,float hi,const float *p,float *out) {
  merlin_bit_polynomial_plan plan={p[0],p[1],{p[2],p[3],p[4],p[5]},p[6],p[7]};
  merlin_f32_interval r=merlin_interval_bit_polynomial(merlin_interval(lo,hi),&plan);
  out[0]=r.lo;out[1]=r.hi;return r.valid;
}
float original(float x,const float*p) {
  if(x<p[0])return 0;
  float scaled=x*p[1],fraction=scaled-floorf(scaled);
  float p0=fmaf(p[2],fraction,p[3]);
  float p1=fmaf(p0,fraction,p[4]);
  float p2=fmaf(p1,fraction,p[5]);
  float bits=fmaf(p[6],scaled-p2,p[7]);
  int32_t i=(int32_t)bits;float result;memcpy(&result,&i,4);return result;
}
int bounded(float lo,float hi,float center,unsigned steps,float*out){return merlin_interval_bf16_bounded_endpoint(merlin_interval(lo,hi),center,steps,out);}
int endpoint(float lo,float hi,float*out){return merlin_interval_bf16_endpoint(merlin_interval(lo,hi),out);}
int arithmetic(float al,float ah,float bl,float bh,float cl,float ch,float*out){
 merlin_f32_interval a=merlin_interval(al,ah),b=merlin_interval(bl,bh),c=merlin_interval(cl,ch);
 merlin_f32_interval r=merlin_interval_fma(a,b,c);out[0]=r.lo;out[1]=r.hi;return r.valid;
}
float original_fma(float a,float b,float c){return fmaf(a,b,c);}
"""


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("native compiler required")
    work = tmp_path_factory.mktemp("interval-endpoint")
    (work / "probe.c").write_text(WRAPPER)
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-I",
            str(HEADER),
            str(work / "probe.c"),
            "-lm",
            "-o",
            str(work / "probe.so"),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(work / "probe.so"))
    f = ctypes.c_float
    fp = ctypes.POINTER(f)
    lib.polynomial.argtypes = [f, f, fp, fp]
    lib.original.argtypes = [f, fp]
    lib.original.restype = f
    lib.bounded.argtypes = [f, f, f, ctypes.c_uint, fp]
    lib.endpoint.argtypes = [f, f, fp]
    lib.arithmetic.argtypes = [f] * 6 + [fp]
    lib.original_fma.argtypes = [f, f, f]
    lib.original_fma.restype = f
    return lib


def f32(x):
    return ctypes.c_float(x).value


def adjacent(x, direction):
    return (
        struct.unpack("f", struct.pack("I", 1 if direction > 0 else 0x80000001))[0]
        if x == 0
        else struct.unpack(
            "f",
            struct.pack("I", struct.unpack("I", struct.pack("f", x))[0] + (1 if (direction > x) == (x > 0) else -1)),
        )[0]
    )


@pytest.mark.parametrize("center", [-87.3365478515625, -80, -16, -8, -1, -0.6931472, -0.001, 0, 1])
def test_source_polynomial_boundaries(native, center):
    p = (ctypes.c_float * 8)(
        -87.3365478515625,
        1.4426950216293335,
        -0.0792042389512062,
        -0.2243383675813675,
        0.3035426139831543,
        0.0001070343496394,
        8388608.0,
        1065353216.0,
    )
    center = f32(center)
    lo = hi = center
    for _ in range(24):
        lo = adjacent(lo, -math.inf)
        hi = adjacent(hi, math.inf)
    out = (ctypes.c_float * 2)()
    assert native.polynomial(lo, hi, p, out)
    values = [lo, hi, center]
    x = lo
    for _ in range(48):
        values.append(x)
        x = adjacent(x, math.inf)
    for x in values:
        value = native.original(x, p)
        assert out[0] <= value <= out[1]


@pytest.mark.parametrize(
    "bounds", [(-4, 3, -7, 2, -2, 9), (1e-30, 1e-20, -1e20, 1e20, -1, 1), (-0.0, 0.0, -1, 1, -0.0, 0.0)]
)
def test_signed_fma_interior_enclosure(native, bounds):
    out = (ctypes.c_float * 2)()
    assert native.arithmetic(*bounds, out)
    for a in [bounds[0], f32((bounds[0] + bounds[1]) / 2), bounds[1]]:
        for b in [bounds[2], f32((bounds[2] + bounds[3]) / 2), bounds[3]]:
            for c in [bounds[4], f32((bounds[4] + bounds[5]) / 2), bounds[5]]:
                x = native.original_fma(a, b, c)
                assert out[0] <= x <= out[1]


@pytest.mark.parametrize("lo,hi", [(math.nan, 1), (1, math.inf), (-1, 1), (-0.0, 0.0), (2, 1), (1.0, 1.0078125)])
def test_endpoint_refuses_unknown_nonfinite_zero_and_two_bins(native, lo, hi):
    out = ctypes.c_float(123)
    assert native.endpoint(lo, hi, ctypes.byref(out)) == 0
    assert out.value == 123


def test_endpoint_tie_even_and_negative(native):
    out = ctypes.c_float()
    assert native.endpoint(1, 1 + 1 / 256, ctypes.byref(out))
    assert out.value == 1
    assert native.endpoint(-1 - 1 / 256, -1, ctypes.byref(out))
    assert out.value == -1
    assert not native.endpoint(1 + 1 / 256, adjacent(f32(1 + 1 / 256), math.inf), ctypes.byref(out))


def test_polynomial_refuses_unproved_conversion_and_large_span(native):
    out = (ctypes.c_float * 2)()
    for p, lo, hi in [
        ([0, 1, 0, 0, 0, 0, 1, 0], -1, 300),
        ([0, 1, 0, 0, 0, 0, 1, 2**31], 0, 1),
        ([0, -1, 0, 0, 0, 0, 1, 0], 0, 1),
        ([0, 1, math.nan, 0, 0, 0, 1, 0], 0, 1),
    ]:
        assert not native.polynomial(lo, hi, (ctypes.c_float * 8)(*p), out)


@pytest.mark.parametrize("coefficients", [(0.8, -0.9, 0.2, 0), (-0.7, 1.4, -0.8, 0.1), (0.1, 0.1, 0.1, 0.1)])
def test_nonmonotone_polynomial_is_structurally_enclosed(native, coefficients):
    p = (ctypes.c_float * 8)(-100, 1, *coefficients, 8388608.0, 1065353216.0)
    out = (ctypes.c_float * 2)()
    for left, right in [(-2.1, -1.9), (-0.9, -0.1), (-0.01, 0.01), (0.1, 0.9)]:
        assert native.polynomial(left, right, p, out)
        for j in range(129):
            x = f32(left + (right - left) * j / 128)
            assert out[0] <= native.original(x, p) <= out[1]


@pytest.mark.parametrize(
    "lo,hi,center", [(1, 1.0078125, 1.007), (1.9921875, 2, 1.999), (-2, -1.9921875, -1.994), (-1.0078125, -1, -1.001)]
)
def test_bounded_endpoint_one_adjacent_bin(native, lo, hi, center):
    output = ctypes.c_float()
    assert native.bounded(lo, hi, center, 1, ctypes.byref(output))
    assert output.value in [lo, hi]
    assert not native.bounded(lo, hi, center, 0, ctypes.byref(output))


@pytest.mark.parametrize(
    "lo,hi,center,steps",
    [
        (1, 1.015625, 1.01, 1),
        (1, 1.0078125, 2, 1),
        (1, 1, 1, 2),
        (1, 1, math.nan, 1),
        (-0.0, 0.0, 0.0, 1),
        (-1, 1, 0.0, 1),
    ],
)
def test_bounded_endpoint_refusal(native, lo, hi, center, steps):
    output = ctypes.c_float(9)
    assert not native.bounded(lo, hi, center, steps, ctypes.byref(output))
    assert output.value == 9


@pytest.mark.parametrize("sign", [-1, 1])
def test_bounded_endpoint_preserves_rounded_zero_sign(native, sign):
    x = f32(sign * 2.0**-140)
    output = ctypes.c_float(9)
    assert native.bounded(x, x, math.copysign(0, sign), 1, ctypes.byref(output))
    assert math.copysign(1, output.value) == sign
    assert not native.bounded(x, x, math.copysign(0, -sign), 1, ctypes.byref(output))
