"""Exact rational packing oracle and independent floating reference comparison."""

from __future__ import annotations

import ctypes
import math
import random
import shutil
import struct
import subprocess
from fractions import Fraction

import pytest

from merlin.common.paths import repo_root

WRAPPER = r"""
#include "bf16_radix_pack.h"
int pack(const float *a,int n,int input_stride,float *r,int output_stride,
         int8_t *p,int digit_stride,int element_stride,int digits,float *step) {
  merlin_fma_bound s=merlin_fma_bound_begin();
  return merlin_bf16_radix_row(&s,a,n,input_stride,r,output_stride,p,
      digit_stride,element_stride,digits,step);
}
int floating_reference(const float *a,int n,int input_stride,float *r,int output_stride,
         int8_t *p,int digit_stride,int element_stride,int digits,float *step) {
  float maximum=0;
  for(int i=0;i<n;i++)maximum=fmaxf(maximum,fabsf(a[i*input_stride]));
  const int e=maximum>0?ilogbf(maximum):0;
  *step=scalbnf(1.0f,e+1-7*digits);
  if(!isfinite(*step)||*step==0)return 0;
  for(int i=0;i<n;i++) {
    const float x=a[i*input_stride];
    long coefficient=lrintf(fabsf(x)/ *step);
    const long maximum_coefficient=(1L<<(7*digits))-1;
    if(coefficient>maximum_coefficient)coefficient=maximum_coefficient;
    const int sign=x<0?-1:x>0?1:0;
    r[i*output_stride]=(float)(sign*coefficient)* *step;
    for(int j=0;j<digits;j++)p[j*digit_stride+i*element_stride]=sign*((coefficient>>(7*j))&127);
  }
  return 1;
}
int bad_rounding(void) {
  int original=fegetround();if(fesetround(FE_UPWARD))return -1;
  const float a=1;float r,step;int8_t p[3];
  int okay=pack(&a,1,1,&r,1,p,1,1,3,&step);
  fesetround(original);return okay;
}
"""


def fbits(value):
    return struct.unpack("I", struct.pack("f", value))[0]


def widened_bf16(raw):
    return struct.unpack("f", struct.pack("I", raw << 16))[0]


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler required for packing checks")
    directory = tmp_path_factory.mktemp("bf16-radix-pack")
    source = directory / "audit.c"
    shared = directory / "audit.so"
    source.write_text(WRAPPER)
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
            str(repo_root() / "merlin/runtime/c"),
            str(source),
            "-lm",
            "-o",
            str(shared),
        ],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(shared))
    for name in ["pack", "floating_reference"]:
        getattr(lib, name).argtypes = [
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_void_p,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_void_p,
        ]
        getattr(lib, name).restype = ctypes.c_int
    lib.bad_rounding.restype = ctypes.c_int
    return lib


def audit(native, values, digits, input_stride=1, output_stride=1, element_stride=1):
    n = len(values)
    digit_stride = n * element_stride + 7
    source = (ctypes.c_float * (n * input_stride))()
    for i, value in enumerate(values):
        source[i * input_stride] = value
    outputs = []
    for function in [native.pack, native.floating_reference]:
        r = (ctypes.c_float * (n * output_stride))()
        p = (ctypes.c_int8 * (digit_stride * digits))(*([-113] * (digit_stride * digits)))
        step = ctypes.c_float()
        okay = function(
            source, n, input_stride, r, output_stride, p, digit_stride, element_stride, digits, ctypes.byref(step)
        )
        outputs.append((okay, list(r), list(p), step.value))
    assert outputs[0] == outputs[1]
    okay, r, p, step = outputs[0]
    if not okay:
        return
    # Fraction.__round__ implements nearest-even integer rounding exactly.
    maximum = max(abs(value) for value in values)
    exponent = math.frexp(maximum)[1] - 1 if maximum else 0
    expected_step = Fraction(2) ** (exponent + 1 - 7 * digits)
    assert Fraction(step) == expected_step
    limit = (1 << (7 * digits)) - 1
    for i, value in enumerate(values):
        coefficient = min(round(abs(Fraction(value)) / expected_step), limit)
        sign = -1 if value < 0 else 1 if value > 0 else 0
        expected = float(sign * coefficient * expected_step)
        assert fbits(r[i * output_stride]) == fbits(expected)
        for j in range(digits):
            assert p[j * digit_stride + i * element_stride] == sign * ((coefficient >> (7 * j)) & 127)
    # Layout padding remains untouched; arithmetic addresses come from strides.
    for j in range(digits):
        addressed = {j * digit_stride + i * element_stride for i in range(n)}
        for offset in range(j * digit_stride, (j + 1) * digit_stride):
            if offset not in addressed:
                assert p[offset] == -113


@pytest.mark.parametrize("digits", [1, 2, 3])
def test_every_finite_bf16_encoding_against_rational_and_source_reference(native, digits):
    # One dynamic row contains every finite BF16 bit pattern, including signed
    # zeros, gradual subnormals and exponent extremes. No reference scale given.
    values = [widened_bf16(raw) for raw in range(1 << 16) if raw & 0x7F80 != 0x7F80]
    audit(native, values, digits)


def test_rounding_ties_saturation_tails_and_strided_layouts(native):
    for digits in (1, 2, 3):
        # BF16 subnormal spacing excludes these coefficient half-ties for low
        # row scales; subnormal rows are audited independently below.
        for exponent in (-100, -1, 0, 100, 127):
            step = math.ldexp(1.0, exponent + 1 - 7 * digits)
            for n in (1, 2, 3, 7, 17):
                values = [math.ldexp(1.9921875, exponent)]
                for i in range(n - 1):
                    value = (i % 5 + 0.5) * step
                    values.append(value if i & 1 else -value)
                assert all(fbits(x) & 0xFFFF == 0 for x in values)
                audit(native, values, digits, 3, 2, 4)


def test_dynamic_random_rows_subnormals_and_zero(native):
    rng = random.Random(383151)
    for digits in (1, 2, 3):
        for length in (1, 2, 7, 31, 64, 129):
            for _ in range(4):
                raw = [rng.randrange(1 << 16) for i in range(length)]
                values = [widened_bf16(x & 0xFF7F if x & 0x7F80 == 0x7F80 else x) for x in raw]
                audit(native, values, digits)
        audit(native, [0.0, -0.0] * 7, digits, 2, 3, 2)
        for raw in (1, 2, 15, 16, 31, 32, 127, 128):
            audit(native, [widened_bf16(raw), -widened_bf16(raw)], digits)


def test_refuses_nonfinite_nonbf16_invalid_digits_and_nonrne(native):
    for value, digits in [(math.inf, 3), (-math.inf, 3), (math.nan, 3), (1.0000001192092896, 3), (1.0, 0), (1.0, 4)]:
        source = (ctypes.c_float * 1)(value)
        r = ctypes.c_float()
        p = (ctypes.c_int8 * 4)()
        step = ctypes.c_float()
        assert native.pack(source, 1, 1, ctypes.byref(r), 1, p, 1, 1, digits, ctypes.byref(step)) == 0
    assert native.bad_rounding() == 0


@pytest.mark.parametrize("flag", ["-ffast-math", "-ffinite-math-only"])
def test_refuses_unsafe_compile_contract(tmp_path, flag):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("native compiler required")
    source = tmp_path / "audit.c"
    executable = tmp_path / "audit"
    source.write_text(
        WRAPPER
        + "\nint main(void) { const float a=1;float r,step;int8_t p[3]; return pack(&a,1,1,&r,1,p,1,1,3,&step) ? 1 : 0; }\n"
    )
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            flag,
            "-I",
            str(repo_root() / "merlin/runtime/c"),
            str(source),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(executable)], check=True, capture_output=True)
