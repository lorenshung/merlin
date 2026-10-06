"""Explicit numerical providers: exact rational enclosure and source FENV."""

import ctypes as C
import math
import random
import shutil
import struct
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

SOURCE = r"""
#include <fenv.h>
#pragma STDC FENV_ACCESS ON
static double provider(double a,double b,int multiply,int rounding){
 fenv_t environment;feholdexcept(&environment);fesetround(rounding);
 volatile double left=a,right=b;
 volatile double result=multiply?left*right:left+right;
 fesetenv(&environment);return result;
}
#define MERLIN_F64_OUTWARD_ADD_UP(a,b) provider(a,b,0,FE_UPWARD)
#define MERLIN_F64_OUTWARD_ADD_DOWN(a,b) provider(a,b,0,FE_DOWNWARD)
#define MERLIN_F64_OUTWARD_MUL_UP(a,b) provider(a,b,1,FE_UPWARD)
#include "ordered_fma_bounds.h"
double result(double a,double b,int op){
 if(op==0)return merlin_fma_up_add(a,b);
 if(op==1)return merlin_fma_down_add(a,b);
 return merlin_fma_up_mul(a,b);
}
double source(double a,double b){return a+b;}
int mode(void){return fegetround();}
int nearest(void){return FE_TONEAREST;}
int raise_flag(void){return feraiseexcept(FE_DIVBYZERO);}
int flags(void){return fetestexcept(FE_ALL_EXCEPT);}
int clear_flags(void){return feclearexcept(FE_ALL_EXCEPT);}
"""


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native compiler required")
    work = tmp_path_factory.mktemp("outward-provider")
    header = Path(__file__).parents[2] / "runtime/c"
    (work / "test.c").write_text(SOURCE)
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
            str(header),
            str(work / "test.c"),
            "-lm",
            "-o",
            str(work / "test.so"),
        ],
        check=True,
    )
    lib = C.CDLL(str(work / "test.so"))
    lib.result.argtypes = [C.c_double, C.c_double, C.c_int]
    lib.result.restype = C.c_double
    lib.source.argtypes = [C.c_double] * 2
    lib.source.restype = C.c_double
    return lib


def finite_pairs():
    rng = random.Random(847)
    points = [
        0.0,
        -0.0,
        2.0**-1074,
        -(2.0**-1074),
        2.0**-1022,
        -(2.0**-1022),
        1.0,
        -1.0,
        2.0**-53,
        -(2.0**-53),
        2.0**500,
        -(2.0**500),
    ]
    yield from ((a, b) for a in points for b in points)
    for _ in range(700):
        values = [struct.unpack("d", struct.pack("Q", rng.getrandbits(64)))[0] for _ in range(2)]
        if all(math.isfinite(v) for v in values):
            yield tuple(values)


@pytest.mark.parametrize("operation", [0, 1, 2])
def test_independent_rational_enclosure(native, operation):
    for a, b in finite_pairs():
        exact = Fraction(a) * Fraction(b) if operation == 2 else Fraction(a) + Fraction(b)
        value = native.result(a, b, operation)
        if math.isfinite(value):
            assert Fraction(value) <= exact if operation == 1 else Fraction(value) >= exact
            # This provider promises correctly directed rounding, so the adjacent
            # representable value in the opposite direction must be too narrow.
            adjacent = math.nextafter(value, math.inf if operation == 1 else -math.inf)
            if math.isfinite(adjacent):
                assert Fraction(adjacent) > exact if operation == 1 else Fraction(adjacent) < exact
        else:
            assert math.copysign(1.0, value) < 0 if operation == 1 else math.copysign(1.0, value) > 0


def test_source_rounding_and_sticky_flags_restored(native):
    assert native.mode() == native.nearest()
    try:
        for operation in (0, 1, 2):
            native.clear_flags()
            native.raise_flag()
            flags = native.flags()
            native.result(1.0, 2.0**-53, operation)
            assert native.mode() == native.nearest() and native.flags() == flags
            assert native.source(1.0, 2.0**-53) == 1.0
    finally:
        native.clear_flags()
