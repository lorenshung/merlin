"""Norm product bounds enclose exact real sums, including underflow/refusals."""

from __future__ import annotations

import ctypes
import math
import random
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

HEADER = Path(__file__).parents[2] / "runtime/c"
SOURCE = r"""
#include "prepared_dot_norms.h"
double bound(const double *a,const double *b,int n,double *norms){
 merlin_fma_bound environment=merlin_fma_bound_begin();
 merlin_dot_norms x=merlin_dot_norms_begin(&environment),y=x;
 for(int i=0;i<n;i++){merlin_dot_norms_add(&x,a[i]);merlin_dot_norms_add(&y,b[i]);}
 merlin_dot_norms_finish(&x);merlin_dot_norms_finish(&y);
 norms[0]=x.l2;norms[1]=y.l2;norms[2]=merlin_fma_up_mul(x.l1,y.maximum);
 return merlin_dot_norms_product_upper(&x,&y);
}
"""


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    w = tmp_path_factory.mktemp("prepared-dot-norms")
    (w / "test.c").write_text(SOURCE)
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-I",
            str(HEADER),
            str(w / "test.c"),
            "-lm",
            "-o",
            str(w / "test.so"),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(w / "test.so"))
    p = ctypes.POINTER(ctypes.c_double)
    lib.bound.argtypes = [p, p, ctypes.c_int, p]
    lib.bound.restype = ctypes.c_double
    return lib


def check(native, a, b):
    out = (ctypes.c_double * 3)()
    bound = native.bound((ctypes.c_double * len(a))(*a), (ctypes.c_double * len(b))(*b), len(a), out)
    exact = sum((Fraction(x) * Fraction(y) for x, y in zip(a, b)), Fraction())
    assert math.isinf(bound) or Fraction(bound) >= exact
    assert bound <= out[2] or math.isinf(out[2])
    for values, norm in [(a, out[0]), (b, out[1])]:
        if math.isfinite(norm):
            assert Fraction(norm) ** 2 >= sum((Fraction(x) ** 2 for x in values), Fraction())
    return bound, out


@pytest.mark.parametrize("scale", [0.0, 2.0**-1074, 2.0**-500, 2.0**-126, 1.0, 2.0**80, 2.0**500, 2.0**1000])
def test_zero_subnormal_normal_and_overflow(native, scale):
    check(native, [scale] * 17, [scale / 2] * 17)


def test_independent_random_exact_fraction_enclosure(native):
    rng = random.Random(149)
    for _ in range(300):
        n = rng.randrange(1, 65)
        a = [math.ldexp(rng.random(), rng.randrange(-140, 120)) for _ in range(n)]
        b = [math.ldexp(rng.random(), rng.randrange(-140, 120)) for _ in range(n)]
        check(native, a, b)


def test_cauchy_improves_peaked_rhs_without_changing_domain(native):
    bound, values = check(native, [1.0] * 64, [8.0] + [0.0] * 63)
    assert bound < values[2] / 7


@pytest.mark.parametrize("bad", [-1.0, math.nan, math.inf])
def test_invalid_magnitude_refuses(native, bad):
    out = (ctypes.c_double * 3)()
    value = native.bound((ctypes.c_double * 2)(1, bad), (ctypes.c_double * 2)(1, 1), 2, out)
    assert math.isinf(value)


def test_admitted_metadata_exact_and_refuses_before_write(tmp_path):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native compiler required")
    (tmp_path / "test.c").write_text(r"""
#include "prepared_dot_norms.h"
#include <assert.h>
#include <string.h>
int main(void){
 merlin_fma_bound e=merlin_fma_bound_begin();
 for(int n=1;n<65;n++){
  merlin_dot_norms a=merlin_dot_norms_begin(&e),b=a;
  for(int i=0;i<n;i++){merlin_dot_norms_add(&a,(i%7)*.125);merlin_dot_norms_add(&b,(i%3)*.25);}
  merlin_dot_norms_finish(&a);merlin_dot_norms_finish(&b);
  merlin_admitted_dot_norms x,y;assert(merlin_dot_norms_admit(&a,&x));assert(merlin_dot_norms_admit(&b,&y));
  double old=merlin_dot_norms_product_upper(&a,&b),now=merlin_dot_norms_admitted_product_upper(&x,&y);assert(!memcmp(&old,&now,8));
  merlin_admitted_dot_norms unchanged=x;a.valid=0;assert(!merlin_dot_norms_admit(&a,&x));assert(!memcmp(&x,&unchanged,sizeof(x)));
  a.valid=1;a.l2=INFINITY;assert(merlin_dot_norms_admit(&a,&x));old=merlin_dot_norms_product_upper(&a,&b);now=merlin_dot_norms_admitted_product_upper(&x,&y);assert(!memcmp(&old,&now,8));
  unchanged=x;a.l2=NAN;assert(!merlin_dot_norms_admit(&a,&x));assert(!memcmp(&x,&unchanged,sizeof(x)));
 }
 return 0;
}
""")
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I",
            str(HEADER),
            str(tmp_path / "test.c"),
            "-lm",
            "-o",
            str(tmp_path / "test"),
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "test")], check=True)
