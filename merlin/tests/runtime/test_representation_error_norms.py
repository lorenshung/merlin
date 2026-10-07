"""Runtime equality specialization encloses original reconstruction error."""

import ctypes
import math
import random
import shutil
import subprocess
from fractions import Fraction
from pathlib import Path

import pytest

from merlin.common.paths import merlin_dir

HEADER = merlin_dir() / "runtime/c"


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    w = tmp_path_factory.mktemp("representation-error")
    (w / "test.c").write_text(r"""#include "representation_error_norms.h"
double bound(double*a,double*r,double*b,int n){
 merlin_fma_bound e=merlin_fma_bound_begin();
 merlin_dot_norms error=merlin_dot_norms_begin(&e),other=error;
 for(int i=0;i<n;i++){merlin_representation_error_add(&error,a[i],r[i]);merlin_dot_norms_add(&other,fabs(b[i]));}
 merlin_representation_error_finish(&error);merlin_dot_norms_finish(&other);
 merlin_admitted_dot_norms x,y;if(!merlin_dot_norms_admit(&error,&x)||!merlin_dot_norms_admit(&other,&y))return INFINITY;
 return merlin_representation_error_product_upper(&x,&y);
}""")
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
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
    lib.bound.argtypes = [p, p, p, ctypes.c_int]
    lib.bound.restype = ctypes.c_double
    return lib


def bound(native, a, r, b):
    return native.bound(*[(ctypes.c_double * len(a))(*v) for v in (a, r, b)], len(a))


def test_equal_signed_zero_subnormals_and_large(native):
    a = [0.0, -0.0, 2.0**-1074, -(2.0**-1074), 2.0**1000, -(2.0**1000)]
    assert bound(native, a, a, [1.0] * len(a)) == 0
    assert bound(native, [0.0], [-0.0], [1.0]) == 0


@pytest.mark.parametrize("bad", [math.inf, -math.inf, math.nan])
def test_nonfinite_refuses_even_equal(native, bad):
    assert math.isinf(bound(native, [bad], [bad], [1.0]))


def test_independent_fraction_error_enclosure(native):
    rng = random.Random(337)
    for _ in range(500):
        n = rng.randrange(1, 40)
        a = [math.ldexp(rng.uniform(-1, 1), rng.randrange(-140, 120)) for _ in range(n)]
        r = [x if rng.randrange(3) else math.nextafter(x, math.inf) for x in a]
        b = [math.ldexp(rng.uniform(-1, 1), rng.randrange(-140, 120)) for _ in range(n)]
        value = bound(native, a, r, b)
        exact = sum((abs(Fraction(x) - Fraction(y)) * abs(Fraction(z)) for x, y, z in zip(a, r, b)), Fraction())
        assert math.isinf(value) or Fraction(value) >= exact
        if a == r:
            assert value == 0
