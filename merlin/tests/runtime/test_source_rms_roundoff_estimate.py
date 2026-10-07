"""A modeled estimate is deliberately not exported as an exact enclosure."""

import ctypes as C
import math
import shutil
import subprocess

import pytest

from merlin.common.paths import merlin_dir


@pytest.fixture
def lib(tmp_path):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler required")
    p = tmp_path / "p.c"
    p.write_text("""#include "source_rms_roundoff_estimate.h"
double ratio(size_t n){merlin_fma_bound e=merlin_fma_bound_begin();merlin_source_rms4_plan p=merlin_source_rms4_prepare(&e,n,0);return p.valid?p.ratio:-1;}
int run(size_t n,double eta,double c,float l,float h,float*out){merlin_fma_bound e=merlin_fma_bound_begin();merlin_source_rms4_plan p=merlin_source_rms4_prepare(&e,n,eta);merlin_source_rms4_estimate v=merlin_source_rms4_apply(&p,c,l,h);out[0]=v.low;out[1]=v.high;return v.available;}
int bad_round(void){int old=fegetround();fesetround(FE_UPWARD);double r=ratio(64);fesetround(old);return r==-1;}
""")
    subprocess.run(
        [
            cc,
            "-O2",
            "-shared",
            "-fPIC",
            "-I",
            str(merlin_dir() / "runtime/c"),
            str(p),
            "-lm",
            "-o",
            str(tmp_path / "p.so"),
        ],
        check=True,
        capture_output=True,
    )
    so = C.CDLL(str(tmp_path / "p.so"))
    so.ratio.argtypes = [C.c_size_t]
    so.ratio.restype = C.c_double
    so.run.argtypes = [C.c_size_t, C.c_double, C.c_double, C.c_float, C.c_float, C.c_void_p]
    return so


def test_fixed_source_length_formula(lib):
    for n in (1, 3, 7, 64, 128, 192, 257):
        assert lib.ratio(n) == min(1, 4 / math.sqrt(3 * n))
    assert lib.ratio(0) == -1 and lib.ratio(1 << 24) == -1


def test_estimate_is_narrower_not_an_exact_bound(lib):
    out = (C.c_float * 2)()
    assert lib.run(64, 0, 0, -1, 1, out) == 1
    assert -1 < out[0] < 0 < out[1] < 1
    # A value inside the rigorous source interval can lie outside this estimate.
    assert 0.9 > out[1]
    assert lib.run(1, 0, 0, -1, 1, out) == 1 and list(out) == [-1, 1]


def test_eta_zero_and_refusals(lib):
    out = (C.c_float * 2)()
    assert lib.run(64, 1, 0, -1, 1, out) == 1 and list(out) == [-1, 1]
    assert lib.run(64, 0, 0, 0, 0, out) == 1 and list(out) == [0, 0]
    assert lib.run(64, -1, 0, -1, 1, out) == 0
    assert lib.run(64, 0, float("nan"), -1, 1, out) == 0
    assert lib.run(64, 0, 3, -1, 1, out) == 0
    assert lib.bad_round() == 1
