"""Source safety and estimate identity; no rigorous enclosure claim."""

import ctypes as C
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
    p.write_text("""#include "source_rms_point_products.h"
int run(float *a,float*b,float*al,float*ah,double*c,float*lo,float*hi,size_t m,size_t n,size_t k,int mode){
 double ar[4096],br[4096],scratch[64];unsigned char af[64],bf[64];
 if(m*k>4096||n*k>4096||m>64||n>64)return -1;
 merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_encoded_row_equality ap=merlin_encoded_rows_widen(&env,a,a,al,ah,ar,af,m,k);
 merlin_encoded_row_equality bp=merlin_encoded_rows_widen(&env,b,b,0,0,br,bf,n,k);
 if(mode==1)ap.source=b;
 if(mode==2)ap.valid=0;
 int old=fegetround();if(mode==3){fesetround(FE_UPWARD);env=merlin_fma_bound_begin();}
 int r=merlin_source_rms4_point_product_estimates(&env,&ap,&bp,a,al,ah,b,ar,br,c,lo,hi,m,n,k,scratch,n);
 fesetround(old);return r;
}""")
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
    so.run.argtypes = [C.c_void_p] * 7 + [C.c_size_t] * 3 + [C.c_int]
    return so


def call(lib, a, b, m, n, k, mode=0, lower=None, upper=None, center=None):
    aa = (C.c_float * len(a))(*a)
    bb = (C.c_float * len(b))(*b)
    al = (C.c_float * len(a))(*lower) if lower is not None else None
    ah = (C.c_float * len(a))(*upper) if upper is not None else None
    centers = center or [
        sum(float(aa[r * k + z]) * float(bb[j * k + z]) for z in range(k)) for r in range(m) for j in range(n)
    ]
    cc = (C.c_double * len(centers))(*centers)
    lo = (C.c_float * (m * n))()
    hi = (C.c_float * (m * n))()
    return lib.run(aa, bb, al, ah, cc, lo, hi, m, n, k, mode), list(lo), list(hi), centers


@pytest.mark.parametrize("m,n,k", [(1, 1, 1), (3, 5, 7), (5, 2, 64)])
def test_independent_point_shapes(lib, m, n, k):
    a = [((i % 9) - 4) * 0.25 for i in range(m * k)]
    b = [((i % 7) - 3) * 0.5 for i in range(n * k)]
    ok, l, h, c = call(lib, a, b, m, n, k)
    assert ok
    assert all(x <= y <= z for x, y, z in zip(l, c, h))


@pytest.mark.parametrize("mode", [1, 2, 3])
def test_owner_or_environment_refusal(lib, mode):
    assert call(lib, [1] * 8, [1] * 8, 1, 1, 8, mode)[0] == 0


def test_nonpoint_nan_overflow_and_impossible_center_refuse(lib):
    assert call(lib, [1] * 8, [1] * 8, 1, 1, 8, lower=[0.5] * 8, upper=[1] * 8)[0] == 0
    assert call(lib, [float("nan")], [1], 1, 1, 1)[0] == 0
    assert call(lib, [3e38], [3e38], 1, 1, 1)[0] == 0
    assert call(lib, [1], [1], 1, 1, 1, center=[2])[0] == 0


def test_zero_and_subnormal_points(lib):
    assert call(lib, [-0.0, 0], [1, -1], 1, 1, 2)[0] == 1
    assert call(lib, [2**-149], [2**-149], 1, 1, 1)[0] == 1
