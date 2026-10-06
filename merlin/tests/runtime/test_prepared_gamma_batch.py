"""Batch specialization preserves checked gamma results and dynamic refusal."""

import ctypes as C
import random
import shutil
import subprocess
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native compiler required")
    w = tmp_path_factory.mktemp("gamma")
    h = Path(__file__).parents[2] / "runtime/c"
    (w / "x.c").write_text("""#include "ordered_fma_bounds.h"
int compare(double center,double absolute,double error,unsigned k,unsigned bad,float *out){
 merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_fma_zero_gamma_plan p=merlin_fma_zero_gamma_prepare(&env,k);
 if(bad==1)p.gamma_upper=INFINITY;if(bad==2)p.valid=0;if(bad==3)p.subnormal_error_upper=0;
 merlin_fma_zero_gamma_batch b=merlin_fma_zero_gamma_batch_prepare(&p);
 merlin_fma_chunk c={center,center,absolute,error,k};
 int x=merlin_fma_zero_gamma_apply(&p,c,out,out+1),y=merlin_fma_zero_gamma_batch_apply(&b,c,out+2,out+3);
 return x+2*y;
}
""")
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-I",
            str(h),
            str(w / "x.c"),
            "-lm",
            "-o",
            str(w / "x.so"),
        ],
        check=True,
    )
    f = C.CDLL(str(w / "x.so")).compare
    f.argtypes = [C.c_double] * 3 + [C.c_uint, C.c_uint, C.POINTER(C.c_float)]
    return f


def test_random_checked_result_bitidentity(lib):
    rng = random.Random(73)
    for _ in range(1000):
        magnitude = 2.0 ** rng.randrange(-130, 120)
        center = rng.uniform(-1, 1) * magnitude
        error = rng.random() * magnitude * 0.01
        out = (C.c_float * 4)(-99, -99, -99, -99)
        r = lib(center, magnitude, error, rng.choice([1, 3, 17, 64, 192, 512]), 0, out)
        assert r in (0, 3)
        assert bytes(out)[:8] == bytes(out)[8:]


@pytest.mark.parametrize(
    "center,absolute,error", [(float("nan"), 1, 0), (1, float("inf"), 0), (2, 1, 0), (0, 1, -1), (0, 1, float("nan"))]
)
def test_dynamic_chunks_still_refuse(lib, center, absolute, error):
    out = (C.c_float * 4)(-99, -99, -99, -99)
    assert lib(center, absolute, error, 64, 0, out) == 0
    assert list(out) == [-99] * 4


@pytest.mark.parametrize("bad", [1, 2, 3])
def test_invalid_plan_admission_refuses(lib, bad):
    out = (C.c_float * 4)(-99, -99, -99, -99)
    assert lib(0.1, 1, 0, 64, bad, out) == 0
    assert list(out) == [-99] * 4
