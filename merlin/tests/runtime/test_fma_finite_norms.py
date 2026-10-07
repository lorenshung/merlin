"""Complete summary equivalence at f32 boundaries and refused FENV contracts."""

import ctypes
import random
import shutil
import struct
import subprocess

import pytest

from merlin.common.paths import repo_root

SOURCE = r"""
#include "fma_product_norms.h"
int summarize(const float*a,const float*b,size_t n,size_t sa,size_t sb,
              merlin_fma_operand_norm*out,int fast,int valid) {
 merlin_fma_bound e=merlin_fma_bound_begin();if(!valid)e.valid=0;
 return fast?merlin_fma_operand_summarize_finite(&e,a,b,n,sa,sb,out):
             merlin_fma_operand_summarize(a,b,n,sa,sb,out);
}
int non_rne(const float*a,merlin_fma_operand_norm*out) {
 int old=fegetround();if(fesetround(FE_UPWARD))return -1;
 merlin_fma_bound e=merlin_fma_bound_begin();
 int result=merlin_fma_operand_summarize_finite(&e,a,a,1,1,1,out);
 fesetround(old);return result;
}
"""


@pytest.fixture(scope="module")
def library(tmp_path_factory):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native compiler required")
    work = tmp_path_factory.mktemp("finite-norms")
    (work / "probe.c").write_text(SOURCE)
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            "-I",
            str(repo_root() / "merlin/runtime/c"),
            str(work / "probe.c"),
            "-lm",
            "-o",
            str(work / "probe.so"),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(work / "probe.so"))
    lib.summarize.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_int,
        ctypes.c_int,
    ]
    lib.non_rne.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return lib


def f32(word):
    return struct.unpack("f", struct.pack("I", word))[0]


def compare(lib, a, b, sa=1, sb=1):
    aa = (ctypes.c_float * (len(a) * sa))()
    bb = (ctypes.c_float * (len(b) * sb))()
    for i, v in enumerate(a):
        aa[i * sa] = v
    for i, v in enumerate(b):
        bb[i * sb] = v
    results = []
    for fast in (0, 1):
        out = (ctypes.c_double * 9)(*([123.0] * 9))
        rc = lib.summarize(aa, bb, len(a), sa, sb, out, fast, 1)
        assert out[7] == out[8] == 123.0
        results.append((rc, bytes(out)[:56]))
    assert results[0] == results[1]
    return results[0]


@pytest.mark.parametrize("sa,sb", [(1, 1), (2, 3), (7, 1)])
def test_full_f32_domain_samples(library, sa, sb):
    rng = random.Random(731)
    words = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0x7F7FFFFF, 0xFF7FFFFF]
    words += [rng.getrandbits(32) for _ in range(2048)]
    values = [f32(w) for w in words if w & 0x7F800000 != 0x7F800000]
    assert compare(library, values, list(reversed(values)), sa, sb)[0] == 1


@pytest.mark.parametrize("values", [[0.0] * 257, [-0.0] * 129, [f32(1)] * 513, [f32(0x7F7FFFFF)] * 1025])
def test_extremes_and_exact_zero_increments(library, values):
    assert compare(library, values, values)[0] == 1


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_refuses_same_partial_summary(library, bad):
    assert compare(library, [1.0, 2.0, bad], [2.0, 1.0, 1.0])[0] == 0


@pytest.mark.parametrize("n", [0, 1 << 24])
def test_invalid_lengths_before_reads(library, n):
    out = (ctypes.c_double * 7)()
    a = (ctypes.c_float * 1)(1.0)
    assert library.summarize(a, a, n, 1, 1, out, 1, 1) == 0


def test_invalid_eligibility_and_rounding(library):
    out = (ctypes.c_double * 7)()
    a = (ctypes.c_float * 1)(1.0)
    assert library.summarize(a, a, 1, 1, 1, out, 1, 0) == 0
    assert library.non_rne(a, out) == 0
