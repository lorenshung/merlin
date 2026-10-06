"""Representation-only metadata matches the legacy full requirement contract."""

import ctypes
import random
import shutil
import struct
import subprocess

import pytest

from merlin.common.paths import repo_root

SOURCE = r"""
#include "fma_product_norms.h"
int probe(const float*a,const float*b,size_t n,size_t sa,size_t sb,
          double*out,int valid) {
 merlin_fma_bound e=merlin_fma_bound_begin();if(!valid)e.valid=0;
 merlin_fma_operand_norm old;
 merlin_fma_representation_norm small;
 int x=merlin_fma_operand_summarize(a,b,n,sa,sb,&old);
 int y=merlin_fma_representation_summarize_finite(&e,a,b,n,sa,sb,&small);
 if(!valid)return y;
 out[0]=old.absolute_sum;out[1]=old.original_max;
 out[2]=old.error_sum;out[3]=old.error_max;
 memcpy(out+4,&small,sizeof small);
 out[8]=merlin_fma_representation_bound(old,old);
 out[9]=merlin_fma_representation_pair_bound(small,small);
 return x==y?y:-1;
}
int non_rne(const float*a,double*out) {
 int old=fegetround();fesetround(FE_DOWNWARD);
 merlin_fma_bound e=merlin_fma_bound_begin();
 merlin_fma_representation_norm n;
 int rc=merlin_fma_representation_summarize_finite(&e,a,a,1,1,1,&n);
 fesetround(old);return rc;
}
"""


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    work = tmp_path_factory.mktemp("representation-norms")
    (work / "test.c").write_text(SOURCE)
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-I",
            str(repo_root() / "merlin/runtime/c"),
            str(work / "test.c"),
            "-lm",
            "-o",
            str(work / "test.so"),
        ],
        check=True,
    )
    result = ctypes.CDLL(str(work / "test.so"))
    result.probe.argtypes = [
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_size_t,
        ctypes.c_void_p,
        ctypes.c_int,
    ]
    result.non_rne.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    return result


def f32(word):
    return struct.unpack("f", struct.pack("I", word))[0]


@pytest.mark.parametrize("sa,sb", [(1, 1), (2, 3), (7, 1)])
@pytest.mark.parametrize("count", [1, 17, 129, 2049])
def test_retained_fields_and_pair_bound(lib, sa, sb, count):
    rng = random.Random(128)
    words = [0, 0x80000000, 1, 0x80000001, 0x7F7FFFFF, 0xFF7FFFFF]
    words += [rng.getrandbits(32) for _ in range(count * 2)]
    values = [f32(w) for w in words if w & 0x7F800000 != 0x7F800000][:count]
    a = (ctypes.c_float * (count * sa))()
    b = (ctypes.c_float * (count * sb))()
    for i, v in enumerate(values):
        a[i * sa] = v
        b[i * sb] = values[-1 - i]
    out = (ctypes.c_double * 12)(*([123.0] * 12))
    assert lib.probe(a, b, count, sa, sb, out, 1) == 1
    raw = bytes(out)
    assert raw[:32] == raw[32:64]
    assert raw[64:72] == raw[72:80]
    assert out[10] == out[11] == 123.0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_refuses_with_same_partial_metadata(lib, bad):
    a = (ctypes.c_float * 3)(1.0, 2.0, bad)
    b = (ctypes.c_float * 3)(2.0, 1.0, 1.0)
    out = (ctypes.c_double * 10)()
    assert lib.probe(a, b, 3, 1, 1, out, 1) == 0
    assert bytes(out)[:32] == bytes(out)[32:64]


@pytest.mark.parametrize("n", [0, 1 << 24])
def test_length_refusal_before_reads(lib, n):
    a = (ctypes.c_float * 1)(1.0)
    out = (ctypes.c_double * 10)()
    assert lib.probe(a, a, n, 1, 1, out, 1) == 0


def test_invalid_token(lib):
    a = (ctypes.c_float * 1)(1.0)
    out = (ctypes.c_double * 10)()
    assert lib.probe(a, a, 1, 1, 1, out, 0) == 0
    assert lib.non_rne(a, out) == 0
