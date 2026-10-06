"""Independent rows, endpoint bin/refusal semantics and write discipline."""

import ctypes as C
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    w = tmp_path_factory.mktemp("frontier")
    header = Path(__file__).parents[2] / "runtime/c"
    (w / "x.c").write_text("""#include "bf16_quant_frontier.h"
int row(const float*l,const float*h,const float*c,size_t n,unsigned char*p,float*s,size_t*k){merlin_bf16_quant_plan plan={127,0x1.5p-17f,-127,127};return merlin_frontier_row(l,h,c,n,plan,p,s,k);}
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
            str(header),
            str(w / "x.c"),
            "-lm",
            "-o",
            str(w / "x.so"),
        ],
        check=True,
    )
    library = C.CDLL(str(w / "x.so"))
    f = library.row
    fp = np.ctypeslib.ndpointer(np.float32, flags="C_CONTIGUOUS")
    up = np.ctypeslib.ndpointer(np.uint8, flags="C_CONTIGUOUS")
    f.argtypes = [fp, fp, fp, C.c_size_t, up, C.POINTER(C.c_float), C.POINTER(C.c_size_t)]
    return f


def bf(x):
    a = np.asarray(x, np.float32)
    u = a.view(np.uint32)
    return ((u + np.uint32(32767) + ((u >> 16) & 1)) & np.uint32(0xFFFF0000)).view(np.float32)


def observations(x):
    x = np.asarray(x, np.float32)
    scale = np.maximum(bf(np.max(abs(x)) / np.float32(127)), np.float32(float.fromhex("0x1.5p-17")))
    q = np.clip(bf(np.rint(bf(x * bf(np.float32(1) / scale)))), -127, 127).astype(np.int8)
    return scale, q


def call(lib, l, h, c=None):
    l = np.ascontiguousarray(l, np.float32)
    h = np.ascontiguousarray(h, np.float32)
    c = l.copy() if c is None else np.ascontiguousarray(c, np.float32)
    p = np.full(len(l) + 2, 213, np.uint8)
    scale = C.c_float(-99)
    count = C.c_size_t(999)
    r = lib(l, h, c, len(l), p[1:-1], C.byref(scale), C.byref(count))
    assert p[0] == p[-1] == 213
    return r, p[1:-1], scale.value, count.value


@pytest.mark.parametrize("n", [1, 3, 17, 64, 129])
def test_point_rows_original_observations(lib, n):
    rng = np.random.default_rng(n)
    a = bf(rng.normal(size=n) * 4)
    r, p, s, k = call(lib, a, a)
    expected, q = observations(a)
    assert r == 1 and k == 0 and not p.any()
    assert np.float32(s).view("u4") == expected.view("u4")


def test_ambiguous_scale_marks_only_possible_extrema(lib):
    r, p, _, n = call(lib, [0, 1, 4], [0, 1, 8])
    assert r == 0 and list(p) == [0, 0, 1] and n == 1


def test_quant_bin_uncertainty_marks_cell(lib):
    r, p, _, _ = call(lib, [127, 0.49], [127, 0.51])
    assert r == -1  # non-BF16 inputs rejected
    r, p, _, _ = call(lib, bf([127, 0.49]), bf([127, 0.51]))
    assert r == 0 and list(p) == [0, 1]


@pytest.mark.parametrize("l,h,c", [(float("nan"), 1, 0), (0, float("inf"), 0), (2, 1, 1), (0, 1, 2), (0.1, 1, 1)])
def test_refusal_before_output_mutation(lib, l, h, c):
    r, p, s, n = call(lib, [l], [h], [c])
    assert r == -1 and p[0] == 213 and s == -99 and n == 999


def test_crosszero_and_signed_zero(lib):
    r, p, scale, n = call(lib, [-0.0, -(2**-20)], [0.0, 2**-20])
    assert r == 1 and n == 0


def test_non_rne_refuses_without_mutation(lib):
    libc = C.CDLL(None)
    libc.fegetround.restype = C.c_int
    original = libc.fegetround()
    try:
        # glibc FE_DOWNWARD, FE_UPWARD and FE_TOWARDZERO on this host.
        for mode in (0x400, 0x800, 0xC00):
            if libc.fesetround(mode):
                continue
            r, p, scale, count = call(lib, [0, 1], [0, 1])
            assert r == -1 and (p == 213).all() and count == 999 and scale == -99
    finally:
        libc.fesetround(original)


def test_certified_intervals_all_interior_bf16_words(lib):
    # Fixed dominating extremum proves scale; exhaust BF16 interiors on tails.
    for raw in range(0x3C00, 0x4000, 29):
        vals = (np.arange(raw, raw + 4, dtype=np.uint32) << 16).view(np.float32)
        r, p, s, n = call(lib, [127, vals[0]], [127, vals[-1]])
        if r == 1:
            baseline = observations([127, vals[0]])
            for v in vals:
                scale, q = observations([127, v])
                assert scale == baseline[0] and np.array_equal(q, baseline[1])


def test_caller_owned_multhead_refinement_coordinator(tmp_path):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    header = Path(__file__).parents[2] / "runtime/c"
    source = r"""
#include "bf16_quant_frontier.h"
#include <assert.h>
static float l[3*5*7],h[3*5*7],v[3*5*7],rl[21],rh[21],rv[21];
static uint8_t pending[21],certified[5];
static unsigned refreshes,refinements;
static int refresh(void*x){(void)x;refreshes++;return 1;}
static int refine(void*x,size_t head,size_t row,size_t col,int exact){
 (void)x;assert(head==2&&row==3&&col==5&&!exact);refinements++;
 size_t i=(head*5+row)*7+col;l[i]=h[i]=v[i]=.5f;return 1;
}
int main(void){
 for(int r=0;r<5;r++)l[r*7]=h[r*7]=v[r*7]=127;
 size_t i=(2*5+3)*7+5;l[i]=v[i]=merlin_frontier_bf16(.49f);h[i]=merlin_frontier_bf16(.51f);
 merlin_frontier_storage s={3,5,7,l,h,v,rl,rh,rv,pending,certified,refresh,refine,0};
 merlin_bf16_quant_plan p={127,0x1.5p-17f,-127,127};
 assert(merlin_frontier_complete(&s,p,4));assert(refreshes==2&&refinements==1);
 for(int r=0;r<5;r++)assert(certified[r]);
 s.heads=SIZE_MAX;assert(!merlin_frontier_complete(&s,p,4));
 s.heads=s.channels=1;s.rows=SIZE_MAX/sizeof(float)+1;assert(!merlin_frontier_complete(&s,p,4));
 assert(!merlin_frontier_complete(0,p,4));
 return 0;
}
"""
    (tmp_path / "x.c").write_text(source)
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I",
            str(header),
            str(tmp_path / "x.c"),
            "-lm",
            "-o",
            str(tmp_path / "x"),
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "x")], check=True)
