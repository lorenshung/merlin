"""Prepared source polynomial plans preserve exact interval outputs."""

from __future__ import annotations

import ctypes
import math
import random
import shutil
import subprocess
from pathlib import Path

import pytest

HEADER = Path(__file__).parents[2] / "runtime/c"
SOURCE = r"""
#include "prepared_bit_polynomial.h"
int compare(float lo,float hi,const float *v,unsigned *out){
 merlin_bit_polynomial_plan p={v[0],v[1],{v[2],v[3],v[4],v[5]},v[6],v[7]};
 merlin_prepared_bit_polynomial prepared=merlin_bit_polynomial_prepare(&p);
 merlin_f32_interval old=merlin_interval_bit_polynomial(merlin_interval(lo,hi),&p);
 /* Preparation owns a copy, not a borrowed mutable coefficient pointer. */
 p.scale=NAN;
 merlin_f32_interval n=merlin_prepared_bit_polynomial_apply(merlin_interval(lo,hi),&prepared);
 out[0]=merlin_interval_bits(old.lo);out[1]=merlin_interval_bits(old.hi);
 out[2]=merlin_interval_bits(n.lo);out[3]=merlin_interval_bits(n.hi);
 return old.valid==n.valid && (!old.valid || (out[0]==out[2] && out[1]==out[3]));
}
"""


@pytest.fixture(scope="module")
def probe(tmp_path_factory):
    cc = shutil.which("clang") or shutil.which("cc")
    if cc is None:
        pytest.skip("native compiler required")
    root = tmp_path_factory.mktemp("prepared-polynomial")
    (root / "test.c").write_text(SOURCE)
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
            str(root / "test.c"),
            "-lm",
            "-o",
            str(root / "test.so"),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(root / "test.so"))
    lib.compare.argtypes = [
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_uint),
    ]
    return lib


PLANS = [
    [-87.33655, 1.442695, -0.07920424, -0.22433837, 0.30354261, 0.000107034, 8388608.0, 1065353216.0],
    [-40.0, 1.0, 0.1, 0.2, 0.3, 0.4, 1048576.0, 1065353216.0],
    [-40.0, 1.0, -0.5, 0.75, -0.125, 0.25, 1048576.0, 1065353216.0],
    [-40.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1048576.0, 1065353216.0],
]


@pytest.mark.parametrize("values", PLANS)
def test_independent_signs_floor_crossings_and_intervals(probe, values):
    p = (ctypes.c_float * 8)(*values)
    out = (ctypes.c_uint * 4)()
    rng = random.Random(414)
    cases = [(-0.0, 0.0), (-88.0, -87.0), (-math.inf, 0), (0, math.inf), (math.nan, 1), (2, 1), (-1000, 1000)]
    cases += [(i / values[1] - 1e-5, i / values[1] + 1e-5) for i in range(-32, 2)]
    for _ in range(2000):
        x = rng.uniform(-42, 2)
        width = 10 ** rng.uniform(-8, 0)
        cases.append((x, x + width))
    for lo, hi in cases:
        assert probe.compare(lo, hi, p, out), (lo, hi, list(out))


@pytest.mark.parametrize(
    "index,value",
    [
        (0, math.nan),
        (1, 0),
        (1, -1),
        (1, math.inf),
        (2, math.nan),
        (3, math.inf),
        (4, -math.inf),
        (5, math.nan),
        (6, math.inf),
        (7, math.nan),
    ],
)
def test_invalid_plans_keep_original_refusal(probe, index, value):
    values = PLANS[0].copy()
    values[index] = value
    assert probe.compare(-1.0, 0.0, (ctypes.c_float * 8)(*values), (ctypes.c_uint * 4)())
