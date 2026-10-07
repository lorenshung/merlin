"""Exact source-FMA containment and checked refusal for separable radii."""

import ctypes as C
import shutil
import subprocess

import numpy as np
import pytest
from test_prepared_fma_product_bounds import SOURCE

from merlin.common.paths import merlin_dir

SOURCE = SOURCE.replace(
    '#include "prepared_fma_product_bounds.h"',
    '#include "prepared_fma_product_bounds.h"\n#include "separable_fma_radius.h"',
)
SOURCE = SOURCE.replace(
    " if(!columns.valid)return -3;",
    " if(!columns.valid)return -3;merlin_fma_exact_columns exact=merlin_fma_exact_columns_prepare(&columns,bn,en);",
)
SOURCE = SOURCE.replace(
    "  *admissions+=row.valid;",
    "  merlin_fma_separable_radius plan=merlin_fma_separable_radius_prepare(&row,&exact,&ap,&ep,used);\n  *admissions+=plan.valid;",
)
SOURCE = SOURCE.replace(
    "int selected=row.valid?", "int selected=plan.valid?merlin_fma_separable_radius_apply(&plan,j,&l2,&h2):row.valid?"
)
SOURCE = SOURCE.replace(
    "if(control!=selected||(control&&(memcmp(&l,&l2,4)||memcmp(&h,&h2,4))))return 0;",
    "float original=0;for(int z=0;z<k;z++)original=fmaf(a[r*k+z],b[j*k+z],original);\n   if(selected&&!(l2<=original&&original<=h2))return 0;\n   if(!plan.valid&&(control!=selected||(control&&(memcmp(&l,&l2,4)||memcmp(&h,&h2,4)))))return 0;",
)
SOURCE += r"""
#include <fenv.h>
int refusals(void){
 merlin_fma_bound env=merlin_fma_bound_begin();merlin_fma_zero_gamma_plan g=merlin_fma_zero_gamma_prepare(&env,1);merlin_fma_zero_gamma_batch gb=merlin_fma_zero_gamma_batch_prepare(&g);
 merlin_admitted_dot_norms a={1,1,1},e={0,0,0};double b=1,center=1;
 merlin_fma_product_columns cols=merlin_fma_product_columns_prepare(&a,&e,&b,1,1);
 merlin_fma_exact_columns exact=merlin_fma_exact_columns_prepare(&cols,&a,&e);
 merlin_fma_product_row row=merlin_fma_product_row_prepare(&gb,&cols,&a,&a,&e,0,0,&center);
 merlin_fma_separable_radius p=merlin_fma_separable_radius_prepare(&row,&exact,&a,&e,0);
 if(!p.valid)return 0;
 if(merlin_fma_separable_radius_prepare(&row,&exact,&a,&e,1).valid)return 0;
 e.l1=0x1p-1074;if(merlin_fma_exact_columns_prepare(&cols,&a,&e).valid)return 0;
 if(merlin_fma_separable_radius_prepare(&row,&exact,&a,&e,0).valid)return 0;
 e=(merlin_admitted_dot_norms){0,0,0};a.l1=DBL_MAX;
 if(merlin_fma_separable_radius_prepare(&row,&exact,&a,&e,0).valid)return 0;
 a.l1=NAN;if(merlin_fma_separable_radius_prepare(&row,&exact,&a,&e,0).valid)return 0;
 a.l1=1;e.l2=INFINITY;if(merlin_fma_exact_columns_prepare(&cols,&a,&e).valid)return 0;
 int saved=fegetround();fesetround(FE_UPWARD);env=merlin_fma_bound_begin();g=merlin_fma_zero_gamma_prepare(&env,1);fesetround(saved);if(g.valid)return 0;
 return 1;
}
"""


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native compiler required")
    p = tmp_path_factory.mktemp("separable-radius")
    (p / "p.c").write_text(SOURCE)
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-I",
            str(merlin_dir() / "runtime/c"),
            str(p / "p.c"),
            "-lm",
            "-o",
            str(p / "p.so"),
        ],
        check=True,
    )
    x = C.CDLL(str(p / "p.so"))
    x.compare.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [C.POINTER(C.c_int)]
    return x


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 7), (2, 17, 31)])
@pytest.mark.parametrize("case", ["zero", "exact", "representation", "uncertain", "overflow", "subnormal", "cancel"])
def test_original_ordered_fma_and_fallback(lib, shape, case):
    m, n, k = shape
    rng = np.random.default_rng(814)
    a = rng.integers(-32, 33, (m, k)).astype(np.float32) / 16
    b = rng.integers(-32, 33, (n, k)).astype(np.float32) / 16
    if case == "zero":
        a[:] = -0.0
        b[:] = 0
    if case == "overflow":
        a[:] = np.finfo(np.float32).max
        b[:] = 2
    if case == "subnormal":
        a[:] = np.float32(2**-149)
        b[:] = 0.5
    if case == "cancel":
        a[:, ::2] = 1
        a[:, 1::2] = -1
        b[:] = 1
    ar = a.astype(np.float64)
    br = b.astype(np.float64)
    if case == "representation":
        ar.flat[0] += 2**-12
        br.flat[-1] -= 2**-12
    count = C.c_int()
    assert (
        lib.compare(
            a.ctypes.data,
            b.ctypes.data,
            ar.ctypes.data,
            br.ctypes.data,
            m,
            n,
            k,
            int(case == "uncertain"),
            C.byref(count),
        )
        == 1
    )
    assert count.value == (0 if case in ("representation", "uncertain", "overflow") else m)


def test_invalid_producer_error_and_environment_refuse(lib):
    assert lib.refusals() == 1
