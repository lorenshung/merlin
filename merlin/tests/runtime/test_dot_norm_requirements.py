"""Consumer-derived norm requirements preserve actual bound results."""

import ctypes as C
import shutil
import subprocess

import numpy as np
import pytest
from test_prepared_fma_product_bounds import SOURCE

from merlin.common.paths import merlin_dir

SOURCE = SOURCE.replace(
    '#include "prepared_fma_product_bounds.h"',
    '#include "prepared_fma_product_bounds.h"\n#include "dot_norm_requirements.h"',
)
SOURCE = SOURCE.replace(
    " merlin_fma_product_columns columns=",
    " merlin_dot_norm_requirements requirements=merlin_reconstruction_norm_requirements(en,n);\n merlin_fma_product_columns columns=",
    1,
)
SOURCE = SOURCE.replace(
    "double uncertainty[31];size_t used=0;",
    "merlin_l1_norm l1=merlin_l1_norm_begin(&env);double uncertainty[31];size_t used=0;",
)
SOURCE = SOURCE.replace(
    "merlin_dot_norms_add(&rn,fabs(ar[r*k+z]))",
    "(requirements.require_l2?merlin_dot_norms_add(&rn,fabs(ar[r*k+z])):merlin_l1_norm_add(&l1,fabs(ar[r*k+z])))",
)
SOURCE = SOURCE.replace(
    "merlin_fma_product_row row=merlin_fma_product_row_prepare(&gamma,&columns,&ap,&rp,&ep,uncertainty,used,center);",
    "merlin_fma_product_row row=requirements.require_l2?merlin_fma_product_row_prepare(&gamma,&columns,&ap,&rp,&ep,uncertainty,used,center):merlin_fma_product_row_prepare_l1(&gamma,&columns,&ap,&l1,&ep,uncertainty,used,center);",
)
SOURCE += r"""
#include <fenv.h>
int environment_refuses(void){
 int saved=fegetround();int modes[3]={FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO};
 for(int i=0;i<3;i++){if(fesetround(modes[i])){fesetround(saved);return 0;}
 merlin_fma_bound env=merlin_fma_bound_begin();merlin_l1_norm n=merlin_l1_norm_begin(&env);
 merlin_l1_norm_add(&n,1);if(n.valid){fesetround(saved);return 0;}}
 fesetround(saved);return 1;
}

int requirements_cases(void){
 merlin_admitted_dot_norms e[2]={{0,0,0},{-0.,0,0}};
 if(merlin_reconstruction_norm_requirements(e,2).require_l2)return 0;
 if(!merlin_reconstruction_norm_requirements(0,2).require_l2)return 0;
 if(!merlin_reconstruction_norm_requirements(e,0).require_l2)return 0;
 e[1].l1=0x1p-1074;if(!merlin_reconstruction_norm_requirements(e,2).require_l2)return 0;
 e[1].l1=NAN;if(!merlin_reconstruction_norm_requirements(e,2).require_l2)return 0;
 e[1]=(merlin_admitted_dot_norms){0,0,INFINITY};
 if(!merlin_reconstruction_norm_requirements(e,2).require_l2)return 0;
 return 1;
}
int compare_norms(const double *values,int n){
 merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_dot_norms a=merlin_dot_norms_begin(&env);merlin_l1_norm b=merlin_l1_norm_begin(&env);

 for(int i=0;i<n;i++){merlin_dot_norms_add(&a,values[i]);merlin_l1_norm_add(&b,values[i]);}
 merlin_dot_norms_finish(&a);
 return a.valid==b.valid && (!a.valid || (a.l1==b.l1));
}
"""


@pytest.fixture(scope="module")
def lib(tmp_path_factory):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    p = tmp_path_factory.mktemp("norm-requirements")
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
    x.compare_norms.argtypes = [C.c_void_p, C.c_int]
    return x


def test_metadata_refusals(lib):
    assert lib.requirements_cases() == 1


@pytest.mark.parametrize("case", ["zero", "exact", "representation", "uncertain", "mixed", "overflow"])
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 7), (2, 17, 31)])
def test_actual_checked_enclosures(lib, case, shape):
    m, n, k = shape
    rng = np.random.default_rng(794)
    a = rng.integers(-32, 33, (m, k)).astype(np.float32) / 16
    b = rng.integers(-32, 33, (n, k)).astype(np.float32) / 16
    if case == "zero":
        a[:] = 0
        b[:] = -0.0
    if case == "overflow":
        a[:] = np.finfo(np.float32).max
        b[:] = 2
    ar = a.astype(np.float64)
    br = b.astype(np.float64)
    if case in ("representation", "mixed"):
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
            int(case in ("uncertain", "mixed")),
            C.byref(count),
        )
        == 1
    )


@pytest.mark.parametrize(
    "values",
    [
        [0.0, -0.0],
        [2**-1074, 2**-1022, 1.0, 2**500],
        [float("inf")],
        [float("nan")],
        [-1.0],
        [np.finfo(np.float64).max] * 2,
    ],
)
def test_l1_linf_identity_and_invalid_values(lib, values):
    a = np.array(values, np.float64)
    assert lib.compare_norms(a.ctypes.data, a.size) == 1


def test_rounding_environment_refusal(lib):
    assert lib.environment_refuses() == 1
