"""Admitted exact-product row bounds retain the checked enclosure bits."""

import ctypes as C
import shutil
import subprocess

import numpy as np
import pytest

from merlin.common.paths import merlin_dir

SOURCE = r"""
#include "prepared_fma_product_bounds.h"
#include <string.h>
int compare(const float *a,const float *b,const double *ar,const double *br,
 int m,int n,int k,int uncertain,int *admissions){
 merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_fma_zero_gamma_plan gp=merlin_fma_zero_gamma_prepare(&env,k);
 merlin_fma_zero_gamma_batch gamma=merlin_fma_zero_gamma_batch_prepare(&gp);
 merlin_admitted_dot_norms bn[17],en[17];double center[17];
 if(m<1||n<1||n>17||k<1||k>31)return -1;
 for(int j=0;j<n;j++){
  merlin_dot_norms x=merlin_dot_norms_begin(&env),e=x;
  for(int z=0;z<k;z++){merlin_dot_norms_add(&x,fabs(b[j*k+z]));merlin_representation_error_add(&e,b[j*k+z],br[j*k+z]);}
  merlin_dot_norms_finish(&x);merlin_representation_error_finish(&e);
  if(!merlin_dot_norms_admit(&x,&bn[j])||!merlin_dot_norms_admit(&e,&en[j]))return -2;
 }
 merlin_fma_product_columns columns=merlin_fma_product_columns_prepare(bn,en,br,n,k);
 if(!columns.valid)return -3;
 *admissions=0;
 for(int r=0;r<m;r++){
  merlin_dot_norms an=merlin_dot_norms_begin(&env),rn=an,ae=an;
  double uncertainty[31];size_t used=0;
  for(int z=0;z<k;z++){
   double delta=uncertain?0x1p-10:0;
   merlin_dot_norms_add(&an,fabs(a[r*k+z])+delta);merlin_dot_norms_add(&rn,fabs(ar[r*k+z]));
   merlin_representation_error_add(&ae,a[r*k+z],ar[r*k+z]);
   if(uncertain)uncertainty[used++]=merlin_fma_next_up(delta);
  }
  merlin_dot_norms_finish(&an);merlin_dot_norms_finish(&rn);merlin_representation_error_finish(&ae);
  merlin_admitted_dot_norms ap,rp,ep;
  if(!merlin_dot_norms_admit(&an,&ap)||!merlin_dot_norms_admit(&rn,&rp)||!merlin_dot_norms_admit(&ae,&ep))return -4;
  for(int j=0;j<n;j++){center[j]=0;for(int z=0;z<k;z++)center[j]+=ar[r*k+z]*br[j*k+z];}
  merlin_fma_product_row row=merlin_fma_product_row_prepare(&gamma,&columns,&ap,&rp,&ep,uncertainty,used,center);
  *admissions+=row.valid;
  for(int j=0;j<n;j++){
   double repr=merlin_fma_up_add(merlin_representation_error_product_upper(&ep,&bn[j]),en[j].l1==0?0:merlin_dot_norms_admitted_product_upper(&rp,&en[j]));
   for(size_t u=0;u<used;u++)repr=merlin_fma_up_add(repr,merlin_fma_up_mul(uncertainty[u],fabs(b[j*k+u])));
   double absolute=merlin_dot_norms_admitted_product_upper(&ap,&bn[j]);
   merlin_fma_chunk chunk={center[j],center[j],absolute,repr,k};
   float l=0,h=0,l2=0,h2=0;
   int control=merlin_fma_zero_gamma_batch_apply(&gamma,chunk,&l,&h);
   int selected=row.valid?merlin_fma_product_row_apply(&row,j,absolute,repr,&l2,&h2):merlin_fma_zero_gamma_batch_apply(&gamma,chunk,&l2,&h2);
   if(control!=selected||(control&&(memcmp(&l,&l2,4)||memcmp(&h,&h2,4))))return 0;
  }
 }
 return 1;
}
int invalid_domains(void){
 merlin_fma_bound env=merlin_fma_bound_begin();merlin_fma_zero_gamma_plan gp=merlin_fma_zero_gamma_prepare(&env,1);
 merlin_fma_zero_gamma_batch gamma=merlin_fma_zero_gamma_batch_prepare(&gp);
 merlin_admitted_dot_norms a={1,1,1},e={0,0,0};double br=1,center=1,u=-1;
 merlin_fma_product_columns columns=merlin_fma_product_columns_prepare(&a,&e,&br,1,1);
 if(!columns.valid)return 0;
 if(merlin_fma_product_columns_prepare(&a,&e,&br,SIZE_MAX,2).valid)return 0;
 if(merlin_fma_product_row_prepare(&gamma,&columns,&a,&a,&e,&u,1,&center).valid)return 0;
 if(merlin_fma_product_row_prepare(&gamma,&columns,&a,&a,&e,0,1,&center).valid)return 0;
 a.l1=INFINITY;if(merlin_fma_product_row_prepare(&gamma,&columns,&a,&a,&e,0,0,&center).valid)return 0;
 br=NAN;if(merlin_fma_product_columns_prepare(&e,&e,&br,1,1).valid)return 0;
 return 1;
}
"""


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    w = tmp_path_factory.mktemp("product-domain")
    (w / "probe.c").write_text(SOURCE)
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
            str(w / "probe.c"),
            "-lm",
            "-o",
            str(w / "probe.so"),
        ],
        check=True,
    )
    lib = C.CDLL(str(w / "probe.so"))
    lib.compare.argtypes = [C.c_void_p] * 4 + [C.c_int] * 4 + [C.POINTER(C.c_int)]
    return lib


@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 5, 7), (2, 17, 31)])
@pytest.mark.parametrize("case", ["zero", "exact", "representation", "uncertain", "mixed", "overflow"])
def test_checked_bits_and_overflow_fallback(native, shape, case):
    m, n, k = shape
    rng = np.random.default_rng(735)
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
    if case in ["representation", "mixed"]:
        ar.flat[0] += 2**-12
        br.flat[-1] -= 2**-12
    count = C.c_int()
    assert (
        native.compare(
            a.ctypes.data,
            b.ctypes.data,
            ar.ctypes.data,
            br.ctypes.data,
            m,
            n,
            k,
            int(case in ["uncertain", "mixed"]),
            C.byref(count),
        )
        == 1
    )
    assert count.value == (0 if case == "overflow" else m)


def test_invalid_admissions(native):
    assert native.invalid_domains() == 1
