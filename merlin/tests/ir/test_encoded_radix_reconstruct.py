import ctypes
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.llvmlower.encoded_radix_reconstruct import c_header
from merlin.llvmlower.radix_product_groups import plan_radix_product_groups


def test_refuse_changed_group_proof_and_nonbyte_digits():
    plan = plan_radix_product_groups(radix_bits=7, digits=3, reduction_length=65)
    with pytest.raises(ValueError):
        c_header(replace(plan, weighted_absolute_bound=1))
    with pytest.raises(ValueError):
        c_header(plan_radix_product_groups(radix_bits=8, digits=1, reduction_length=1))
    with pytest.raises(ValueError):
        c_header(replace(plan, reduction_length=2049))


@pytest.mark.parametrize("block_rows", [(1, 1), (4, 7), (16, 16)])
def test_compiled_allzero_tail_and_exact_cancellation_prefixes(tmp_path, block_rows):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")
    plan = plan_radix_product_groups(radix_bits=7, digits=3, reduction_length=65)
    (tmp_path / "reconstruct.h").write_text(c_header(plan))
    source = r"""#include "reconstruct.h"
#include <string.h>
#define M 17
#define N 19
#define K 65
#define AB 16
#define BB 16
#define AC ((M+AB-1)/AB)
#define BC ((N+BB-1)/BB)
static int8_t a[3*M*K],b[3*K*N];static uint8_t az[3*AC],bz[3*BC];
static double control[M*N],candidate[M*N];static int32_t source[M*N];
int test(void){
 for(unsigned scenario=0;scenario<6;scenario++){
 for(unsigned i=0;i<3*AC;i++)az[i]=0;
 for(unsigned i=0;i<3*BC;i++)bz[i]=0;
 for(unsigned p=0;p<3;p++)for(unsigned r=0;r<M;r++)for(unsigned k=0;k<K;k++){
  int8_t v=(r*13+k*29+p*43)%255-127;
  if(scenario==0||(scenario==1&&r<16)||(scenario==2&&p==0))v=0;
  if(scenario==5)v=r<16?0:p==0?(k==0?1:k==1?127:0):p==1&&k==0?-1:0;
  a[p*M*K+r*K+k]=v;az[p*AC+r/AB]|=v!=0;}
 for(unsigned p=0;p<3;p++)for(unsigned k=0;k<K;k++)for(unsigned c=0;c<N;c++){
  int8_t v=(c*7+k*19+p*31)%255-127;
  if(scenario==0||(scenario==3&&c<16)||(scenario==4&&p==2))v=0;
  if(scenario==5)v=p==0&&k<2?1:0;
  b[p*K*N+k*N+c]=v;bz[p*BC+c/BB]|=v!=0;}
 for(unsigned i=0;i<M*N;i++)control[i]=candidate[i]=0.0;
 for(unsigned degree=0;degree<5;degree++){
  for(unsigned r=0;r<M;r++)for(unsigned c=0;c<N;c++){
   int32_t sum=0;
   for(unsigned p=0;p<3;p++){int q=(int)degree-(int)p;if(q<0||q>=3)continue;
    for(unsigned k=0;k<K;k++)sum+=(int32_t)a[p*M*K+r*K+k]*b[q*K*N+k*N+c];}
   source[r*N+c]=sum;
   const double weight=(double)((uint64_t)1<<(7*degree));control[r*N+c]+=(double)sum*weight;
  }
  merlin_radix_group_accumulate_exact_f64(candidate,source,az,bz,M,N,AB,BB,degree);
  if(memcmp(control,candidate,sizeof(control)))return 10+scenario;
 }
 }
 return 0;}
"""
    source = source.replace("#define AB 16", f"#define AB {block_rows[0]}").replace(
        "#define BB 16", f"#define BB {block_rows[1]}"
    )
    (tmp_path / "test.c").write_text(source)
    library = tmp_path / f"test_blocks_{block_rows[0]}_{block_rows[1]}.so"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(tmp_path / "test.c"),
            "-o",
            str(library),
        ],
        check=True,
    )
    assert ctypes.CDLL(str(library)).test() == 0


def test_compiled_largest_accepted_absolute_prefix(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")
    plan = plan_radix_product_groups(radix_bits=7, digits=3, reduction_length=2048)
    (tmp_path / "reconstruct.h").write_text(c_header(plan))
    groups = ",".join(str(g.accumulator_bound) for g in plan.groups)
    (tmp_path / "test.c").write_text(
        """#include "reconstruct.h"
#include <string.h>
int test(void){const int32_t groups[5]={"""
        + groups
        + """};
 uint8_t flags[3]={1,1,1};double control=0.0,candidate=0.0;
 for(unsigned d=0;d<5;d++){control+=(double)groups[d]*(double)((uint64_t)1<<(7*d));
 merlin_radix_group_accumulate_exact_f64(&candidate,groups+d,flags,flags,1,1,1,1,d);
 if(memcmp(&control,&candidate,sizeof(control)))return 1;}
 return 0;}"""
    )
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(tmp_path / "test.c"),
            "-o",
            str(tmp_path / "test.so"),
        ],
        check=True,
    )
    assert ctypes.CDLL(str(tmp_path / "test.so")).test() == 0
