"""Independent checked source DAG versus prepared private endpoint rows."""

import dataclasses
import shutil
import subprocess

import pytest

from merlin.common.paths import data_path
from merlin.llvmlower.prepared_endpoint_dag import PreparedEndpointContract, c_header


@pytest.mark.parametrize("field", [f.name for f in dataclasses.fields(PreparedEndpointContract)])
def test_permission_required(field):
    p = PreparedEndpointContract(*([True] * 9))
    with pytest.raises(ValueError):
        c_header(dataclasses.replace(p, **{field: False}))


@pytest.mark.parametrize("rows,columns,parts_per_factor", [(1, 1, 1), (3, 7, 3), (2, 13, 2)])
def test_native_source_order_refinement_epoch_and_overflow(tmp_path, rows, columns, parts_per_factor):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler required")
    (tmp_path / "prepared.h").write_text(c_header(PreparedEndpointContract(*([True] * 9))))
    code = (
        r"""
#include "prepared.h"
#include <assert.h>
#include <math.h>
#include <string.h>
enum{R=ROWS,C=COLS,Z=PARTS,P=2*Z,N=R*C};
static int reference(const float*pl,const float*ph,const float*centers,const float*f,
 float dl,float dh,size_t row,float*l,float*h,float*v){
 merlin_f32_interval den=merlin_interval(dl,dh);if(dl==0&&dh==0)den=merlin_interval_point(1);
 merlin_f32_interval reciprocal=merlin_interval_recip_positive(den);if(!reciprocal.valid)return 0;
 float de=(float)(((double)dl+dh)*.5);if(de==0)de=1;float point_reciprocal=1.0f/de;
 for(size_t j=0;j<C;j++){
  merlin_f32_interval acc=merlin_interval_point(0);float point=0;
  for(size_t t=0;t<2;t++){
   acc=merlin_interval_nonnegative_scale(acc,f[t]);point*=f[t];
   for(size_t z=0;z<Z;z++){
    size_t ix=((t*Z+z)*R+row)*C+j;
    acc=merlin_interval_add(acc,merlin_interval(pl[ix],ph[ix]));point+=pl[ix]==ph[ix]?pl[ix]:centers[ix];
   }
  }
  merlin_f32_interval e=merlin_interval_positive_rhs_product(acc,reciprocal);if(!e.valid)return 0;
  point*=point_reciprocal;
  l[j]=merlin_interval_bf16(e.lo);h[j]=merlin_interval_bf16(e.hi);
  float q=merlin_interval_bf16(point);if(!isfinite(l[j])||!isfinite(h[j])||!isfinite(q))return 0;
  v[j]=fmaxf(l[j],fminf(h[j],q));
 }
 return 1;
}
int main(void){
 float low[P*N],high[P*N],centers[P*N],a[N],b[N];double c[N];uint32_t maxima[R];unsigned char dirty[R];
 float got[3*C],expect[3*C];unsigned epoch=0,foreign=0;
 assert(fesetround(FE_TONEAREST)==0);merlin_fma_bound env=merlin_fma_bound_begin();assert(env.valid);
 for(int scenario=0;scenario<193;scenario++){
  merlin_endpoint_span_owner owner=merlin_endpoint_span_begin(&env,low,high,centers,maxima,dirty,R,R,C,P,&epoch);
  assert(owner.valid);
  for(size_t part=0;part<P;part++){
   for(size_t i=0;i<N;i++){
    float x=(float)((int)((i*7+part*13+scenario*19)%37)-18)*0.0625f;
    a[i]=x-0.125f;b[i]=x+0.125f;c[i]=x;
    if(scenario>=97){
     uint32_t raw=((uint32_t)((scenario*17+i*11+part*3)%181+27)<<23)|
       ((uint32_t)(scenario*3719+i*173+part*53)&UINT32_C(0x7fffff));
     if((i+part+scenario)%2)raw|=UINT32_C(0x80000000);
     x=merlin_interval_float(raw);a[i]=nextafterf(x,-INFINITY);b[i]=nextafterf(x,INFINITY);c[i]=x;
    }
    if(scenario==192){x=(float)(i%5)*0x1p-149f;a[i]=-x;b[i]=x;c[i]=0;}
    if(scenario==0){a[i]=-0.0f;b[i]=0.0f;c[i]=-0.0;}
   }
   assert(merlin_endpoint_span_record(&owner,&epoch,part,a,b,c));
  }
  float f[2]={scenario%3==0?0.0f:0.5f,scenario%2==0?1.0f:0.125f};
  float dl=scenario==0?0:0.75f,dh=scenario==0?0:1.25f;
  for(size_t row=0;row<R;row++){
   assert(reference(low,high,centers,f,dl,dh,row,expect,expect+C,expect+2*C));
   assert(merlin_endpoint_prepared_row(&owner,&epoch,row,f,2,Z,dl,dh,got,got+C,got+2*C));
   assert(!memcmp(got,expect,sizeof(got)));
  }
  assert(!merlin_endpoint_prepared_row(&owner,&foreign,0,f,2,Z,dl,dh,got,got+C,got+2*C));
  assert(!merlin_endpoint_prepared_row(&owner,&epoch,0,f,2,Z,2,1,got,got+C,got+2*C));
  assert(!merlin_endpoint_span_invalidate_row(&owner,&foreign,0));
  assert(merlin_endpoint_span_invalidate_row(&owner,&epoch,0));
  low[0]=high[0]=centers[0];
  assert(!merlin_endpoint_prepared_row(&owner,&epoch,0,f,2,Z,dl,dh,got,got+C,got+2*C));
  assert(!memcmp(got,expect,sizeof(got)));
 }
 merlin_endpoint_span_owner owner=merlin_endpoint_span_begin(&env,low,high,centers,maxima,dirty,R,R,C,P,&epoch);
 for(size_t i=0;i<N;i++){a[i]=b[i]=FLT_MAX/2;c[i]=a[i];}
 assert(!merlin_endpoint_span_record(&owner,&epoch,1,a,b,c));
 for(size_t p=0;p<P;p++)assert(merlin_endpoint_span_record(&owner,&epoch,p,a,b,c));
 float f[2]={1,1};memset(got,0x12,sizeof(got));memcpy(expect,got,sizeof(got));
 assert(!merlin_endpoint_prepared_row(&owner,&epoch,0,f,2,Z,1,1,got,got+C,got+2*C));
 assert(!memcmp(got,expect,sizeof(got)));
 owner=merlin_endpoint_span_begin(&env,low,high,centers,maxima,dirty,R,R,C,P,&epoch);
 a[0]=NAN;assert(!merlin_endpoint_span_record(&owner,&epoch,0,a,b,c));assert(!owner.valid);
 owner=merlin_endpoint_span_begin(&env,low,high,centers,maxima,dirty,R,R,C,P,&epoch);
 a[0]=1;b[0]=0;c[0]=0.5;assert(!merlin_endpoint_span_record(&owner,&epoch,0,a,b,c));
 const int modes[]={FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 for(size_t i=0;i<3;i++){assert(!fesetround(modes[i]));env=merlin_fma_bound_begin();
  owner=merlin_endpoint_span_begin(&env,low,high,centers,maxima,dirty,R,R,C,P,&epoch);assert(!owner.valid);}
 assert(!fesetround(FE_TONEAREST));
}
""".replace("ROWS", str(rows))
        .replace("COLS", str(columns))
        .replace("PARTS", str(parts_per_factor))
    )
    (tmp_path / "test.c").write_text(code)
    subprocess.run(
        [
            cc,
            "-O2",
            "-ffp-contract=off",
            "-frounding-math",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-I",
            str(data_path("runtime", "c")),
            str(tmp_path / "test.c"),
            "-lm",
            "-o",
            str(tmp_path / "test"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "test")], check=True, capture_output=True)
