"""Independent source DAG and private narrowing/cache refusal checks."""

import dataclasses
import shutil
import subprocess

import pytest

from merlin.common.paths import data_path
from merlin.llvmlower.endpoint_narrowing import EndpointNarrowingContract, c_header
from merlin.llvmlower.prepared_endpoint_dag import (
    PreparedEndpointContract,
)
from merlin.llvmlower.prepared_endpoint_dag import (
    c_header as prepared_header,
)


@pytest.mark.parametrize("field", [item.name for item in dataclasses.fields(EndpointNarrowingContract)])
@pytest.mark.parametrize("value", [False, 1, None])
def test_complete_typed_permission_required(field, value):
    contract = EndpointNarrowingContract(*([True] * 14))
    with pytest.raises(ValueError):
        c_header(
            PreparedEndpointContract(*([True] * 9)),
            dataclasses.replace(contract, **{field: value}),
        )


def test_original_header_is_composed_without_changes():
    prepared = PreparedEndpointContract(*([True] * 9))
    assert c_header(prepared, EndpointNarrowingContract(*([True] * 14))).startswith(prepared_header(prepared))
    with pytest.raises(ValueError):
        c_header(prepared, None)


@pytest.mark.parametrize("rows,columns,parts_per_factor,factor_count", [(1, 1, 1, 1), (3, 7, 3, 2), (2, 13, 2, 3)])
def test_source_order_and_dependency_local_reuse(tmp_path, rows, columns, parts_per_factor, factor_count):
    cc = shutil.which("cc")
    if not cc:
        pytest.skip("C compiler required")
    (tmp_path / "narrowing.h").write_text(
        c_header(PreparedEndpointContract(*([True] * 9)), EndpointNarrowingContract(*([True] * 14)))
    )
    code = (
        r"""
#include "narrowing.h"
#include <assert.h>
#include <math.h>
#include <string.h>
enum{R=ROWS,C=COLS,Z=PARTS,T=FACTORS,P=T*Z,N=R*C};
static float low[P*N],high[P*N],centers[P*N],a[N],b[N],f[R*T],dl[R],dh[R];
static double center[N];
static uint32_t maxima[R],fw[R*T],dwlo[R],dwhi[R];
static unsigned char dirty[R];
static uint64_t generations[N],observed[N];
static size_t pending_columns[R];
static unsigned char row_state[R];
static float cached_low[N],cached_high[N],cached_candidate[N];
static unsigned epoch,foreign;

/* Independent original interval constructors, source multiply/add schedule,
 * midpoint reciprocal, source BF16 conversion and final clamp. */
static int reference(size_t row,float *l,float *h,float *v){
 merlin_f32_interval den=merlin_interval(dl[row],dh[row]);
 if(dl[row]==0&&dh[row]==0)den=merlin_interval_point(1);
 merlin_f32_interval reciprocal=merlin_interval_recip_positive(den);
 if(!reciprocal.valid)return 0;
 float de=(float)(((double)dl[row]+dh[row])*.5);if(de==0)de=1;
 float point_reciprocal=1.0f/de;
 for(size_t j=0;j<C;j++){
  merlin_f32_interval acc=merlin_interval_point(0);float point=0;
  for(size_t t=0;t<T;t++){
   acc=merlin_interval_nonnegative_scale(acc,f[row*T+t]);point*=f[row*T+t];
   for(size_t z=0;z<Z;z++){
    size_t ix=((t*Z+z)*R+row)*C+j;
    acc=merlin_interval_add(acc,merlin_interval(low[ix],high[ix]));
    point+=low[ix]==high[ix]?low[ix]:centers[ix];
   }
  }
  merlin_f32_interval end=merlin_interval_positive_rhs_product(acc,reciprocal);
  if(!end.valid)return 0;point*=point_reciprocal;
  l[j]=merlin_interval_bf16(end.lo);h[j]=merlin_interval_bf16(end.hi);
  float p=merlin_interval_bf16(point);
  if(!isfinite(l[j])||!isfinite(h[j])||!isfinite(p))return 0;
  v[j]=fmaxf(l[j],fminf(h[j],p));
 }
 return 1;
}
static int compare(merlin_endpoint_narrowing_owner *owner,size_t row,size_t expected){
 float result[3*C];size_t count=SIZE_MAX;
 assert(reference(row,result,result+C,result+2*C));
 if(!merlin_endpoint_narrowing_refresh_row(owner,&epoch,row,&count)){
  int zero=0;for(size_t i=0;i<3*C;i++)zero|=result[i]==0.0f;
  assert(zero&&owner->span->dirty[row]&&count==SIZE_MAX);
  /* Retained source fallback uses this unchanged reference; refused private
   * cache words are never published or consumed in another mutation. */
  return 0;
 }
 assert(count==expected);
 assert(!memcmp(owner->cached_lower+row*C,result,C*sizeof(float)));
 assert(!memcmp(owner->cached_upper+row*C,result+C,C*sizeof(float)));
 assert(!memcmp(owner->cached_candidate+row*C,result+2*C,C*sizeof(float)));
 return 1;
}
static merlin_endpoint_narrowing_owner begin(merlin_endpoint_span_owner *span){
 return merlin_endpoint_narrowing_begin(span,&epoch,f,T,Z,dl,dh,fw,R*T,
  dwlo,dwhi,R,generations,observed,pending_columns,row_state,
  cached_low,cached_high,cached_candidate,N);
}
static merlin_endpoint_span_owner produce(int scenario){
 merlin_fma_bound env=merlin_fma_bound_begin();assert(env.valid);
 merlin_endpoint_span_owner span=merlin_endpoint_span_begin(&env,low,high,centers,
  maxima,dirty,R,R,C,P,&epoch);assert(span.valid);
 for(size_t part=0;part<P;part++){
  for(size_t i=0;i<N;i++){
   float x=(float)((int)((i*7+part*13+scenario*19)%37)-18)*0.0625f;
   a[i]=x-0.125f;b[i]=x+0.125f;center[i]=x;
   if(scenario>=97){
    uint32_t word=((uint32_t)((scenario*17+i*11+part*3)%181+27)<<23)|
     ((uint32_t)(scenario*3719+i*173+part*53)&UINT32_C(0x7fffff));
    if((i+part+scenario)%2)word|=UINT32_C(0x80000000);
    x=merlin_interval_float(word);a[i]=nextafterf(x,-INFINITY);
    b[i]=nextafterf(x,INFINITY);center[i]=x;
   }
   if(scenario==192){x=(float)(i%5)*0x1p-149f;a[i]=-x;b[i]=x;center[i]=0;}
   if(scenario==0){a[i]=-0.0f;b[i]=0.0f;center[i]=-0.0;}
   if(scenario==1000){a[i]=1;b[i]=2;center[i]=1.5;}
  }
  assert(merlin_endpoint_span_record(&span,&epoch,part,a,b,center));
 }
 for(size_t row=0;row<R;row++){
  for(size_t t=0;t<T;t++)f[row*T+t]=(scenario+t)%3==0?0.0f:(t%2?0.125f:0.5f);
  dl[row]=scenario==0?0.0f:0.75f;dh[row]=scenario==0?0.0f:1.25f;
 }
 return span;
}
int main(void){
 assert(!fesetround(FE_TONEAREST));
 for(int scenario=0;scenario<193;scenario++){
  merlin_endpoint_span_owner span=produce(scenario);
  merlin_endpoint_narrowing_owner owner=begin(&span);assert(owner.valid);
  int active[R];
  uint32_t old_maxima[R];memcpy(old_maxima,maxima,sizeof(maxima));
  for(size_t row=0;row<R;row++){
   active[row]=compare(&owner,row,C);if(active[row])assert(compare(&owner,row,0));
  }
  for(size_t part=0;part<P;part++)for(size_t row=0;row<R;row++)for(size_t col=0;col<C;col++){
   if(!active[row])continue;
   size_t index=(part*R+row)*C+col;float point=centers[index];
   assert(merlin_endpoint_narrowing_partial(&owner,&epoch,part,row,col,point,point));
   float original[3*C];
   assert(!merlin_endpoint_prepared_row(&span,&epoch,row,f+row*T,T,Z,dl[row],dh[row],
    original,original+C,original+2*C));
   active[row]=compare(&owner,row,1);if(active[row])assert(compare(&owner,row,0));
   if(active[row]){
    assert(merlin_endpoint_narrowing_partial(&owner,&epoch,part,row,col,point,point));
    assert(compare(&owner,row,0));
   }
   if(R>1&&active[(row+1)%R])assert(compare(&owner,(row+1)%R,0));
   assert(!memcmp(old_maxima,maxima,sizeof(maxima)));
  }
  for(size_t row=0;row<R;row++){
   if(!active[row])continue;
   float point=scenario==0?0.0f:1.0f;
   assert(merlin_endpoint_narrowing_denominator(&owner,&epoch,row,point,point));
   active[row]=compare(&owner,row,C);if(active[row])assert(compare(&owner,row,0));
   if(active[row]){
    assert(merlin_endpoint_narrowing_denominator(&owner,&epoch,row,point,point));
    assert(compare(&owner,row,0));
   }
   if(R>1&&active[(row+1)%R])assert(compare(&owner,(row+1)%R,0));
  }
 }
 /* Unknown and failed updates invalidate before any source/cache write. */
 for(int failure=0;failure<20;failure++){
  merlin_endpoint_span_owner span=produce(1000);
  merlin_endpoint_narrowing_owner owner=begin(&span);assert(owner.valid);
  assert(compare(&owner,0,C));float cache[3*N],previous_low=low[0],previous_high=high[0];
  memcpy(cache,cached_low,sizeof(cached_low));memcpy(cache+N,cached_high,sizeof(cached_high));
  memcpy(cache+2*N,cached_candidate,sizeof(cached_candidate));
  size_t count=SIZE_MAX;
  if(failure==0){assert(!merlin_endpoint_narrowing_partial(&owner,&epoch,0,0,0,-FLT_MAX,FLT_MAX));}
  if(failure==1){assert(!merlin_endpoint_narrowing_partial(&owner,&epoch,0,0,0,NAN,NAN));}
  if(failure==2){assert(!merlin_endpoint_narrowing_partial(&owner,&epoch,0,0,0,INFINITY,INFINITY));}
  if(failure==3){assert(!merlin_endpoint_narrowing_partial(&owner,&epoch,0,0,0,1,-1));}
  if(failure==4){assert(!merlin_endpoint_narrowing_denominator(&owner,&epoch,0,0.5f,2));}
  if(failure==5){assert(!merlin_endpoint_narrowing_denominator(&owner,&epoch,0,-1,1));}
  if(failure==6){assert(!merlin_endpoint_narrowing_denominator(&owner,&epoch,0,NAN,NAN));}
  if(failure==7){f[0]=1;}
  if(failure==8){dl[0]=1;}
  if(failure==9){assert(merlin_endpoint_narrowing_invalidate_row(&owner,&epoch,0));low[0]=0;}
  if(failure==10){assert(!fesetround(FE_DOWNWARD));}
  if(failure==11){generations[0]=UINT64_MAX;
   assert(!merlin_endpoint_narrowing_partial(&owner,&epoch,0,0,0,centers[0],centers[0]));}
  if(failure==12){generations[0]=UINT64_MAX;
   assert(!merlin_endpoint_narrowing_denominator(&owner,&epoch,0,1,1));}
  if(failure==13){span.lower=a;}
  if(failure==14){span.columns=C+1;}
  if(failure==15){span.epoch=&foreign;}
  if(failure==16){span.recorded=0;}
  if(failure==17){assert(!merlin_endpoint_narrowing_partial(&owner,&epoch,P,0,0,0,0));}
  if(failure==18){assert(!merlin_endpoint_narrowing_partial(&owner,&epoch,0,0,C,0,0));}
  if(failure==19){assert(!merlin_endpoint_narrowing_partial(&owner,&epoch,0,0,0,1,1.25f));}
  assert(!merlin_endpoint_narrowing_refresh_row(&owner,&epoch,0,&count));
  assert(count==SIZE_MAX);
  assert(!memcmp(cache,cached_low,sizeof(cached_low)));
  assert(!memcmp(cache+N,cached_high,sizeof(cached_high)));
  assert(!memcmp(cache+2*N,cached_candidate,sizeof(cached_candidate)));
  if(failure!=9){assert(low[0]==previous_low&&high[0]==previous_high);}
  assert(!fesetround(FE_TONEAREST));
 }
 /* Wrong caller epoch refuses without poisoning the valid original owner. */
 merlin_endpoint_span_owner span=produce(1000);
 merlin_endpoint_narrowing_owner owner=begin(&span);size_t count=SIZE_MAX;
 assert(!merlin_endpoint_narrowing_refresh_row(&owner,&foreign,0,&count));
 assert(!merlin_endpoint_narrowing_partial(&owner,&foreign,0,0,0,0,0));
 assert(compare(&owner,0,C));
 /* Begin never accepts aliases, insufficient capacities, incomplete producers,
  * unsupported modes, or nonfinite/negative factors and denominators. */
 assert(!merlin_endpoint_narrowing_begin(&span,&epoch,f,T,Z,dl,dh,fw,R*T,
  dwlo,dwhi,R,generations,observed,pending_columns,row_state,
  low,cached_high,cached_candidate,N).valid);
 assert(!merlin_endpoint_narrowing_begin(&span,&epoch,f,T,Z,dl,dh,fw,R*T,
  dwlo,dwhi,R,generations,observed,pending_columns,row_state,
  cached_low,cached_high,cached_candidate,N-1).valid);
 assert(!merlin_endpoint_narrowing_begin(&span,&epoch,f,T,Z,dl,dh,fw,R*T-1,
  dwlo,dwhi,R,generations,observed,pending_columns,row_state,
  cached_low,cached_high,cached_candidate,N).valid);
 assert(!merlin_endpoint_narrowing_begin(&span,&epoch,f,T,Z,dl,dh,fw,R*T,
  dwlo,dwhi,R,(uint64_t*)((unsigned char*)generations+1),observed,pending_columns,row_state,
  cached_low,cached_high,cached_candidate,N).valid);
 f[0]=NAN;assert(!begin(&span).valid);f[0]=-1;assert(!begin(&span).valid);f[0]=0.5f;
 dl[0]=-1;assert(!begin(&span).valid);dl[0]=0.75f;dh[0]=INFINITY;assert(!begin(&span).valid);
 dh[0]=1.25f;span.recorded=0;assert(!begin(&span).valid);span.recorded=P;
 assert(!fesetround(FE_UPWARD));assert(!begin(&span).valid);assert(!fesetround(FE_TONEAREST));
 /* Signed-zero word membership is stricter than numeric <= equality. */
 assert(!merlin_endpoint_narrowing_subset(0.0f,0.0f,-0.0f,-0.0f));
 assert(!merlin_endpoint_narrowing_subset(-0.0f,-0.0f,0.0f,0.0f));
 assert(merlin_endpoint_narrowing_subset(-0.0f,0.0f,-0.0f,-0.0f));
 assert(merlin_endpoint_narrowing_subset(-1,1,-0.0f,0.0f));
 assert(!merlin_endpoint_narrowing_subset(0.0f,1,-0.0f,0.0f));
}
""".replace("ROWS", str(rows))
        .replace("COLS", str(columns))
        .replace("PARTS", str(parts_per_factor))
        .replace("FACTORS", str(factor_count))
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
