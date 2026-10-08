"""All finite BF16 points, source scales, mixed intervals and write refusal."""

import subprocess

from merlin.common.paths import data_path
from merlin.llvmlower.frontier_point_cells import FrontierPointCellsContract, c_header


def test_complete_finite_bf16_point_domain_and_mixed_rows(tmp_path):
    runtime = data_path("runtime", "c")
    source = r"""
#include <math.h>
#include <assert.h>
#include <stdint.h>
#include <string.h>
static unsigned rounding_calls;
static float counted_nearbyint(float value) {rounding_calls++;return nearbyintf(value);}
#define nearbyintf counted_nearbyint
@HEADER@
static float word(uint32_t value) {float x;memcpy(&x,&value,4);return x;}
static uint32_t bits(float x) {uint32_t value;memcpy(&value,&x,4);return value;}
static void compare(const float*l,const float*h,const float*c,size_t n,merlin_bf16_quant_plan plan,int point) {
 uint8_t a[19],b[19];memset(a,213,sizeof(a));memset(b,213,sizeof(b));
 float sa=-99,sb=-99;size_t ka=999,kb=999;rounding_calls=0;
 int ra=merlin_frontier_row(l,h,c,n,plan,a+1,&sa,&ka);unsigned old_calls=rounding_calls;
 rounding_calls=0;int rb=merlin_frontier_row_finite_points(l,h,c,n,plan,b+1,&sb,&kb);
 assert(ra==rb&&ka==kb&&bits(sa)==bits(sb)&&!memcmp(a,b,sizeof(a)));
 assert(a[0]==213&&a[n+1]==213);
 if(point&&ra==1){assert(old_calls==2*n&&rounding_calls==0&&ka==0);}
 else assert(rounding_calls<=old_calls);
}
int main(void) {
 merlin_bf16_quant_plan plans[]={
  {127,0x1.5p-17f,-127,127},{127,0x1p-126f,-128,127},
  {1,1,-7,3},{1000000,0x1p-120f,-128,127},{.5f,0x1p-133f,-127,127},
  {127,0x1.5p-17f,4,7},{127,0x1.5p-17f,-7,-4},{127,0x1.5p-17f,-3,-3}};
 unsigned finite=0;
 for(uint32_t raw=0;raw<65536;raw++) {
  if((raw&0x7f80)==0x7f80)continue;finite++;float x=word(raw<<16);
  for(unsigned p=0;p<sizeof(plans)/sizeof(plans[0]);p++)compare(&x,&x,&x,1,plans[p],1);
 }
 assert(finite==65280);
 float l[3]={-0.0f,0.0f,-0.0f},h[3]={0.0f,-0.0f,0.0f},c[3]={0.0f,-0.0f,-0.0f};
 for(unsigned p=0;p<sizeof(plans)/sizeof(plans[0]);p++) {
  compare(l,h,c,3,plans[p],1);float inv=merlin_frontier_bf16(1.0f/plans[p].epsilon);
  int expected=plans[p].lower>0?plans[p].lower:plans[p].upper<0?plans[p].upper:0;
  if(isfinite(inv)){
   assert(merlin_frontier_quant_prepared(-0.0f,inv,plans[p])==expected);
   assert(merlin_frontier_quant_prepared(0.0f,inv,plans[p])==expected);
  }
 }
 /* Stable source scale, one uncertain quantization bin and two exact points. */
 l[0]=h[0]=c[0]=127;l[1]=c[1]=word(0x3efb0000);h[1]=word(0x3f020000);l[2]=h[2]=c[2]=word(0x00010000);
 compare(l,h,c,3,plans[0],0);rounding_calls=0;uint8_t pending[3];float scale;size_t count;
 assert(merlin_frontier_row_finite_points(l,h,c,3,plans[0],pending,&scale,&count)==0);
 assert(rounding_calls==2&&pending[0]==0&&pending[1]==1&&pending[2]==0&&count==1);
 /* Unstable scale retains every possible extremum, including exact points. */
 l[0]=h[0]=c[0]=1;l[1]=c[1]=4;h[1]=8;l[2]=h[2]=c[2]=0;compare(l,h,c,3,plans[0],0);
 /* Saturated final integer observations retain source clamps. */
 assert(merlin_frontier_quant_prepared(1,1000000,plans[3])==127);
 assert(merlin_frontier_quant_prepared(-1,1000000,plans[3])==-128);
 for(int mode=0;mode<4;mode++) {
  int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};assert(!fesetround(modes[mode]));
  float x=1;compare(&x,&x,&x,1,plans[0],mode==0);
 }
 assert(!fesetround(FE_TONEAREST));
 /* Invalid endpoints, candidate or inverse refuse before touching outputs. */
 float bad[]={NAN,INFINITY,.1f};for(unsigned i=0;i<3;i++)compare(&bad[i],&bad[i],&bad[i],1,plans[0],0);
 float x=1,y=2,z=3;compare(&y,&x,&x,1,plans[0],0);compare(&x,&y,&z,1,plans[0],0);compare(&x,&x,&x,0,plans[0],0);
 return 0;
}
"""
    source = source.replace("@HEADER@", c_header(FrontierPointCellsContract(*([True] * 7))))
    path = tmp_path / "point.c"
    path.write_text(source)
    subprocess.run(
        [
            "/usr/bin/cc",
            "-O2",
            "-fno-builtin",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            "-I",
            str(runtime),
            str(path),
            "-lm",
            "-o",
            str(tmp_path / "point"),
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "point")], check=True)
