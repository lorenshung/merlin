"""All BF16 words, original product/scales, mixed rows, guards and effects."""

import subprocess

import pytest

from merlin.common.paths import data_path
from merlin.llvmlower.bf16_integer_observer import (
    BF16IntegerObserverContract,
    c_header,
    prepare_bf16_integer_observer,
)
from merlin.llvmlower.frontier_point_cells import (
    FrontierPointCellsContract,
)
from merlin.llvmlower.frontier_point_cells import (
    c_header as point_header,
)


@pytest.mark.parametrize("points", [False, True])
def test_exhaustive_words_and_complete_source_rows_ubsan(tmp_path, points):
    runtime = data_path("runtime", "c")
    original = (
        (runtime / "bf16_quant_frontier.h")
        .read_text()
        .replace("MERLIN_BF16_QUANT_FRONTIER_H", "ORIGINAL_MERLIN_BF16_QUANT_FRONTIER_H")
        .replace("merlin_", "original_merlin_")
    )
    selected = c_header(BF16IntegerObserverContract(*([True] * 7)))
    if points:
        selected = prepare_bf16_integer_observer(
            point_header(FrontierPointCellsContract(*([True] * 7))),
            contract=BF16IntegerObserverContract(*([True] * 7)),
        )
    source = (
        original
        + "\n"
        + selected
        + r"""
#include <assert.h>
static float word(uint32_t bits){float result;memcpy(&result,&bits,4);return result;}
static uint32_t bits(float value){uint32_t result;memcpy(&result,&value,4);return result;}
static void compare_row(const float*lo,const float*hi,const float*c,size_t n,
 merlin_bf16_quant_plan p){
 original_merlin_bf16_quant_plan old={p.divisor,p.epsilon,p.lower,p.upper};
 uint8_t a[19],b[19];memset(a,213,sizeof(a));memset(b,213,sizeof(b));
 float sa=-99,sb=-99;size_t ca=999,cb=999;
 int ra=original_merlin_frontier_row(lo,hi,c,n,old,a+1,&sa,&ca);
 int rb=merlin_frontier_row(lo,hi,c,n,p,b+1,&sb,&cb);
 assert(ra==rb&&ca==cb&&bits(sa)==bits(sb)&&!memcmp(a,b,sizeof(a)));
 assert(a[0]==213&&a[n+1]==213);
}
int main(void){
 assert(!fesetround(FE_TONEAREST));
 merlin_bf16_quant_plan plans[]={
  {127,0x1.5p-17f,-127,127},{127,0x1p-126f,-128,127},
  {1,1,-7,3},{1000000,0x1p-120f,-128,127},{.5f,0x1p-133f,-127,127},
  {127,0x1.5p-17f,4,7},{127,0x1.5p-17f,-7,-4},{127,0x1.5p-17f,-3,-3}};
 float inverses[]={0,.5f,1,1.5f,127,0x1p-133f,0x1.fep+127f,INFINITY};
 for(uint32_t raw=0;raw<65536;raw++){
  float x=word(raw<<16);
  for(size_t j=0;j<sizeof(plans)/sizeof(plans[0]);j++){
   merlin_bf16_quant_plan p=plans[j];
   original_merlin_bf16_quant_plan old={p.divisor,p.epsilon,p.lower,p.upper};
   int reference=original_merlin_frontier_quant_prepared(x,1,old);
   assert(reference==merlin_bf16_rne_clamped_integer_word((uint16_t)raw,p.lower,p.upper));
   for(size_t k=0;k<sizeof(inverses)/sizeof(inverses[0]);k++)
    assert(original_merlin_frontier_quant_prepared(x,inverses[k],old)==
           merlin_frontier_quant_prepared(x,inverses[k],p));
   compare_row(&x,&x,&x,1,p);
  }
 }
 /* Half-integer ties on either side of zero, integer parity and saturation. */
 for(int integer=-128;integer<128;integer++){
  float x=integer+.5f;
  int expected=(integer&1)?integer+1:integer;
  if(expected>127)expected=127;
  assert(merlin_frontier_quant_prepared(x,1,plans[1])==expected);
 }
 float lo[3]={127,word(0x3efb0000),-0.0f};
 float hi[3]={127,word(0x3f020000),0.0f};float c[3]={127,lo[1],0.0f};
 compare_row(lo,hi,c,3,plans[0]);
 lo[0]=hi[0]=c[0]=1;lo[1]=c[1]=4;hi[1]=8;compare_row(lo,hi,c,3,plans[0]);
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 for(size_t mode=0;mode<4;mode++){
  assert(!fesetround(modes[mode]));float x=1;compare_row(&x,&x,&x,1,plans[0]);
 }
 assert(!fesetround(FE_TONEAREST));
 float x=1,y=2,z=3,bad[]={NAN,INFINITY,-INFINITY,.1f};
 for(size_t j=0;j<sizeof(bad)/sizeof(bad[0]);j++)compare_row(&bad[j],&bad[j],&bad[j],1,plans[0]);
 compare_row(&y,&x,&x,1,plans[0]);compare_row(&x,&y,&z,1,plans[0]);
 compare_row(&x,&x,&x,0,plans[0]);
 merlin_bf16_quant_plan invalid=plans[0];invalid.upper=128;compare_row(&x,&x,&x,1,invalid);
 invalid=plans[0];invalid.lower=-129;compare_row(&x,&x,&x,1,invalid);
 invalid=plans[0];invalid.epsilon=0;compare_row(&x,&x,&x,1,invalid);
 invalid=plans[0];invalid.divisor=NAN;compare_row(&x,&x,&x,1,invalid);
 return 0;
}"""
    )
    if points:
        source = source.replace("int rb=merlin_frontier_row(", "int rb=merlin_frontier_row_finite_points(")
    path = tmp_path / "observer.c"
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
            str(tmp_path / "observer"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "observer")], check=True, capture_output=True)
