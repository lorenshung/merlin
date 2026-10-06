import os
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.common.paths import merlin_dir
from merlin.llvmlower.source_numeric_capability import SourceNumericContract, emit_source_numeric_capability

GOOD = SourceNumericContract(True, True, True, True, True, True, True)


def test_default_prefix_is_empty():
    assert emit_source_numeric_capability(GOOD) == ""


@pytest.mark.parametrize(
    "field",
    ["round_to_nearest_even", "fused_single_rounding", "errno_unobserved", "nontrapping", "exception_flags_unobserved"],
)
def test_fma_contract_refuses(field):
    with pytest.raises(ValueError):
        emit_source_numeric_capability(replace(GOOD, **{field: False}), inline_fma=True)


@pytest.mark.parametrize("field", ["standard_bitcast_copy", "copy_interposition_unobserved"])
def test_copy_contract_refuses(field):
    with pytest.raises(ValueError):
        emit_source_numeric_capability(replace(GOOD, **{field: False}), inline_bitcasts=True)


CLASSIFICATION = replace(GOOD, standard_fp_classification=True, classification_interposition_unobserved=True)


@pytest.mark.parametrize(
    "field",
    [
        "standard_fp_classification",
        "classification_interposition_unobserved",
        "nontrapping",
        "exception_flags_unobserved",
    ],
)
def test_classification_contract_refuses(field):
    with pytest.raises(ValueError):
        emit_source_numeric_capability(replace(CLASSIFICATION, **{field: False}), inline_classification=True)


def test_existing_contract_does_not_admit_classification():
    with pytest.raises(ValueError):
        emit_source_numeric_capability(GOOD, inline_classification=True)


@pytest.mark.parametrize(
    "choice",
    [
        "inline_fma",
        "inline_bitcasts",
        "inline_classification",
        "inline_absolute_values",
        "inline_min_max",
        "inline_floor",
    ],
)
def test_nonboolean_choice_refuses(choice):
    with pytest.raises(ValueError):
        emit_source_numeric_capability(CLASSIFICATION, **{choice: 1})


FLOOR = replace(GOOD, standard_floor_values=True, floor_interposition_unobserved=True)


@pytest.mark.parametrize(
    "field",
    [
        "standard_floor_values",
        "floor_interposition_unobserved",
        "errno_unobserved",
        "nontrapping",
        "exception_flags_unobserved",
    ],
)
def test_floor_contract_refuses(field):
    with pytest.raises(ValueError):
        emit_source_numeric_capability(replace(FLOOR, **{field: False}), inline_floor=True)


def test_existing_contract_does_not_admit_floor():
    with pytest.raises(ValueError):
        emit_source_numeric_capability(GOOD, inline_floor=True)


def test_floor_has_no_source_rounding_mode_restriction():
    # Floor is independent of the current rounding direction. It grants no FMA
    # rounding permission and the selected helper must preserve signed zero.
    emit_source_numeric_capability(replace(FLOOR, round_to_nearest_even=False), inline_floor=True)


SOURCE = r"""
#include "source_f32_math.h"
#include <stdint.h>
#include <fenv.h>
static uint32_t bits(float x){uint32_t n;__builtin_memcpy(&n,&x,4);return n;}
static float value(uint32_t n){float x;__builtin_memcpy(&x,&n,4);return x;}
int main(void){
 float (*volatile original)(float,float,float)=fmaf;
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 uint32_t rng=713;
 for(int mode=0;mode<4;mode++){
  if(fesetround(modes[mode]))return 1;
  for(int i=0;i<10000;i++){
   rng=rng*1664525u+1013904223u;float a=value(rng&0xfeffffffu);
   rng=rng*1664525u+1013904223u;float b=value(rng&0xfeffffffu);
   rng=rng*1664525u+1013904223u;float c=value(rng&0xfeffffffu);
   float expected=original(a,b,c),actual=MERLIN_SOURCE_F32_FMA(a,b,c);
   if(bits(expected)!=bits(actual))return 2;
  }
 }
 unsigned char a[19],b[19];for(int i=0;i<19;i++)a[i]=(unsigned char)(13*i);
 for(int n=2;n<=8;n*=2)for(int off=0;off<8;off++){
  for(int i=0;i<19;i++)b[i]=0xa5;
  MERLIN_SOURCE_BITCAST_COPY(b+off,a+off,n);
  for(int i=0;i<19;i++)if(b[i]!=(i>=off&&i<off+n?a[i]:0xa5))return 3;
 }
 return 0;
}
"""


@pytest.mark.parametrize("fma,copy", [(False, False), (True, False), (False, True), (True, True)])
def test_actual_native_rounding_and_unaligned_copy(tmp_path, fma, copy):
    cc = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not cc:
        pytest.skip("Clang builtin capability test requires Clang")
    c = tmp_path / "test.c"
    c.write_text(emit_source_numeric_capability(GOOD, inline_fma=fma, inline_bitcasts=copy) + SOURCE)
    exe = tmp_path / "test"
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-builtin",
            "-fno-fast-math",
            "-frounding-math",
            "-ffp-contract=off",
            "-I",
            str(merlin_dir() / "runtime/c"),
            str(c),
            "-lm",
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


def test_capability_cannot_arrive_after_numeric_headers(tmp_path):
    cc = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not cc:
        pytest.skip("Clang capability test requires Clang")
    c = tmp_path / "late.c"
    c.write_text(
        '#include "source_f32_math.h"\n'
        + emit_source_numeric_capability(GOOD, inline_fma=True)
        + "int main(void){return 0;}\n"
    )
    result = subprocess.run(
        [cc, "-I", str(merlin_dir() / "runtime/c"), "-c", str(c), "-o", str(tmp_path / "late.o")],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert result.returncode != 0
    assert "must precede numeric headers" in result.stderr


CLASSIFICATION_SOURCE = r"""
#include "source_f32_math.h"
#include <stdint.h>
#include <fenv.h>
#include <errno.h>
static float f32(uint32_t n){float x;__builtin_memcpy(&x,&n,4);return x;}
static double f64(uint64_t n){double x;__builtin_memcpy(&x,&n,8);return x;}
static int seen;
static double once(void){seen++;return 1;}
static int check32(uint32_t n){
 return !!MERLIN_SOURCE_ISFINITE(f32(n))==((n&0x7f800000u)!=0x7f800000u);
}
static int check64(uint64_t n){
 return !!MERLIN_SOURCE_ISFINITE(f64(n))==((n&UINT64_C(0x7ff0000000000000))!=UINT64_C(0x7ff0000000000000));
}
int main(void){
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 uint64_t rng=713;
 for(int mode=0;mode<4;mode++){
  if(fesetround(modes[mode]))return 1;
  errno=123;
  /* Entire BF16 representation, including both signed zeros, infinities,
     quiet/signaling NaNs; no floating conversion supplies the oracle. */
  for(uint32_t i=0;i<65536;i++)if(!check32(i<<16))return 2;
  uint32_t mf[]={0,1,0x3fffff,0x400000,0x7ffffe,0x7fffff};
  uint64_t md[]={0,1,UINT64_C(0x7ffffffffffff),UINT64_C(0x8000000000000),UINT64_C(0xffffffffffffe),UINT64_C(0xfffffffffffff)};
  for(unsigned s=0;s<2;s++)for(unsigned e=0;e<256;e++)for(unsigned m=0;m<6;m++)
   if(!check32((s<<31)|(e<<23)|mf[m]))return 3;
  for(unsigned s=0;s<2;s++)for(unsigned e=0;e<2048;e++)for(unsigned m=0;m<6;m++)
   if(!check64(((uint64_t)s<<63)|((uint64_t)e<<52)|md[m]))return 4;
  for(unsigned i=0;i<10000;i++){
   rng=rng*UINT64_C(6364136223846793005)+1;
   if(!check32((uint32_t)rng)||!check64(rng))return 5;
  }
  seen=0;if(!MERLIN_SOURCE_ISFINITE(once())||seen!=1)return 6;
  if(fegetround()!=modes[mode]||errno!=123)return 7;
 }
 return 0;
}
"""


@pytest.mark.parametrize("inline", [False, True])
def test_classification_all_bf16_and_f32_f64_boundaries(tmp_path, inline):
    cc = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not cc:
        pytest.skip("Clang classification capability test requires Clang")
    c = tmp_path / "classification.c"
    c.write_text(emit_source_numeric_capability(CLASSIFICATION, inline_classification=inline) + CLASSIFICATION_SOURCE)
    exe = tmp_path / "classification"
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-builtin",
            "-fno-fast-math",
            "-frounding-math",
            "-ffp-contract=off",
            "-I",
            str(merlin_dir() / "runtime/c"),
            str(c),
            "-lm",
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


ABSOLUTE = replace(
    GOOD,
    standard_absolute_value=True,
    absolute_value_interposition_unobserved=True,
    absolute_value_nan_payload_unobserved=True,
)


@pytest.mark.parametrize(
    "field",
    [
        "standard_absolute_value",
        "absolute_value_interposition_unobserved",
        "absolute_value_nan_payload_unobserved",
        "errno_unobserved",
        "nontrapping",
        "exception_flags_unobserved",
    ],
)
def test_absolute_value_contract_refuses(field):
    with pytest.raises(ValueError):
        emit_source_numeric_capability(replace(ABSOLUTE, **{field: False}), inline_absolute_values=True)


ABSOLUTE_SOURCE = r"""
#include "source_f32_math.h"
#include <stdint.h>
#include <fenv.h>
static float f32(uint32_t n){float x;__builtin_memcpy(&x,&n,4);return x;}
static double f64(uint64_t n){double x;__builtin_memcpy(&x,&n,8);return x;}
static uint32_t b32(float x){uint32_t n;__builtin_memcpy(&n,&x,4);return n;}
static uint64_t b64(double x){uint64_t n;__builtin_memcpy(&n,&x,8);return n;}
static float (*volatile source32)(float)=fabsf;
static double (*volatile source64)(double)=fabs;
static int check32(uint32_t n){
 uint32_t expected=n&UINT32_C(0x7fffffff);
 return b32(source32(f32(n)))==expected&&b32(MERLIN_SOURCE_F32_ABS(f32(n)))==expected;
}
static int check64(uint64_t n){
 uint64_t expected=n&UINT64_C(0x7fffffffffffffff);
 return b64(source64(f64(n)))==expected&&b64(MERLIN_SOURCE_F64_ABS(f64(n)))==expected;
}
static int seen;
static double once(void){seen++;return -1;}
int main(void){
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 uint64_t rng=713;
 for(int mode=0;mode<4;mode++){
  if(fesetround(modes[mode]))return 1;
  for(uint32_t i=0;i<65536;i++)if(!check32(i<<16))return 2;
  uint32_t mf[]={0,1,0x3fffff,0x400000,0x7ffffe,0x7fffff};
  uint64_t md[]={0,1,UINT64_C(0x7ffffffffffff),UINT64_C(0x8000000000000),UINT64_C(0xffffffffffffe),UINT64_C(0xfffffffffffff)};
  for(unsigned s=0;s<2;s++)for(unsigned e=0;e<256;e++)for(unsigned m=0;m<6;m++)
   if(!check32((s<<31)|(e<<23)|mf[m]))return 3;
  for(unsigned s=0;s<2;s++)for(unsigned e=0;e<2048;e++)for(unsigned m=0;m<6;m++)
   if(!check64(((uint64_t)s<<63)|((uint64_t)e<<52)|md[m]))return 4;
  for(unsigned i=0;i<10000;i++){
   rng=rng*UINT64_C(6364136223846793005)+1;
   if(!check32((uint32_t)rng)||!check64(rng))return 5;
  }
  seen=0;if(MERLIN_SOURCE_F64_ABS(once())!=1||seen!=1)return 6;
  if(fegetround()!=modes[mode])return 7;
 }
 return 0;
}
"""


@pytest.mark.parametrize("inline", [False, True])
def test_absolute_values_all_bf16_and_f32_f64_boundaries(tmp_path, inline):
    cc = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not cc:
        pytest.skip("Clang absolute-value capability test requires Clang")
    c = tmp_path / "absolute.c"
    c.write_text(emit_source_numeric_capability(ABSOLUTE, inline_absolute_values=inline) + ABSOLUTE_SOURCE)
    exe = tmp_path / "absolute"
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-builtin",
            "-fno-fast-math",
            "-frounding-math",
            "-ffp-contract=off",
            "-I",
            str(merlin_dir() / "runtime/c"),
            str(c),
            "-lm",
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)


MINMAX = replace(
    GOOD, standard_min_max=True, min_max_interposition_unobserved=True, min_max_signed_zero_nan_payload_unobserved=True
)


@pytest.mark.parametrize(
    "field",
    [
        "standard_min_max",
        "min_max_interposition_unobserved",
        "min_max_signed_zero_nan_payload_unobserved",
        "errno_unobserved",
        "nontrapping",
        "exception_flags_unobserved",
    ],
)
def test_min_max_contract_refuses(field):
    with pytest.raises(ValueError):
        emit_source_numeric_capability(replace(MINMAX, **{field: False}), inline_min_max=True)


def test_existing_contract_does_not_admit_min_max():
    with pytest.raises(ValueError):
        emit_source_numeric_capability(ABSOLUTE, inline_min_max=True)


MINMAX_SOURCE = r"""
#include <math.h>
#include "source_f32_math.h"
#include <fenv.h>
#include <stdint.h>
#include <stdio.h>
static uint32_t b32(float x){uint32_t n;__builtin_memcpy(&n,&x,4);return n;}
static uint64_t b64(double x){uint64_t n;__builtin_memcpy(&n,&x,8);return n;}
static float f32(uint32_t n){float x;__builtin_memcpy(&x,&n,4);return x;}
static double f64(uint64_t n){double x;__builtin_memcpy(&x,&n,8);return x;}
static float(*volatile original_min32)(float,float)=fminf;
static float(*volatile original_max32)(float,float)=fmaxf;
static double(*volatile original_min64)(double,double)=fmin;
static double(*volatile original_max64)(double,double)=fmax;

static int equal32(float a,float b){
 uint32_t x=b32(a),y=b32(b);
 if((x&UINT32_C(0x7fffffff))==0&&(y&UINT32_C(0x7fffffff))==0)return 1;
 if((x&UINT32_C(0x7fffffff))>UINT32_C(0x7f800000)&&
    (y&UINT32_C(0x7fffffff))>UINT32_C(0x7f800000))return 1;
 return x==y;
}
static int equal64(double a,double b){
 uint64_t x=b64(a),y=b64(b);
 if((x&UINT64_C(0x7fffffffffffffff))==0&&(y&UINT64_C(0x7fffffffffffffff))==0)return 1;
 if((x&UINT64_C(0x7fffffffffffffff))>UINT64_C(0x7ff0000000000000)&&
    (y&UINT64_C(0x7fffffffffffffff))>UINT64_C(0x7ff0000000000000))return 1;
 return x==y;
}

static unsigned long checks;
static int pair32(uint32_t a,uint32_t b){
 float x=f32(a),y=f32(b);checks++;
 return equal32(original_min32(x,y),MERLIN_SOURCE_F32_MIN(x,y))&&
        equal32(original_max32(x,y),MERLIN_SOURCE_F32_MAX(x,y));
}
static int pair64(uint64_t a,uint64_t b){
 double x=f64(a),y=f64(b);checks++;
 return equal64(original_min64(x,y),MERLIN_SOURCE_F64_MIN(x,y))&&
        equal64(original_max64(x,y),MERLIN_SOURCE_F64_MAX(x,y));
}
static int observed;
static double once(void){observed++;return observed==1?3:4;}
int main(void){
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 uint64_t rng=713;
 const uint32_t df[]={0,0x80000000,1,0x80000001,0x007fffff,0x807fffff,0x00800000,0x80800000,0x3f800000,0xbf800000,0x7f7fffff,0xff7fffff,0x7f800000,0xff800000,0x7fc00000,0xffc00000,0x7f800001,0xff800001,0x7fffffff,0xffffffff};
 const uint64_t dd[]={0,UINT64_C(0x8000000000000000),1,UINT64_C(0x8000000000000001),UINT64_C(0x000fffffffffffff),UINT64_C(0x800fffffffffffff),UINT64_C(0x0010000000000000),UINT64_C(0x8010000000000000),UINT64_C(0x3ff0000000000000),UINT64_C(0xbff0000000000000),UINT64_C(0x7fefffffffffffff),UINT64_C(0xffefffffffffffff),UINT64_C(0x7ff0000000000000),UINT64_C(0xfff0000000000000),UINT64_C(0x7ff8000000000000),UINT64_C(0xfff8000000000000),UINT64_C(0x7ff0000000000001),UINT64_C(0xfff0000000000001),UINT64_C(0x7fffffffffffffff),UINT64_C(0xffffffffffffffff)};
 for(unsigned long mode=0;mode<4;mode++){
  if(fesetround(modes[mode]))return 10;
  for(unsigned i=0;i<20;i++)for(unsigned j=0;j<20;j++){
   if(!pair32(df[i],df[j]))return 1;
   if(!pair64(dd[i],dd[j]))return 2;
  }
  for(uint32_t i=0;i<65536;i++){
   rng=rng*UINT64_C(6364136223846793005)+1;
   if(!pair32(i<<16,(uint32_t)rng))return 3;
   if(!pair32((uint32_t)rng,i<<16))return 4;
  }
  for(unsigned i=0;i<20000;i++){
   rng=rng*UINT64_C(6364136223846793005)+1;uint64_t a=rng;
   rng=rng*UINT64_C(6364136223846793005)+1;uint64_t b=rng;
   if(!pair32((uint32_t)a,(uint32_t)b))return 5;
   if(!pair64(a,b))return 6;
  }
  observed=0;if(MERLIN_SOURCE_F64_MIN(once(),once())!=3||observed!=2)return 7;
  observed=0;if(MERLIN_SOURCE_F64_MAX(once(),once())!=4||observed!=2)return 8;
  if(fegetround()!=modes[mode])return 9;
 }

 if(checks!=687488)return 11;
 return 0;
}
"""


@pytest.mark.parametrize("inline", [False, True])
def test_min_max_source_values_and_unobserved_zero_nan_distinctions(tmp_path, inline):
    cc = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    if not cc:
        pytest.skip("Clang min/max capability test requires Clang")
    c = tmp_path / "minmax.c"
    c.write_text(emit_source_numeric_capability(MINMAX, inline_min_max=inline) + MINMAX_SOURCE)
    exe = tmp_path / "minmax"
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-builtin",
            "-fno-fast-math",
            "-frounding-math",
            "-ffp-contract=off",
            "-I",
            str(merlin_dir() / "runtime/c"),
            str(c),
            "-lm",
            "-o",
            str(exe),
        ],
        check=True,
    )
    subprocess.run([str(exe)], check=True)
