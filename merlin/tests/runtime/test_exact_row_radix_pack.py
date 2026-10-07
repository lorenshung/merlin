import ctypes as C
import math
import shutil
import struct
import subprocess

import pytest

from merlin.common.paths import data_path
from merlin.llvmlower.exact_row_radix_pack import c_header, prepare_exact_row_radix
from merlin.llvmlower.fused_encoded_witness import fused_encoded_row_header


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native compiler required")
    w = tmp_path_factory.mktemp("exact-row-radix")
    (w / "old.h").write_text(
        fused_encoded_row_header()
        .replace("MERLIN_FUSED_ENCODED_ROW_H", "OLD_ROW_H")
        .replace("merlin_bf16_radix_row_widen", "old_row")
    )
    (w / "new.h").write_text(c_header())
    (w / "test.c").write_text(r"""
#include "old.h"
#include "new.h"
#include <string.h>
int check(const float*source,size_t length,unsigned digits,unsigned transpose,unsigned endpoints,int mode){
 double old[256],fresh[256];int8_t op[768],np[768];unsigned char of,nf;float os,ns;
 float low[256],high[256];
 if(!length||length>64)return -2;
 memset(old,0xa5,sizeof old);memcpy(fresh,old,sizeof old);
 memset(op,0x5a,sizeof op);memcpy(np,op,sizeof op);
 for(size_t i=0;i<length;i++){low[i]=source[i];high[i]=source[i];}
 if(endpoints==2)low[0]=-INFINITY;
 int saved=fegetround();if(mode)fesetround(FE_UPWARD);
 merlin_fma_bound env=merlin_fma_bound_begin();
 size_t stride=transpose?3:1;
 int a=old_row(&env,source,length,1,old,stride,op,256,stride,digits,&os,
 endpoints?low:0,endpoints?high:0,&of);
 int b=merlin_bf16_radix_row_widen(&env,source,length,1,fresh,stride,np,256,stride,digits,&ns,
 endpoints?low:0,endpoints?high:0,&nf);
 fesetround(saved);
 if(a!=b)return -1;
 if(!a)return 0;
 return memcmp(old,fresh,sizeof old)==0&&memcmp(op,np,sizeof op)==0&&memcmp(&os,&ns,4)==0&&of==nf?1:-1;
}
""")
    subprocess.run(
        [
            cc,
            "-O2",
            "-shared",
            "-fPIC",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I" + str(data_path("runtime", "c")),
            str(w / "test.c"),
            "-lm",
            "-o",
            str(w / "test.so"),
        ],
        check=True,
    )
    lib = C.CDLL(str(w / "test.so"))
    lib.check.argtypes = [C.POINTER(C.c_float), C.c_size_t] + [C.c_uint] * 3 + [C.c_int]
    return lib, w, cc


@pytest.mark.parametrize("maximum", [1.0, 2**100, 2**-100])
def test_all_bf16_words_and_three_digit_counts(native, maximum):
    lib, _, _ = native
    for bits in range(65536):
        value = struct.unpack("!f", struct.pack("!I", bits << 16))[0]
        data = (C.c_float * 2)(maximum, value)
        for digits in (1, 2, 3):
            assert lib.check(data, 2, digits, 0, 0, 0) >= 0


@pytest.mark.parametrize("length", [1, 3, 7, 17, 64])
@pytest.mark.parametrize("endpoints", [0, 1, 2])
def test_strided_planes_tails_point_and_uncertain_inputs(native, length, endpoints):
    data = (C.c_float * length)(*[math.ldexp((-1) ** i * (128 + i % 127), -20 - i % 25) for i in range(length)])
    assert native[0].check(data, length, 3, 1, endpoints, 0) == 1


def test_zero_and_environment_refusal(native):
    data = (C.c_float * 3)(0, -0.0, 1)
    assert native[0].check(data, 3, 3, 0, 0, 0) == 1
    assert native[0].check(data, 3, 3, 0, 0, 1) == 0


def test_changed_producer_refuses():
    with pytest.raises(ValueError):
        prepare_exact_row_radix("")


def test_ubsan(native):
    _, w, cc = native
    (w / "main.c").write_text(
        '#include "test.c"\nint main(void){float x[4]={0,-0.0f,1,0x1p-30f};return check(x,4,3,1,0,0)==1?0:1;}'
    )
    subprocess.run(
        [
            cc,
            "-O2",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-I" + str(data_path("runtime", "c")),
            str(w / "main.c"),
            "-lm",
            "-o",
            str(w / "main"),
        ],
        check=True,
    )
    subprocess.run([str(w / "main")], check=True)
