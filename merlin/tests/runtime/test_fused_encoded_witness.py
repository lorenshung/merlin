"""Canonical encoder differential proof, including its private equality epoch."""

import ctypes as C
import shutil
import subprocess

import numpy as np
import pytest
from test_source_attention_frontier import PLAN

from merlin.common.paths import merlin_dir
from merlin.llvmlower.fused_encoded_witness import fused_encoded_row_header, prepare_fused_encoded_witness
from merlin.llvmlower.source_attention_frontier import emit_source_attention_frontier

C_SOURCE = r"""
#include "encoded_row_equality.h"
#include <string.h>
int compare(const float*source,int length,int stride,int transpose,int uncertainty){
 float rf[512],step1=42,step2=42,lo[512],hi[512];double old[512],new[512];
 int8_t a[4096],b[4096];unsigned char flag1=0xa5,flag2=0xa5;
 if(length<=0||length*stride>512)return -10;
 memset(a,0xa5,sizeof(a));memset(b,0xa5,sizeof(b));memset(old,0xa5,sizeof(old));memset(new,0xa5,sizeof(new));
 for(int i=0;i<length;i++){lo[i*stride]=hi[i*stride]=source[i*stride];}
 if(uncertainty)lo[(length-1)*stride]=nextafterf(lo[(length-1)*stride],-INFINITY);
 int ds=length*3,es=transpose?3:1;
 merlin_fma_bound env=merlin_fma_bound_begin();
 int x=merlin_bf16_radix_row(&env,source,length,stride,rf,1,a,ds,es,3,&step1);
 int y=merlin_bf16_radix_row_widen(&env,source,length,stride,new,1,b,ds,es,3,&step2,lo,hi,&flag2);
 if(x!=y)return -1;if(!x)return 0;
 int same=1;for(int i=0;i<length;i++){old[i]=(double)rf[i];same=same&&isfinite(source[i*stride])&&isfinite(rf[i])&&source[i*stride]==rf[i]&&lo[i*stride]==source[i*stride]&&hi[i*stride]==source[i*stride];}flag1=same;
 if(memcmp(old,new,sizeof(old))||memcmp(a,b,sizeof(a))||memcmp(&step1,&step2,4)||flag1!=flag2)return -2;
 return 1;
}
int environment_refuses(void){int old=fegetround();float a=1;int modes[]={FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO};for(int i=0;i<3;i++){fesetround(modes[i]);if(compare(&a,1,1,0,0)!=0){fesetround(old);return 0;}}fesetround(old);return 1;}
"""


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("compiler required")
    w = tmp_path_factory.mktemp("fused-encoded")
    (w / "p.c").write_text(fused_encoded_row_header() + C_SOURCE)
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
            str(w / "p.c"),
            "-lm",
            "-o",
            str(w / "p.so"),
        ],
        check=True,
    )
    lib = C.CDLL(str(w / "p.so"))
    lib.compare.argtypes = [C.c_void_p, C.c_int, C.c_int, C.c_int, C.c_int]
    return lib


def test_every_bf16_and_environment(native):
    raw = (np.arange(65536, dtype=np.uint32) << 16).view(np.float32)
    for i in range(65536):
        assert native.compare(raw[i : i + 1].ctypes.data, 1, 1, 0, 0) >= 0
    assert native.environment_refuses() == 1


@pytest.mark.parametrize("length,stride,transpose", [(1, 1, 0), (3, 2, 1), (17, 3, 0), (64, 1, 1), (127, 2, 1)])
@pytest.mark.parametrize("uncertainty", [0, 1])
def test_random_rows_transpose_dirty_guards(native, length, stride, transpose, uncertainty):
    rng = np.random.default_rng(312 + length)
    for _ in range(30):
        raw = (rng.integers(0, 65536, length * stride, dtype=np.uint32) << 16).view(np.float32)
        assert native.compare(raw.ctypes.data, length, stride, transpose, uncertainty) >= 0


def test_default_and_admission():
    assert emit_source_attention_frontier(PLAN, symbol="p") == emit_source_attention_frontier(
        PLAN, symbol="p", fuse_encoded_witness=False
    )
    for bad in [1, None, "yes"]:
        with pytest.raises(ValueError, match="must be bool"):
            emit_source_attention_frontier(PLAN, symbol="p", fuse_encoded_witness=bad)
    with pytest.raises(ValueError, match="admitted encoded"):
        emit_source_attention_frontier(PLAN, symbol="p", fuse_encoded_witness=True)
    with pytest.raises(ValueError, match="unknown producer"):
        prepare_fused_encoded_witness("unknown")


@pytest.fixture(scope="module")
def group_native(tmp_path_factory):
    from test_source_attention_frontier import EXTRA, View

    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("compiler required")
    flags = dict(
        prepare_endpoint_rows=True,
        word_interval_enclosure=True,
        prepare_product_domain=True,
        prepare_required_norms=True,
        retain_certified_rows=True,
        separable_source_radius=True,
        prepare_softmax_domain=True,
        integer_reconstruction=True,
        prepare_probability_bins=True,
        prepare_encoded_rows=True,
        prepare_softmax_spans=True,
        polynomial_batch_four=True,
        fuse_encoded_witness=True,
    )
    w = tmp_path_factory.mktemp("fused-complete")
    (w / "p.c").write_text(emit_source_attention_frontier(PLAN, symbol="test_provider", **flags) + EXTRA)
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
            str(w / "p.c"),
            "-lm",
            "-o",
            str(w / "p.so"),
        ],
        check=True,
    )
    lib = C.CDLL(str(w / "p.so"))
    lib.test_provider_workspace_bytes.restype = C.c_size_t
    lib.run.argtypes = [C.POINTER(View), C.POINTER(View), C.c_void_p, C.c_size_t, C.c_int]
    lib.oracle.argtypes = [C.POINTER(View), C.c_void_p]
    return lib


@pytest.mark.parametrize(
    "seed,masked,strided", [(1, False, False), (2, False, True), (3, True, True), (4, False, False)]
)
def test_complete_rectangular_source_gate(group_native, seed, masked, strided):
    from test_source_attention_frontier import test_independent_source_quant_and_reused_dirty_workspace

    test_independent_source_quant_and_reused_dirty_workspace(group_native, seed, masked, strided)


@pytest.mark.parametrize(
    "failure", ["capacity", "alignment", "callback", "nonfinite", "stride", "shape", "overlapping_output"]
)
def test_complete_refusal_preserves_output(group_native, failure):
    from test_source_attention_frontier import test_refusal_preserves_public_destination

    test_refusal_preserves_public_destination(group_native, failure)
