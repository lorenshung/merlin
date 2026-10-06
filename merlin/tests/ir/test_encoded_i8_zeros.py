import ctypes
import shutil
import subprocess

import pytest

from merlin.llvmlower.encoded_i8_zeros import EncodedI8Nonzero, c_header, summarize_encoded_i8


@pytest.mark.parametrize("transposed", [False, True])
def test_actual_signed_byte_rows_and_tail(transposed):
    rows, k = 17, 3
    data = [0] * (2 * rows * k)
    address = lambda p, r, z: p * rows * k + (z * rows + r if transposed else r * k + z)
    data[address(0, 16, 2)] = -128
    data[address(1, 4, 0)] = 127
    summary = summarize_encoded_i8(data, planes=2, rows=rows, reduction_length=k, block_rows=16, transposed=transposed)
    assert summary.nonzero == (False, True, True, False)
    data[address(0, 0, 1)] = 1
    assert (
        summarize_encoded_i8(data, planes=2, rows=rows, reduction_length=k, block_rows=16, transposed=transposed)
        != summary
    )


@pytest.mark.parametrize("data,kwargs", [([128], {}), ([-129], {}), ([0], {"rows": 2}), ([0], {"block_rows": 0})])
def test_refuse_unencoded_or_incomplete_domain(data, kwargs):
    args = dict(planes=1, rows=1, reduction_length=1, block_rows=1)
    args.update(kwargs)
    with pytest.raises(ValueError):
        summarize_encoded_i8(data, **args)


@pytest.mark.parametrize("flags", [(False,), (False, True, False), (0, 1), [False, True]])
def test_refuse_incomplete_or_mutable_metadata(flags):
    with pytest.raises(ValueError):
        EncodedI8Nonzero(1, 17, 3, 16, flags)


def test_compiled_producer_join_and_mutated_metadata_refusal(tmp_path):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")
    (tmp_path / "summary.h").write_text(c_header())
    (tmp_path / "test.c").write_text("""#include "summary.h"
int test(void){int8_t values[2*17*3]={0};uint8_t flags[4]={9,9,9,9};
 values[16*3+2]=-128;values[17*3+4*3]=127;
 merlin_encoded_i8_nonzero_begin(flags,2,17,16);
 for(unsigned p=0;p<2;p++)for(unsigned r=0;r<17;r++){
 uint8_t row=0;for(unsigned k=0;k<3;k++)row|=(uint8_t)values[p*17*3+r*3+k];
 merlin_encoded_i8_nonzero_join_row(flags,p,r,17,16,row);}
 if(flags[0]||flags[1]!=1||flags[2]!=1||flags[3])return 1;
 if(!merlin_encoded_i8_nonzero_verify(values,flags,2,17,3,16,0))return 2;
 values[0]=1;if(merlin_encoded_i8_nonzero_verify(values,flags,2,17,3,16,0))return 3;
 values[0]=0;flags[3]=1;if(merlin_encoded_i8_nonzero_verify(values,flags,2,17,3,16,0))return 4;
 return 0;}""")
    subprocess.run(
        [cc, "-std=c11", "-O2", "-shared", "-fPIC", str(tmp_path / "test.c"), "-o", str(tmp_path / "test.so")],
        check=True,
    )
    assert ctypes.CDLL(str(tmp_path / "test.so")).test() == 0
