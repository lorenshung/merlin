"""Output-only guards avoid input scans without changing ordered source math."""

import ctypes
import hashlib
import subprocess
from copy import deepcopy

import numpy as np
import pytest

from merlin.llvmlower.quantized_affine_pair import (
    derive,
    derive_output_value_guard,
    emit_correction,
    predictor_table,
    source_table,
)

SOURCE = dict(
    lhs_scale=0.011258588172495365, rhs_scale=0.00940733402967453, output_scale=0.011643771082162857, relu=True
)
CONTRACTS = [
    (SOURCE, dict(p=298, q=249, scale=0.0032446938566863537)),
    (SOURCE, dict(p=73, q=61, scale=0.013245166279375553)),
    (dict(lhs_scale=0.003, rhs_scale=0.004, output_scale=0.002, relu=False), dict(p=3, q=4, scale=0.5)),
    (dict(lhs_scale=1.0, rhs_scale=1.0, output_scale=1.0, relu=False), dict(p=1, q=1, scale=1.0)),
]


def test_complete_output_guard_and_old_defaults():
    proof = derive(**SOURCE, **CONTRACTS[0][1])
    guard = derive_output_value_guard(proof)
    assert proof["mismatched_pairs"] == 1 and guard["pairs"] == 65536
    assert guard["predicate"] == dict(mask=255, value=107)
    assert guard["inexact_raw_byte_values"] == guard["accepted_raw_byte_values"] == [107]
    old = derive(**SOURCE, **CONTRACTS[1][1])
    for packed, sha in [
        (False, "3ed675b68ce68b9d61ddba7fbcd905637aaaddbc2554202af10b12a35224caea"),
        (True, "69fd8b50b32b45560c3c52543b114e8c469fc8e5844e312302a7f36f36d45486"),
    ]:
        assert hashlib.sha256(emit_correction(old, "correct", packed_prefix=packed).encode()).hexdigest() == sha


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("unaligned", [False, True])
@pytest.mark.parametrize("source,predictor", CONTRACTS)
def test_actual_ordered_c_all65536_pairs_readonly_inputs_and_tails(tmp_path, packed, unaligned, source, predictor):
    proof = derive(**source, **predictor)
    emitted = emit_correction(proof, "correct", packed_prefix=packed, output_value_guard=True)
    c = tmp_path / "correct.c"
    c.write_text(emitted)
    lib = tmp_path / (hashlib.sha256(emitted.encode()).hexdigest() + ".so")
    subprocess.run(
        ["cc", "-O2", "-fno-fast-math", "-ffp-contract=off", "-shared", "-fPIC", str(c), "-o", str(lib)], check=True
    )
    correct = ctypes.CDLL(str(lib)).correct
    correct.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t]
    indices = np.arange(65536, dtype=np.int64)
    indices = (indices * 40503 + 29) % 65536
    a = (indices // 256 - 128).astype(np.int8)
    b = (indices % 256 - 128).astype(np.int8)
    offset = int(unaligned)
    arrays = [np.full(65536 + 32, 37, dtype=np.int8) for _ in range(3)]
    arrays[0][offset : offset + 65536] = a
    arrays[1][offset : offset + 65536] = b
    predicted = predictor_table(**predictor, relu=source["relu"]).ravel()[indices]
    expected = source_table(**source).ravel()[indices]
    for length in (65536, 1013, 0, 1, 7, 8, 9):
        arrays[2][:] = 37
        arrays[2][offset : offset + length] = predicted[:length]
        correct(*(v.ctypes.data + offset for v in arrays), length)
        np.testing.assert_array_equal(arrays[2][offset : offset + length], expected[:length])
        assert np.all(arrays[2][:offset] == 37) and np.all(arrays[2][offset + length :] == 37)
        np.testing.assert_array_equal(arrays[0][offset : offset + 65536], a)
        np.testing.assert_array_equal(arrays[1][offset : offset + 65536], b)


def test_rejected_output_values_do_not_read_input_storage(tmp_path):
    proof = derive(**SOURCE, **CONTRACTS[0][1])
    guard = derive_output_value_guard(proof)
    safe = [v for v in range(256) if v not in guard["accepted_raw_byte_values"]]
    # This is a direct implementation test of the rejected branch, not a legal
    # predictor invocation: real callers must keep original A/B alive.
    source = tmp_path / "safe.c"
    source.write_text(
        emit_correction(proof, "correct", output_value_guard=True)
        + f"""
#include <stdlib.h>
int main(void) {{
 unsigned char *c=malloc(2048);unsigned safe[]={{{",".join(map(str, safe))}}};
 for(unsigned offset=0;offset<8;offset++){{
  for(unsigned i=0;i<1021;i++)c[offset+i]=safe[i%{len(safe)}];
  correct(NULL,NULL,(int8_t*)c+offset,1021);
  for(unsigned i=0;i<1021;i++)if(c[offset+i]!=safe[i%{len(safe)}])return 1;
 }}
 free(c);return 0;
}}
"""
    )
    exe = tmp_path / "safe"
    subprocess.run(["cc", "-O2", "-fno-fast-math", "-ffp-contract=off", str(source), "-o", str(exe)], check=True)
    subprocess.run([str(exe)], check=True)


def test_changed_certificate_and_nonboolean_policy_refuse():
    proof = derive(**SOURCE, **CONTRACTS[0][1])
    changed = deepcopy(proof)
    changed["predictor_bit_guard"]["mask"] = 0
    with pytest.raises(ValueError, match="unchanged complete pair"):
        derive_output_value_guard(changed)
    with pytest.raises(ValueError, match="unchanged complete pair"):
        emit_correction(changed, "correct", output_value_guard=True)
    with pytest.raises(ValueError):
        emit_correction(proof, "correct", output_value_guard=1)


def test_conservative_word_zero_predicate_all_lane_masks_values_and_borrows(tmp_path):
    source = tmp_path / "word.c"
    source.write_text("""#include <stdint.h>
uint64_t word(uint64_t value,uint64_t mask,uint64_t wanted){
 uint64_t d=(value&mask)^wanted;
 return (d-UINT64_C(0x0101010101010101))&~d&UINT64_C(0x8080808080808080);
}""")
    lib = tmp_path / "word.so"
    subprocess.run(["cc", "-O2", "-shared", "-fPIC", str(source), "-o", str(lib)], check=True)
    call = ctypes.CDLL(str(lib)).word
    call.argtypes = [ctypes.c_uint64] * 3
    call.restype = ctypes.c_uint64
    for mask, wanted in [(255, 107), (1, 1), (0, 0), (127, 63)]:
        for flags in range(256):
            for value in range(256):
                lanes = [wanted if flags & (1 << i) else value for i in range(8)]
                word = sum(v << (8 * i) for i, v in enumerate(lanes))
                required = sum(0x80 << (8 * i) for i, v in enumerate(lanes) if v & mask == wanted)
                actual = call(word, mask * 0x0101010101010101, wanted * 0x0101010101010101)
                assert actual & required == required
