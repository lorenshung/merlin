import dataclasses
import hashlib

import pytest

from merlin.llvmlower.rounded_polynomial_monotonicity import (
    RoundedPolynomialMonotonicity,
    consume_rounded_polynomial_monotonicity,
)
from merlin.llvmlower.source_numeric_capability import SourceNumericContract

HEADER = "  out.rounding_error=total_error;out.word_budget=(uint32_t)budget;out.fast_valid=1;"
WORDS = (0xC2AEAC50, 0x3FB8AA3B, 0xBDA235D5, 0xBE65B8F5, 0x3E9B69F0, 0x38E077A1, 0x4B000000, 0x4E7E0000)


def fixture():
    proof = RoundedPolynomialMonotonicity(
        WORDS, 1118743633, "a" * 64, "b" * 64, hashlib.sha256(HEADER.encode()).hexdigest(), True, True, True
    )
    contract = SourceNumericContract(*([True] * 7), standard_floor_values=True, floor_interposition_unobserved=True)
    return proof, contract


def test_default_identity():
    assert consume_rounded_polynomial_monotonicity(HEADER, None) == HEADER


def test_explicit_preparation_only():
    proof, contract = fixture()
    result = consume_rounded_polynomial_monotonicity(HEADER, proof, contract)
    assert "upper==0.0f" in result
    assert "FE_TONEAREST" in result
    assert "if(same)out.word_budget=0" in result
    assert HEADER in result


@pytest.mark.parametrize(
    "field,value",
    [
        ("checked_negative_words", 1118743632),
        ("complete_domain", False),
        ("gradual_underflow", False),
        ("evaluator_equivalent", False),
        ("evidence_sha256", "unknown"),
        ("plan_words", (0,) * 8),
    ],
)
def test_incomplete_proof_refuses(field, value):
    proof, contract = fixture()
    with pytest.raises(ValueError):
        consume_rounded_polynomial_monotonicity(HEADER, dataclasses.replace(proof, **{field: value}), contract)


def test_mutated_source_refuses():
    proof, contract = fixture()
    with pytest.raises(ValueError):
        consume_rounded_polynomial_monotonicity(HEADER + "changed", proof, contract)


@pytest.mark.parametrize("field", ["exception_flags_unobserved", "round_to_nearest_even", "standard_floor_values"])
def test_effect_contract_refuses(field):
    proof, contract = fixture()
    with pytest.raises(ValueError):
        consume_rounded_polynomial_monotonicity(HEADER, proof, dataclasses.replace(contract, **{field: False}))


def test_compiled_preparation_guards_and_endpoint_enclosure(tmp_path):
    import ctypes
    import random
    import shutil
    import struct
    import subprocess

    from merlin.common.paths import data_path

    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    headers = data_path("runtime", "c")
    original = (headers / "monotone_bit_polynomial.h").read_text()
    proof, contract = fixture()
    proof = dataclasses.replace(proof, header_sha256=hashlib.sha256(original.encode()).hexdigest())
    (tmp_path / "selected.h").write_text(consume_rounded_polynomial_monotonicity(original, proof, contract))
    (tmp_path / "probe.c").write_text(r"""
#include "selected.h"
void probe(int changed,int upper_negative,int mode,float lo,float hi,uint32_t*out){
 uint32_t words[8]={0xc2aeac50,0x3fb8aa3b,0xbda235d5,0xbe65b8f5,
  0x3e9b69f0,0x38e077a1,0x4b000000,0x4e7e0000};
 words[7]+=(changed!=0);float a[8];memcpy(a,words,sizeof(a));
 merlin_bit_polynomial_plan s={a[0],a[1],{a[2],a[3],a[4],a[5]},a[6],a[7]};
 int old=fegetround();const int modes[4]={FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO};
 fesetround(modes[mode]);merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&env,&s,upper_negative?-1.0f:0.0f);
 out[0]=p.fast_valid;out[1]=p.word_budget;
 if(p.fast_valid){
  merlin_f32_interval v=merlin_monotone_bit_polynomial_apply_words(merlin_interval(lo,hi),&p);
  out[2]=merlin_interval_bits(v.lo);out[3]=merlin_interval_bits(v.hi);out[4]=v.valid;
  out[5]=lo<s.cutoff?0:merlin_monotone_polynomial_source_word(lo,&s);
  out[6]=hi<s.cutoff?0:merlin_monotone_polynomial_source_word(hi,&s);
 }
 fesetround(old);
}
""")
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-frounding-math",
            "-shared",
            "-fPIC",
            "-I" + str(headers),
            str(tmp_path / "probe.c"),
            "-lm",
            "-o",
            str(tmp_path / "probe.so"),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(tmp_path / "probe.so"))
    lib.probe.argtypes = [ctypes.c_int] * 3 + [ctypes.c_float] * 2 + [ctypes.POINTER(ctypes.c_uint32)]
    out = (ctypes.c_uint32 * 7)()
    for mode in range(4):
        lib.probe(0, 0, mode, -2, -1, out)
        assert out[0] == (mode == 0)
        if mode == 0:
            assert out[1] == 0
    for changed, upper in ((1, 0), (0, 1)):
        lib.probe(changed, upper, 0, -2, -1, out)
        assert out[0] and out[1] > 0
    rng = random.Random(71)
    cases = [(-100.0, -87.0), (-0.0, 0.0), (-87.3365478515625, -87.0)]
    for _ in range(256):
        values = [struct.unpack("!f", struct.pack("!I", rng.randrange(0x80000000, WORDS[0] + 1)))[0] for _ in range(2)]
        cases.append(tuple(sorted(values)))
    for lo, hi in cases:
        lib.probe(0, 0, 0, lo, hi, out)
        assert out[0] and out[1] == 0 and out[4]
        assert out[2] <= out[5] <= out[6] <= out[3]
