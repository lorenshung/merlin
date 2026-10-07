"""Refuse specialization without complete rounded source/effect authority."""

import hashlib
from dataclasses import replace

import pytest

from merlin.llvmlower.prepared_polynomial_constants import c_header
from merlin.llvmlower.rounded_polynomial_monotonicity import RoundedPolynomialMonotonicity
from merlin.llvmlower.source_numeric_capability import SourceNumericContract

HEADER = "  out.rounding_error=total_error;out.word_budget=(uint32_t)budget;out.fast_valid=1;"
WORDS = (0xC2AEAC50, 0x3FB8AA3B, 0xBDA235D5, 0xBE65B8F5, 0x3E9B69F0, 0x38E077A1, 0x4B000000, 0x4E7E0000)


def proof():
    return RoundedPolynomialMonotonicity(
        WORDS,
        WORDS[0] - 0x80000000 + 1,
        "a" * 64,
        "b" * 64,
        hashlib.sha256(HEADER.encode()).hexdigest(),
        True,
        True,
        True,
    )


def contract():
    return SourceNumericContract(*([True] * 7), standard_floor_values=True, floor_interposition_unobserved=True)


@pytest.mark.parametrize("field", ["gradual_underflow", "complete_domain", "evaluator_equivalent"])
def test_incomplete_source_theorem_refuses(field):
    with pytest.raises(ValueError):
        c_header(HEADER, replace(proof(), **{field: False}), contract())


@pytest.mark.parametrize(
    "field",
    [
        "round_to_nearest_even",
        "fused_single_rounding",
        "errno_unobserved",
        "nontrapping",
        "exception_flags_unobserved",
        "standard_bitcast_copy",
        "copy_interposition_unobserved",
        "standard_floor_values",
        "floor_interposition_unobserved",
    ],
)
def test_observable_or_unproved_effect_refuses(field):
    with pytest.raises(ValueError):
        c_header(HEADER, proof(), replace(contract(), **{field: False}))


def test_changed_consumer_refuses():
    with pytest.raises(ValueError, match="changed consumer"):
        c_header(HEADER + "\n", proof(), contract())


def test_partial_domain_refuses():
    with pytest.raises(ValueError, match="complete cutoff"):
        c_header(HEADER, replace(proof(), checked_negative_words=5), contract())


def test_nonfinite_source_plan_refuses():
    words = list(WORDS)
    words[3] = 0x7F800000
    with pytest.raises(ValueError, match="finite source plan"):
        c_header(HEADER, replace(proof(), plan_words=tuple(words)), contract())


def test_constants_are_derived_from_selected_source_words():
    output = c_header(HEADER, proof(), contract())
    other_words = (*WORDS[:-1], WORDS[-1] + 1)
    other = c_header(HEADER, replace(proof(), plan_words=other_words), contract())
    assert output != other
    assert "@" not in output
    # Exact plan/RNE/budget admission and original fallback remain in the emitted API.
    assert "fegetround()!=FE_TONEAREST" in output
    assert "p->word_budget!=0" in output
    assert "merlin_polynomial_words_four(x,p,out)" in output


def test_packaged_headers_native_rounding_and_refusal(tmp_path):
    import shutil
    import subprocess

    from merlin.common.paths import data_path

    compiler = shutil.which("gcc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    runtime = data_path("runtime", "c")
    for path in runtime.glob("*.h"):
        shutil.copyfile(path, tmp_path / path.name)
    original = (tmp_path / "monotone_bit_polynomial.h").read_text()
    selected = replace(proof(), header_sha256=hashlib.sha256(original.encode()).hexdigest())
    from merlin.llvmlower.rounded_polynomial_monotonicity import consume_rounded_polynomial_monotonicity

    (tmp_path / "monotone_bit_polynomial.h").write_text(
        consume_rounded_polynomial_monotonicity(original, selected, contract())
    )
    (tmp_path / "constants.h").write_text(c_header(original, selected, contract()))
    source = r"""
#include "constants.h"
#include <stdlib.h>
int main(void){
 uint32_t words[]={WORDS};float f[8];memcpy(f,words,sizeof(f));
 merlin_bit_polynomial_plan plan={f[0],f[1],{f[2],f[3],f[4],f[5]},f[6],f[7]};
 merlin_fma_bound e=merlin_fma_bound_begin();
 merlin_monotone_bit_polynomial p=merlin_monotone_bit_polynomial_prepare(&e,&plan,0);
 if(!merlin_polynomial_constants_admit(&p))return 1;
 for(unsigned i=0;i<2048;i++){
  float lo=-100.0f*(float)i/2048.0f, hi=lo+0.00001f;if(hi>0)hi=0;
  merlin_f32_interval x[4],a[4],b[4];
  for(int j=0;j<4;j++)x[j]=(merlin_f32_interval){lo,hi,1};
  merlin_polynomial_words_four(x,&p,a);merlin_polynomial_constants_four(x,&p,1,b);
  for(int j=0;j<4;j++)if(a[j].valid!=b[j].valid||
   merlin_interval_bits(a[j].lo)!=merlin_interval_bits(b[j].lo)||
   merlin_interval_bits(a[j].hi)!=merlin_interval_bits(b[j].hi))return 2;
 }
 int modes[]={FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 for(int i=0;i<3;i++){fesetround(modes[i]);if(merlin_polynomial_constants_admit(&p))return 3;}
 fesetround(FE_TONEAREST);p.checked.source.scale=nextafterf(p.checked.source.scale,INFINITY);
 if(merlin_polynomial_constants_admit(&p))return 4;
 return 0;
}
""".replace("WORDS", ",".join(hex(word) + "u" for word in WORDS))
    (tmp_path / "main.c").write_text(source)
    subprocess.run(
        [
            compiler,
            "-O2",
            "-std=c11",
            "-fno-fast-math",
            "-frounding-math",
            "-ffp-contract=off",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            "-I",
            str(tmp_path),
            str(tmp_path / "main.c"),
            "-lm",
            "-o",
            str(tmp_path / "check"),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(tmp_path / "check")], check=True, capture_output=True)
