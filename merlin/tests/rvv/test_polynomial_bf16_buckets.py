import ctypes
import hashlib
import math
import random
import shutil
import struct
import subprocess
from dataclasses import replace

import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower.polynomial_bf16_buckets import (
    _add,
    _bf16,
    _floor,
    _fma,
    _from_key,
    _key,
    _multiply,
    _rne,
    c_header,
    prepare_polynomial_bf16_buckets,
    source_polynomial_word,
    validate_polynomial_bf16_buckets,
)
from merlin.llvmlower.rounded_polynomial_monotonicity import RoundedPolynomialMonotonicity
from merlin.llvmlower.source_numeric_capability import SourceNumericContract


def word(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def value(bits):
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def rounded(number):
    return value(word(number))


@pytest.fixture(scope="module")
def binding():
    header = (runtime_dir() / "c/monotone_bit_polynomial.h").read_text()
    # Independent simple source theorem: zero Horner polynomial, positive scale
    # and positive encoded affine FMA/conversion are all monotone. No sampled
    # output supplies its legality or numeric policy.
    plan = (word(-1), word(1), 0, 0, 0, 0, word(8388608), word(1065353216))
    proof = RoundedPolynomialMonotonicity(
        plan,
        plan[0] - 0x80000000 + 1,
        "1" * 64,
        "2" * 64,
        hashlib.sha256(header.encode()).hexdigest(),
        True,
        True,
        True,
    )
    contract = SourceNumericContract(*([True] * 7), standard_floor_values=True, floor_interposition_unobserved=True)
    return header, proof, contract


@pytest.fixture(scope="module")
def buckets(binding):
    return prepare_polynomial_bf16_buckets(*binding)


def test_independent_complete_partition(buckets):
    validate_polynomial_bf16_buckets(buckets)
    assert len(buckets.cells) == 129
    assert buckets.cells[0].first_input_key == _key(word(-1))
    assert buckets.cells[-1].last_input_key == _key(0)
    assert buckets.cells[-1].first_input_key <= _key(0x80000000)
    assert all(a.last_input_key + 1 == b.first_input_key for a, b in zip(buckets.cells, buckets.cells[1:]))
    for cell in buckets.cells:
        for key in (cell.first_input_key, cell.last_input_key, (cell.first_input_key + cell.last_input_key) // 2):
            assert (
                _bf16(source_polynomial_word(_from_key(key), buckets.source_plan_words)) == cell.probability_bf16_word
            )


@pytest.mark.parametrize("which", [0, 1, 2, 63, 64, 127, 128])
def test_every_selected_bucket_boundary_and_neighbors(buckets, which):
    cell = buckets.cells[which]
    for key in (cell.first_input_key, cell.last_input_key):
        output = source_polynomial_word(_from_key(key), buckets.source_plan_words)
        assert cell.first_probability_word <= output <= cell.last_probability_word
        assert _bf16(output) == cell.probability_bf16_word
    if which:
        assert (
            _bf16(source_polynomial_word(_from_key(cell.first_input_key - 1), buckets.source_plan_words))
            < cell.probability_bf16_word
        )
    if which + 1 < len(buckets.cells):
        assert (
            _bf16(source_polynomial_word(_from_key(cell.last_input_key + 1), buckets.source_plan_words))
            > cell.probability_bf16_word
        )


def test_source_floor_cutoff_and_signed_zero(binding):
    _, proof, _ = binding
    assert source_polynomial_word(word(-1.0000001192092896), proof.plan_words) == 0
    assert source_polynomial_word(word(-1), proof.plan_words) == word(0.5)
    assert source_polynomial_word(0, proof.plan_words) == word(1)
    assert source_polynomial_word(0x80000000, proof.plan_words) == word(1)
    assert _floor(0x80000000) == 0x80000000
    assert _floor(0x80000001) == word(-1)
    assert _floor(1) == 0
    assert _add(0x80000000, 0x80000000) == 0x80000000
    assert _add(0x80000000, 0) == 0


def test_exact_rne_subnormal_ties_and_overflow():
    assert _rne(1, -150) == 0
    assert _rne(3, -150) == 2
    assert _rne(5, -150) == 2
    assert _rne(-1, -150) == 0x80000000
    assert _rne(0xFFFFFF, -149) == 0xFFFFFF
    assert _fma(word(-1), 0, 0x80000000) == 0x80000000
    assert _multiply(0x80000000, word(1)) == 0x80000000
    with pytest.raises(ValueError, match="overflow"):
        _multiply(0x7F7FFFFF, word(2))


def test_integer_fma_against_independent_native_rounding():
    lib = ctypes.CDLL("libm.so.6")
    lib.fmaf.argtypes = [ctypes.c_float] * 3
    lib.fmaf.restype = ctypes.c_float
    assert lib.fegetround() == 0
    rng = random.Random(60831)
    for _ in range(2048):
        triple = tuple(rng.randrange(0x100000000) for _ in range(3))
        if any(((bits >> 23) & 255) == 255 for bits in triple):
            continue
        actual = lib.fmaf(*(value(bits) for bits in triple))
        if math.isfinite(actual):
            assert _fma(*triple) == word(actual)
        else:
            with pytest.raises(ValueError, match="overflow"):
                _fma(*triple)


@pytest.mark.parametrize("word_value", [0x7F800000, 0xFF800000, 0x7FC00000, 0xFFC00000])
def test_nonfinite_words_refuse(binding, word_value):
    with pytest.raises(ValueError, match="finite"):
        source_polynomial_word(word_value, binding[1].plan_words)


@pytest.mark.parametrize("bad", [0, -1, True, 1.5, None])
def test_preparation_quota_types_refuse(binding, bad):
    with pytest.raises(ValueError, match="quota"):
        prepare_polynomial_bf16_buckets(*binding, max_cells=bad)


def test_preparation_quota_exhaustion_is_not_partial_admission(binding):
    with pytest.raises(ValueError, match="evaluation quota"):
        prepare_polynomial_bf16_buckets(*binding, max_evaluations=1)
    with pytest.raises(ValueError, match="cell quota"):
        prepare_polynomial_bf16_buckets(*binding, max_cells=1)


@pytest.mark.parametrize("field", ["complete_domain", "gradual_underflow", "evaluator_equivalent"])
def test_missing_source_theorem_refuses_before_cells(binding, field):
    header, proof, contract = binding
    with pytest.raises(ValueError):
        prepare_polynomial_bf16_buckets(header, replace(proof, **{field: False}), contract)


@pytest.mark.parametrize(
    "field",
    [
        "round_to_nearest_even",
        "nontrapping",
        "exception_flags_unobserved",
        "standard_floor_values",
        "floor_interposition_unobserved",
    ],
)
def test_incomplete_source_effect_permission_refuses(binding, field):
    header, proof, contract = binding
    with pytest.raises(ValueError):
        prepare_polynomial_bf16_buckets(header, proof, replace(contract, **{field: False}))


@pytest.mark.parametrize(
    "field",
    ["first_input_key", "last_input_key", "first_probability_word", "last_probability_word", "probability_bf16_word"],
)
def test_bucket_corruption_refuses(buckets, field):
    first = buckets.cells[0]
    changed = replace(first, **{field: getattr(first, field) + 1})
    with pytest.raises(ValueError):
        validate_polynomial_bf16_buckets(replace(buckets, cells=(changed,) + buckets.cells[1:]))


def test_changed_source_identity_refuses(binding, buckets):
    header, proof, contract = binding
    with pytest.raises(ValueError, match="changed consumer"):
        prepare_polynomial_bf16_buckets(header + "\n", proof, contract)
    with pytest.raises(ValueError, match="source plan"):
        validate_polynomial_bf16_buckets(replace(buckets, source_plan_words=(word(-2),) + proof.plan_words[1:]))


def test_emitted_membership_keeps_original_f32_bounds(buckets):
    text = c_header(buckets)
    assert "fegetround()!=FE_TONEAREST" in text
    assert "p->word_budget" in text
    assert "x.hi<p->checked.source.cutoff" in text
    assert "a<merlin_polynomial_bf16_cells[first][0]" in text
    assert "b>merlin_polynomial_bf16_cells[first][1]" in text
    assert "merlin_interval_float(merlin_polynomial_bf16_cells[first][2])" in text
    assert "merlin_interval_float(merlin_polynomial_bf16_cells[first][3])" in text
    assert "MERLIN_SOURCE_ISFINITE" not in text


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_compiled_membership_source_words_modes_and_refusals(tmp_path, buckets, optimization):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.fail("compiled source-consumer qualification requires a C compiler")
    header = tmp_path / "buckets.h"
    header.write_text(c_header(buckets))
    # This simple complete affine source law is independent of a captured
    # workload. Test the emitted consumer and original source evaluator, not
    # merely a second use of the Python table evaluator.
    source = tmp_path / "consumer.c"
    source.write_text(
        r"""
#include "buckets.h"
#include <assert.h>
static uint32_t input_word(uint32_t key){
 return key>>31?key^UINT32_C(0x80000000):~key;
}
static int lookup(uint32_t a,uint32_t b,int admitted,
 const merlin_monotone_bit_polynomial *p){
 merlin_f32_interval x={merlin_interval_float(a),merlin_interval_float(b),1},y;
 return merlin_polynomial_bf16_bucket(x,admitted,p,&y);
}
int main(void){
 merlin_fma_bound env=merlin_fma_bound_begin();
 merlin_bit_polynomial_plan plan={.cutoff=-1,.scale=1,
  .coefficients={0,0,0,0},.bit_multiplier=8388608,.bit_bias=1065353216};
 assert(env.valid);
 /* The independent exact affine theorem supplies this private prepared
  * witness. The legacy real-polynomial builder conservatively rejects the
  * zero jump after next_down(0); its refusal is not a membership failure. */
 merlin_monotone_bit_polynomial p={.checked=merlin_bit_polynomial_prepare(&plan),
  .upper=0,.word_budget=0,.fast_valid=1};
 assert(p.checked.valid);
 assert(merlin_polynomial_bf16_bucket_admit(&p));
 for(size_t i=0;i<sizeof(merlin_polynomial_bf16_cells)/sizeof(*merlin_polynomial_bf16_cells);i++){
  const uint32_t *cell=merlin_polynomial_bf16_cells[i];
  float lo=merlin_interval_float(input_word(cell[0]));
  float hi=merlin_interval_float(input_word(cell[1]));
  merlin_f32_interval x={lo,hi,1},y;
  assert(merlin_polynomial_bf16_bucket(x,1,&p,&y));
  assert(merlin_interval_bits(y.lo)==cell[2]&&merlin_interval_bits(y.hi)==cell[3]);
  assert(merlin_monotone_polynomial_source_word(lo,&plan)==cell[2]);
  assert(merlin_monotone_polynomial_source_word(hi,&plan)==cell[3]);
 }
 merlin_f32_interval x={-2,-2,1},y;
 assert(merlin_polynomial_bf16_bucket(x,1,&p,&y));
 assert(merlin_interval_bits(y.lo)==0&&merlin_interval_bits(y.hi)==0);
 const uint32_t refused[][2]={{0x7fc00000,0},{0xff800000,0},
  {0xbf800000,0x3f800000},{0xbf800000,0xc0000000},{0xbf800001,0xbf800000}};
 for(size_t i=0;i<sizeof(refused)/sizeof(*refused);i++)
  assert(!lookup(refused[i][0],refused[i][1],1,&p));
 assert(!lookup(0,0,0,&p));
 x.valid=0;assert(!merlin_polynomial_bf16_bucket(x,1,&p,&y));
 assert(!merlin_polynomial_bf16_bucket(x,1,0,&y));
 p.word_budget=1;assert(!merlin_polynomial_bf16_bucket_admit(&p));p.word_budget=0;
 p.upper=1;assert(!merlin_polynomial_bf16_bucket_admit(&p));p.upper=0;
 p.fast_valid=0;assert(!merlin_polynomial_bf16_bucket_admit(&p));p.fast_valid=1;
 p.checked.source.scale=2;assert(!merlin_polynomial_bf16_bucket_admit(&p));
 p.checked.source.scale=1;
 const int modes[]={FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 for(size_t i=0;i<sizeof(modes)/sizeof(*modes);i++){
  assert(fesetround(modes[i])==0);assert(!merlin_polynomial_bf16_bucket_admit(&p));
 }
 assert(fesetround(FE_TONEAREST)==0);
 assert(merlin_polynomial_bf16_bucket_admit(&p));
 return 0;
}
"""
    )
    binary = tmp_path / "consumer"
    subprocess.run(
        [
            compiler,
            optimization,
            "-fno-builtin",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-I",
            str(runtime_dir() / "c"),
            str(source),
            "-lm",
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run([str(binary)], check=True, capture_output=True, text=True)
