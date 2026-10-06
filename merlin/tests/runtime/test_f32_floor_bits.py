"""Original floor values from independent integer/rational and libm oracles."""

import ctypes
import math
import random
import shutil
import struct
import subprocess
from fractions import Fraction

import pytest

from merlin.common.paths import merlin_dir

HEADERS = merlin_dir() / "runtime" / "c"


@pytest.fixture(scope="module", params=["source-copy", "builtin-copy", "builtin-floor"])
def native(tmp_path_factory, request):
    cc = shutil.which("clang") or shutil.which("cc")
    if not cc:
        pytest.skip("native C compiler required")
    work = tmp_path_factory.mktemp("f32-floor-bits")
    from merlin.llvmlower.source_numeric_capability import SourceNumericContract, emit_source_numeric_capability

    prefix = emit_source_numeric_capability(
        SourceNumericContract(
            True, True, True, True, True, True, True, standard_floor_values=True, floor_interposition_unobserved=True
        ),
        inline_bitcasts=request.param != "source-copy",
        inline_floor=request.param == "builtin-floor",
    )
    if request.param == "builtin-floor":
        prefix += "#define TEST_BUILTIN_FLOOR 1\n"
    (work / "probe.c").write_text(
        prefix
        + r"""
#include "f32_floor_bits.h"
#include <fenv.h>
void probe(const uint32_t *inputs, uint32_t *candidate, uint32_t *source, int n) {
  for (int i=0;i<n;i++) {
    float x,y,z;memcpy(&x,&inputs[i],4);
#ifdef TEST_BUILTIN_FLOOR
    y=MERLIN_SOURCE_F32_FLOOR(x);
#else
    y=merlin_f32_floor_bits(x);
#endif
    z=floorf(x);
    memcpy(&candidate[i],&y,4);memcpy(&source[i],&z,4);
  }
}
int select_rounding(int index) {
  const int modes[]={FE_TONEAREST,FE_UPWARD,FE_DOWNWARD,FE_TOWARDZERO};
  return fesetround(modes[index]);
}
int current_rounding(void) {return fegetround();}
static int evaluations;
static float once(void) { evaluations++; return -1.25f; }
int floor_single_evaluation(void) {
  evaluations=0;
  return MERLIN_SOURCE_F32_FLOOR(once()) == -2.0f && evaluations == 1;
}
"""
    )
    subprocess.run(
        [
            cc,
            "-O2",
            "-fno-fast-math",
            "-fno-builtin-floorf",
            "-frounding-math",
            "-shared",
            "-fPIC",
            "-I",
            str(HEADERS),
            str(work / "probe.c"),
            "-lm",
            "-o",
            str(work / "probe.so"),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(work / "probe.so"))
    pointer = ctypes.POINTER(ctypes.c_uint32)
    lib.probe.argtypes = [pointer, pointer, pointer, ctypes.c_int]
    return lib


def oracle(word):
    value = struct.unpack("<f", struct.pack("<I", word))[0]
    if not math.isfinite(value):
        return None
    if value == 0:
        return word
    integer = Fraction(value).__floor__()
    return struct.unpack("<I", struct.pack("<f", integer))[0]


def run(native, words, mode=0):
    assert native.select_rounding(mode) == 0
    previous = native.current_rounding()
    array = ctypes.c_uint32 * len(words)
    candidate, source = array(), array()
    try:
        native.probe(array(*words), candidate, source, len(words))
        assert native.current_rounding() == previous
        assert list(candidate) == list(source)
        assert native.floor_single_evaluation() == 1
        return list(candidate)
    finally:
        assert native.select_rounding(0) == 0


@pytest.mark.parametrize("mode", range(4))
def test_all_bf16_words_preserve_value_sign_and_rounding(native, mode):
    words = [word << 16 for word in range(65536)]
    actual = run(native, words, mode)
    for word, result in zip(words, actual):
        expected = oracle(word)
        if expected is not None:
            assert result == expected


@pytest.mark.parametrize("mode", range(4))
def test_independent_all_exponent_fraction_boundaries(native, mode):
    fractions = [0, 1, 2, 0x3FFFFF, 0x400000, 0x7FFFFE, 0x7FFFFF]
    words = [
        sign | (exponent << 23) | fraction
        for sign in (0, 0x80000000)
        for exponent in range(256)
        for fraction in fractions
    ]
    rng = random.Random(8521)
    words.extend(rng.getrandbits(32) for _ in range(25000))
    actual = run(native, words, mode)
    for word, result in zip(words, actual):
        expected = oracle(word)
        if expected is not None:
            assert result == expected
