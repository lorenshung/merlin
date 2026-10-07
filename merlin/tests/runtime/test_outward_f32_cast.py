"""Independent exact-real enclosure checks for optional narrowing capabilities."""

import ctypes
import math
import struct
import subprocess
from fractions import Fraction

import pytest

from merlin.common.paths import merlin_dir


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    work = tmp_path_factory.mktemp("outward-narrowing")
    source = work / "probe.c"
    source.write_text("""#include <fenv.h>
#pragma STDC FENV_ACCESS ON
static float directed(double value,int mode){
 fenv_t env;feholdexcept(&env);fesetround(mode);
 volatile double input=value;volatile float output=(float)input;
 fesetenv(&env);return output;
}
#ifdef SELECT_CAPABILITY
#define MERLIN_F32_OUTWARD_FROM_F64_DOWN(x) directed((x),FE_DOWNWARD)
#define MERLIN_F32_OUTWARD_FROM_F64_UP(x) directed((x),FE_UPWARD)
#endif
#include "ordered_fma_bounds.h"
float lower(double x){return merlin_fma_down_cast_f32(x);}
float upper(double x){return merlin_fma_up_cast_f32(x);}
int rounding(void){return fegetround();}
int set_mode(int n){const int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};return fesetround(modes[n]);}
""")
    result = {}
    for label, flags in [("default", []), ("directed", ["-DSELECT_CAPABILITY=1"])]:
        output = work / (label + ".so")
        subprocess.run(
            [
                "cc",
                "-O2",
                "-frounding-math",
                "-fno-fast-math",
                "-shared",
                "-fPIC",
                *flags,
                "-I",
                str(merlin_dir() / "runtime/c"),
                str(source),
                "-lm",
                "-o",
                str(output),
            ],
            check=True,
        )
        lib = ctypes.CDLL(str(output))
        for name in ("lower", "upper"):
            function = getattr(lib, name)
            function.argtypes = [ctypes.c_double]
            function.restype = ctypes.c_float
        result[label] = lib
    return result


def f32(word):
    return struct.unpack("<f", struct.pack("<I", word))[0]


def words(value):
    return struct.unpack("<I", struct.pack("<f", value))[0]


def cases():
    output = [0.0, -0.0, 2.0**-1074, -(2.0**-1074), 2.0**128, -(2.0**128), float.fromhex("0x1.fffffffffffffp+1023")]
    for exponent in range(255):
        for fraction in (0, 1, 0x3FFFFF, 0x7FFFFE, 0x7FFFFF):
            word = exponent * 2**23 + fraction
            value = f32(word)
            if word == 0x7F7FFFFF:
                output.extend([value, -value])
                continue
            other = f32(word + 1)
            halfway = float((Fraction.from_float(value) + Fraction.from_float(other)) / 2)
            output.extend(
                [
                    value,
                    -value,
                    halfway,
                    -halfway,
                    math.nextafter(halfway, -math.inf),
                    math.nextafter(halfway, math.inf),
                ]
            )
    return output


def check_enclosure(lib, value, *, tight):
    lo, hi = lib.lower(value), lib.upper(value)
    exact = Fraction.from_float(value)
    assert math.isinf(lo) and lo < 0 or Fraction.from_float(lo) <= exact
    assert math.isinf(hi) and hi > 0 or exact <= Fraction.from_float(hi)
    if tight and value != 0:
        # Independent adjacency optimality: moving either finite endpoint one
        # binary32 value inward would violate the real enclosure.
        if math.isfinite(lo):
            raw = words(lo)
            inward = f32(raw - 1 if raw >> 31 else raw + 1)
            assert math.isinf(inward) or Fraction.from_float(inward) > exact
        if math.isfinite(hi):
            raw = words(hi)
            inward = f32(raw + 1 if raw >> 31 else (0x80000001 if raw == 0 else raw - 1))
            assert math.isinf(inward) or Fraction.from_float(inward) < exact


def test_default_encloses_all_finite_boundaries_under_source_rne(native):
    lib = native["default"]
    try:
        assert lib.set_mode(0) == 0
        for value in cases():
            check_enclosure(lib, value, tight=False)
    finally:
        lib.set_mode(0)


@pytest.mark.parametrize("mode", range(4))
def test_explicit_directed_capability_is_tight_and_preserves_ambient_mode(native, mode):
    lib = native["directed"]
    try:
        assert lib.set_mode(mode) == 0
        rounding = lib.rounding()
        for value in cases():
            check_enclosure(lib, value, tight=True)
        assert lib.rounding() == rounding
        for value in (0.0, -0.0):
            assert words(lib.lower(value)) == words(value)
            assert words(lib.upper(value)) == words(value)
    finally:
        lib.set_mode(0)
