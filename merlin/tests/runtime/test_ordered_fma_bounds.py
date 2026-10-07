"""Independent exact-rational and native-FMA audits for chunk certificates."""

from __future__ import annotations

import ctypes
import math
import random
import shutil
import struct
import subprocess
import sys
from fractions import Fraction

import pytest

from merlin.common.paths import repo_root

WRAPPER = r"""
#include "ordered_fma_bounds.h"
#include "fma_product_norms.h"
int bound(const double *lo, const double *hi, const double *abs,
          const double *repr, const int *length, int chunks,
          float *outlo, float *outhi, double *radius) {
  merlin_fma_bound s = merlin_fma_bound_begin();
  for (int i = 0; i < chunks; ++i)
    if (!merlin_fma_bound_push(&s, (merlin_fma_chunk){lo[i], hi[i], abs[i],
          repr[i], (size_t)length[i]})) return 0;
  *radius = merlin_fma_up_add(s.rounding_error, s.representation_error);
  return merlin_fma_bound_finish(&s, outlo, outhi);
}
float source(const float *a, const float *b, int length) {
  float s = 0.0f;
  for (int i = 0; i < length; ++i) s = fmaf(a[i], b[i], s);
  return s;
}
int bad_rounding(void) {
  int original = fegetround();
  if (fesetround(FE_UPWARD)) return -1;
  merlin_fma_bound s = merlin_fma_bound_begin();
  fesetround(original);
  return s.valid;
}
int norms(const float *a, const float *b, const float *ar, const float *br,
          int length, double *absolute, double *error) {
  merlin_fma_operand_norm left,right;
  if (!merlin_fma_operand_summarize(a,ar,length,1,1,&left) ||
      !merlin_fma_operand_summarize(b,br,length,1,1,&right)) return 0;
  merlin_fma_product_summarize(left,right,absolute,error);
  return isfinite(*absolute) && isfinite(*error);
}
double nextup(double x) { return merlin_fma_next_up(x); }
double nextdown(double x) { return merlin_fma_next_down(x); }
double half_ulp(double x) { return merlin_fma_half_ulp(x); }
float nextup32(float x) { return merlin_fma_next_up_f32(x); }
float nextdown32(float x) { return merlin_fma_next_down_f32(x); }
float lib_nextup32(float x) { return nextafterf(x, INFINITY); }
float lib_nextdown32(float x) { return nextafterf(x, -INFINITY); }
int gamma_one(double lo,double hi,double absolute,double repr,int length,float *outlo,float *outhi) {
  merlin_fma_bound s=merlin_fma_bound_begin();
  return merlin_fma_zero_chunk_gamma(&s,(merlin_fma_chunk){lo,hi,absolute,repr,(size_t)length},outlo,outhi);
}
int half_ulp_one(double lo,double hi,double absolute,double repr,int length,float *outlo,float *outhi) {
  merlin_fma_bound s=merlin_fma_bound_begin();
  return merlin_fma_zero_chunk_half_ulp(&s,(merlin_fma_chunk){lo,hi,absolute,repr,(size_t)length},outlo,outhi);
}
int prepared_one(double lo,double hi,double absolute,double repr,int length,float *outlo,float *outhi) {
  merlin_fma_bound s=merlin_fma_bound_begin();
  const merlin_fma_zero_gamma_plan p=merlin_fma_zero_gamma_prepare(&s,(size_t)length);
  return merlin_fma_zero_gamma_apply(&p,(merlin_fma_chunk){lo,hi,absolute,repr,(size_t)length},outlo,outhi);
}
int bad_prepared_length(void) {
  merlin_fma_bound s=merlin_fma_bound_begin();
  const merlin_fma_zero_gamma_plan p=merlin_fma_zero_gamma_prepare(&s,64);
  float lo,hi;
  return merlin_fma_zero_gamma_apply(&p,(merlin_fma_chunk){0,0,1,0,63},&lo,&hi);
}
"""


@pytest.fixture(scope="module")
def native(tmp_path_factory):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("a native C compiler is needed for the numeric certificate")
    directory = tmp_path_factory.mktemp("ordered-fma-bound")
    source = directory / "audit.c"
    shared = directory / "audit.so"
    source.write_text(WRAPPER)
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            "-I",
            str(repo_root() / "merlin/runtime/c"),
            str(source),
            "-lm",
            "-o",
            str(shared),
        ],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(shared))
    lib.bound.argtypes = [ctypes.c_void_p] * 5 + [ctypes.c_int] + [ctypes.c_void_p] * 3
    lib.bound.restype = ctypes.c_int
    lib.source.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int]
    lib.source.restype = ctypes.c_float
    lib.bad_rounding.restype = ctypes.c_int
    lib.norms.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_int] + [ctypes.c_void_p] * 2
    lib.norms.restype = ctypes.c_int
    for name in ["nextup", "nextdown", "half_ulp"]:
        function = getattr(lib, name)
        function.argtypes = [ctypes.c_double]
        function.restype = ctypes.c_double
    for name in ["nextup32", "nextdown32", "lib_nextup32", "lib_nextdown32"]:
        function = getattr(lib, name)
        function.argtypes = [ctypes.c_float]
        function.restype = ctypes.c_float
    lib.gamma_one.argtypes = [ctypes.c_double] * 4 + [ctypes.c_int] + [ctypes.c_void_p] * 2
    lib.gamma_one.restype = ctypes.c_int
    lib.half_ulp_one.argtypes = lib.gamma_one.argtypes
    lib.half_ulp_one.restype = ctypes.c_int
    lib.prepared_one.argtypes = lib.gamma_one.argtypes
    lib.prepared_one.restype = ctypes.c_int
    return lib


def bits(value):
    return struct.unpack("I", struct.pack("f", value))[0]


def unbits(value):
    return struct.unpack("f", struct.pack("I", value))[0]


def bf16_bits(value):
    raw = bits(value)
    return (raw + 0x7FFF + ((raw >> 16) & 1)) & 0xFFFF0000


def exact_round32(value):
    """Integer nearest-even division; independent of certificate arithmetic."""
    if value == 0:
        return 0.0
    sign = 0x80000000 if value < 0 else 0
    value = abs(value)
    exponent = value.numerator.bit_length() - value.denominator.bit_length()
    if value < Fraction(2) ** exponent:
        exponent -= 1
    scale_exponent = max(-149, exponent - 23)
    scaled = value / (Fraction(2) ** scale_exponent)
    quotient, remainder = divmod(scaled.numerator, scaled.denominator)
    if 2 * remainder > scaled.denominator or (2 * remainder == scaled.denominator and quotient & 1):
        quotient += 1
    # The exact result is binary32 representable; Python binary64 preserves it.
    return unbits(bits(math.ldexp(float(quotient), scale_exponent)) | sign)


def rational_source(a, b):
    result = 0.0
    for lhs, rhs in zip(a, b, strict=True):
        product = Fraction(lhs) * Fraction(rhs)
        exact = product + Fraction(result)
        if product == 0 and result == 0.0:
            product_negative = math.copysign(1.0, lhs) * math.copysign(1.0, rhs) < 0
            result = -0.0 if product_negative and math.copysign(1.0, result) < 0 else 0.0
        else:
            result = exact_round32(exact)
    return result


def outward(value):
    result = float(value)
    lo = math.nextafter(result, -math.inf) if Fraction(result) > value else result
    hi = math.nextafter(result, math.inf) if Fraction(result) < value else result
    return lo, hi


def summaries(a, b, chunk_size, reconstructed=None):
    original = [Fraction(x) * Fraction(y) for x, y in zip(a, b, strict=True)]
    reconstructed = original if reconstructed is None else reconstructed
    sums, absolutes, errors, lengths = [], [], [], []
    for start in range(0, len(a), chunk_size):
        original_chunk = original[start : start + chunk_size]
        recon_chunk = reconstructed[start : start + chunk_size]
        sums.append(outward(sum(recon_chunk)))
        absolutes.append(outward(sum(abs(x) for x in recon_chunk))[1])
        errors.append(outward(sum(abs(x - y) for x, y in zip(original_chunk, recon_chunk, strict=True)))[1])
        lengths.append(len(original_chunk))
    return sums, absolutes, errors, lengths


def certificate(native, summary):
    sums, absolute, repr_error, lengths = summary
    arrays = [
        (ctypes.c_double * len(sums))(*[pair[0] for pair in sums]),
        (ctypes.c_double * len(sums))(*[pair[1] for pair in sums]),
        (ctypes.c_double * len(sums))(*absolute),
        (ctypes.c_double * len(sums))(*repr_error),
        (ctypes.c_int * len(sums))(*lengths),
    ]
    lo, hi, radius = ctypes.c_float(), ctypes.c_float(), ctypes.c_double()
    okay = native.bound(*arrays, len(sums), ctypes.byref(lo), ctypes.byref(hi), ctypes.byref(radius))
    return okay, lo.value, hi.value, radius.value


def audit(native, a, b, chunk_size, reconstructed=None):
    left, right = (ctypes.c_float * len(a))(*a), (ctypes.c_float * len(b))(*b)
    actual = native.source(left, right, len(a))
    oracle = rational_source(a, b)
    assert bits(actual) == bits(oracle)
    okay, lo, hi, radius = certificate(native, summaries(a, b, chunk_size, reconstructed))
    assert okay and lo <= actual <= hi
    one_sum, one_abs, one_repr, one_len = summaries(a, b, len(a), reconstructed)
    gl, gh = ctypes.c_float(), ctypes.c_float()
    gamma_endpoints = None
    for helper in (native.gamma_one, native.prepared_one, native.half_ulp_one):
        assert helper(*one_sum[0], one_abs[0], one_repr[0], one_len[0], ctypes.byref(gl), ctypes.byref(gh))
        assert gl.value <= actual <= gh.value
        if helper == native.gamma_one:
            gamma_endpoints = (bits(gl.value), bits(gh.value))
        if helper == native.prepared_one:
            assert (bits(gl.value), bits(gh.value)) == gamma_endpoints
        if not gl.value <= 0.0 <= gh.value and bf16_bits(gl.value) == bf16_bits(gh.value):
            assert bf16_bits(gl.value) == bf16_bits(actual)
    # Signed zero is not certified by a real interval that contains zero.
    if not lo <= 0.0 <= hi and bf16_bits(lo) == bf16_bits(hi):
        assert bf16_bits(lo) == bf16_bits(actual)
    return radius


@pytest.mark.parametrize(
    "length,chunk_size",
    [
        (1, 1),
        (2, 7),
        (3, 2),
        (7, 3),
        (16, 7),
        (31, 16),
        (64, 64),
        (65, 16),
        (192, 64),
        (193, 64),
        (257, 31),
        (512, 192),
    ],
)
def test_independent_bf16_random_cancellation_and_tails(native, length, chunk_size):
    rng = random.Random(711 + length)
    for trial in range(12):
        # BF16 mantissas/exponents; cancellation and mixed-sign products.
        a = [math.ldexp(float(rng.randrange(-255, 256)), rng.randrange(-12, 7)) for _ in range(length)]
        b = [math.ldexp(float(rng.randrange(-255, 256)), rng.randrange(-12, 7)) for _ in range(length)]
        assert all(bits(x) & 0xFFFF == 0 for x in a + b)
        audit(native, a, b, chunk_size)
        if trial == 0:
            products = [Fraction(x) * Fraction(y) for x, y in zip(a, b, strict=True)]
            # Independently provided lossy product representation, not an answer.
            reconstructed = [Fraction(round(x * 256), 256) for x in products]
            audit(native, a, b, chunk_size, reconstructed)


def test_bf16_midpoints_and_nearby_rounding_bins(native):
    for exponent in (-100, -10, 0, 10, 100):
        base = math.ldexp(1.0, exponent)
        half_bf16 = math.ldexp(1.0, exponent - 8)
        near = math.ldexp(1.0, exponent - 24)
        for direction in (-1.0, 0.0, 1.0):
            a = [base, half_bf16, direction * near]
            audit(native, a, [1.0] * 3, 1)
            audit(native, [-x for x in a], [1.0] * 3, 2)


def test_gradual_underflow_and_signed_cancellation(native):
    tiny_bf16 = math.ldexp(1.0, -133)
    audit(native, [tiny_bf16] * 65, [math.ldexp(1.0, -20)] * 65, 16)
    audit(native, [-tiny_bf16, -0.0, -0.0], [math.ldexp(1.0, -20), 1.0, 1.0], 2)
    audit(native, [tiny_bf16, -tiny_bf16] * 96, [1.0] * 192, 31)
    a = [32768.0] + [math.ldexp(1.0, -9)] * 191 + [-32768.0]
    audit(native, a, [1.0] * len(a), 64)
    # All repeated zeros remain enclosed; the compiler's BF16 gate replays zero.
    audit(native, [0.0] * 17, [-1.0] * 17, 7)


def test_signed_prefix_bound_improves_global_abs_gamma(native):
    a = [1.0, -1.0] * 32
    b = [1.0] * 64
    radius = audit(native, a, b, 64)
    old = 64 * math.ldexp(1.0, -24) / (1 - 64 * math.ldexp(1.0, -24)) * 64
    assert radius < old * 0.51
    chunk_radius = audit(native, a, b, 8)
    assert chunk_radius < radius * 0.13


@pytest.mark.parametrize(
    "summary",
    [
        ([(0.0, 0.0)], [-1.0], [0.0], [1]),
        ([(2.0, 2.0)], [1.0], [0.0], [1]),
        ([(math.nan, 0.0)], [1.0], [0.0], [1]),
        ([(0.0, 0.0)], [math.inf], [0.0], [1]),
        ([(0.0, 0.0)], [1.0], [-1.0], [1]),
        ([(1.0, 0.0)], [1.0], [0.0], [1]),
        ([(0.0, 0.0)], [1.0], [0.0], [0]),
        ([(0.0, 0.0)], [1.0], [0.0], [1 << 24]),
        ([(1e39, 1e39)], [1e39], [0.0], [1]),
    ],
)
def test_refuses_malformed_or_overflowing_summary(native, summary):
    assert certificate(native, summary)[0] == 0
    sums, absolute, repr_error, length = summary
    lo, hi = ctypes.c_float(), ctypes.c_float()
    assert native.gamma_one(*sums[0], absolute[0], repr_error[0], length[0], ctypes.byref(lo), ctypes.byref(hi)) == 0
    assert native.half_ulp_one(*sums[0], absolute[0], repr_error[0], length[0], ctypes.byref(lo), ctypes.byref(hi)) == 0
    assert native.prepared_one(*sums[0], absolute[0], repr_error[0], length[0], ctypes.byref(lo), ctypes.byref(hi)) == 0


def test_prepared_plan_refuses_actual_source_length_mismatch(native):
    assert native.bad_prepared_length() == 0


def test_half_ulp_zero_chunk_near_binade_boundaries_and_long_lengths(native):
    # Boundaries require recomputing the larger binade before claiming induction.
    for exponent in (-149, -126, -100, 0, 100):
        for mantissa in (1.0, float.fromhex("0x1.fffffep0")):
            magnitude = math.ldexp(mantissa, exponent)
            for length in (1, 64, 192, 512, (1 << 24) - 1):
                lo, hi = ctypes.c_float(), ctypes.c_float()
                assert native.half_ulp_one(
                    magnitude, magnitude, magnitude, 0.0, length, ctypes.byref(lo), ctypes.byref(hi)
                )
                # One nonzero product followed by exact-zero products is an
                # admissible independent source with this sum/absolute/count.
                source = exact_round32(Fraction(magnitude))
                assert lo.value <= source <= hi.value


def test_refuses_non_rne_environment(native):
    assert native.bad_rounding() == 0


def test_inline_ieee_adjacency_matches_independent_nextafter(native):
    rng = random.Random(66781)
    values = [
        0.0,
        -0.0,
        math.inf,
        -math.inf,
        math.ldexp(1.0, -1074),
        -math.ldexp(1.0, -1074),
        math.ldexp(1.0, -1022),
        float.fromhex("0x1.fffffffffffffp1023"),
        -float.fromhex("0x1.fffffffffffffp1023"),
    ]
    for _ in range(2000):
        raw = rng.getrandbits(64)
        value = struct.unpack("d", struct.pack("Q", raw))[0]
        if math.isfinite(value):
            values.append(value)
    for value in values:
        assert struct.pack("d", native.nextup(value)) == struct.pack("d", math.nextafter(value, math.inf))
        assert struct.pack("d", native.nextdown(value)) == struct.pack("d", math.nextafter(value, -math.inf))
    nan = struct.unpack("d", struct.pack("Q", 0x7FF8000000000137))[0]
    assert struct.pack("d", native.nextup(nan)) == struct.pack("d", nan)
    assert struct.pack("d", native.nextdown(nan)) == struct.pack("d", nan)


def test_half_ulp_exponent_decode_matches_arithmetic(native):
    for exponent in range(-1074, 128):
        for mantissa in (1.0, 1.5, 1.999):
            value = math.ldexp(mantissa, exponent)
            expected = max(math.ldexp(1.0, -150), math.ldexp(1.0, math.frexp(value)[1] - 25))
            assert native.half_ulp(value) == expected
    assert native.half_ulp(0.0) == math.ldexp(1.0, -150)


def test_inline_binary32_adjacency_matches_independent_libm(native):
    rng = random.Random(66251)
    raw_values = [0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000, 0x7F800000, 0xFF800000, 0x7F7FFFFF, 0xFF7FFFFF]
    raw_values.extend(rng.getrandbits(32) for _ in range(2000))
    for raw in raw_values:
        value = unbits(raw)
        if math.isnan(value):
            continue
        assert bits(native.nextup32(value)) == bits(native.lib_nextup32(value))
        assert bits(native.nextdown32(value)) == bits(native.lib_nextdown32(value))


@pytest.mark.parametrize("flag", ["-ffast-math", "-ffinite-math-only"])
def test_refuses_unsafe_compilation_modes(tmp_path, flag):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("a native C compiler is needed for the numeric certificate")
    source, executable = tmp_path / "unsafe.c", tmp_path / "unsafe"
    source.write_text(
        '#include "ordered_fma_bounds.h"\nint main(void) { return merlin_fma_bound_begin().valid ? 1 : 0; }\n'
    )
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            flag,
            "-I",
            str(repo_root() / "merlin/runtime/c"),
            str(source),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    # Some fast-math shared-library constructors change the caller's FENV;
    # isolate the unsafe compile contract rather than contaminating pytest.
    subprocess.run([str(executable)], check=True, capture_output=True)


def test_refuses_runtime_flushed_subnormals_without_float_zero_comparison(tmp_path):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("native compiler required")
    strict_source = tmp_path / "strict.c"
    strict = tmp_path / "strict.so"
    strict_source.write_text(r"""
#include "ordered_fma_bounds.h"
int eligible(void) { return merlin_fma_bound_begin().valid; }
uint32_t float_subnormal_bits(void) {
 volatile float tiny=0x1p-149f;volatile float sum=tiny+tiny;
 float observed=sum;uint32_t raw;memcpy(&raw,&observed,sizeof(raw));return raw;
}
uint64_t double_subnormal_bits(void) {
 volatile double tiny=0x1p-1074;volatile double sum=tiny+tiny;
 double observed=sum;uint64_t raw;memcpy(&raw,&observed,sizeof(raw));return raw;
}
""")
    unsafe_source = tmp_path / "unsafe.c"
    unsafe = tmp_path / "unsafe.so"
    unsafe_source.write_text("int untouched(void) { return 0; }\n")
    subprocess.run(
        [
            compiler,
            "-std=c11",
            "-O2",
            "-fno-fast-math",
            "-shared",
            "-fPIC",
            "-I",
            str(repo_root() / "merlin/runtime/c"),
            str(strict_source),
            "-lm",
            "-o",
            str(strict),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [compiler, "-std=c11", "-O2", "-ffast-math", "-shared", "-fPIC", str(unsafe_source), "-lm", "-o", str(unsafe)],
        check=True,
        capture_output=True,
    )
    # A separate process observes the actual constructor effect; targets whose
    # toolchain does not change underflow behavior do not manufacture evidence.
    script = """import ctypes,sys
s=ctypes.CDLL(sys.argv[1]);s.float_subnormal_bits.restype=ctypes.c_uint32
s.double_subnormal_bits.restype=ctypes.c_uint64
assert s.eligible()==1
u=ctypes.CDLL(sys.argv[2])
flushed=s.float_subnormal_bits()!=2 or s.double_subnormal_bits()!=2
assert not flushed or s.eligible()==0
print(int(flushed))
"""
    result = subprocess.run(
        [sys.executable, "-c", script, str(strict), str(unsafe)], check=True, capture_output=True, text=True
    )
    if result.stdout.strip() != "1":
        pytest.skip("toolchain constructor did not flush subnormals")


@pytest.mark.parametrize("length", [1, 7, 64, 192, 257])
def test_holder_norms_enclose_exact_original_product_error(native, length):
    rng = random.Random(391 + length)
    for trial in range(16):
        a = [math.ldexp(float(rng.randrange(-255, 256)), rng.randrange(-100, 101)) for _ in range(length)]
        b = [math.ldexp(float(rng.randrange(-255, 256)), rng.randrange(-100, 101)) for _ in range(length)]
        # Exact input BF16 values; deliberate independently chosen lossy reconstructions.
        ar = [x if (i + trial) % 3 else 0.0 for i, x in enumerate(a)]
        br = [x if (i + trial) % 5 else x * 0.5 for i, x in enumerate(b)]
        arrays = [(ctypes.c_float * length)(*row) for row in (a, b, ar, br)]
        absolute, error = ctypes.c_double(), ctypes.c_double()
        assert native.norms(*arrays, length, ctypes.byref(absolute), ctypes.byref(error))
        exact_abs = sum(abs(Fraction(x) * Fraction(y)) for x, y in zip(ar, br, strict=True))
        exact_error = sum(
            abs(Fraction(x) * Fraction(y) - Fraction(rx) * Fraction(ry))
            for x, y, rx, ry in zip(a, b, ar, br, strict=True)
        )
        assert Fraction(absolute.value) >= exact_abs
        assert Fraction(error.value) >= exact_error
