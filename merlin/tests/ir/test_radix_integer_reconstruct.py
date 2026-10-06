import ctypes
import shutil
import struct
import subprocess
from dataclasses import replace

import pytest

from merlin.llvmlower.radix_integer_reconstruct import c_header
from merlin.llvmlower.radix_product_groups import plan_radix_product_groups


def test_refuses_mutated_group_and_prefix_proof():
    plan = plan_radix_product_groups(radix_bits=7, digits=3, reduction_length=65)
    with pytest.raises(ValueError):
        c_header(replace(plan, weighted_absolute_bound=1))
    with pytest.raises(ValueError):
        c_header(replace(plan, groups=plan.groups[:-1]))
    with pytest.raises(ValueError):
        c_header(replace(plan, reduction_length=2049))


@pytest.mark.parametrize("k", [1, 65, 2048])
def test_compiled_exact_prefixes_cancellation_tails_and_flags(tmp_path, k):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")
    plan = plan_radix_product_groups(radix_bits=7, digits=3, reduction_length=k)
    (tmp_path / "reconstruct.h").write_text(c_header(plan))
    limits = ",".join(str(g.accumulator_bound) for g in plan.groups)
    source = r"""#include "reconstruct.h"
#include <fenv.h>
#include <string.h>
enum { COUNT = 17 };
static const int32_t limits[5] = {LIMITS};
int test(void) {
  if (fesetround(FE_TONEAREST)) return 1;
  for (unsigned method = 0; method < 2; ++method)
  for (unsigned scenario = 0; scenario < 5; ++scenario) {
    double control[COUNT], candidate[COUNT];
    int64_t storage[COUNT + 2]; int64_t *integer = storage + 1;
    int32_t group[COUNT];
    storage[0] = storage[COUNT + 1] = INT64_C(0x13579bdf);
    for (unsigned t = 0; t < COUNT; ++t) {
      control[t] = 0.0; candidate[t] = -19.0; integer[t] = INT64_C(-98341);
    }
    if (!method) merlin_radix_integer_begin_exact(integer, COUNT);
    for (unsigned d = 0; d < 5; ++d) {
      for (unsigned t = 0; t < COUNT; ++t) {
        int32_t v = limits[d];
        if (scenario == 0) v = 0;
        if (scenario == 2) v = -v;
        if (scenario == 3) v = ((t + d) & 1) ? -v : v;
        /* Exactly 128 + (-1)*128 = +0; other entries exercise tails. */
        if (scenario == 4) v = d == 0 ? 128 : d == 1 ? -1 : 0;
        group[t] = v;
        control[t] += (double)v * (double)(INT64_C(1) << (7 * d));
      }
      feclearexcept(FE_ALL_EXCEPT);
      if (method && !d)
        merlin_radix_integer_begin_from_first_group_exact_i64(integer, group, COUNT);
      else merlin_radix_integer_accumulate_exact_i64(integer, group, COUNT, d);
      merlin_radix_integer_finish_exact_f64(candidate, integer, COUNT);
      if (fetestexcept(FE_ALL_EXCEPT)) return 2;
      if (memcmp(control, candidate, sizeof(control))) return 10 + scenario;
      if (storage[0] != INT64_C(0x13579bdf) ||
          storage[COUNT + 1] != INT64_C(0x13579bdf)) return 3;
    }
  }
  return 0;
}
""".replace("LIMITS", limits)
    (tmp_path / "test.c").write_text(source)
    # Passed pytest temp directories may be removed and their paths reused
    # between parameters. A loaded CDLL remains mapped, so every distinct
    # numerical plan also needs a distinct library basename.
    library = tmp_path / f"test_k{k}.so"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(tmp_path / "test.c"),
            "-o",
            str(library),
            "-lm",
        ],
        check=True,
    )
    assert ctypes.CDLL(str(library)).test() == 0
    # UBSan independently checks defined negative multiplication/addition and
    # the near-2**53 positive/negative prefixes in the actual generated C.
    (tmp_path / "runner.c").write_text("int test(void); int main(void){return test();}\n")
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O1",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=undefined",
            str(tmp_path / "test.c"),
            str(tmp_path / "runner.c"),
            "-o",
            str(tmp_path / "test"),
            "-lm",
        ],
        check=True,
    )
    subprocess.run([str(tmp_path / "test")], check=True)


@pytest.mark.parametrize("radix_bits,digits,k", [(1, 1, 1), (4, 2, 65), (8, 1, 127)])
def test_independent_radices_and_group_counts(tmp_path, radix_bits, digits, k):
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("C compiler unavailable")
    plan = plan_radix_product_groups(radix_bits=radix_bits, digits=digits, reduction_length=k)
    (tmp_path / "reconstruct.h").write_text(c_header(plan))
    (tmp_path / "test.c").write_text("""#include "reconstruct.h"
void begin(int64_t *d,size_t n){merlin_radix_integer_begin_exact(d,n);}
void update(int64_t *d,const int32_t *s,size_t n,unsigned g){
 merlin_radix_integer_accumulate_exact_i64(d,s,n,g);}
void finish(double *d,const int64_t *s,size_t n){
 merlin_radix_integer_finish_exact_f64(d,s,n);}
void first(int64_t *d,const int32_t *s,size_t n){
 merlin_radix_integer_begin_from_first_group_exact_i64(d,s,n);}
""")
    library = tmp_path / f"test_radix{radix_bits}_digits{digits}_k{k}.so"
    subprocess.run(
        [
            cc,
            "-std=c11",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-shared",
            "-fPIC",
            str(tmp_path / "test.c"),
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(library))
    count = 19
    integer = (ctypes.c_int64 * count)(*([-1821] * count))
    output = (ctypes.c_double * count)(*([-17.0] * count))
    lib.begin.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
    lib.update.argtypes = [
        ctypes.POINTER(ctypes.c_int64),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_size_t,
        ctypes.c_uint,
    ]
    lib.finish.argtypes = [ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_int64), ctypes.c_size_t]
    lib.first.argtypes = [ctypes.POINTER(ctypes.c_int64), ctypes.POINTER(ctypes.c_int32), ctypes.c_size_t]
    exact = [0] * count
    for ordinal, group in enumerate(plan.groups):
        bound = group.accumulator_bound
        values = [(t * 917 + ordinal * 31) % (2 * bound + 1) - bound for t in range(count)]
        source = (ctypes.c_int32 * count)(*values)
        if ordinal == 0:
            lib.first(integer, source, count)
        else:
            lib.update(integer, source, count, ordinal)
        lib.finish(output, integer, count)
        exact = [old + value * (1 << group.exponent) for old, value in zip(exact, values)]
        assert list(integer) == exact
        assert bytes(output) == b"".join(struct.pack("=d", float(x)) for x in exact)
