"""Neutral baseline for the guest work done by the lossless output encoder.

The callback consumes each encoded byte in memory.  Guest cycle counts therefore
exclude console transport, so a later encoder change can be measured separately
from FESVR throughput.  Timings are observations, never correctness thresholds.
"""

from __future__ import annotations

import base64
import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.backends import spike
from merlin.runtime.backends.base import parse_console

_SOURCE = r"""
#include <stdint.h>
#include "out_b64.h"
#ifdef __riscv
#include "htif.h"
static uint64_t tick(void) {
  uint64_t value;
  /* The bare-metal Spike harness stays in machine mode; its unprivileged
   * cycle CSR may be disabled even though mcycle is available. */
  __asm__ volatile("csrr %0, mcycle" : "=r"(value));
  return value;
}
static void number(uint64_t value) { htif_putd((long)value); htif_putc(' '); }
static void newline(void) { htif_putc('\n'); }
#else
#include <stdio.h>
#include <time.h>
static uint64_t tick(void) {
  struct timespec value;
  clock_gettime(CLOCK_MONOTONIC, &value);
  return (uint64_t)value.tv_sec * UINT64_C(1000000000) + (uint64_t)value.tv_nsec;
}
static void number(uint64_t value) { printf("%llu ", (unsigned long long)value); }
static void newline(void) { putchar('\n'); }
#endif

#ifndef MERLIN_PROFILE_WORDS
#error "profile size must be selected explicitly"
#endif
static int32_t words[MERLIN_PROFILE_WORDS];
static uint64_t sink_hash = UINT64_C(1469598103934665603);
static uint64_t sink_calls;
static void sink(const char *line) {
  uint64_t value = sink_hash;
  for (const unsigned char *p = (const unsigned char *)line; *p; ++p)
    value = (value ^ *p) * UINT64_C(1099511628211);
  sink_hash = value;
  ++sink_calls;
#ifdef MERLIN_PROFILE_EMIT
  fputs(line, stdout);
#endif
}

int main(void) {
  merlin_out_b64_range range;
  merlin_out_b64 out;
  for (uint64_t i = 0; i < MERLIN_PROFILE_WORDS; ++i)
    words[i] = (int32_t)((i * UINT64_C(37) + UINT64_C(11)) % UINT64_C(155));
  uint64_t start = tick();
  merlin_out_b64_range_init(&range, MERLIN_PROFILE_WORDS, 4u, 1);
  for (uint64_t i = 0; i < MERLIN_PROFILE_WORDS; ++i)
    if (!merlin_out_b64_range_signed(&range, words[i])) return 2;
  unsigned width = merlin_out_b64_range_width(&range);
  int signed_words = merlin_out_b64_range_wire_signed(&range);
  uint64_t scan_ticks = tick() - start;
  if (width != 1u || signed_words != 0) return 3;

#ifdef MERLIN_PROFILE_EMIT
  printf("OUT_B64_BEGIN v1 P 1 %u %u %c\n", (unsigned)MERLIN_PROFILE_WORDS,
         width, signed_words ? 's' : 'u');
#endif
  start = tick();
  merlin_out_b64_init(&out, MERLIN_PROFILE_WORDS, width, signed_words, sink);
#ifdef MERLIN_PROFILE_BULK
  if (!merlin_out_b64_words_i32(&out, words, MERLIN_PROFILE_WORDS)) return 4;
#else
  for (uint64_t i = 0; i < MERLIN_PROFILE_WORDS; ++i)
    if (!merlin_out_b64_word(&out, (uint64_t)words[i])) return 4;
#endif
  if (!merlin_out_b64_finish(&out)) return 5;
  uint64_t pack_ticks = tick() - start;
  if (range.seen_words != MERLIN_PROFILE_WORDS || out.seen_words != MERLIN_PROFILE_WORDS ||
      sink_calls != (MERLIN_PROFILE_WORDS + MERLIN_OUT_B64_RAW_CAP - 1u) / MERLIN_OUT_B64_RAW_CAP)
    return 6;

#ifndef MERLIN_PROFILE_EMIT
  number(scan_ticks);
  number(pack_ticks);
  number(sink_hash);
  number(sink_calls);
  number(range.seen_words);
  number(out.seen_words);
  number(width);
  number((uint64_t)signed_words);
  newline();
#else
  (void)scan_ticks;
  (void)pack_ticks;
  puts("OUT_B64_END");
  puts("DONE");
#endif
#ifdef __riscv
  htif_exit(0);
#endif
  return 0;
}
"""


_CASE_SOURCE = r"""
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "out_b64.h"

static int32_t words[769];
static void sink(const char *line) { fputs(line, stdout); }
static int32_t value(unsigned selected, unsigned index) {
  static const int32_t cases[8][4] = {
    {0, 127, 128, 255},
    {-128, -1, 0, 127},
    {0, 255, 256, 65535},
    {-32768, -129, 128, 32767},
    {0, 65535, 65536, INT32_MAX},
    {INT32_MIN, -65536, 0, INT32_MAX},
    {-1, INT32_MIN, 0, INT32_MAX},
    {INT32_MIN, -1, 0, INT32_MAX},
  };
  return cases[selected][index % 4u];
}

static int refusals(void) {
  merlin_out_b64 out;
  merlin_out_b64_range range;
  int32_t values[2] = {1, 2};
  merlin_out_b64_init(&out, 1, 3, 0, sink);
  if (merlin_out_b64_words_i32(&out, values, 1)) return 1;
  merlin_out_b64_init(&out, 1, 1, 0, sink);
  if (merlin_out_b64_words_i32(&out, values, 2) || out.seen_words) return 2;
  if (merlin_out_b64_words_i32(&out, 0, 1)) return 3;
  if (merlin_out_b64_words_i32(&out, values, 0) != 1 || merlin_out_b64_finish(&out)) return 4;
  merlin_out_b64_range_init(&range, 2, 4, 1);
  if (!merlin_out_b64_range_signed(&range, values[0]) ||
      merlin_out_b64_range_width(&range)) return 5;
  merlin_out_b64_range_init(&range, 1, 4, 1);
  if (!merlin_out_b64_range_signed(&range, values[0]) ||
      merlin_out_b64_range_signed(&range, values[1])) return 6;
  values[0] = 256; /* The actual buffer changed AFTER a valid u8 range scan. */
  merlin_out_b64_init(&out, 1, 1, 0, sink);
  if (merlin_out_b64_words_i32(&out, values, 1)) return 7;
  values[0] = -129;
  merlin_out_b64_init(&out, 1, 1, 1, sink);
  if (merlin_out_b64_words_i32(&out, values, 1)) return 8;
  values[0] = 1;
  merlin_out_b64_init(&out, 2, 1, 0, sink);
  if (!merlin_out_b64_words_i32(&out, values, 1) || merlin_out_b64_finish(&out)) return 9;
  merlin_out_b64_init(&out, 1, 1, 0, sink);
  out.valid = 1;
  out.word_bytes = 3; /* A malformed post-init width cannot reach a shift. */
  if (merlin_out_b64_words_i32(&out, values, 1)) return 10;
  merlin_out_b64_init(&out, 1, 1, 0, sink);
  if (merlin_out_b64_words_i32(&out, values, UINT64_MAX)) return 11;
  out.raw_len = MERLIN_OUT_B64_RAW_CAP + 1u;
  if (merlin_out_b64_words_i32(&out, values, 1)) return 12;
  out.raw_len = 0;
  out.signed_words = 2;
  if (merlin_out_b64_words_i32(&out, values, 1)) return 13;
  out.signed_words = 0;
  out.seen_words = 2;
  if (merlin_out_b64_words_i32(&out, values, 1)) return 14;
  puts("REFUSALS_OK");
  return 0;
}

int main(int argc, char **argv) {
  if (argc == 2 && !strcmp(argv[1], "refusals")) return refusals();
  if (argc != 4) return 20;
  unsigned selected = (unsigned)atoi(argv[1]);
  unsigned count = (unsigned)atoi(argv[2]);
  int bulk = atoi(argv[3]);
  if (selected >= 8 || count == 0 || count > 769) return 21;
  static const unsigned widths[8] = {1, 1, 2, 2, 4, 4, 8, 8};
  static const int signs[8] = {0, 1, 0, 1, 0, 1, 0, 1};
  unsigned width = widths[selected];
  int sign = signs[selected];
  for (unsigned i = 0; i < count; ++i) words[i] = value(selected, i);
  printf("OUT_B64_BEGIN v1 P 1 %u %u %c\n", count, width, sign ? 's' : 'u');
  merlin_out_b64 out;
  merlin_out_b64_init(&out, count, width, sign, sink);
  if (bulk) {
    if (!merlin_out_b64_words_i32(&out, words, count)) return 22;
  } else {
    for (unsigned i = 0; i < count; ++i)
      if (!merlin_out_b64_word(&out, (uint64_t)words[i])) return 23;
  }
  if (!merlin_out_b64_finish(&out)) return 24;
  puts("OUT_B64_END");
  puts("DONE");
  return 0;
}
"""


def _wire_lines(size: int) -> list[str]:
    payload = bytes((index * 37 + 11) % 155 for index in range(size))
    lines = [f"OUT_B64_BEGIN v1 P 1 {size} 1 u"]
    for ordinal, start in enumerate(range(0, size, 768)):
        block = payload[start : start + 768]
        lines.append(f"OUT_B64_CHUNK {ordinal:08x} {len(block):04x} {base64.b64encode(block).decode('ascii')}")
    lines.extend(("OUT_B64_END", "DONE"))
    return lines


def _measure(cmd: list[str], *, size: int) -> tuple[int, ...]:
    result = subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=90)
    values = tuple(int(value) for value in result.stdout.split())
    assert len(values) == 8
    scan, pack, actual_hash, calls, scanned, packed, width, signed = values
    assert scan > 0 and pack > 0
    assert calls == (size + 767) // 768
    assert scanned == packed == size
    assert (width, signed) == (1, 0)
    expected_hash = 1469598103934665603
    for line in _wire_lines(size)[1:-2]:
        for byte in (line + "\n").encode("ascii"):
            expected_hash = ((expected_hash ^ byte) * 1099511628211) & ((1 << 64) - 1)
    assert actual_hash & ((1 << 64) - 1) == expected_hash
    return values


@pytest.fixture
def case_executable(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    source = tmp_path / "encoder_cases.c"
    executable = tmp_path / "encoder_cases"
    source.write_text(_CASE_SOURCE, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c99",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-function",
            "-I",
            str(runtime_dir() / "baremetal"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return executable


@pytest.mark.parametrize("selected", range(8))
@pytest.mark.parametrize("size", [1, 193, 769])
def test_typed_bulk_matches_scalar_exact_wire(case_executable, selected, size):
    old = subprocess.run(
        [str(case_executable), str(selected), str(size), "0"], check=True, capture_output=True, text=True, timeout=30
    ).stdout
    bulk = subprocess.run(
        [str(case_executable), str(selected), str(size), "1"], check=True, capture_output=True, text=True, timeout=30
    ).stdout
    assert bulk == old
    outputs, metrics = parse_console(bulk)
    assert metrics == {}
    signed_values = [
        (0, 127, 128, 255),
        (-128, -1, 0, 127),
        (0, 255, 256, 65535),
        (-32768, -129, 128, 32767),
        (0, 65535, 65536, (1 << 31) - 1),
        (-(1 << 31), -65536, 0, (1 << 31) - 1),
        (-1, -(1 << 31), 0, (1 << 31) - 1),
        (-(1 << 31), -1, 0, (1 << 31) - 1),
    ][selected]
    expected = [signed_values[index % 4] for index in range(size)]
    if selected == 6:
        expected = [value & ((1 << 64) - 1) for value in expected]
    assert outputs == {"P": [expected]}


def test_typed_bulk_refuses_count_width_and_changed_values(case_executable):
    result = subprocess.run([str(case_executable), "refusals"], check=True, capture_output=True, text=True, timeout=30)
    assert result.stdout == "REFUSALS_OK\n"


@pytest.mark.parametrize("size", [4093, 65521])
@pytest.mark.parametrize("bulk", [False, True])
def test_host_encoder_profile(tmp_path, size, bulk):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    source = tmp_path / "encoder_profile.c"
    executable = tmp_path / "encoder_profile"
    source.write_text(_SOURCE, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c99",
            "-D_POSIX_C_SOURCE=200809L",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-function",
            f"-DMERLIN_PROFILE_WORDS={size}",
            *(["-DMERLIN_PROFILE_BULK"] if bulk else []),
            "-I",
            str(runtime_dir() / "baremetal"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    values = _measure([str(executable)], size=size)
    print(f"host bulk={bulk} size={size} scan_ns={values[0]} pack_ns={values[1]} hash={values[2]}")


@pytest.mark.parametrize("size", [4093, 65521])
@pytest.mark.parametrize("bulk", [False, True])
def test_host_encoder_wire_is_exact(tmp_path, size, bulk):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    source = tmp_path / "encoder_wire.c"
    executable = tmp_path / "encoder_wire"
    source.write_text(_SOURCE, encoding="utf-8")
    subprocess.run(
        [
            compiler,
            "-std=c99",
            "-D_POSIX_C_SOURCE=200809L",
            "-O2",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-Wno-unused-function",
            "-DMERLIN_PROFILE_EMIT",
            f"-DMERLIN_PROFILE_WORDS={size}",
            *(["-DMERLIN_PROFILE_BULK"] if bulk else []),
            "-I",
            str(runtime_dir() / "baremetal"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    actual = subprocess.run([str(executable)], check=True, capture_output=True, text=True, timeout=60).stdout
    assert actual == "\n".join(_wire_lines(size)) + "\n"


@pytest.mark.parametrize("size", [4093, 65521])
@pytest.mark.parametrize("bulk", [False, True])
def test_spike_encoder_profile(tmp_path, size, bulk):
    gcc, simulator = spike.gcc_path(), spike.spike_path()
    if not gcc.is_file() or not simulator.is_file():
        pytest.skip("selected bare-metal GCC and Spike unavailable")
    runtime = runtime_dir() / "baremetal"
    harness = runtime / "spike"
    source = tmp_path / "encoder_profile.c"
    elf = tmp_path / "encoder_profile.elf"
    source.write_text(_SOURCE, encoding="utf-8")
    subprocess.run(
        [
            str(gcc),
            "-march=rv64gc",
            "-mabi=lp64d",
            "-mcmodel=medany",
            "-O2",
            "-ffreestanding",
            "-fno-builtin",
            f"-DMERLIN_PROFILE_WORDS={size}",
            *(["-DMERLIN_PROFILE_BULK"] if bulk else []),
            "-I",
            str(runtime),
            "-I",
            str(harness),
            "-nostdlib",
            "-nostartfiles",
            "-static",
            "-T",
            str(harness / "link.ld"),
            str(source),
            str(harness / "crt.S"),
            str(harness / "htif.c"),
            str(harness / "libc_min.c"),
            "-lgcc",
            "-o",
            str(elf),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    values = _measure([str(simulator), "--isa=rv64gc", "-m128", str(elf)], size=size)
    print(f"Spike bulk={bulk} size={size} scan_cycles={values[0]} pack_cycles={values[1]} hash={values[2]}")
