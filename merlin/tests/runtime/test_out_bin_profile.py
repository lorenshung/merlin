"""Observe, without grading on, guest cost of binary packing and checksum.

The sink reads the actual bytes in memory. The selected Spike run measures
guest work separately from FESVR transport; no numerical result is inferred.
"""

from __future__ import annotations

import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.backends import spike


_SOURCE = r"""
#include <stdint.h>
#include "out_b64.h"
#include "out_bin.h"
#include "htif.h"

#ifndef MERLIN_PROFILE_WORDS
#error "profile size must be selected explicitly"
#endif

static int32_t words[MERLIN_PROFILE_WORDS];
static uint64_t sink_hash = UINT64_C(0xcbf29ce484222325);
static uint64_t sink_calls;

static uint64_t tick(void) {
  uint64_t value;
  __asm__ volatile("csrr %0, mcycle" : "=r"(value));
  return value;
}
static void number(uint64_t value) { htif_putd((long)value); htif_putc(' '); }
static void fail(int code) { htif_puts("PROFILE_FAIL "); htif_putd(code); htif_putc('\n'); htif_exit(code); }
static int sink(const void *data, size_t length) {
  const unsigned char *bytes = (const unsigned char *)data;
  for (size_t index = 0; index < length; ++index)
    sink_hash = (sink_hash ^ bytes[index]) * UINT64_C(1099511628211);
  ++sink_calls;
  return 1;
}

int main(void) {
  merlin_out_b64_range range;
  merlin_out_bin out;
  for (uint64_t i = 0; i < MERLIN_PROFILE_WORDS; ++i)
    words[i] = (int32_t)((i * UINT64_C(37) + UINT64_C(11)) % UINT64_C(155));
  uint64_t start = tick();
  merlin_out_b64_range_init(&range, MERLIN_PROFILE_WORDS, 4u, 1);
  for (uint64_t i = 0; i < MERLIN_PROFILE_WORDS; ++i)
    if (!merlin_out_b64_range_signed(&range, words[i])) fail(2);
  unsigned width = merlin_out_b64_range_width(&range);
  int signed_words = merlin_out_b64_range_wire_signed(&range);
  uint64_t scan_ticks = tick() - start;
  if (width != 1u || signed_words != 0) fail(3);

  start = tick();
  merlin_out_bin_init(&out, MERLIN_PROFILE_WORDS, width, signed_words, sink);
  for (uint64_t i = 0; i < MERLIN_PROFILE_WORDS; ++i)
    if (!merlin_out_bin_word(&out, (uint64_t)words[i])) fail(4);
  if (!merlin_out_bin_finish(&out)) fail(5);
  uint64_t pack_ticks = tick() - start;

  start = tick();
  uint64_t independent = UINT64_C(0xcbf29ce484222325);
  for (uint64_t i = 0; i < MERLIN_PROFILE_WORDS; ++i)
    independent = (independent ^ (unsigned char)words[i]) * UINT64_C(1099511628211);
  uint64_t hash_ticks = tick() - start;
  if (independent != out.checksum || sink_hash != independent ||
      range.seen_words != MERLIN_PROFILE_WORDS || out.seen_words != MERLIN_PROFILE_WORDS ||
      sink_calls != (MERLIN_PROFILE_WORDS + MERLIN_OUT_BIN_RAW_CAP - 1u) / MERLIN_OUT_BIN_RAW_CAP)
    fail(6);
  number(scan_ticks);
  number(pack_ticks);
  number(hash_ticks);
  number(sink_calls);
  htif_putc('\n');
  htif_exit(0);
  return 0;
}
"""


@pytest.mark.parametrize("size", [4093, 65521])
def test_selected_spike_binary_encoder_profile(tmp_path, size):
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
            str(gcc), "-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2",
            "-ffreestanding", "-fno-builtin", f"-DMERLIN_PROFILE_WORDS={size}",
            "-I", str(runtime), "-I", str(harness), "-nostdlib", "-nostartfiles",
            "-static", "-T", str(harness / "link.ld"), str(source),
            str(harness / "crt.S"), str(harness / "htif.c"),
            str(harness / "libc_min.c"), "-lgcc", "-o", str(elf),
        ],
        check=True, capture_output=True, timeout=60,
    )
    from merlin.targetgen.rtl_engine_policy import gsim_runtime_slot

    try:
        with gsim_runtime_slot(wait_timeout_s=60):
            result = subprocess.run(
                [str(simulator), "--isa=rv64gc", "-m128", str(elf)],
                check=True, capture_output=True, text=True, timeout=90,
            )
    except TimeoutError:
        pytest.skip("shared native slot unavailable for optional Spike profile")
    values = [int(value) for value in result.stdout.split()]
    assert len(values) == 4
    scan, pack, hash_ticks, calls = values
    assert min(scan, pack, hash_ticks, calls) > 0
    assert calls == (size + 1023) // 1024
    print(f"Spike binary size={size} scan_cycles={scan} pack_cycles={pack} checksum_cycles={hash_ticks}")
