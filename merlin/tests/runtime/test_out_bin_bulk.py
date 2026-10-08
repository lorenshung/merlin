"""The contiguous int32 path must preserve the scalar binary wire exactly."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.out_bin import parse_binary_console

_SOURCE = r"""
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include "out_bin.h"

#define COUNT 1031u
static unsigned char scalar_bytes[COUNT * 8u], bulk_bytes[COUNT * 8u];
static unsigned char *destination;
static size_t length;
static unsigned calls, fail_call;
static int32_t *mutate_next;

static int sink(const void *bytes, size_t count) {
  ++calls;
  if (calls == fail_call) return 0;
  if (length > sizeof(scalar_bytes) - count) return 0;
  memcpy(destination + length, bytes, count);
  length += count;
  if (mutate_next) { *mutate_next = 256; mutate_next = 0; }
  return 1;
}

static void select_sink(unsigned char *bytes) {
  destination = bytes;
  length = 0;
  calls = 0;
  fail_call = 0;
  mutate_next = 0;
}

static int same_wire(unsigned width, int sign) {
  int32_t words[COUNT];
  merlin_out_bin scalar, bulk;
  size_t scalar_length;
  unsigned scalar_calls;
  for (unsigned i = 0; i < COUNT; ++i) {
    if (width == 1u) words[i] = sign ? (i & 1u ? INT8_MIN : INT8_MAX) : (i & 1u ? 0 : UINT8_MAX);
    else if (width == 2u) words[i] = sign ? (i & 1u ? INT16_MIN : INT16_MAX) : (i & 1u ? 0 : UINT16_MAX);
    else words[i] = sign ? (i & 1u ? INT32_MIN : INT32_MAX) : (i & 1u ? 0 : INT32_MAX);
  }
  select_sink(scalar_bytes);
  merlin_out_bin_init(&scalar, COUNT, width, sign, sink);
  for (unsigned i = 0; i < COUNT; ++i)
    if (!merlin_out_bin_word(&scalar, (uint64_t)(int64_t)words[i])) return 0;
  if (!merlin_out_bin_finish(&scalar)) return 0;
  scalar_length = length;
  scalar_calls = calls;

  select_sink(bulk_bytes);
  merlin_out_bin_init(&bulk, COUNT, width, sign, sink);
  if (!merlin_out_bin_words_i32(&bulk, words, 17u) ||
      !merlin_out_bin_words_i32(&bulk, words + 17u, 1000u) ||
      !merlin_out_bin_words_i32(&bulk, words + 1017u, COUNT - 1017u) ||
      !merlin_out_bin_finish(&bulk)) return 0;
  return length == scalar_length && calls == scalar_calls &&
         memcmp(scalar_bytes, bulk_bytes, length) == 0 &&
         bulk.checksum == scalar.checksum && bulk.seen_words == scalar.seen_words;
}

static int refusals(void) {
  int32_t values[] = {-129, 128, 256, 32768};
  merlin_out_bin out;
  select_sink(bulk_bytes);
  if (merlin_out_bin_words_i32(0, values, 1)) return 0;
  merlin_out_bin_init(&out, 1, 1, 1, sink);
  if (merlin_out_bin_words_i32(&out, 0, 1) || merlin_out_bin_finish(&out)) return 0;
  merlin_out_bin_init(&out, 1, 1, 1, sink);
  if (!merlin_out_bin_words_i32(&out, 0, 0) ||
      merlin_out_bin_words_i32(&out, values, 2) || out.seen_words ||
      merlin_out_bin_finish(&out)) return 0;
  merlin_out_bin_init(&out, 1, 1, 1, sink);
  if (merlin_out_bin_words_i32(&out, values, 1) || out.seen_words) return 0;
  merlin_out_bin_init(&out, 1, 1, 1, sink);
  if (merlin_out_bin_words_i32(&out, values + 1, 1) || out.seen_words) return 0;
  merlin_out_bin_init(&out, 1, 1, 0, sink);
  if (merlin_out_bin_words_i32(&out, values, 1) ||
      merlin_out_bin_words_i32(&out, values + 2, 1)) return 0;
  merlin_out_bin_init(&out, 1, 2, 1, sink);
  if (merlin_out_bin_words_i32(&out, values + 3, 1)) return 0;
  merlin_out_bin_init(&out, 1, 4, 0, sink);
  if (merlin_out_bin_words_i32(&out, values, 1)) return 0;
  merlin_out_bin_init(&out, 1, 8, 0, sink);
  if (merlin_out_bin_words_i32(&out, values, 1)) return 0;
  merlin_out_bin_init(&out, 1, 1, 0, 0);
  if (merlin_out_bin_words_i32(&out, values + 1, 1)) return 0;
  merlin_out_bin_init(&out, 1, 1, 0, sink);
  if (!merlin_out_bin_words_i32(&out, values + 1, 1) ||
      merlin_out_bin_words_i32(&out, values + 1, 1) ||
      merlin_out_bin_finish(&out)) return 0;
  merlin_out_bin_init(&out, UINT64_MAX, 1, 0, sink);
  if (merlin_out_bin_words_i32(&out, values, UINT64_MAX) ||
      calls || out.seen_words || out.raw_len || merlin_out_bin_finish(&out)) return 0;
  return 1;
}

static int failures(void) {
  int32_t words[1025] = {0};
  merlin_out_bin out;
  select_sink(bulk_bytes);
  merlin_out_bin_init(&out, 1025, 1, 0, sink);
  mutate_next = &words[1024];
  if (merlin_out_bin_words_i32(&out, words, 1025) ||
      out.seen_words != 1024 || merlin_out_bin_finish(&out)) return 0;
  select_sink(bulk_bytes);
  merlin_out_bin_init(&out, 1025, 1, 0, sink);
  fail_call = 1;
  if (merlin_out_bin_words_i32(&out, words, 1025) || calls != 1 ||
      merlin_out_bin_finish(&out)) return 0;
  select_sink(bulk_bytes);
  merlin_out_bin_init(&out, 1, 3, 0, sink);
  if (merlin_out_bin_words_i32(&out, words, 1)) return 0;
  merlin_out_bin_init(&out, 1, 2, 0, sink);
  out.raw_len = 1;
  if (merlin_out_bin_words_i32(&out, words, 1)) return 0;
  merlin_out_bin_init(&out, 1, 1, 0, sink);
  out.raw_len = MERLIN_OUT_BIN_RAW_CAP + 1u;
  return !merlin_out_bin_words_i32(&out, words, 1);
}

int main(void) {
  for (unsigned width = 1; width <= 8; width *= 2)
    for (int sign = 0; sign <= 1; ++sign)
      if (!same_wire(width, sign)) return 1;
  if (!refusals() || !failures()) return 2;
  int32_t final[] = {INT32_MIN, 0, INT32_MAX};
  merlin_out_bin out;
  select_sink(bulk_bytes);
  merlin_out_bin_init(&out, 3, 4, 1, sink);
  if (!merlin_out_bin_words_i32(&out, final, 3) || !merlin_out_bin_finish(&out)) return 3;
  printf("OUT_BIN_BEGIN v1 I 1 3 4 s 12\n");
  if (fwrite(bulk_bytes, 1, length, stdout) != length) return 4;
  printf("OUT_BIN_END v1 %016llx\nDONE\n", (unsigned long long)out.checksum);
  return 0;
}
"""


def test_contiguous_i32_bulk_wire_and_refusals(tmp_path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    source = tmp_path / "out_bin_bulk.c"
    source.write_text(_SOURCE, encoding="utf-8")
    executable = tmp_path / "out_bin_bulk"
    subprocess.run(
        [
            compiler,
            "-std=c99",
            "-Wall",
            "-Wextra",
            "-Werror",
            "-I",
            str(runtime_dir() / "baremetal"),
            str(source),
            "-o",
            str(executable),
        ],
        check=True,
        capture_output=True,
    )
    raw = subprocess.run([str(executable)], check=True, capture_output=True).stdout
    outputs, metrics = parse_binary_console(raw)
    assert metrics == {}
    assert outputs == {"I": [[-(1 << 31), 0, (1 << 31) - 1]]}
