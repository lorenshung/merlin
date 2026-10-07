"""Full-value output framing, including the actual freestanding C packer."""

from __future__ import annotations

import shutil
import subprocess

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.backends.base import parse_console


def _host_encoded_console(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    source = tmp_path / "out_b64.c"
    source.write_text(
        """
#include <stdint.h>
#include <stdio.h>
#include "out_b64.h"
static void sink(const char *line) { fputs(line, stdout); }
int main(void) {
  merlin_out_b64 out;
  merlin_out_b64_range range;
  puts("OUT_B64_BEGIN v1 S 1 4 1 s");
  merlin_out_b64_init(&out, 4, 1, 1, sink);
  const int8_t narrow[] = {-128, -1, 0, 127};
  for (unsigned i = 0; i < 4; ++i)
    if (!merlin_out_b64_word(&out, (uint64_t)narrow[i])) return 2;
  if (!merlin_out_b64_finish(&out)) return 3;
  puts("OUT_B64_END");

  puts("OUT_B64_BEGIN v1 H 1 2 2 u");
  merlin_out_b64_init(&out, 2, 2, 0, sink);
  const uint16_t half_bits[] = {UINT16_C(0x8000), UINT16_C(0x7e00)};
  for (unsigned i = 0; i < 2; ++i)
    if (!merlin_out_b64_word(&out, half_bits[i])) return 10;
  if (!merlin_out_b64_finish(&out)) return 11;
  puts("OUT_B64_END");

  puts("OUT_B64_BEGIN v1 U 1 2 4 u");
  const uint32_t float_bits[] = {UINT32_C(0x80000000), UINT32_C(0x7fc12345)};
  merlin_out_b64_range_init(&range, 2, 4, 0);
  for (unsigned i = 0; i < 2; ++i)
    if (!merlin_out_b64_range_unsigned(&range, float_bits[i])) return 27;
  if (merlin_out_b64_range_width(&range) != 4 ||
      merlin_out_b64_range_wire_signed(&range) != 0) return 28;
  merlin_out_b64_init(&out, 2, 4, 0, sink);
  for (unsigned i = 0; i < 2; ++i)
    if (!merlin_out_b64_word(&out, float_bits[i])) return 4;
  if (!merlin_out_b64_finish(&out)) return 5;
  puts("OUT_B64_END");

  puts("OUT_B64_BEGIN v1 W 1 3 8 s");
  merlin_out_b64_init(&out, 3, 8, 1, sink);
  const int64_t wide[] = {INT64_MIN, -1, INT64_MAX};
  for (unsigned i = 0; i < 3; ++i)
    if (!merlin_out_b64_word(&out, (uint64_t)wide[i])) return 6;
  if (!merlin_out_b64_finish(&out)) return 7;
  puts("OUT_B64_END");

  puts("OUT_B64_BEGIN v1 T 1 193 4 s");
  merlin_out_b64_init(&out, 193, 4, 1, sink);
  for (int i = 0; i < 193; ++i)
    if (!merlin_out_b64_word(&out, (uint64_t)i)) return 8;
  if (!merlin_out_b64_finish(&out)) return 9;
  puts("OUT_B64_END");

  const int32_t positive[] = {127, 128, 154, 255};
  merlin_out_b64_range_init(&range, 4, 4, 1);
  for (unsigned i = 0; i < 4; ++i)
    if (!merlin_out_b64_range_signed(&range, positive[i])) return 12;
  if (merlin_out_b64_range_width(&range) != 1 ||
      merlin_out_b64_range_wire_signed(&range) != 0) return 13;
  puts("OUT_B64_BEGIN v1 P 1 4 1 u");
  merlin_out_b64_init(&out, 4, 1, 0, sink);
  for (unsigned i = 0; i < 4; ++i)
    if (!merlin_out_b64_word(&out, (uint64_t)positive[i])) return 14;
  if (!merlin_out_b64_finish(&out)) return 15;
  puts("OUT_B64_END");

  const int32_t mixed[] = {-129, -128, -1, 0, 127, 128, 255, 256};
  merlin_out_b64_range_init(&range, 8, 4, 1);
  for (unsigned i = 0; i < 8; ++i)
    if (!merlin_out_b64_range_signed(&range, mixed[i])) return 16;
  if (merlin_out_b64_range_width(&range) != 2 ||
      merlin_out_b64_range_wire_signed(&range) != 1) return 17;
  puts("OUT_B64_BEGIN v1 R 1 8 2 s");
  merlin_out_b64_init(&out, 8, 2, 1, sink);
  for (unsigned i = 0; i < 8; ++i)
    if (!merlin_out_b64_word(&out, (uint64_t)mixed[i])) return 18;
  if (!merlin_out_b64_finish(&out)) return 19;
  puts("OUT_B64_END");

  merlin_out_b64_range_init(&range, 1, 8, 0);
  if (!merlin_out_b64_range_unsigned(&range, UINT64_MAX) ||
      merlin_out_b64_range_width(&range) != 8 ||
      merlin_out_b64_range_wire_signed(&range) != 0) return 20;
  puts("OUT_B64_BEGIN v1 Z 1 1 8 u");
  merlin_out_b64_init(&out, 1, 8, 0, sink);
  if (!merlin_out_b64_word(&out, UINT64_MAX) || !merlin_out_b64_finish(&out)) return 21;
  puts("OUT_B64_END");

  merlin_out_b64_range_init(&range, 1, 1, 1);
  if (merlin_out_b64_range_signed(&range, 128)) return 22;
  if (merlin_out_b64_range_width(&range)) return 23;
  merlin_out_b64_init(&out, 1, 1, 1, sink);
  if (merlin_out_b64_word(&out, 128)) return 24;
  merlin_out_b64_init(&out, 1, 1, 0, sink);
  if (merlin_out_b64_word(&out, 256)) return 25;
  merlin_out_b64_init(&out, 1, 1, 1, sink);
  if (merlin_out_b64_word(&out, (uint64_t)(int64_t)-129)) return 26;
  puts("DONE");
  return 0;
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "out_b64"
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
        text=True,
    )
    return subprocess.run([str(executable)], check=True, capture_output=True, text=True).stdout


def test_real_c_packer_roundtrips_every_container_word(tmp_path):
    outputs, metrics = parse_console(_host_encoded_console(tmp_path))
    assert metrics == {}
    assert outputs == {
        "S": [[-128, -1, 0, 127]],
        "H": [[0x8000, 0x7E00]],
        "U": [[0x80000000, 0x7FC12345]],
        "W": [[-(1 << 63), -1, (1 << 63) - 1]],
        "T": [list(range(193))],
        "P": [[127, 128, 154, 255]],
        "R": [[-129, -128, -1, 0, 127, 128, 255, 256]],
        "Z": [[(1 << 64) - 1]],
    }


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda lines: lines[:-1], "DONE"),
        (lambda lines: [*lines[:-2], lines[-1]], "END"),
        (lambda lines: [*lines[:2], lines[1], *lines[2:]], "chunk order"),
        (lambda lines: [*lines[:1], lines[1].replace("0004", "0005", 1), *lines[2:]], "payload"),
        (lambda lines: [lines[0].replace("1 4 1 s", "1 5 1 s"), *lines[1:]], "complete"),
        (lambda lines: [*lines[:-1], "OUT_B64_BEGIN v1 S 1 1 1 s", lines[-1]], "duplicate"),
    ],
)
def test_corrupted_frames_fail_closed(tmp_path, mutate, error):
    lines = _host_encoded_console(tmp_path).splitlines()
    # The first output is enough for frame mutation; keep the terminal DONE.
    first = [*lines[:3], "DONE"]
    damaged = mutate(first)
    with pytest.raises(RuntimeError, match=error):
        parse_console("\n".join(damaged) + "\n")


def test_legacy_out_remains_full_value():
    assert parse_console("OUT Y0 1 3 -1 0 7\nDONE\n")[0] == {"Y0": [[-1, 0, 7]]}


@pytest.mark.parametrize("before", [True, False])
def test_packed_output_refuses_a_done_before_or_inside_its_frame(tmp_path, before):
    frame = _host_encoded_console(tmp_path).splitlines()[:3]
    lines = ["DONE", *frame] if before else [*frame[:2], "DONE", frame[2]]
    with pytest.raises(RuntimeError, match="after DONE|interrupted"):
        parse_console("\n".join(lines) + "\n")


@pytest.mark.parametrize("legacy_first", [True, False])
def test_legacy_and_packed_output_cannot_claim_the_same_name(tmp_path, legacy_first):
    frame = _host_encoded_console(tmp_path).splitlines()[:3]
    legacy = "OUT S 1 4 -128 -1 0 127"
    lines = [legacy, *frame] if legacy_first else [*frame, legacy]
    with pytest.raises(RuntimeError, match="duplicate"):
        parse_console("\n".join([*lines, "DONE"]) + "\n")
