"""Closed memory packets expose every actual value and exact float32 bits."""

from __future__ import annotations

import math
import shutil
import subprocess
from dataclasses import replace

import pytest

from merlin.common.paths import runtime_dir
from merlin.runtime.out_packet import (
    ExpectedOutput,
    OutPacketError,
    decode_out_bin_packet,
    out_bin_packet_capacity,
)


def _closed_packet(tmp_path) -> bytes:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("host C compiler unavailable")
    source = tmp_path / "packet.c"
    source.write_text(
        r"""
#include <stdint.h>
#include <stdio.h>
#include "out_bin.h"
#include "out_bin_memory.h"

static unsigned char arena[512];
static uint64_t published_used;
static merlin_out_bin_memory *active;

static int sink(const void *bytes, size_t count) {
  return merlin_out_bin_memory_append(active, bytes, count);
}

static int frame(merlin_out_bin_memory *packet, const char *name,
                 unsigned rows, unsigned cols, const uint64_t *words,
                 unsigned width, int signed_words) {
  merlin_out_bin encoder;
  if (!merlin_out_bin_memory_cstr(packet, "OUT_BIN_BEGIN v1 ") ||
      !merlin_out_bin_memory_cstr(packet, name) ||
      !merlin_out_bin_memory_cstr(packet, " ") ||
      !merlin_out_bin_memory_decimal(packet, rows) ||
      !merlin_out_bin_memory_cstr(packet, " ") ||
      !merlin_out_bin_memory_decimal(packet, cols) ||
      !merlin_out_bin_memory_cstr(packet, " ") ||
      !merlin_out_bin_memory_decimal(packet, width) ||
      !merlin_out_bin_memory_cstr(packet, signed_words ? " s " : " u ") ||
      !merlin_out_bin_memory_decimal(packet, (uint64_t)rows * cols * width) ||
      !merlin_out_bin_memory_cstr(packet, "\n")) return 0;
  active = packet; /* Scoped by this one-shot, single-threaded test harness. */
  merlin_out_bin_init(&encoder, (uint64_t)rows * cols, width, signed_words, sink);
  for (unsigned i = 0; i < rows * cols; ++i)
    if (!merlin_out_bin_word(&encoder, words[i])) return 0;
  if (!merlin_out_bin_finish(&encoder) ||
      !merlin_out_bin_memory_cstr(packet, "OUT_BIN_END v1 ") ||
      !merlin_out_bin_memory_hex16(packet, encoder.checksum) ||
      !merlin_out_bin_memory_cstr(packet, "\n")) return 0;
  active = 0;
  return 1;
}

int main(void) {
  merlin_out_bin_memory packet;
  const uint64_t ints[] = {(uint64_t)INT64_MIN, INT64_MAX};
  const uint64_t floats[] = {UINT32_C(0x80000000), UINT32_C(0x7fc12345)};
  const uint64_t narrow[] = {0, 255};
  const uint64_t boolean[] = {0, 1};
  merlin_out_bin_memory_init(&packet, arena, sizeof(arena), &published_used);
  if (!frame(&packet, "I", 1, 2, ints, 8, 1) ||
      !frame(&packet, "F", 1, 2, floats, 4, 0) ||
      !frame(&packet, "N", 2, 1, narrow, 1, 0) ||
      !frame(&packet, "B", 1, 2, boolean, 1, 0) ||
      !merlin_out_bin_memory_cstr(&packet, "DONE\n") ||
      published_used != 0 || !merlin_out_bin_memory_finish(&packet) ||
      published_used != packet.used) return 2;
  return fwrite(arena, 1, published_used, stdout) == published_used ? 0 : 3;
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "packet"
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
    return subprocess.run([str(executable)], check=True, capture_output=True).stdout


def _roster() -> tuple[ExpectedOutput, ...]:
    return (
        ExpectedOutput("I", 1, 2, "i64", True, 8, (2,)),
        ExpectedOutput("F", 1, 2, "f32", False, 4, (2,)),
        ExpectedOutput("N", 2, 1, "i32", True, 4, (2, 1)),
        ExpectedOutput("B", 1, 2, "i1", False, 1, (2,)),
    )


def test_actual_c_packet_roundtrips_complete_roster_and_f32_bits(tmp_path) -> None:
    packet = _closed_packet(tmp_path)
    assert out_bin_packet_capacity(_roster()) == len(packet) + 6  # N narrows two i32 containers to u8.
    decoded = decode_out_bin_packet(packet, _roster())
    assert tuple(decoded) == ("I", "F", "N", "B")
    assert decoded["I"].values == (-(1 << 63), (1 << 63) - 1)
    assert decoded["N"].values == ((0,), (255,))
    assert decoded["N"].wire_bytes == 1 and not decoded["N"].wire_signed
    assert decoded["B"].values == (0, 1)
    assert decoded["F"].f32_bits == (0x80000000, 0x7FC12345)
    assert math.copysign(1.0, decoded["F"].values[0]) == -1.0
    assert math.isnan(decoded["F"].values[1])


def test_capacity_counts_exact_maximum_frame_and_decimal_growth() -> None:
    footer = b"OUT_BIN_END v1 0123456789abcdef\n"
    maximum = (ExpectedOutput("I", 1, 2, "i64", True, 8, (2,)),)
    assert out_bin_packet_capacity(maximum) == (
        len(b"OUT_BIN_BEGIN v1 I 1 2 8 s 16\n") + 16 + len(footer) + len(b"DONE\n")
    )
    narrowable = (ExpectedOutput("T", 1, 3, "i32", True, 4, (3,)),)
    maximum_bytes = out_bin_packet_capacity(narrowable)
    selected_narrow_bytes = len(b"OUT_BIN_BEGIN v1 T 1 3 1 u 3\n") + 3 + len(footer) + 5
    assert maximum_bytes - selected_narrow_bytes == 10  # Nine data bytes plus one count digit.
    with pytest.raises(OutPacketError, match="bounded host arena"):
        out_bin_packet_capacity((ExpectedOutput("BIG", 1, 67_108_864, "i32", True, 4, (67_108_864,)),))


@pytest.mark.parametrize(
    ("alter", "error"),
    [
        (lambda packet: b"METRIC cycles 1\n" + packet, "maximum capacity"),
        (lambda packet: b"\n" + packet, "exact frames"),
        (lambda packet: packet + b"EXTRA\n", "exact frames"),
        (lambda packet: packet[:-5], "DONE"),
        (lambda packet: packet.replace(b"OUT_BIN_BEGIN v1 N 2 1", b"OUT_BIN_BEGIN v1 N 1 2"), "geometry"),
        (lambda packet: packet.replace(b"OUT_BIN_BEGIN v1 N 2 1", b"OUT_BIN_BEGIN v1 N 2 1 "), "header"),
        (
            lambda packet: packet.replace(b"OUT_BIN_BEGIN v1 F 1 2 4 u 8", b"OUT_BIN_BEGIN v1 F 1 2 4 s 8"),
            "signed wire",
        ),
        (lambda packet: packet.replace(b"OUT_BIN_END v1 ", b"OUT_BIN_END v2 ", 1), "END"),
    ],
)
def test_packet_refuses_stray_or_malformed_wire(tmp_path, alter, error) -> None:
    with pytest.raises(OutPacketError, match=error):
        decode_out_bin_packet(alter(_closed_packet(tmp_path)), _roster())


def test_packet_refuses_missing_extra_or_forged_output_declaration(tmp_path) -> None:
    packet = _closed_packet(tmp_path)
    roster = _roster()
    for wrong in (
        roster[:-1],
        (roster[1], roster[0], *roster[2:]),
        (*roster, roster[-1]),
        (replace(roster[0], rows=2, cols=1), *roster[1:]),
        (*roster[:2], replace(roster[2], logical_dtype="i1", container_signed=False), roster[3]),
        (roster[0], replace(roster[1], container_max_bytes=2), *roster[2:]),
        (roster[0], replace(roster[1], container_signed=True), *roster[2:]),
        (*roster[:3], replace(roster[3], logical_shape=(True, 2))),
    ):
        with pytest.raises(OutPacketError):
            decode_out_bin_packet(packet, wrong)
