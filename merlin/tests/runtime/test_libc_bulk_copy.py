"""Run the actual freestanding copy functions across alignment and tail cases."""

import ctypes
import subprocess

import pytest

from merlin.common.paths import runtime_dir


@pytest.fixture(scope="module")
def libc(tmp_path_factory):
    work = tmp_path_factory.mktemp("bulk_libc")
    library = work / "bulk.so"
    console = work / "console.c"
    console.write_text(
        "void htif_putc(char c) { (void)c; }\n"
        "void htif_puts(const char *s) { (void)s; }\n"
        "__attribute__((noreturn)) void htif_exit(int code) { (void)code; __builtin_trap(); }\n"
    )
    subprocess.run(
        [
            "cc",
            "-O2",
            "-fno-builtin",
            "-fPIC",
            "-shared",
            "-Dmemcpy=merlin_test_memcpy",
            "-Dmemset=merlin_test_memset",
            "-Dmemmove=merlin_test_memmove",
            str(runtime_dir() / "baremetal/spike/libc_min.c"),
            str(console),
            "-o",
            str(library),
        ],
        check=True,
    )
    lib = ctypes.CDLL(str(library))
    lib.merlin_test_memcpy.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    lib.merlin_test_memcpy.restype = ctypes.c_void_p
    lib.merlin_test_memset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
    lib.merlin_test_memset.restype = ctypes.c_void_p
    return lib


@pytest.mark.parametrize("src_offset", range(ctypes.sizeof(ctypes.c_void_p)))
def test_copy_all_dest_alignments_lengths_and_guards(libc, src_offset):
    payload = bytes((i * 73 + 11) % 256 for i in range(320))
    src = ctypes.create_string_buffer(payload, len(payload))
    for dst_offset in range(ctypes.sizeof(ctypes.c_void_p)):
        for length in range(130):
            dst = ctypes.create_string_buffer(b"\xa5" * 320, 320)
            address = ctypes.addressof(dst) + dst_offset
            result = libc.merlin_test_memcpy(address, ctypes.addressof(src) + src_offset, length)
            expected = bytearray(b"\xa5" * 320)
            expected[dst_offset : dst_offset + length] = payload[src_offset : src_offset + length]
            assert result == address
            assert dst.raw == expected
            assert src.raw == payload


@pytest.mark.parametrize("fill", [0, 1, 0xAB, 257, -1])
def test_fill_all_alignments_lengths_and_guards(libc, fill):
    for offset in range(ctypes.sizeof(ctypes.c_void_p)):
        for length in range(130):
            dst = ctypes.create_string_buffer(b"\xa5" * 320, 320)
            address = ctypes.addressof(dst) + offset
            result = libc.merlin_test_memset(address, fill, length)
            expected = bytearray(b"\xa5" * 320)
            expected[offset : offset + length] = bytes([fill & 255]) * length
            assert result == address
            assert dst.raw == expected
