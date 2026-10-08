"""Checked packet alignment preserves byte-storage contracts."""

import ctypes
import mmap
import os
import subprocess
from pathlib import Path

import pytest

from merlin.llvmlower.enclosed_readout import emit_pair_scan, prove
from merlin.llvmlower.requantization import quantized


def library(tmp_path, checked_alignment):
    certificate = prove([1.0], [0.99, 1.01], -90, 90)
    path = tmp_path / f"scan{int(checked_alignment)}.c"
    path.write_text(
        emit_pair_scan(certificate, "scan", copy_policy="compiler_builtin", checked_alignment=checked_alignment)
    )
    obj = path.with_suffix(".so")
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-fno-builtin", str(path), "-o", str(obj)], check=True)
    function = ctypes.CDLL(str(obj)).scan
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    return function


@pytest.mark.parametrize("checked_alignment", [False, True])
def test_all_pointer_residues_complete_packets_tails_and_stable_second(tmp_path, checked_alignment):
    scan = library(tmp_path, checked_alignment)
    for first_residue in range(8):
        for second_residue in range(8):
            for count in (0, 1, 7, 8, 9, 15, 16, 17, 180, 181):
                first = (ctypes.c_ubyte * (count + 32))(*([173] * (count + 32)))
                second = (ctypes.c_ubyte * (count + 32))(*([219] * (count + 32)))
                fi = 8 + (first_residue - ctypes.addressof(first)) % 8
                si = 8 + (second_residue - ctypes.addressof(second)) % 8
                source = list(range(-90, -90 + count))
                for i, value in enumerate(source):
                    first[fi + i] = quantized(value, [0.99]) & 255
                    second[si + i] = quantized(value, [1.01]) & 255
                second_before = bytes(second)
                scan(ctypes.byref(first, fi), ctypes.byref(second, si), count)
                assert bytes(first[fi : fi + count]) == bytes(value & 255 for value in source)
                assert bytes(first[:fi]) == bytes([173] * fi)
                assert bytes(first[fi + count :]) == bytes([173] * (len(first) - fi - count))
                assert bytes(second) == second_before


def test_read_only_second_and_inaccessible_packet_tail(tmp_path):
    scan = library(tmp_path, True)
    libc = ctypes.CDLL(None)
    libc.mprotect.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
    for count in (0, 1, 7, 8, 9, 15, 16, 17, 180, 181):
        # Each live span ends exactly at an inaccessible page. The second live
        # page is read-only after initialization; packet copies may not read
        # past count or write second even on disagreement lanes.
        a = mmap.mmap(-1, mmap.PAGESIZE * 2)
        b = mmap.mmap(-1, mmap.PAGESIZE * 2)
        av = (ctypes.c_ubyte * (mmap.PAGESIZE * 2)).from_buffer(a)
        bv = (ctypes.c_ubyte * (mmap.PAGESIZE * 2)).from_buffer(b)
        abase, bbase = ctypes.addressof(av), ctypes.addressof(bv)
        start = mmap.PAGESIZE - count
        values = list(range(-90, -90 + count))
        for i, value in enumerate(values):
            av[start + i] = quantized(value, [0.99]) & 255
            bv[start + i] = quantized(value, [1.01]) & 255
        before = bytes(bv[: mmap.PAGESIZE])
        assert libc.mprotect(abase + mmap.PAGESIZE, mmap.PAGESIZE, 0) == 0
        assert libc.mprotect(bbase + mmap.PAGESIZE, mmap.PAGESIZE, 0) == 0
        assert libc.mprotect(bbase, mmap.PAGESIZE, 1) == 0
        scan(abase + start, bbase + start, count)
        assert bytes(av[start : mmap.PAGESIZE]) == bytes(value & 255 for value in values)
        assert bytes(bv[: mmap.PAGESIZE]) == before
        for base in (abase, bbase):
            assert libc.mprotect(base, mmap.PAGESIZE * 2, 3) == 0
        del av, bv
        a.close()
        b.close()


@pytest.mark.parametrize("offset", [0, 1])
def test_unknown_pair_still_traps_in_aligned_and_byte_paths(tmp_path, offset):
    certificate = prove([1.0], [0.99, 1.01], -90, 90)
    code = emit_pair_scan(certificate, "scan", copy_policy="compiler_builtin", checked_alignment=True)
    code += f"""
int main(void){{
 unsigned char a[16] __attribute__((aligned(8)))={{0}};
 const unsigned char b[16] __attribute__((aligned(8)))={{1,1,1,1,1,1,1,1,1}};
 scan(a+{offset},b+{offset},8);return 0;
}}
"""
    source = tmp_path / "trap.c"
    source.write_text(code)
    exe = tmp_path / "trap"
    subprocess.run(["cc", "-O2", str(source), "-o", str(exe)], check=True)
    import resource

    def no_core():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    assert subprocess.run([str(exe)], preexec_fn=no_core).returncode < 0


def test_explicit_checked_alignment_contract_and_unchanged_default():
    p = prove([1.0], [0.99, 1.01], -90, 90)
    for copy_policy in ("runtime", "compiler_builtin"):
        assert emit_pair_scan(p, "scan", copy_policy=copy_policy) == emit_pair_scan(
            p, "scan", copy_policy=copy_policy, checked_alignment=False
        )
        assert "assume_aligned" not in emit_pair_scan(p, "scan", copy_policy=copy_policy)
    with pytest.raises(ValueError, match="compiler builtin"):
        emit_pair_scan(p, "scan", checked_alignment=True)
    for value in (None, 0, 1, "true"):
        with pytest.raises(ValueError, match="boolean"):
            emit_pair_scan(p, "scan", copy_policy="compiler_builtin", checked_alignment=value)


def test_scan_preserves_existing_floating_environment(tmp_path):
    p = prove([1.0], [0.99, 1.01], -90, 90)
    source = tmp_path / "effects.c"
    code = emit_pair_scan(p, "scan", copy_policy="compiler_builtin", checked_alignment=True)
    code += """
#include <fenv.h>
int verify(void){
 fenv_t original;fegetenv(&original);
 int modes[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
 for(unsigned mode=0;mode<4;mode++)for(unsigned offset=0;offset<2;offset++){
  unsigned char a[24] __attribute__((aligned(8)))={0};
  const unsigned char b[24] __attribute__((aligned(8)))={0};
  fesetround(modes[mode]);feclearexcept(FE_ALL_EXCEPT);
  feraiseexcept(FE_INVALID|FE_INEXACT|FE_UNDERFLOW);
  int before=fetestexcept(FE_ALL_EXCEPT);
  scan(a+offset,b+offset,17);
  if(fegetround()!=modes[mode]||fetestexcept(FE_ALL_EXCEPT)!=before){fesetenv(&original);return 1;}
 }
 fesetenv(&original);return 0;
}
"""
    source.write_text(code)
    so = source.with_suffix(".so")
    subprocess.run(["cc", "-shared", "-fPIC", "-O2", "-frounding-math", str(source), "-lm", "-o", str(so)], check=True)
    assert ctypes.CDLL(str(so)).verify() == 0


def test_actual_rv64_object_has_checked_word_loads_and_byte_fallback(tmp_path):
    clang = os.environ.get("MERLIN_CLANG")
    if not clang:
        pytest.skip("selected compiler required")
    p = prove([1.0], [0.99, 1.01], -90, 90)
    histograms = {}
    for selected in (False, True):
        source = tmp_path / f"rv{int(selected)}.c"
        source.write_text(emit_pair_scan(p, "scan", copy_policy="compiler_builtin", checked_alignment=selected))
        obj = source.with_suffix(".o")
        subprocess.run(
            [
                clang,
                "--target=riscv64-unknown-elf",
                "-march=rv64gc",
                "-ffreestanding",
                "-fno-builtin",
                "-O2",
                "-c",
                str(source),
                "-o",
                str(obj),
            ],
            check=True,
            capture_output=True,
        )
        tools = Path(clang).parent
        undefined = subprocess.run(
            [str(tools / "llvm-nm"), "--undefined-only", str(obj)], check=True, text=True, capture_output=True
        )
        assert not undefined.stdout.strip()
        result = subprocess.run(
            [str(tools / "llvm-objdump"), "-d", "--no-show-raw-insn", str(obj)],
            check=True,
            text=True,
            capture_output=True,
        )
        histogram = {}
        for line in result.stdout.splitlines():
            prefix, colon, body = line.partition(":")
            if colon and prefix.strip() and all(c in "0123456789abcdef" for c in prefix.strip()) and body.split():
                op = body.split()[0]
                histogram[op] = histogram.get(op, 0) + 1
        histograms[selected] = histogram
    # Stack restore ld instructions occur in both. Only the selected path adds
    # the two packet word loads; the unsupported alignment path retains lbu.
    assert histograms[True].get("ld", 0) >= histograms[False].get("ld", 0) + 2
    assert histograms[True].get("lbu", 0) >= 16
