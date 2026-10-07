"""Execute the model compiler's BF16 ABI against the actual bare-metal runtime."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import runtime_dir
from merlin.llvmlower import toolchain
from merlin.runtime.backends import spike, spike_model


def test_runtime_compiler_uses_model_abi_and_environment_headers(tmp_path, monkeypatch):
    include = tmp_path / "include"
    include.mkdir()
    (include / "math.h").touch()
    calls = []

    def run(cmd):
        calls.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, str(tmp_path) + "\n", "")

    monkeypatch.setattr(spike_model, "_run", run)
    flags = ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany"]
    result = spike_model._mlir_runtime_compiler(Path("model-clang"), Path("environment-gcc"), flags)
    assert calls == [[Path("environment-gcc"), "-print-sysroot"]]
    assert result[0] == "model-clang"
    assert result[-3:] == flags
    assert result[1] == spike_model.CLANG_TARGET
    assert result[result.index("-isystem") + 1] == str(include)


@pytest.mark.parametrize("value", ["", "relative/path", "/missing/sysroot"])
def test_runtime_compiler_refuses_missing_environment_headers(monkeypatch, value):
    monkeypatch.setattr(spike_model, "_run", lambda cmd: subprocess.CompletedProcess(cmd, 0, value, ""))
    with pytest.raises(spike_model.SpikeModelError, match="absolute sysroot"):
        spike_model._mlir_runtime_compiler(Path("clang"), Path("gcc"), [])


def test_actual_spike_bf16_conversion_roundtrip(tmp_path):
    gcc, clang, simulator = spike.gcc_path(), toolchain.clang(), spike.spike_path()
    if not all(p.is_file() for p in (gcc, clang, simulator)):
        pytest.skip("bare-metal GCC, model Clang, or Spike unavailable")
    harness = runtime_dir() / "baremetal/spike"
    flags = [
        "-march=rv64gc",
        "-mabi=lp64d",
        "-mcmodel=medany",
        "-O2",
        "-ffreestanding",
        "-fno-builtin",
        "-ffunction-sections",
        "-fdata-sections",
    ]
    compiler = spike_model._mlir_runtime_compiler(clang, gcc, flags)
    # RNE ties, both signed zeros, finite signs, subnormal boundaries, infinities,
    # and quiet/signaling NaNs with payloads exercise conversion rather than ABI alone.
    bits = [
        0x3F808000,
        0x3F818000,
        0xC0300000,
        0,
        0x80000000,
        1,
        0x80000001,
        0x00008000,
        0x00018000,
        0x007FFFFF,
        0x00800000,
        0x7F7FFFFF,
        0x7F800000,
        0xFF800000,
        0x7FC12345,
        0x7F812345,
        0xFFC12345,
    ]
    expected = []
    for value in bits:
        if value & 0x7F800000 == 0x7F800000:
            half = (value >> 16) | (0x40 if value & 0x7FFFFF else 0)
        else:
            half = ((value + 0x7FFF + ((value >> 16) & 1)) & 0xFFFFFFFF) >> 16
        expected.append(half << 16)
    source = tmp_path / "caller.c"
    source.write_text(
        """#include <stdint.h>
#include "htif.h"
volatile uint32_t values[]={"""
        + ",".join(str(v) + "u" for v in bits)
        + """};
int main(long hart) {
  if(hart) htif_exit(0);
  for(unsigned i=0;i<sizeof(values)/sizeof(values[0]);i++) {
    uint32_t raw=values[i];float f;__builtin_memcpy(&f,&raw,4);
    __bf16 b=(__bf16)f;f=(float)b;__builtin_memcpy(&raw,&f,4);
    htif_putd(raw);htif_putc(' ');
  }
  htif_putc('\\n');htif_exit(0);
}
"""
    )

    def run(cmd):
        return subprocess.run(list(map(str, cmd)), check=True, capture_output=True, text=True, timeout=60)

    caller = tmp_path / "caller.o"
    run([*compiler, "-I", harness, "-c", source, "-o", caller])
    common = [caller]
    for name, suffix in (("crt", ".S"), ("htif", ".c"), ("libc_min", ".c")):
        obj = tmp_path / (name + ".o")
        run([gcc, *flags, "-I", harness, "-c", harness / (name + suffix), "-o", obj])
        common.append(obj)
    outputs = {}
    for name, command in (("gcc_fallback", [gcc, *flags]), ("model_abi", compiler)):
        obj, elf = tmp_path / (name + ".o"), tmp_path / (name + ".elf")
        run([*command, "-c", runtime_dir() / "abi/mlir_runtime.c", "-o", obj])
        run(
            [
                gcc,
                *flags,
                "-nostdlib",
                "-nostartfiles",
                "-Wl,--gc-sections",
                "-T",
                harness / "link.ld",
                *common,
                obj,
                "-lm",
                "-lgcc",
                "-o",
                elf,
            ]
        )
        result = run([simulator, "--isa=rv64gc", "-m128", elf])
        outputs[name] = [int(v) for v in result.stdout.split()]
    assert outputs["gcc_fallback"] != expected, "negative control must expose the old ABI mismatch"
    assert outputs["model_abi"] == expected
