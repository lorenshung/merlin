"""Embedded host inputs retain their captured IEEE-754 bits, including NaN payloads."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower.c_runtime import _embed_array

_F32_BITS = [
    0x00000000, 0x80000000, 0x3FA00000, 0x7F800000, 0xFF800000,
    0x7FC00001, 0xFFC12345, 0x7F800001, 0xFF812345,
]
_F64_BITS = [
    0x0000000000000000, 0x8000000000000000, 0x3FF4000000000000,
    0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000001,
    0xFFF8ABCDEF123456, 0x7FF0000000000001, 0xFFF0123456789ABC,
]


@pytest.mark.parametrize(
    ("dtype", "bits", "c_type", "c_int_type", "digits"),
    [
        (
            "f32",
            _F32_BITS,
            "float", "uint32_t", 8,
        ),
        (
            "f64",
            _F64_BITS,
            "double", "uint64_t", 16,
        ),
    ],
)
@pytest.mark.parametrize("compiler", ["gcc", "clang"])
@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
def test_nonfinite_static_inputs_preserve_raw_bits(
    tmp_path, dtype, bits, c_type, c_int_type, digits, compiler, optimization,
):
    cc = shutil.which(compiler)
    if cc is None:
        pytest.skip(f"{compiler} unavailable")
    integer_dtype = np.uint32 if dtype == "f32" else np.uint64
    float_dtype = np.float32 if dtype == "f32" else np.float64
    samples = np.array(bits, dtype=integer_dtype).view(float_dtype)
    emitted = _embed_array(samples, dtype)
    assert emitted.split(",")[:3] == ["0.0", "-0.0", "1.25"]
    source = f"""
        #include <stdint.h>
        #include <stdio.h>
        #include <string.h>
        static const {c_type} values[] = {{{emitted}}};
        int main(void) {{
            for (unsigned i = 0; i < sizeof(values) / sizeof(values[0]); ++i) {{
                {c_int_type} raw;
                memcpy(&raw, &values[i], sizeof(raw));
                printf("%0{digits}llx\\n", (unsigned long long) raw);
            }}
            return 0;
        }}
    """
    executable = tmp_path / f"{compiler}-{optimization[1:]}-{dtype}"
    built = subprocess.run(
        [cc, "-std=gnu11", optimization, "-x", "c", "-", "-o", str(executable)],
        input=source, text=True, capture_output=True, check=False,
    )
    assert built.returncode == 0, built.stderr
    observed = subprocess.run([str(executable)], capture_output=True, text=True, check=True)
    assert observed.stdout.splitlines() == [f"{bit:0{digits}x}" for bit in bits]


@pytest.mark.parametrize(("dtype", "bits", "c_type"), [
    ("f32", _F32_BITS, "float"),
    ("f64", _F64_BITS, "double"),
])
def test_selected_cross_object_keeps_nonfinite_static_bits(tmp_path, dtype, bits, c_type):
    cross = os.environ.get("MERLIN_RISCV_GCC")
    if not cross:
        pytest.skip("selected RISC-V compiler was not supplied")
    objcopy = Path(cross).with_name(Path(cross).name.replace("-gcc", "-objcopy"))
    if not objcopy.is_file():
        pytest.skip("selected RISC-V objcopy is unavailable")
    integer_dtype = np.uint32 if dtype == "f32" else np.uint64
    float_dtype = np.float32 if dtype == "f32" else np.float64
    samples = np.array(bits, dtype=integer_dtype).view(float_dtype)
    source = (
        f'__attribute__((section(".merlin_ieee"), used)) '
        f'static const {c_type} values[] = {{{_embed_array(samples, dtype)}}};'
    )
    object_path = tmp_path / f"ieee-{dtype}.o"
    emitted = subprocess.run(
        [cross, "-std=gnu11", "-O2", "-c", "-x", "c", "-", "-o", str(object_path)],
        input=source, text=True, capture_output=True, check=False,
    )
    assert emitted.returncode == 0, emitted.stderr
    section_path = tmp_path / f"ieee-{dtype}.bin"
    subprocess.run(
        [str(objcopy), "--dump-section", f".merlin_ieee={section_path}", str(object_path)],
        capture_output=True, text=True, check=True,
    )
    assert section_path.read_bytes() == np.array(bits, dtype=integer_dtype).tobytes()
