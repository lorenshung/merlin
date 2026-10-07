"""Execute the emitted scalar IR against the audited source arithmetic."""

from __future__ import annotations

import ctypes
import shutil
import struct
import subprocess

import numpy as np
import pytest
from xdsl.dialects.builtin import ModuleOp, TensorType, f32, f64
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region

from merlin.llvmlower import toolchain
from merlin.llvmlower.cpu_flash_exp import (
    CPU_FLASH_BF16_AVX2_POLICY,
    build_cpu_flash_exp_nonpositive,
)


def _module():
    block = Block(arg_types=[f32])
    ops, result = build_cpu_flash_exp_nonpositive(
        block.args[0], backend_policy=CPU_FLASH_BF16_AVX2_POLICY, domain="nonpositive"
    )
    block.add_ops([*ops, ReturnOp(result)])
    module = ModuleOp([FuncOp("approx", ([f32], [f32]), Region(block))])
    module.verify()
    return module


@pytest.mark.parametrize("dtype", [f64, TensorType(f32, [8])])
def test_refuse_non_scalar_f32(dtype):
    with pytest.raises(TypeError, match="scalar f32"):
        build_cpu_flash_exp_nonpositive(
            Block(arg_types=[dtype]).args[0],
            backend_policy=CPU_FLASH_BF16_AVX2_POLICY,
            domain="nonpositive",
        )


@pytest.mark.parametrize("policy,domain", [("math", "nonpositive"), (CPU_FLASH_BF16_AVX2_POLICY, "finite")])
def test_require_explicit_policy_and_domain(policy, domain):
    with pytest.raises(ValueError):
        build_cpu_flash_exp_nonpositive(Block(arg_types=[f32]).args[0], backend_policy=policy, domain=domain)


@pytest.fixture(scope="module")
def lowered(tmp_path_factory):
    clang = toolchain.clang()
    translate = toolchain.mlir_translate()
    opt = translate.with_name("mlir-opt")
    if not all(shutil.which(str(p)) for p in [clang, translate, opt]):
        pytest.skip("standalone MLIR/LLVM toolchain unavailable")
    root = tmp_path_factory.mktemp("cpu_flash_exp")
    text = str(_module())
    assert text.count("math.fma") == 4
    assert "fastmath" not in text
    src = root / "input.mlir"
    src.write_text(text)
    lowered = root / "llvm.mlir"
    subprocess.run(
        [
            str(opt),
            str(src),
            "--convert-math-to-llvm",
            "--convert-arith-to-llvm",
            "--convert-func-to-llvm",
            "--reconcile-unrealized-casts",
            "-o",
            str(lowered),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    llvm = root / "input.ll"
    subprocess.run(
        [str(translate), "--mlir-to-llvmir", str(lowered), "-o", str(llvm)], check=True, capture_output=True, text=True
    )
    assert "llvm.fma.f32" in llvm.read_text()
    shared = root / "generated.so"
    subprocess.run(
        [str(clang), "-O2", "-fPIC", "-shared", str(llvm), "-lm", "-o", str(shared)],
        check=True,
        capture_output=True,
        text=True,
    )
    # Independent reference transcribes the audited SIMD formula using libc fmaf,
    # not Python arithmetic or the xDSL builder. Rounding is enforced by fmaf and
    # disabled implicit contraction; float bit reinterpretation uses memcpy.
    reference = root / "reference.c"
    reference.write_text("""
#include <math.h>
#include <stdint.h>
#include <string.h>
static float bits(uint32_t b) { float f; memcpy(&f,&b,4); return f; }
float reference(float x) {
  if (x < bits(0xc2aeac50)) return 0.f;
  float s = x * bits(0x3fb8aa3b);
  float f = s - floorf(s);
  float p = fmaf(f, -0.079204240219773236f, -0.22433836478672356f);
  p = fmaf(f, p, 0.30354260500649682f);
  p = fmaf(f, p, 0.00010703434948458272f);
  s = s - p;
  float t = fmaf(8388608.f, s, 1065353216.f);
  int32_t i = (int32_t)t;
  float y; memcpy(&y,&i,4); return y;
}
""")
    refso = root / "reference.so"
    subprocess.run(
        [str(clang), "-O2", "-ffp-contract=off", "-fPIC", "-shared", str(reference), "-lm", "-o", str(refso)],
        check=True,
        capture_output=True,
        text=True,
    )
    return root, llvm, ctypes.CDLL(str(shared)), ctypes.CDLL(str(refso))


def test_native_matches_source_finite_zero_underflow_and_negative_infinity(lowered):
    _, _, generated, reference = lowered
    actual = generated.approx
    expected = reference.reference
    for fn in [actual, expected]:
        fn.argtypes = [ctypes.c_float]
        fn.restype = ctypes.c_float
    threshold = np.array(0xC2AEAC50, dtype=np.uint32).view(np.float32)
    rng = np.random.default_rng(713)
    values = np.concatenate(
        [
            np.linspace(threshold, 0, 10001, dtype=np.float32),
            -rng.uniform(0, 100, 10000).astype(np.float32),
            rng.integers(0x80000000, 0xFF800001, 10000, dtype=np.uint32).view(np.float32),
            np.array(
                [
                    0.0,
                    -0.0,
                    -np.inf,
                    -np.finfo(np.float32).max,
                    np.nextafter(threshold, -np.inf, dtype=np.float32),
                    threshold,
                    np.nextafter(threshold, np.inf, dtype=np.float32),
                ],
                dtype=np.float32,
            ),
        ]
    )
    for x in values:
        assert struct.pack("f", actual(float(x))) == struct.pack("f", expected(float(x))), x
    # fexp_u20(0) intentionally differs from correctly rounded exp(0).
    assert actual(0.0) != 1.0
    assert actual(float("-inf")) == 0.0


def test_scalar_rv64gc_codegen_retains_fma(lowered):
    root, llvm, _, _ = lowered
    asm = root / "scalar.s"
    subprocess.run(
        [
            str(toolchain.clang()),
            "--target=riscv64-unknown-elf",
            "-march=rv64gc",
            "-mabi=lp64d",
            "-O2",
            "-S",
            str(llvm),
            "-o",
            str(asm),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    text = asm.read_text()
    assert "fmadd.s" in text
    assert "vset" not in text
