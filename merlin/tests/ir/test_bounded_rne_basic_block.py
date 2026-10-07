"""Original source proofs, independent inputs, barriers, boundaries and execution."""

import ctypes
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower.bounded_rne_basic_block import merlin_host_llvm_transform, rewrite
from merlin.llvmlower.toolchain import clang


def source(width=2):
    lines = ["define void @quant(ptr %input, ptr %output) {"]
    for lane in range(width):
        lines += [
            f"  %p{lane} = getelementptr float, ptr %input, i64 {lane}",
            f"  %x{lane} = load float, ptr %p{lane}, align 4",
        ]
    for lane in range(width):
        p = f"%a{lane}."
        lines += [
            f"  {p}1 = call float @llvm.maximum.f32(float %x{lane}, float -128.0)",
            f"  {p}2 = call float @llvm.minimum.f32(float {p}1, float 127.0)",
            f"  {p}3 = fptosi float {p}2 to i8",
            f"  {p}4 = sitofp i8 {p}3 to float",
            f"  {p}5 = fsub float {p}2, {p}4",
            f"  {p}6 = fneg float {p}5",
            f"  {p}7 = call float @llvm.maximum.f32(float {p}5, float {p}6)",
            f"  {p}8 = fcmp ogt float {p}7, 0.5",
            f"  {p}9 = fcmp oeq float {p}7, 0.5",
            f"  {p}10 = and i8 {p}3, 1",
            f"  {p}11 = icmp ne i8 {p}10, 0",
            f"  {p}12 = and i1 {p}9, {p}11",
            f"  {p}13 = or i1 {p}8, {p}12",
            f"  {p}14 = fcmp olt float {p}2, 0.0",
            f"  {p}15 = select i1 {p}14, i8 -1, i8 1",
            f"  {p}16 = select i1 {p}13, i8 {p}15, i8 0",
        ]
    for lane in range(width):
        lines += [f"  %r{lane} = add i8 %a{lane}.3, %a{lane}.16"]
    for lane in range(width):
        lines += [
            f"  %o{lane} = getelementptr i8, ptr %output, i64 {lane}",
            f"  store i8 %r{lane}, ptr %o{lane}, align 1",
        ]
    lines += [
        "  ret void",
        "}",
        "declare float @llvm.maximum.f32(float, float)",
        "declare float @llvm.minimum.f32(float, float)",
    ]
    return "\n".join(lines) + "\n"


def test_empty_policy_exact_and_explicit_parameters():
    text = source()
    assert rewrite(text)[0] == text and not rewrite(text)[1]["routes"]
    for policy in ("unknown", "rv64gcv"):
        with pytest.raises(ValueError):
            rewrite(text, host_isa=policy)
    for width in (True, 1, 5, 2.0):
        with pytest.raises(ValueError):
            rewrite(text, host_isa="rv64gc", width=width)


@pytest.mark.parametrize("width", [2, 3, 4])
def test_independent_packets_and_scalar_tail(width):
    selected, report = rewrite(source(width), host_isa="rv64gc", width=width)
    assert len(report["routes"]) == 1 and report["routes"][0]["lanes"] == width
    assert selected.index(f"fmax.s ft{width - 1}") < selected.index("fmin.s ft0")
    assert selected.index(f"fmin.s ft{width - 1}") < selected.index("fcvt.w.s $0")
    assert selected.count("store i8") == width and "%a0.3 = fptosi" in selected
    _, report = rewrite(source(3), host_isa="rv64gc", width=2)
    assert sum(row["lanes"] for row in report["routes"]) == 2


@pytest.mark.parametrize(
    "between",
    [
        "  store i8 1, ptr %output",
        "  call void @unknown()",
        "  br label %next\nnext:",
        "  %intervening = add i8 %r0, 1",
    ],
)
def test_any_intervening_statement_or_label_refuses(between):
    text = source().replace("  %r1 =", between + "\n  %r1 =")
    selected, report = rewrite(text, host_isa="rv64gc")
    assert selected == text and not report["routes"]


@pytest.mark.parametrize(
    "old,new",
    [
        ("float 127.0", "float 128.0"),
        ("fsub float", "fsub fast float"),
        ("and i8 %a1.3, 1", "and i8 %a1.3, 2"),
        ("add i8 %a1.3", "add nsw i8 %a1.3"),
        ("ogt float %a1.7, 0.5", "ogt float %a1.7, 0.4"),
    ],
)
def test_incomplete_numeric_proof_refuses(old, new):
    text = source().replace(old, new)
    assert rewrite(text, host_isa="rv64gc")[0] == text


def test_input_defined_after_insertion_or_in_other_block_refuses():
    text = source().replace("  %x1 = load float, ptr %p1, align 4\n", "")
    text = text.replace("  %r1 =", "  %x1 = load float, ptr %p1, align 4\n  %r1 =")
    assert rewrite(text, host_isa="rv64gc")[0] == text
    text = source().replace("  %a0.1 =", "  br label %next\nnext:\n  %a0.1 =")
    assert rewrite(text, host_isa="rv64gc")[0] == text


@pytest.mark.parametrize(
    "marker",
    ["attributes #0 = { strictfp }", "declare float @llvm.experimental.constrained.roundeven.f32(float, metadata)"],
)
def test_strict_and_constrained_refuse(marker):
    text = source() + marker + "\n"
    assert rewrite(text, host_isa="rv64gc")[0] == text


def test_verified_target_callback_and_paired_portable(tmp_path):
    text = tmp_path / "source.ll"
    text.write_text(source(4))
    selected = merlin_host_llvm_transform(Path(clang()).resolve().parent, host_isa="rv64gc", width=4)(
        text, tmp_path / "build"
    )
    assert "call { i32, i32, i32, i32 } asm" in selected.read_text()
    assert "asm " not in (tmp_path / "build/model.native.ll").read_text()
    assert text.read_text() == source(4)


def test_original_and_portable_boundary_results_exact_in_host_rounding_modes(tmp_path):
    c = Path(clang()).resolve()
    libraries = []
    text = source(4)
    portable, report = rewrite(text, host_isa="portable", width=4)
    assert report["routes"]
    for name, llvm in (("source", text), ("portable", portable)):
        path = tmp_path / (name + ".ll")
        path.write_text(llvm)
        so = path.with_suffix(".so")
        subprocess.run([str(c), "-shared", "-fPIC", "-O2", str(path), "-o", str(so)], check=True)
        lib = ctypes.CDLL(str(so))
        lib.quant.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        libraries.append(lib)
    halves = np.arange(-130.5, 130.5, dtype=np.float32)
    values = np.concatenate(
        [
            halves,
            np.nextafter(halves, np.float32(np.inf)),
            np.nextafter(halves, np.float32(-np.inf)),
            np.array([0.0, -0.0, 1e-40, -1e-40, np.inf, -np.inf], np.float32),
        ]
    )
    oracle = np.rint(np.clip(values, -128, 127)).astype(np.int8)
    libc = ctypes.CDLL(None)
    previous = libc.fegetround()
    try:
        for mode in (0, 0x400, 0x800, 0xC00):
            assert libc.fesetround(mode) == 0
            for offset in range(0, len(values), 4):
                lane = values[offset : offset + 4]
                padded = np.pad(lane, (0, 4 - len(lane))).astype(np.float32)
                for lib in libraries:
                    result = np.empty(4, np.int8)
                    lib.quant(padded.ctypes.data, result.ctypes.data)
                    assert np.array_equal(result[: len(lane)], oracle[offset : offset + 4])
    finally:
        assert libc.fesetround(previous) == 0
