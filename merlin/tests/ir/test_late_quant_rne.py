"""Typed SSA legalization, independent refusal cases and portable execution."""

import ctypes
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower.late_quant_rne import merlin_host_llvm_transform, rewrite
from merlin.llvmlower.toolchain import clang

SOURCE = """define i8 @quant(float %x) {
  %v1 = call float @llvm.maximum.f32(float %x, float -1.280000e+02)
  %v2 = call float @llvm.minimum.f32(float %v1, float 1.270000e+02)
  %v3 = fptosi float %v2 to i8
  %v4 = sitofp i8 %v3 to float
  %v5 = fsub float %v2, %v4
  %v6 = fneg float %v5
  %v7 = call float @llvm.maximum.f32(float %v5, float %v6)
  %v8 = fcmp ogt float %v7, 5.000000e-01
  %v9 = fcmp oeq float %v7, 5.000000e-01
  %v10 = and i8 %v3, 1
  %v11 = icmp ne i8 %v10, 0
  %v12 = and i1 %v9, %v11
  %v13 = or i1 %v8, %v12
  %v14 = fcmp olt float %v2, 0.000000e+00
  %v15 = select i1 %v14, i8 -1, i8 1
  %v16 = select i1 %v13, i8 %v15, i8 0
  %v17 = add i8 %v3, %v16
  ret i8 %v17
}
declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
"""


def test_default_off_and_explicit_cpu_policy():
    assert rewrite(SOURCE)[0] == SOURCE
    assert not rewrite(SOURCE)[1]["routes"]
    with pytest.raises(ValueError, match="host ISA policy"):
        rewrite(SOURCE, host_isa="unknown")
    with pytest.raises(ValueError, match="requires"):
        rewrite(SOURCE, host_isa="portable", combine_clamp=True)
    with pytest.raises(TypeError, match="bool"):
        rewrite(SOURCE, host_isa="rv64gc", combine_clamp="true")


@pytest.mark.parametrize(
    "old,new",
    [
        ("-1.280000e+02", "-1.290000e+02"),
        ("1.270000e+02", "1.280000e+02"),
        ("and i8 %v3, 1", "and i8 %v3, 2"),
        ("5.000000e-01", "4.000000e-01"),
        ("fcmp olt", "fcmp ole"),
        ("i8 -1, i8 1", "i8 1, i8 -1"),
        ("fsub float %v2, %v4", "fsub float %x, %v4"),
        ("@llvm.minimum.f32", "@llvm.minnum.f32"),
        ("fsub float", "fsub fast float"),
        ("add i8", "add nsw i8"),
        ("fcmp ogt", "fcmp nnan ogt"),
        ("call float @llvm.minimum", "call nnan float @llvm.minimum"),
    ],
)
def test_refuse_incomplete_or_different_contract(old, new):
    source = SOURCE.replace(old, new)
    selected, proof = rewrite(source, host_isa="rv64gc")
    assert not proof["routes"] and selected == source


@pytest.mark.parametrize(
    "marker",
    [
        "attributes #0 = { strictfp }",
        "declare float @llvm.experimental.constrained.roundeven.f32(float, metadata)",
        'declare float @"llvm.experimental.constrained.roundeven.f32"(float, metadata)',
    ],
)
def test_strict_or_constrained_module_refused(marker):
    source = SOURCE + marker + "\n"
    selected, proof = rewrite(source, host_isa="rv64gc")
    assert selected == source and not proof["routes"] and proof["refusal"]


def test_structural_tokens_accept_renaming_comments_and_multiline_calls():
    source = SOURCE.replace("%x", '%"input value"')
    for i in range(17, 0, -1):
        source = source.replace(f"%v{i}", f'%"value {i}"')
    source = source.replace("float -1.280000e+02)", "float 0xC060000000000000) ; strictfp is only a comment")
    source = source.replace("float 1.270000e+02)", "\n      float 127.0)")
    source = source.replace("@llvm.maximum.f32", '@"llvm.maximum.f32"')
    selected, proof = rewrite(source, host_isa="rv64gc", combine_clamp=True)
    assert len(proof["routes"]) == 1
    assert 'float %"input value"' in selected and "~{ft0}" in selected
    assert "; strictfp is only a comment" in selected


def test_source_uses_and_existing_native_declaration_preserved():
    source = SOURCE.replace("  ret i8 %v17", "  %other = add i8 %v3, 2\n  ret i8 %v17")
    source += "declare float @llvm.roundeven.f32(float) #0\nattributes #0 = { nounwind }\n"
    selected, proof = rewrite(source, host_isa="portable")
    assert len(proof["routes"]) == 1
    assert "%other = add i8 %v3, 2" in selected and "%v3 = fptosi" in selected
    assert selected.count("declare float @llvm.roundeven.f32") == 1
    assert "declare float @llvm.roundeven.f32(float) #0" in selected


def test_scopes_are_independent_and_unused_quoted_argument_cannot_collide():
    second = SOURCE[: SOURCE.index("declare")].replace("@quant", "@quant2")
    source = SOURCE + second
    source = source.replace("float %x)", 'float %x, i8 %"merlin.rne.\\30")', 1)
    selected, proof = rewrite(source, host_isa="portable")
    assert len(proof["routes"]) == 2
    assert "%merlin.rne.1 = call" in selected


def test_width_is_derived_from_current_integer_type_and_clamp_bounds():
    source = SOURCE.replace("i8", "i16").replace("-1.280000e+02", "-32768.0").replace("1.270000e+02", "32767.0")
    selected, proof = rewrite(source, host_isa="rv64gc", combine_clamp=True)
    assert len(proof["routes"]) == 1
    assert proof["routes"][0]["bounds"] == [-32768, 32767]
    assert "trunc i32 %merlin.rne.0 to i16" in selected
    assert rewrite(SOURCE.replace("i8", "i16"), host_isa="rv64gc")[0] == SOURCE.replace("i8", "i16")


def test_unknown_tail_metadata_refuses_without_changing_source():
    source = SOURCE.replace("add i8 %v3, %v16", "add i8 %v3, %v16, !dbg !1")
    assert rewrite(source, host_isa="rv64gc")[0] == source


def test_verified_callback_selects_target_and_paired_portable_module(tmp_path):
    source = tmp_path / "input.ll"
    source.write_text(SOURCE)
    work = tmp_path / "callback"
    selected = merlin_host_llvm_transform(Path(clang()).resolve().parent, host_isa="rv64gc", combine_clamp=True)(
        source, work
    )
    proof = json.loads((work / "receipt.json").read_text())
    assert proof["host_isa"] == "rv64gc" and len(proof["routes"]) == 1
    assert source.read_text() == SOURCE
    assert selected == work / "model.ll"
    assert "fcvt.w.s" in selected.read_text()
    native = (work / "model.native.ll").read_bytes()
    assert b"@llvm.roundeven.f32" in native and b"asm " not in native
    assert proof["native_oracle_sha256"] == hashlib.sha256(native).hexdigest()


@pytest.mark.parametrize("bits", [8, 16])
def test_portable_native_boundary_oracle(tmp_path: Path, bits: int):
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    source = (
        SOURCE.replace("i8", f"i{bits}")
        .replace("-1.280000e+02", str(float(lower)))
        .replace("1.270000e+02", str(float(upper)))
    )
    baseline = tmp_path / "baseline.ll"
    selected = tmp_path / "selected.ll"
    baseline.write_text(source)
    selected.write_text(rewrite(source, host_isa="portable")[0])
    functions = []
    for stem, path in [("baseline", baseline), ("selected", selected)]:
        library = tmp_path / (stem + str(bits) + ".so")
        subprocess.run(
            [clang(), "-O2", "-shared", "-fPIC", str(path), "-lm", "-o", str(library)], check=True, capture_output=True
        )
        fn = ctypes.CDLL(str(library)).quant
        fn.argtypes = [ctypes.c_float]
        fn.restype = ctypes.c_int8 if bits == 8 else ctypes.c_int16
        functions.append(fn)
    half = np.arange(lower - 2, upper + 3, dtype=np.float32) + np.float32(0.5)
    rng = np.random.default_rng(715)
    random = rng.integers(0, 2**32, 20000, dtype=np.uint32).view(np.float32)
    random = random[~np.isnan(random)]
    values = np.concatenate(
        [
            half,
            np.nextafter(half, np.float32(np.inf)),
            np.nextafter(half, np.float32(-np.inf)),
            random,
            np.array([np.inf, -np.inf, 0.0, -0.0, 1e-40, -1e-40], dtype=np.float32),
        ]
    )
    expected = np.rint(np.clip(values, lower, upper)).astype(np.int64)
    for fn in functions:
        actual = np.fromiter((fn(float(x)) for x in values), dtype=np.int64)
        np.testing.assert_array_equal(actual, expected)
