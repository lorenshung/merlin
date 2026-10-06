"""Exact strided copies preserve suffix byte tails and refuse uncertain aliasing."""

import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower.contiguous_suffix_copy import RUNNER_PRELUDE


def fixture(kind="i8", same_root=False, dynamic=False, skip=False):
    srcstride = 2 if skip else 1
    cols = 3 if skip else 5
    base = "%a" if same_root else "%b"
    return f"""module {{
 func.func @forward(%in: memref<5x9x{kind}>, %out: memref<7x11x{kind}>) attributes {{llvm.emit_c_interface}} {{
  %a = memref.alloc() : memref<5x9x{kind}>
  %b = memref.alloc() : memref<7x11x{kind}>
  memref.copy %in,%a : memref<5x9x{kind}> to memref<5x9x{kind}>
  memref.copy %out,%b : memref<7x11x{kind}> to memref<7x11x{kind}>
  %s = memref.subview %a[0,1][3,{cols}][2,{srcstride}] : memref<5x9x{kind}> to memref<3x{cols}x{kind},strided<[18,{srcstride}],offset:1>>
  %d = memref.subview {base}[1,3][3,{cols}][2,{srcstride}] : memref<{("5x9" if same_root else "7x11")}x{kind}> to memref<3x{cols}x{kind},strided<[{(18 if same_root else 22)},{srcstride}],offset:{(12 if same_root else 14)}>>
  memref.copy %s,%d : memref<3x{cols}x{kind},strided<[18,{srcstride}],offset:1>> to memref<3x{cols}x{kind},strided<[{(18 if same_root else 22)},{srcstride}],offset:{(12 if same_root else 14)}>>
  memref.copy %b,%out : memref<7x11x{kind}> to memref<7x11x{kind}>
  return
 }}
}}"""


def transform(tmp_path, source):
    from merlin.llvmlower.toolchain import m2m_python

    src = tmp_path / "src.mlir"
    dst = tmp_path / "dst.mlir"
    script = tmp_path / "rewrite.py"
    src.write_text(source)
    script.write_text(
        "from torch_mlir import ir\nfrom pathlib import Path\n"
        + RUNNER_PRELUDE
        + "\nctx=ir.Context()\nwith ctx:\n m=ir.Module.parse(Path("
        + repr(str(src))
        + ").read_text())\n n=_specialize_contiguous_copies(ctx,m)\n m.operation.verify()\n Path("
        + repr(str(dst))
        + ").write_text(str(m))\n print(n)\n"
    )
    p = subprocess.run([str(m2m_python()), str(script)], check=True, capture_output=True, text=True)
    return int(p.stdout.strip()), dst.read_text()


@pytest.mark.parametrize("kind", ["i8", "f32"])
def test_native_odd_suffix_misalignment_nonuniform_bits_and_guards(tmp_path, kind):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    source = fixture(kind)
    count, folded = transform(tmp_path, source)
    assert count == 1 and "scf.for" in folded
    words = np.uint8 if kind == "i8" else np.uint32
    dtype = np.int8 if kind == "i8" else np.float32
    inp = (np.arange(45, dtype=words) * 17 + 13).reshape(5, 9)
    if kind == "f32":
        inp.ravel()[::5] = 0x7F800001
        inp.ravel()[1::5] = 0x80000000
    for label, code in [("control", source), ("candidate", folded)]:
        ll = tmp_path / (label + ".ll")
        ir = lower_to_llvm_ir(code, workdir=tmp_path / label, vectorize=False)
        ll.write_text(ir)
        if label == "candidate":
            assert "call void @memrefCopy" not in ir
        lib = tmp_path / (label + "_" + kind + ".so")
        subprocess.run(
            [str(clang()), "-O2", "-shared", "-fPIC", str(ll), str(mlir_runtime_c()), "-o", str(lib)],
            check=True,
            capture_output=True,
        )
        invoke = HostModel.load(str(lib))
        x = inp.view(dtype)
        for _ in range(3):
            backing = np.full(7 * 11 + 18, 91, dtype=words)
            out = backing[9:-9].reshape(7, 11).view(dtype)
            expected = backing.copy()
            expected[9:-9].reshape(7, 11)[1:6:2, 3:8] = inp[0:5:2, 1:6]
            invoke([(x.ctypes.data, x.shape), (out.ctypes.data, out.shape)])
            np.testing.assert_array_equal(backing, expected)
            np.testing.assert_array_equal(x.view(words), inp)


@pytest.mark.parametrize(
    "case", ["same_root", "no_suffix", "unknown_roots", "zero_extent", "dynamic_extent", "negative_stride"]
)
def test_refuses_unproved_disjoint_or_contiguous_storage(tmp_path, case):
    if case == "same_root":
        # Valid overlapping views in one allocation; no promotion to row memcpy.
        source = (
            fixture()
            .replace("%b[1,3][3,5][2,1]", "%a[0,2][3,5][2,1]")
            .replace(
                "memref<7x11xi8> to memref<3x5xi8,strided<[22,1],offset:14>>",
                "memref<5x9xi8> to memref<3x5xi8,strided<[18,1],offset:2>>",
            )
            .replace("to memref<3x5xi8,strided<[22,1],offset:14>>", "to memref<3x5xi8,strided<[18,1],offset:2>>")
        )
    elif case == "no_suffix":
        source = fixture(skip=True)
    elif case == "unknown_roots":
        source = "module { func.func @f(%a: memref<3x5xi8,strided<[10,1]>>, %b: memref<3x5xi8>) { memref.copy %a,%b : memref<3x5xi8,strided<[10,1]>> to memref<3x5xi8> return } }"
    else:
        shape = "0x5" if case == "zero_extent" else "?x5" if case == "dynamic_extent" else "3x5"
        stride = "-1" if case == "negative_stride" else "1"
        # Distinct fresh roots exercise the shape/stride gate itself.
        args = "%n:index" if case == "dynamic_extent" else ""
        size = "%n" if case == "dynamic_extent" else ""
        source = f"module {{ func.func @f({args}) {{ %a=memref.alloc({size}) : memref<{shape}xi8,strided<[10,{stride}]>> %b=memref.alloc({size}) : memref<{shape}xi8> memref.copy %a,%b : memref<{shape}xi8,strided<[10,{stride}]>> to memref<{shape}xi8> return }} }}"
    count, result = transform(tmp_path, source)
    assert count == 0 and "memref.copy" in result


def test_normal_pipeline_reports_selected_copy_and_retains_memcpy(tmp_path):
    import json

    from merlin.llvmlower.impr_features import normalize
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    assert "specialize_contiguous_copy" not in normalize(set())
    control = lower_to_llvm_ir(fixture(), workdir=tmp_path / "control", vectorize=False)
    result = lower_to_llvm_ir(
        fixture(), workdir=tmp_path / "candidate", vectorize=False, features=frozenset({"specialize_contiguous_copy"})
    )
    assert "call void @memrefCopy" in control
    assert "call void @memrefCopy" not in result and "llvm.memcpy" in result
    assert json.loads((tmp_path / "candidate/contiguous_suffix_copy.json").read_text())["copies_specialized"] == 1


@pytest.mark.parametrize(
    "text", ["", "OK specialize_contiguous_copy -1", "OK specialize_contiguous_copy 1\nOK specialize_contiguous_copy 1"]
)
def test_missing_or_ambiguous_report_refuses(tmp_path, text):
    from merlin.llvmlower.contiguous_suffix_copy import require_report

    with pytest.raises(ValueError):
        require_report(text, tmp_path)


def test_multiple_outer_dimensions_native(tmp_path):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    source = """module {
 func.func @forward(%in: memref<3x5x7xi8>, %out: memref<4x7x9xi8>) attributes {llvm.emit_c_interface} {
  %a=memref.alloc() : memref<3x5x7xi8>
  %b=memref.alloc() : memref<4x7x9xi8>
  memref.copy %in,%a : memref<3x5x7xi8> to memref<3x5x7xi8>
  memref.copy %out,%b : memref<4x7x9xi8> to memref<4x7x9xi8>
  %s=memref.subview %a[0,0,1][2,3,5][2,2,1] : memref<3x5x7xi8> to memref<2x3x5xi8,strided<[70,14,1],offset:1>>
  %d=memref.subview %b[0,1,2][2,3,5][2,2,1] : memref<4x7x9xi8> to memref<2x3x5xi8,strided<[126,18,1],offset:11>>
  memref.copy %s,%d : memref<2x3x5xi8,strided<[70,14,1],offset:1>> to memref<2x3x5xi8,strided<[126,18,1],offset:11>>
  memref.copy %b,%out : memref<4x7x9xi8> to memref<4x7x9xi8>
  return
 }
}"""
    count, folded = transform(tmp_path, source)
    assert count == 1 and folded.count("scf.for") == 2
    ll = tmp_path / "multi.ll"
    ll.write_text(lower_to_llvm_ir(folded, workdir=tmp_path / "lower", vectorize=False))
    lib = tmp_path / "multi_suffix.so"
    subprocess.run(
        [str(clang()), "-O2", "-shared", "-fPIC", str(ll), str(mlir_runtime_c()), "-o", str(lib)],
        check=True,
        capture_output=True,
    )
    x = np.arange(105, dtype=np.int8).reshape(3, 5, 7)
    y = np.full((4, 7, 9), -73, dtype=np.int8)
    expected = y.copy()
    expected[0:4:2, 1:6:2, 2:7] = x[0:3:2, 0:5:2, 1:6]
    HostModel.load(str(lib))([(x.ctypes.data, x.shape), (y.ctypes.data, y.shape)])
    np.testing.assert_array_equal(y, expected)
