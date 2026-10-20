"""Default copy lowering preserves storage bits, strided tails and guards.

The optional performance rewrite was removed; these exactness checks now
exercise the ordinary upstream copy implementation.
"""
import subprocess
import numpy as np
import pytest

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

@pytest.mark.parametrize("kind", ["i8", "f32"])
def test_native_odd_suffix_misalignment_nonuniform_bits_and_guards(tmp_path, kind):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    source = fixture(kind)
    folded = source
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
    folded = source
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
