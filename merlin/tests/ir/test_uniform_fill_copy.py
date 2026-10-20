"""Default copy lowering preserves storage bits, strided tails and guards.

The optional performance rewrite was removed; these exactness checks now
exercise the ordinary upstream copy implementation.
"""
import subprocess
import numpy as np
import pytest

def fixture(kind="i8"):
    return f"""module {{
  func.func @forward(%value: memref<1x{kind}>, %out: memref<5x7x{kind}>) attributes {{llvm.emit_c_interface}} {{
    %c0 = arith.constant 0 : index
    %v = memref.load %value[%c0] : memref<1x{kind}>
    %tmp = memref.alloc() : memref<2x3x{kind}>
    linalg.generic {{indexing_maps=[affine_map<(i,j)->(i,j)>],iterator_types=["parallel","parallel"]}} outs(%tmp : memref<2x3x{kind}>) {{
    ^bb0(%old: {kind}):
      linalg.yield %v : {kind}
    }}
    %left = memref.subview %out[1,1][2,3][1,1] : memref<5x7x{kind}> to memref<2x3x{kind},strided<[7,1],offset:8>>
    memref.copy %tmp,%left : memref<2x3x{kind}> to memref<2x3x{kind},strided<[7,1],offset:8>>
    %right = memref.subview %out[3,3][2,3][1,1] : memref<5x7x{kind}> to memref<2x3x{kind},strided<[7,1],offset:24>>
    memref.copy %tmp,%right : memref<2x3x{kind}> to memref<2x3x{kind},strided<[7,1],offset:24>>
    memref.dealloc %tmp : memref<2x3x{kind}>
    return
  }}
}}"""

@pytest.mark.parametrize(
    "kind,bits", [("i8", [0, 127, 128, 255]), ("f32", [0, 0x80000000, 0x7FC12345, 0x7F800001, 0x3F812345])]
)
def test_native_strided_fills_preserve_all_bits_and_untouched_regions(tmp_path, kind, bits):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    source = fixture(kind)
    folded = source
    unsigned = np.uint8 if kind == "i8" else np.uint32
    dtype = np.int8 if kind == "i8" else np.float32
    for label, ir in [("control", source), ("candidate", folded)]:
        # The ordinary pipeline owns deallocation; direct rewrite tests above also
        # verify explicit same-block frees. Remove the fixture's explicit free
        # before invoking that ownership pipeline.
        ir = "\n".join(line for line in ir.splitlines() if "memref.dealloc" not in line)
        ll = tmp_path / (label + ".ll")
        ll.write_text(lower_to_llvm_ir(ir, workdir=tmp_path / label, vectorize=False))
        lib = tmp_path / (label + "_" + kind + ".so")
        subprocess.run(
            [str(clang()), "-O2", "-shared", "-fPIC", str(ll), str(mlir_runtime_c()), "-o", str(lib)],
            check=True,
            capture_output=True,
        )
        invoke = HostModel.load(str(lib))
        for bit in bits:
            value = np.array([bit], dtype=unsigned).view(dtype)
            backing = np.full((5 * 7 + 16,), 77, dtype=unsigned)
            out = backing[8:-8].reshape(5, 7).view(dtype)
            expected = backing.copy()
            e = expected[8:-8].reshape(5, 7)
            e[1:3, 1:4] = bit
            e[3:5, 3:6] = bit
            for _ in range(3):
                invoke([(value.ctypes.data, value.shape), (out.ctypes.data, out.shape)])
                np.testing.assert_array_equal(backing, expected)
