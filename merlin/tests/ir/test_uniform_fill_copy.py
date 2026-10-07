"""Private uniform buffers may be replaced by exact destination stores only."""

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower.uniform_fill_copy import RUNNER_PRELUDE


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


def transform(tmp_path, source):
    from merlin.llvmlower.toolchain import m2m_python

    src = tmp_path / "input.mlir"
    dst = tmp_path / "output.mlir"
    script = tmp_path / "rewrite.py"
    src.write_text(source)
    script.write_text(
        "from torch_mlir import ir\nfrom pathlib import Path\n"
        + RUNNER_PRELUDE
        + "\nctx=ir.Context()\nwith ctx:\n m=ir.Module.parse(Path("
        + repr(str(src))
        + ").read_text())\n before=str(m)\n n=_fold_uniform_fill_copies(ctx,m)\n m.operation.verify()\n Path("
        + repr(str(dst))
        + ").write_text(str(m))\n print(n)\n"
    )
    p = subprocess.run([str(m2m_python()), str(script)], capture_output=True, text=True, check=True)
    return int(p.stdout.strip()), dst.read_text()


@pytest.mark.parametrize(
    "kind,bits", [("i8", [0, 127, 128, 255]), ("f32", [0, 0x80000000, 0x7FC12345, 0x7F800001, 0x3F812345])]
)
def test_native_strided_fills_preserve_all_bits_and_untouched_regions(tmp_path, kind, bits):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang

    source = fixture(kind)
    count, folded = transform(tmp_path, source)
    assert count == 2 and "memref.copy" not in folded and "memref.alloc" not in folded
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


@pytest.mark.parametrize(
    "change", ["alias", "external", "read_old", "nonidentity", "write_after", "nested_copy", "before_fill"]
)
def test_refuses_unproven_ownership_uniformity_or_order(tmp_path, change):
    source = fixture()
    if change == "alias":
        source = source.replace(
            "    %left =", "    %alias = memref.cast %tmp : memref<2x3xi8> to memref<?x?xi8>\n    %left ="
        )
    elif change == "external":
        source = source.replace("module {", "module {\n func.func private @escape(memref<2x3xi8>)").replace(
            "    %left =", "    func.call @escape(%tmp) : (memref<2x3xi8>) -> ()\n    %left ="
        )
    elif change == "read_old":
        source = source.replace("linalg.yield %v : i8", "linalg.yield %old : i8")
    elif change == "nonidentity":
        source = source.replace("affine_map<(i,j)->(i,j)>", "affine_map<(i,j)->(j,i)>")
    elif change == "write_after":
        source = source.replace("    %left =", "    memref.store %v,%tmp[%c0,%c0] : memref<2x3xi8>\n    %left =")
    elif change == "nested_copy":
        source = source.replace(
            "    memref.copy %tmp,%left",
            "    %yes = arith.constant true\n    scf.if %yes {\n    memref.copy %tmp,%left",
        ).replace("    %right =", "    }\n    %right =")
    elif change == "before_fill":
        begin = source.index("    linalg.generic")
        end = source.index("    %left =")
        writer = source[begin:end]
        source = source[:begin] + source[end:]
        source = source.replace("    %right =", writer + "    %right =")
    count, out = transform(tmp_path, source)
    assert count == 0 and "memref.copy" in out


def test_optional_normal_pipeline_hook_and_default_identity(tmp_path):
    from merlin.llvmlower.impr_features import normalize
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    source = fixture()
    source = "\n".join(line for line in source.splitlines() if "memref.dealloc" not in line)
    assert "fold_uniform_fill_copy" not in normalize(set())
    baseline = lower_to_llvm_ir(source, workdir=tmp_path / "off", vectorize=False)
    candidate = lower_to_llvm_ir(
        source, workdir=tmp_path / "on", vectorize=False, features=frozenset({"fold_uniform_fill_copy"})
    )
    assert "call void @memrefCopy" in baseline
    assert "call void @memrefCopy" not in candidate
    assert baseline == lower_to_llvm_ir(source, workdir=tmp_path / "off_again", vectorize=False)


@pytest.mark.parametrize(
    "stdout",
    [
        "",
        "OK fold_uniform_fill_copy -1",
        "OK fold_uniform_fill_copy x",
        "OK fold_uniform_fill_copy 2\nOK fold_uniform_fill_copy 3",
    ],
)
def test_selected_feature_requires_a_complete_count_report(tmp_path, stdout):
    from merlin.llvmlower.uniform_fill_copy import require_report

    with pytest.raises(ValueError):
        require_report(stdout, tmp_path)


def test_zero_matches_is_an_explicit_valid_report(tmp_path):
    from merlin.llvmlower.uniform_fill_copy import require_report

    assert require_report("OK fold_uniform_fill_copy 0", tmp_path)["copies_folded"] == 0
    assert json.loads((tmp_path / "uniform_fill_copy.json").read_text())["copies_folded"] == 0
