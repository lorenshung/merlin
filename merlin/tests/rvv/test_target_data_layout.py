"""A host loop is lowered under the cross target's own data layout, so its accesses are aligned."""

from __future__ import annotations

import pytest

from merlin.llvmlower import target_data_layout as TDL


def test_the_layout_line_is_read_structurally():
    ir = 'source_filename = "-"\ntarget datalayout = "e-m:e-p:64:64-i64:64"\ntarget triple = "x"\n'
    assert TDL.parse(ir) == "e-m:e-p:64:64-i64:64"
    assert TDL.parse("no layout here\n") is None


def test_the_layout_is_asked_of_the_compiler_for_its_flags():
    from merlin.llvmlower import toolchain

    clang = toolchain.clang()
    if not clang or not __import__("pathlib").Path(str(clang)).is_file():
        pytest.skip("no cross clang in this checkout")
    layout = TDL.of(clang, ["--target=riscv64-unknown-elf", "-march=rv64gc", "-O2"])
    assert layout.startswith("e-") and "i64:64" in layout


def test_a_lowering_with_the_layout_aligns_64_bit_accesses_naturally(tmp_path):
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    src = """module {
      func.func @f(%a: memref<8xi64>, %b: memref<8xf64>) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c8 = arith.constant 8 : index
        scf.for %i = %c0 to %c8 step %c1 {
          %x = memref.load %a[%i] : memref<8xi64>
          %y = arith.sitofp %x : i64 to f64
          memref.store %y, %b[%i] : memref<8xf64>
        }
        return
      }
    }"""
    default = lower_to_llvm_ir(src, workdir=tmp_path / "default")
    assert "load i64, ptr" in default and "align 4" in default, "LLVM's default layout under-aligns i64"
    layout = "e-m:e-p:64:64-i64:64-i128:128-n32:64-S128"
    got = lower_to_llvm_ir(src, workdir=tmp_path / "target", data_layout=layout)
    assert f'target datalayout = "{layout}"' in got
    loads = [line for line in got.splitlines() if "load i64" in line or "store double" in line]
    assert loads and all("align 8" in line for line in loads)
