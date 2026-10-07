"""Explicit loop retention survives the actual upstream translation boundary."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest
from xdsl.context import Context
from xdsl.dialects import llvm
from xdsl.dialects.builtin import Builtin
from xdsl.parser import Parser

from merlin.llvmlower.llvm_loop_metadata import disable_loop_unroll

SOURCE = """module {
 llvm.func @consume(i64)
 llvm.func @loop() {
  %zero = llvm.mlir.constant(0 : i64) : i64
  %one = llvm.mlir.constant(1 : i64) : i64
  %eight = llvm.mlir.constant(8 : i64) : i64
  llvm.br ^header(%zero : i64)
 ^header(%i : i64):
  %cmp = llvm.icmp "slt" %i, %eight : i64
  llvm.cond_br %cmp, ^body, ^exit
 ^body:
  llvm.call @consume(%i) : (i64) -> ()
  %next = llvm.add %i, %one : i64
  llvm.br ^header(%next : i64)
 ^exit:
  llvm.return
 }
}"""


def fixture():
    context = Context(allow_unregistered=True)
    context.load_dialect(Builtin)
    context.load_dialect(llvm.LLVM)
    module = Parser(context, SOURCE).parse_module()
    latch = [op for op in module.walk() if isinstance(op, llvm.BrOp)][-1]
    return module, latch


def test_helper_refuses_existing_choices_and_nonbranches():
    module, latch = fixture()
    disable_loop_unroll(latch)
    module.verify()
    before = str(module)
    with pytest.raises(ValueError, match="composition"):
        disable_loop_unroll(latch)
    assert str(module) == before
    with pytest.raises(ValueError, match="branch"):
        disable_loop_unroll(next(op for op in module.walk() if isinstance(op, llvm.CallOp)))


def test_actual_upstream_translation_and_optimizer_preserve_disable(tmp_path):
    translate = os.environ.get("MERLIN_MLIR_TRANSLATE") or shutil.which("mlir-translate")
    if not translate:
        pytest.skip("upstream MLIR translator unavailable")
    opt = Path(translate).parent / "opt"
    if not opt.exists():
        pytest.skip("upstream LLVM optimizer unavailable")
    module, latch = fixture()
    disable_loop_unroll(latch)
    source, ir, optimized = [tmp_path / name for name in ("input.mlir", "input.ll", "optimized.ll")]
    source.write_text(str(module) + "\n")
    subprocess.run(
        [translate, "--mlir-to-llvmir", str(source), "-o", str(ir)], check=True, capture_output=True, text=True
    )
    assert '!"llvm.loop.unroll.disable"' in ir.read_text()
    subprocess.run(
        [str(opt), "-passes=default<O2>", "-S", str(ir), "-o", str(optimized)],
        check=True,
        capture_output=True,
        text=True,
    )
    text = optimized.read_text()
    assert '!"llvm.loop.unroll.disable"' in text
    assert text.count("call void @consume") == 1
