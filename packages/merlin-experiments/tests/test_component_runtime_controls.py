"""Private primitive controls use parsed upstream semantics and real stock tools."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from merlin_experiments.phase2.component_runtime_controls import (
    emit_primitive_llvm,
    parse_primitive,
    verify_primitive_llvm,
)

from merlin.common import invocation_record


def source(shape="1x3", *, add=True):
    tensor = "tensor<" + shape + "xf32>"
    body = f"%sum = arith.addf %a, %b : {tensor} " if add else ""
    value = "%sum" if add else "%a"
    return (
        f"module {{ func.func @main(%a: {tensor}, %b: {tensor}) -> ({tensor}, {tensor}) {{ "
        + body
        + f"func.return {value}, %b : {tensor}, {tensor} }} }}\n"
    )


@pytest.mark.parametrize("shape", ["1x1", "2x3", "1x7"])
@pytest.mark.parametrize("add", [True, False])
def test_upstream_control_semantics_cover_full_output_extents(shape, add):
    original = source(shape, add=add)
    program = parse_primitive(original)
    lowered = emit_primitive_llvm(program, "control_entry")
    proof = verify_primitive_llvm(original, lowered, entry_symbol="control_entry")
    assert proof["logical_output_stores"] == 2 * program.count
    assert "runtime/effects unqualified" in proof["scope"]


@pytest.mark.parametrize("defect", ["wrong_operand", "missing_output", "extent", "input_write", "extra_instruction"])
def test_symbolic_validation_rejects_concrete_lowered_defects(defect):
    original = source()
    lowered = emit_primitive_llvm(parse_primitive(original), "control_entry")
    lines = lowered.splitlines()
    if defect == "wrong_operand":
        lowered = lowered.replace("llvm.fadd %v2, %v4", "llvm.fadd %v2, %v2", 1)
    elif defect == "missing_output":
        index = next(index for index, line in enumerate(lines) if "llvm.store" in line)
        lowered = "\n".join(lines[:index] + lines[index + 1 :])
    elif defect == "extent":
        lowered = lowered.replace("%ptr0[0]", "%ptr0[3]", 1)
    elif defect == "input_write":
        lowered = lowered.replace("%ptr2[0]", "%ptr0[0]", 1)
    else:
        lowered = lowered.replace("llvm.fadd", "llvm.fmul", 1)
    assert lowered != emit_primitive_llvm(parse_primitive(original), "control_entry")
    with pytest.raises(ValueError):
        verify_primitive_llvm(original, lowered, entry_symbol="control_entry")


def test_unknown_source_operation_is_refused_before_emission():
    with pytest.raises(ValueError, match="unsupported operation"):
        parse_primitive(source().replace("arith.addf", "arith.mulf"))


@pytest.mark.parametrize("attribute", ["function", "return"])
def test_unproved_source_semantic_attributes_are_refused(attribute):
    original = source()
    if attribute == "function":
        original = original.replace(") { %sum", ") attributes {private_control = true} { %sum")
    else:
        original = original.replace("func.return %sum, %b", "func.return {private_control = true} %sum, %b")
    with pytest.raises(ValueError):
        parse_primitive(original)


@pytest.mark.parametrize("add", [False, True])
def test_real_stock_translation_link_and_execution(tmp_path, add):
    selected = os.environ.get("MERLIN_TEST_MLIR_TRANSLATE")
    if not selected:
        pytest.skip("requires explicitly selected stock MLIR translation tool")
    translator = Path(selected).resolve(strict=True)
    compiler = Path(os.environ.get("MERLIN_TEST_HOST_CLANG") or shutil.which("clang")).resolve(strict=True)
    original = source(add=add)
    lowered = emit_primitive_llvm(parse_primitive(original), "control_entry")
    verify_primitive_llvm(original, lowered, entry_symbol="control_entry")
    mlir, ir, harness, obj, elf = (
        tmp_path / name for name in ("control.mlir", "control.ll", "main.c", "control.o", "control.elf")
    )
    mlir.write_text(lowered)
    harness.write_text("""#include <stdio.h>
extern void control_entry(void *,void *,void *,void *);
int main(void) {
  float a[3]={1.25f,-3.5f,0.125f},b[3]={2.0f,0.5f,-0.25f},s[3]={0},c[3]={0};
  control_entry(a,b,s,c);
  for(int i=0;i<3;i++) printf("%.9g %.9g\\n",s[i],c[i]);
  return 0;
}
""")
    invocation_record.run(
        [str(translator), "--mlir-to-llvmir", str(mlir), "-o", str(ir)],
        directory=tmp_path,
        stage="llvm_translation",
        inputs=(mlir,),
        outputs=(ir,),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    invocation_record.run(
        [str(compiler), "-c", str(ir), "-o", str(obj)],
        directory=tmp_path,
        stage="kernel_object",
        inputs=(ir,),
        outputs=(obj,),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    invocation_record.run(
        [str(compiler), str(obj), str(harness), "-o", str(elf)],
        directory=tmp_path,
        stage="elf_link",
        inputs=(obj, harness),
        outputs=(elf,),
        capture_output=True,
        text=True,
        timeout=30,
        check=True,
    )
    result = invocation_record.run(
        [str(elf)],
        directory=tmp_path,
        stage="execution",
        inputs=(elf,),
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    observed = [[float(word) for word in line.split()] for line in result.stdout.splitlines()]
    assert observed == (
        [[3.25, 2.0], [-3.0, 0.5], [-0.125, -0.25]] if add else [[1.25, 2.0], [-3.5, 0.5], [0.125, -0.25]]
    )
    records = [invocation_record.verify(path) for path in tmp_path.rglob("invocation.json")]
    assert {row["stage"] for row in records} == {"llvm_translation", "kernel_object", "elf_link", "execution"}
    assert all(row["kind"] == "subprocess" for row in records)
