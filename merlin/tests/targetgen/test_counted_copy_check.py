"""Actual public LLVM query and independent conditional copy proof controls.

These evaluator-private IRs exercise a static checker, never seed a compiler,
issue experimental origin or grant physical/numerical/runtime authority.
"""

import os
import shutil
import subprocess
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common import invocation_record
from merlin.common.digest import sha256_file
from merlin.llvmlower.counted_copy_check import check_counted_copy
from merlin.llvmlower.layout_observation import observe_compiled_layout, observe_llvm_layout
from merlin.targetgen.contract.compile_only import CompileOnlySourceAbi, CompileOnlyTensor
from merlin.targetgen.contract.pointer_storage import OriginalPointerStorageContract, PointerStoragePolicy


def storage(shape=(2, 3), dtype="i32", *, byte_order="little", alignment=8):
    return OriginalPointerStorageContract(
        CompileOnlySourceAbi((CompileOnlyTensor("input", shape, dtype),), (CompileOnlyTensor("output", shape, dtype),)),
        PointerStoragePolicy(
            "row_major_contiguous",
            "inputs_then_outputs",
            "disjoint",
            "static_original_tensor_type",
            byte_order,
            alignment,
        ),
    )


def source(shape=(2, 3), dtype="i32"):
    typ = "tensor<" + "x".join(map(str, shape)) + "x" + dtype + ">"
    return f"module {{ func.func @original(%a: {typ}) -> {typ} {{ func.return %a : {typ} }} }}"


def emitted(count=6, *, index_bits=64, element_bits=32):
    return f"""module {{
 llvm.func @copy_entry(%a: !llvm.ptr, %b: !llvm.ptr) {{
  %zero = llvm.mlir.constant(0 : i{index_bits}) : i{index_bits}
  %one = llvm.mlir.constant(1 : i{index_bits}) : i{index_bits}
  %count = llvm.mlir.constant({count} : i{index_bits}) : i{index_bits}
  llvm.br ^loop(%zero : i{index_bits})
 ^loop(%i: i{index_bits}):
  %src = llvm.getelementptr %a[%i] : (!llvm.ptr, i{index_bits}) -> !llvm.ptr, i{element_bits}
  %dst = llvm.getelementptr %b[%i] : (!llvm.ptr, i{index_bits}) -> !llvm.ptr, i{element_bits}
  %v = llvm.load %src {{alignment = 4 : i64}} : !llvm.ptr -> i{element_bits}
  llvm.store %v, %dst {{alignment = 4 : i64}} : i{element_bits}, !llvm.ptr
  %next = llvm.add %i, %one : i{index_bits}
  %done = llvm.icmp "eq" %next, %count : i{index_bits}
  llvm.cond_br %done, ^exit, ^loop(%next : i{index_bits})
 ^exit:
  llvm.return
 }}
}}"""


@pytest.fixture(scope="module")
def tools():
    cc = shutil.which("g++")
    config = os.environ.get("MERLIN_TEST_LLVM_CONFIG") or shutil.which("llvm-config")
    if not cc or not config:
        pytest.skip("explicit public native LLVM C++ tools unavailable")
    return dict(native_compiler=Path(cc).resolve(strict=True), llvm_config=Path(config).resolve(strict=True))


@pytest.fixture(scope="module")
def layout(tools, tmp_path_factory):
    # Deliberately omit a p0 declaration. Its default is observed by LLVM's
    # native API, never assumed by the Merlin text parser or this checker.
    return observe_llvm_layout(
        layout="e-i64:64", integer_bits=32, output_root=tmp_path_factory.mktemp("native") / "query", **tools
    )


def check(layout, code=None, original=None, selected=None):
    return check_counted_copy(
        original_source=source() if original is None else original,
        emitted_llvm=emitted() if code is None else code,
        storage=storage() if selected is None else selected,
        layout_observation=layout,
        entry_symbol="copy_entry",
    )


def test_actual_native_accessor_and_complete_large_copy_prove_without_unrolling(layout):
    assert len(layout.records) == 4 and layout.verify()["allocation_stride"] == 4
    result = check(layout)
    assert result["status"] == "PROVED" and set(result["facets"].values()) == {"PROVED"}
    assert result["facts"]["byte_interval"] == [0, 23]
    shape = (1 << 35, 7)
    count = shape[0] * shape[1]
    result = check(layout, emitted(count), source(shape), storage(shape))
    assert result["status"] == "PROVED" and result["facts"]["iterations"] == count
    assert "resource_legality" not in result["facets"] and "machine-code" in result["scope"]


@pytest.mark.parametrize("defect", ["bound", "initial", "step", "pointer", "element", "alignment", "store_value"])
def test_actual_counter_address_output_and_alignment_defects_are_refuted(layout, defect):
    code = emitted()
    mutations = {
        "bound": ("constant(6 : i64)", "constant(5 : i64)"),
        "initial": ("constant(0 : i64)", "constant(1 : i64)"),
        "step": ("constant(1 : i64)", "constant(2 : i64)"),
        "pointer": ("getelementptr %b[%i]", "getelementptr %a[%i]"),
        "element": ("!llvm.ptr, i32", "!llvm.ptr, i16"),
        "alignment": ("alignment = 4", "alignment = 8"),
        "store_value": ("llvm.store %v, %dst", "llvm.store %one, %dst"),
    }
    old, new = mutations[defect]
    result = check(layout, code.replace(old, new))
    # A malformed LLVM store width is stock verifier UNKNOWN, not a theorem.
    assert result["status"] == ("UNKNOWN" if defect == "store_value" else "REFUTED")


@pytest.mark.parametrize(
    "defect", ["opaque_call", "missing_store", "inbounds", "extra_definition", "wrong_index", "metadata"]
)
def test_unsupported_operations_flags_control_and_layout_remain_unknown(layout, defect):
    code = emitted()
    mutations = {
        "opaque_call": ("  llvm.return", "  llvm.call @opaque() : () -> ()\n  llvm.return"),
        "missing_store": ("  llvm.store %v, %dst {alignment = 4 : i64} : i32, !llvm.ptr\n", ""),
        "inbounds": ("llvm.getelementptr %a", "llvm.getelementptr inbounds %a"),
        "extra_definition": ("module {", "module { llvm.func @extra() { llvm.return }"),
        "wrong_index": ("i64", "i32"),
        "metadata": ("llvm.return", "llvm.return {unexpected = 1 : i32}"),
    }
    old, new = mutations[defect]
    result = check(layout, code.replace(old, new))
    assert result["status"] == "UNKNOWN" and set(result["facets"].values()) == {"UNKNOWN"}


def test_missing_or_forged_native_observation_cannot_issue_a_theorem(layout):
    for claim in (None, layout.record(), replace(layout)):
        assert check(claim)["status"] == "UNKNOWN"
    shape = (1 << 62, 7)
    assert check(layout, emitted(shape[0] * shape[1]), source(shape), storage(shape))["status"] != "PROVED"
    assert check(layout, selected=storage(byte_order="big"))["status"] == "REFUTED"
    assert check(layout, original="module {}")["status"] == "UNKNOWN"


def test_unsupported_dense_splat_is_refused_before_any_shaped_parser_allocation(layout, monkeypatch):
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr

    def refuse(*args, **kwargs):
        pytest.fail("closed scalar checker allocated an unsupported aggregate literal")

    monkeypatch.setattr(DenseIntOrFPElementsAttr, "__init__", refuse)
    aggregate = "module attributes {payload = dense<0> : tensor<1099511627776xi8>} {}"
    for arguments in ((None, aggregate), (aggregate, None)):
        result = check(layout, code=arguments[0], original=arguments[1])
        assert result["status"] == "UNKNOWN" and "before shaped parser allocation" in result["detail"]


def test_actual_llvm_allocation_padding_cannot_match_a_dense_original_byte_contract(tools, tmp_path):
    observed = observe_llvm_layout(layout="e-i24:32", integer_bits=24, output_root=tmp_path / "native", **tools)
    assert observed.allocation_stride == 4
    result = check(observed, emitted(element_bits=24), source(dtype="i24"), storage(dtype="i24"))
    assert result["status"] == "REFUTED" and "allocation stride" in result["detail"]


def test_native_records_and_external_header_bytes_are_reopened(tools, tmp_path):
    observed = observe_llvm_layout(layout="e-i32:32", integer_bits=32, output_root=tmp_path / "native", **tools)
    assert observed.verify()
    Path(observed.records[-1][0]).unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        observed.verify()


def test_selected_actual_object_compiler_layout_is_observed_from_same_ir_and_options(tools, tmp_path):
    from merlin.llvmlower import codegen, toolchain

    translator, clang = toolchain.mlir_translate(), Path(toolchain.clang())
    if not translator.is_file() or not clang.is_file():
        pytest.skip("requires selected stock translator and object compiler")
    mlir, ir, obj = (tmp_path / name for name in ("emitted.mlir", "kernel.ll", "kernel.o"))
    mlir.write_text(emitted())
    result = invocation_record.run(
        [str(translator), "--mlir-to-llvmir", str(mlir), "-o", str(ir)],
        directory=tmp_path,
        stage="llvm_translation",
        inputs=(mlir,),
        outputs=(ir,),
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stderr
    codegen.compile_ll(ir, obj, "riscv", timeout_s=30)
    records = [
        path for path in tmp_path.rglob("invocation.json") if invocation_record.verify(path)["stage"] == "object"
    ]
    observed = observe_compiled_layout(
        object_record=records[0], integer_bits=32, output_root=tmp_path / "layout", **tools
    )
    assert observed.object_record == (str(records[0]), sha256_file(records[0]))
    assert len(observed.records) == 5 and observed.verify()["allocation_stride"] == 4
    assert check(observed)["status"] == "PROVED"
    obj.write_bytes(obj.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="dependency or invocation changed"):
        observed.verify()


def test_actual_tool_timeout_retains_failed_invocation_and_issues_no_layout(tools, tmp_path):
    stalled = tmp_path / "stalled-config"
    stalled.write_text("#!/bin/sh\nsleep 5\n")
    stalled.chmod(0o700)
    root = tmp_path / "failed-layout"
    with pytest.raises(subprocess.TimeoutExpired):
        observe_llvm_layout(
            layout="e-i32:32",
            integer_bits=32,
            native_compiler=tools["native_compiler"],
            llvm_config=stalled,
            output_root=root,
            timeout_s=0.01,
        )
    paths = tuple(root.rglob("invocation.json"))
    assert len(paths) == 1 and not (root / "layout_observation.json").exists()
    with pytest.raises(ValueError, match="did not complete"):
        invocation_record.verify(paths[0])
