"""Executed LLVM outlining tests, including ordered sums and visible stores."""

import ctypes
import shutil
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.impr_features import apply_pipeline, normalize
from merlin.llvmlower.llvm_loop_outline import FEATURE, MERGE_FEATURE, _defined_functions, outline_loops
from merlin.llvmlower.pipeline import PipelineError, lower_to_llvm_ir
from merlin.llvmlower.toolchain import clang, llvm_opt

SOURCE = """
define float @sum(ptr %values, ptr %visits, i64 %n) {
entry:
  %empty = icmp eq i64 %n, 0
  br i1 %empty, label %exit, label %first.pre
first.pre:
  br label %loop
loop:
  %i = phi i64 [0, %first.pre], [%next, %loop]
  %acc = phi float [0.0, %first.pre], [%added, %loop]
  %p = getelementptr float, ptr %values, i64 %i
  %value = load volatile float, ptr %p
  %added = fadd float %acc, %value
  %next = add i64 %i, 1
  store volatile i64 %next, ptr %visits
  %done = icmp eq i64 %next, %n
  br i1 %done, label %second.pre, label %loop
second.pre:
  br label %second
second:
  %j = phi i64 [0, %second.pre], [%jnext, %second]
  %again = phi float [%added, %second.pre], [%again_added, %second]
  %q = getelementptr float, ptr %values, i64 %j
  %v = load volatile float, ptr %q
  %again_added = fadd float %again, %v
  %jnext = add i64 %j, 1
  %jdone = icmp eq i64 %jnext, %n
  br i1 %jdone, label %exit, label %second
exit:
  %result = phi float [0.0, %entry], [%again_added, %second]
  ret float %result
}
"""


@pytest.fixture
def optimizer():
    if not shutil.which(str(llvm_opt())):
        pytest.skip("selected LLVM opt unavailable")
    return llvm_opt()


def test_feature_is_explicit_and_keeps_mlir_pipeline():
    assert normalize({FEATURE}) == frozenset({FEATURE})
    baseline = ["canonicalize", "convert-scf-to-cf"]
    assert apply_pipeline(baseline, {FEATURE}) == baseline
    assert apply_pipeline(baseline, set()) == baseline
    assert normalize({MERGE_FEATURE}) == frozenset({MERGE_FEATURE})
    assert apply_pipeline(baseline, {MERGE_FEATURE}) == baseline


@pytest.mark.parametrize("level", ["-O0", "-O2"])
@pytest.mark.parametrize("merge_identical", [False, True])
def test_executed_order_and_side_effects(tmp_path, optimizer, level, merge_identical):
    if not shutil.which(str(clang())):
        pytest.skip("native clang unavailable")
    transformed = outline_loops(SOURCE, tmp_path / "transform", merge_identical=merge_identical)
    assert "define internal" in transformed
    assert "noinline" in transformed and "alwaysinline" not in transformed
    assert "fadd fast" not in transformed
    functions = []
    for label, source in [("original", SOURCE), ("outlined", transformed)]:
        ll, so = tmp_path / f"{label}.ll", tmp_path / f"{label}.so"
        ll.write_text(source)
        subprocess.run(
            [str(clang()), level, "-shared", "-fPIC", str(ll), "-o", str(so)],
            check=True,
            capture_output=True,
        )
        fn = ctypes.CDLL(str(so)).sum
        fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
        fn.restype = ctypes.c_float
        functions.append(fn)
    cases = [
        [],
        [2**24, 1, -(2**24), 1],  # Reassociation changes the answer.
        [-0.0],
        [np.inf, -np.inf],
        [np.nan, 1],
        [np.finfo(np.float32).tiny, -np.finfo(np.float32).tiny],
        np.random.default_rng(32).standard_normal(257),
    ]
    for values in cases:
        array = np.asarray(values, dtype=np.float32)
        results = []
        for fn in functions:
            visits = ctypes.c_int64(-1)
            value = np.float32(fn(array.ctypes.data, ctypes.byref(visits), array.size))
            assert visits.value == (array.size if array.size else -1)
            results.append(value.view(np.uint32))
        assert results[0] == results[1]


def test_missing_explicit_optimizer_fails_closed(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_LLVM_OPT", str(tmp_path / "absent-opt"))
    with pytest.raises(FileNotFoundError):
        outline_loops(SOURCE, tmp_path / "transform")


def test_invalid_llvm_fails_closed(tmp_path, optimizer):
    with pytest.raises(RuntimeError, match="command failed"):
        outline_loops("invalid LLVM", tmp_path)


def test_lowering_feature_and_audit_stage(tmp_path, optimizer):
    source = """module {
      func.func @ordered(%a: memref<?xf32>, %n: index) -> f32 {
        %zero = arith.constant 0.0 : f32
        %lo = arith.constant 0 : index
        %step = arith.constant 1 : index
        %result = scf.for %i = %lo to %n step %step iter_args(%sum = %zero) -> f32 {
          %v = memref.load %a[%i] : memref<?xf32>
          %next = arith.addf %sum, %v : f32
          scf.yield %next : f32
        }
        %again = scf.for %i = %lo to %n step %step iter_args(%sum = %result) -> f32 {
          %v = memref.load %a[%i] : memref<?xf32>
          %next = arith.addf %sum, %v : f32
          scf.yield %next : f32
        }
        return %again : f32
      }
    }"""
    baseline = lower_to_llvm_ir(source, workdir=tmp_path / "baseline")
    selected = lower_to_llvm_ir(source, workdir=tmp_path / "selected", features={FEATURE})
    assert "define internal" not in baseline
    assert "define internal" in selected and "noinline" in selected
    assert (tmp_path / "selected/llvm-loop-outline/loops-outlined.ll").is_file()


def test_audit_records_actual_optimizer_command(tmp_path, optimizer):
    class Audit:
        def command(self, command, **kwargs):
            self.argv = command

        def stage(self, name, text, **kwargs):
            self.last_stage = name, text, kwargs

    audit = Audit()
    output = outline_loops(SOURCE, tmp_path, audit=audit)
    assert audit.argv[0] == str(optimizer)
    assert "-passes=forceattrs,verify" in audit.argv
    assert audit.last_stage == ("llvm-loops-outlined", output, {"format": "llvm-ir"})


def test_preserves_original_alwaysinline(tmp_path, optimizer):
    original = SOURCE.replace("i64 %n) {", "i64 %n) alwaysinline {")
    original += "\ndefine float @original_helper(float %x) alwaysinline { ret float %x }\n"
    result = outline_loops(original, tmp_path)
    assert "alwaysinline" in result and "noinline" in result
    attributes = {
        line.split(" = ", 1)[0].split()[1]: line.split(" = ", 1)[1]
        for line in result.splitlines()
        if line.startswith("attributes #")
    }
    helpers = [line for line in result.splitlines() if line.startswith("define internal")]
    assert helpers
    for helper in helpers:
        group = helper.rsplit(" ", 2)[1]
        assert "noinline" in attributes[group] and "alwaysinline" not in attributes[group]
    selectors = (tmp_path / "helper-attributes.rsp").read_text()
    assert "=sum:noinline" not in selectors
    assert "=sum:alwaysinline" not in selectors
    assert "=original_helper:" not in selectors


def test_exact_merging_retains_public_function_addresses_and_effects(tmp_path, optimizer):
    original = SOURCE + SOURCE.replace("@sum(", "@alternate(")
    plain = outline_loops(original, tmp_path / "plain")
    selected = outline_loops(original, tmp_path / "merged", merge_identical=True)
    assert (tmp_path / "merged/loops-merged.bc").is_file()
    assert selected.count("define internal") < plain.count("define internal")
    for label, text in [("original", original), ("merged", selected)]:
        ll, so = tmp_path / (label + ".ll"), tmp_path / (label + ".so")
        ll.write_text(text)
        subprocess.run(
            [str(clang()), "-O2", "-shared", "-fPIC", str(ll), "-o", str(so)], check=True, capture_output=True
        )
        library = ctypes.CDLL(str(so))
        first, second = library.sum, library.alternate
        assert ctypes.cast(first, ctypes.c_void_p).value != ctypes.cast(second, ctypes.c_void_p).value
        for values in [
            [],
            [2**24, 1, -(2**24), 1],
            [-0.0],
            [np.inf, -np.inf],
            np.random.default_rng(791).standard_normal(73),
        ]:
            array = np.asarray(values, dtype=np.float32)
            results = []
            for fn in [first, second]:
                fn.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_int64), ctypes.c_int64]
                fn.restype = ctypes.c_float
                visits = ctypes.c_int64(-1)
                result = np.float32(fn(array.ctypes.data, ctypes.byref(visits), array.size))
                assert visits.value == (array.size if array.size else -1)
                results.append(result.view(np.uint32))
            assert results[0] == results[1]


def test_merge_requires_boolean_before_any_output_mutation(tmp_path):
    directory = tmp_path / "untouched"
    with pytest.raises(ValueError, match="explicit boolean"):
        outline_loops(SOURCE, directory, merge_identical=1)
    assert not directory.exists()


def test_normal_merge_policy_and_existing_default(tmp_path, optimizer):
    source = """module {
      func.func @sum(%a: memref<?xf32>, %n: index) -> f32 {
        %zero = arith.constant 0.0 : f32
        %lo = arith.constant 0 : index
        %step = arith.constant 1 : index
        %result = scf.for %i = %lo to %n step %step iter_args(%v = %zero) -> f32 {
          %x = memref.load %a[%i] : memref<?xf32>
          %next = arith.addf %v, %x : f32
          scf.yield %next : f32
        }
        %again = scf.for %i = %lo to %n step %step iter_args(%v = %result) -> f32 {
          %x = memref.load %a[%i] : memref<?xf32>
          %next = arith.addf %v, %x : f32
          scf.yield %next : f32
        }
        return %again : f32
      }
    }"""
    baseline = lower_to_llvm_ir(source, workdir=tmp_path / "default")
    selected = lower_to_llvm_ir(source, workdir=tmp_path / "merged", features={MERGE_FEATURE})
    assert "define internal" not in baseline
    assert "define internal" in selected and "noinline" in selected
    assert (tmp_path / "merged/llvm-loop-outline/loops-merged.bc").is_file()
    assert lower_to_llvm_ir(source, workdir=tmp_path / "default_again") == baseline
    with pytest.raises(PipelineError, match="Select one explicit"):
        lower_to_llvm_ir(source, workdir=tmp_path / "conflict", features={FEATURE, MERGE_FEATURE})


@pytest.mark.parametrize("record", ["quoted name T -------- 0", "bad:name T -------- 0", "broken"])
def test_ambiguous_symbol_names_refused(record):
    with pytest.raises(ValueError):
        _defined_functions(record)
