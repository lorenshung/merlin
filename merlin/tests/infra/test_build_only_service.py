"""The optional compile service has no runtime registry/cache dependency."""

import builtins
import hashlib
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe, KernelStackFramePolicy
from merlin.targetgen.contract.build_service import BuildOnlyService, load_build_package


def service(tmp_path):
    source = tmp_path / "renderer.py"
    source.write_text("trusted build fixture")
    recipe = HarnessBuildRecipe(
        Path("/usr/bin/cc"), (), (), Path("/fixture/link.ld"), 0, ("-march=rv64gc", "-mabi=lp64d")
    )
    return BuildOnlyService(
        "fixture",
        recipe,
        lambda cb, **kwargs: "C text",
        ((str(source), hashlib.sha256(source.read_bytes()).hexdigest()),),
    )


def test_pure_branch_never_imports_registry_or_cache(tmp_path, monkeypatch):
    cap = service(tmp_path)
    original = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.startswith("merlin.runtime.backends") or "build_cache" in name:
            raise AssertionError("pure build attempted runtime registry/cache import")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    calls = []

    def obj(text, work, **kwargs):
        calls.append(("object", kwargs))
        return work / "kernel.o"

    def link(cb, obj, work, **kwargs):
        calls.append(("link", kwargs))
        return work / "kernel.elf"

    monkeypatch.setattr(compiler, "llvm_mlir_to_object", obj)
    monkeypatch.setattr(compiler, "link_elf", link)
    assert (
        compiler.compile_lowered_to_elf({}, "llvm", tmp_path, target="fixture", inputs={"a": [1]}, _build_service=cap)
        == tmp_path / "kernel.elf"
    )
    assert [name for name, _ in calls] == ["object", "link"]
    assert all(kwargs["_build_service"] is cap for _, kwargs in calls)


@pytest.mark.parametrize("defect", ["stale", "target", "inputs", "authority"])
def test_invalid_capability_before_compile(tmp_path, monkeypatch, defect):
    cap = service(tmp_path)
    kwargs = {"target": "fixture", "inputs": {"a": [1]}, "_build_service": cap}
    if defect == "stale":
        Path(cap.source_pins[0][0]).write_text("changed")
    elif defect == "target":
        kwargs["target"] = "other"
    elif defect == "inputs":
        kwargs["inputs"] = None
    else:
        kwargs["prepack_authorizations"] = {}
    monkeypatch.setattr(compiler, "llvm_mlir_to_object", lambda *a, **k: pytest.fail("compiled"))
    with pytest.raises(ValueError):
        compiler.compile_lowered_to_elf({}, "llvm", tmp_path, **kwargs)


def test_package_reuse_refuses_changed_source(tmp_path):
    initializer = tmp_path / "__init__.py"
    initializer.write_text("VALUE = 1")
    original_modules = set(sys.modules)
    try:
        assert load_build_package(initializer).VALUE == 1
        initializer.write_text("VALUE = 2")
        with pytest.raises(ValueError, match="changed"):
            load_build_package(initializer)
    finally:
        # The production cache must survive the mutation check. Afterwards,
        # remove only new modules loaded from this disposable fixture; keeping
        # them would correctly fail the installed package origin guard.
        for name, module in tuple(sys.modules.items()):
            filename = getattr(module, "__file__", None)
            if (
                name not in original_modules
                and name.startswith("merlin._pure_build_packages.")
                and filename
                and Path(filename).resolve().is_relative_to(tmp_path.resolve())
            ):
                sys.modules.pop(name, None)


def test_build_translation_refuses_non_llvm_before_tool(tmp_path, monkeypatch):
    cap = service(tmp_path)
    monkeypatch.setattr(compiler.subprocess, "run", lambda *a, **k: pytest.fail("translation ran"))
    for text in (
        'builtin.module { %x = "builtin.unrealized_conversion_cast"() : () -> i32 }',
        'builtin.module { "unknown.target_operation"() : () -> () }',
    ):
        with pytest.raises(ValueError, match="LLVM/Builtin"):
            compiler.llvm_mlir_to_object(text, tmp_path, target="fixture", _build_service=cap)


def test_build_translation_preserves_upstream_llvm_metadata(tmp_path, monkeypatch):
    """Stock LLVM metadata is validated by the selected translator, not xDSL's older schema."""
    from types import SimpleNamespace

    from merlin.llvmlower import codegen

    cap = service(tmp_path)
    text = """#unroll = #llvm.loop_unroll<disable = true>
#annotation = #llvm.loop_annotation<unroll = #unroll>
"builtin.module"() ({
  "llvm.func"() <{function_type = !llvm.func<void ()>, sym_name = "fixture_entry"}> ({
    %c = "llvm.mlir.constant"() <{value = true}> : () -> i1
    "llvm.cond_br"(%c)[^bb1, ^bb1] <{loop_annotation = #annotation,
      operandSegmentSizes = array<i32: 1, 0, 0>}> : (i1) -> ()
  ^bb1:
    "llvm.return"() : () -> ()
  }) : () -> ()
}) : () -> ()"""
    calls = []

    def translate(command, **kwargs):
        calls.append(command)
        assert Path(command[2]).read_text() == text
        return SimpleNamespace(returncode=1, stderr="fixture translator rejected metadata")

    monkeypatch.setattr(compiler.subprocess, "run", translate)
    monkeypatch.setattr(codegen, "compile_ll", lambda *a, **k: pytest.fail("compiled rejected LLVM"))
    with pytest.raises(RuntimeError, match="translator rejected metadata"):
        compiler.llvm_mlir_to_object(text, tmp_path, target="fixture", _build_service=cap)
    assert len(calls) == 1
    assert calls[0][1] == "--mlir-to-llvmir"


def test_explicit_public_object_budget_bounds_translation_and_compile(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin.llvmlower import codegen, toolchain

    cap = service(tmp_path)
    cap = replace(cap, recipe=replace(cap.recipe, kernel_stack_frame=KernelStackFramePolicy("fixture_entry", 4096)))
    seen = []
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: Path("/fixture/mlir-translate"))

    def translate(command, **kwargs):
        seen.append(("translate", kwargs["timeout"]))
        Path(command[-1]).write_text("define void @fixture_entry() { ret void }\n")
        return SimpleNamespace(returncode=0, stderr="")

    def compile_ll(source, output, target, *, extra_flags, timeout_s):
        seen.append(("compile", timeout_s))
        Path(output).write_bytes(b"object")
        Path(output).with_suffix(".su").write_text(f"{source}:fixture_entry\t32\tstatic\n")
        return output

    monkeypatch.setattr(compiler.subprocess, "run", translate)
    monkeypatch.setattr(codegen, "compile_ll", compile_ll)
    result = compiler.llvm_mlir_to_object(
        "builtin.module { llvm.func @fixture_entry() { llvm.return } }",
        tmp_path / "build", target="fixture", _build_service=cap, build_timeout_s=2,
    )
    assert result.is_file()
    assert [part for part, _seconds in seen] == ["translate", "compile"]
    assert all(0 < seconds <= 2 for _part, seconds in seen)


def test_explicit_public_object_budget_refuses_translator_timeout(tmp_path, monkeypatch):
    import subprocess

    from merlin.llvmlower import codegen, toolchain

    cap = service(tmp_path)
    monkeypatch.setattr(toolchain, "mlir_translate", lambda: Path("/fixture/mlir-translate"))
    monkeypatch.setattr(
        compiler.subprocess, "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(subprocess.TimeoutExpired(args[0], kwargs["timeout"])),
    )
    monkeypatch.setattr(codegen, "compile_ll", lambda *args, **kwargs: pytest.fail("compiled after timeout"))
    with pytest.raises(TimeoutError, match="translation budget expired"):
        compiler.llvm_mlir_to_object(
            "builtin.module { llvm.func @fixture_entry() { llvm.return } }",
            tmp_path / "build", target="fixture", _build_service=cap, build_timeout_s=1,
        )


def test_explicit_object_compiler_limit_is_tighter_than_global_default(tmp_path, monkeypatch):
    import subprocess

    from merlin.llvmlower import codegen

    seen = []
    monkeypatch.setattr(codegen, "clang", lambda: Path("/fixture/clang"))
    def observed(command, **kwargs):
        seen.append(kwargs["timeout"])
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(codegen._proc, "run_checked", observed)
    codegen.compile_ll(tmp_path / "fixture.ll", tmp_path / "fixture.o", timeout_s=0.5)
    assert seen == [0.5]
    for invalid in (0, -1, True, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="explicit compile timeout"):
            codegen.compile_ll(tmp_path / "fixture.ll", tmp_path / "fixture.o", timeout_s=invalid)


def test_legacy_object_path_retains_original_lowering(tmp_path, monkeypatch):
    from merlin.llvmlower import codegen, pipeline

    calls = []

    def lower(text, *, workdir):
        calls.append((text, workdir))
        return "; existing lowering"

    def compile_ll(source, output, target, *, extra_flags):
        assert source.read_text() == "; existing lowering"
        assert target == "riscv" and extra_flags == ()
        return output

    monkeypatch.setattr(pipeline, "lower_to_llvm_ir", lower)
    monkeypatch.setattr(codegen, "compile_ll", compile_ll)
    assert compiler.llvm_mlir_to_object("legacy source", tmp_path) == tmp_path / "kernel.o"
    assert calls == [("legacy source", tmp_path)]
