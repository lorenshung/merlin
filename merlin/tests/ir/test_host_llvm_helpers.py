"""Real LLVM linkage, required call removal and preserved source side effects."""

import ctypes
import json
import subprocess

import pytest

from merlin.common.digest import sha256_file
from merlin.llvmlower import toolchain
from merlin.llvmlower.host_llvm_helpers import HelperIR, link_and_inline, merlin_host_llvm_transform
from merlin.runtime.backends.spike_model import _transform_host_ir

SOURCE = """
declare i32 @merlin_test_helper(i32, ptr)
define i32 @fallback(i32 %x) noinline {
  %answer = sub i32 %x, 7
  ret i32 %answer
}
define i32 @evaluate(i32 %x, ptr %visits) {
  %answer = call i32 @merlin_test_helper(i32 %x, ptr %visits)
  ret i32 %answer
}
"""
HELPER = """
declare i32 @fallback(i32)
define i32 @merlin_test_helper(i32 %x, ptr %visits) alwaysinline {
  %old = load volatile i32, ptr %visits
  %next = add i32 %old, 1
  store volatile i32 %next, ptr %visits
  %negative = icmp slt i32 %x, 0
  br i1 %negative, label %source, label %direct
source:
  %answer = call i32 @fallback(i32 %x)
  ret i32 %answer
direct:
  %square = mul i32 %x, %x
  ret i32 %square
}
"""


@pytest.fixture
def modules(tmp_path):
    main, helper = tmp_path / "source.ll", tmp_path / "helper.ll"
    main.write_text(SOURCE)
    helper.write_text(HELPER)
    return main, helper


@pytest.fixture
def tools():
    for tool in (toolchain.clang(), toolchain.llvm_opt(), toolchain.llvm_link()):
        if not tool.is_file():
            pytest.skip("selected upstream LLVM unavailable")


def test_empty_selection_preserves_default_bytes_and_needs_no_tools(modules, tmp_path, monkeypatch):
    source, _helper = modules
    before = source.read_bytes()
    monkeypatch.setenv("MERLIN_LLVM_OPT", str(tmp_path / "missing"))
    monkeypatch.setenv("MERLIN_LLVM_LINK", str(tmp_path / "missing-link"))
    work = tmp_path / "empty"
    selected, receipt = _transform_host_ir(source, work, merlin_host_llvm_transform())
    assert selected == source.resolve() and source.read_bytes() == before
    assert not list(work.iterdir())
    assert receipt["source_sha256"] == receipt["selected_sha256"]


def test_normal_callback_links_and_inlines_actual_helper_preserving_fallback_and_stores(modules, tmp_path, tools):
    source, helper = modules
    before = source.read_bytes()
    policy = merlin_host_llvm_transform(
        helpers=(HelperIR(helper, sha256_file(helper)),), required_inlined_symbols=("merlin_test_helper",)
    )
    selected, receipt = _transform_host_ir(source, tmp_path / "normal", policy)
    assert source.read_bytes() == before
    text = selected.read_text()
    assert "call i32 @merlin_test_helper" not in text and "call i32 @fallback" in text
    assert "load volatile i32" in text and "store volatile i32" in text
    assert receipt["selected_sha256"] == sha256_file(selected)
    recipe = json.loads((selected.parent / "compilation_recipe.json").read_text())
    assert recipe["status"] == "completed" and "executable" not in recipe
    assert recipe["llvm_ir"]["sha256"] == receipt["selected_sha256"]
    assert recipe["remaining_helper_references"] == []
    assert recipe["original_public_functions"] == ["evaluate", "fallback"]
    assert len(recipe["commands"]) == 4
    functions = []
    for label, sources in (("original", [source, helper]), ("selected", [selected])):
        library = tmp_path / (label + ".so")
        subprocess.run(
            [str(toolchain.clang()), "-O2", "-fPIC", "-shared", *map(str, sources), "-o", str(library)],
            check=True,
            capture_output=True,
        )
        function = ctypes.CDLL(str(library)).evaluate
        function.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
        function.restype = ctypes.c_int32
        functions.append(function)
    for value in (-12345, -1, 0, 1, 65536, 2**31 - 1):
        results = []
        for function in functions:
            visits = ctypes.c_int32(23)
            results.append(function(value, ctypes.byref(visits)))
            assert visits.value == 24
        assert results[0] == results[1]


@pytest.mark.parametrize(
    "mutation",
    [
        "stale",
        "missing",
        "digest",
        "types",
        "aggregate",
        "varargs",
        "cc",
        "unused",
        "duplicate",
        "context",
        "override",
        "fallback_type",
        "extension",
        "pointer_abi",
        "target_triple",
    ],
)
def test_refusal_before_compiler_outputs(modules, tmp_path, mutation):
    source, helper = modules
    digest = sha256_file(helper)
    if mutation == "stale":
        helper.write_text(HELPER + "; change\n")
    elif mutation == "missing":
        helper = helper.with_name("absent.ll")
    elif mutation == "digest":
        digest = "bad"
    elif mutation == "types":
        helper.write_text(HELPER.replace("ptr %visits", "i64 %visits"))
    elif mutation == "aggregate":
        source.write_text(
            SOURCE.replace(
                "declare i32 @merlin_test_helper(i32, ptr)", "declare { i32, i32 } @merlin_test_helper(i32, ptr)"
            )
        )
    elif mutation == "varargs":
        source.write_text(
            SOURCE.replace(
                "declare i32 @merlin_test_helper(i32, ptr)", "declare i32 @merlin_test_helper(i32, ptr, ...)"
            )
        )
    elif mutation == "cc":
        source.write_text(SOURCE.replace("declare i32 @merlin_test_helper", "declare fastcc i32 @merlin_test_helper"))
    elif mutation == "unused":
        source.write_text("declare i32 @merlin_test_helper(i32, ptr)\ndefine i32 @empty(){ret i32 0}\n")
    elif mutation == "duplicate":
        pass
    elif mutation == "override":
        helper.write_text(
            HELPER.replace("declare i32 @fallback(i32)", "define weak i32 @fallback(i32 %x) { ret i32 %x }")
        )
    elif mutation == "fallback_type":
        helper.write_text(HELPER.replace("declare i32 @fallback(i32)", "declare i64 @fallback(i32)"))
    elif mutation == "extension":
        helper.write_text(HELPER.replace("declare i32 @fallback(i32)", "declare signext i32 @fallback(i32)"))
    elif mutation == "pointer_abi":
        source.write_text(
            SOURCE.replace(
                "declare i32 @merlin_test_helper(i32, ptr)", "declare i32 @merlin_test_helper(i32, ptr byval(i32))"
            )
        )
    elif mutation == "target_triple":
        source.write_text('target triple = "x86_64-unknown-linux-gnu"\n' + SOURCE)
        helper.write_text('target triple = "riscv64-unknown-elf"\n' + HELPER)
    else:
        source.write_text('target datalayout = "e-p:64:64"\n' + SOURCE)
        helper.write_text('target datalayout = "e-p:32:32"\n' + HELPER)
    if mutation not in {"stale", "missing", "digest"}:
        digest = sha256_file(helper)
    bound = HelperIR(helper, digest)
    work = tmp_path / "refused"
    with pytest.raises(
        (ValueError, FileNotFoundError), match="replace an original" if mutation == "override" else None
    ):
        link_and_inline(
            source,
            work,
            helpers=(bound, bound) if mutation == "duplicate" else (bound,),
            required_inlined_symbols=("merlin_test_helper",),
        )
    assert not work.exists()


def test_always_inline_annotation_alone_does_not_grant_call_removal(modules, tmp_path, tools):
    source, helper = modules
    helper.write_text(HELPER.replace("alwaysinline", "noinline"))
    with pytest.raises(ValueError, match="still has a call/address"):
        link_and_inline(
            source,
            tmp_path / "retained",
            helpers=(HelperIR(helper, sha256_file(helper)),),
            required_inlined_symbols=("merlin_test_helper",),
        )
    recipe = json.loads((tmp_path / "retained/compilation_recipe.json").read_text())
    assert recipe["status"] != "completed"


def test_address_escape_refuses_even_after_direct_call_inlining(modules, tmp_path, tools):
    source, helper = modules
    source.write_text("@escaped = global ptr @merlin_test_helper\n" + SOURCE)
    with pytest.raises(ValueError, match="still has a call/address"):
        link_and_inline(
            source,
            tmp_path / "escape",
            helpers=(HelperIR(helper, sha256_file(helper)),),
            required_inlined_symbols=("merlin_test_helper",),
        )


def test_missing_selected_linker_fails_closed(modules, tmp_path, tools, monkeypatch):
    source, helper = modules
    monkeypatch.setenv("MERLIN_LLVM_LINK", str(tmp_path / "no-linker"))
    with pytest.raises(FileNotFoundError):
        link_and_inline(
            source,
            tmp_path / "missing-tool",
            helpers=(HelperIR(helper, sha256_file(helper)),),
            required_inlined_symbols=("merlin_test_helper",),
        )


def test_linker_locator_follows_selected_optimizer_and_explicit_override(tmp_path, monkeypatch):
    monkeypatch.setenv("MERLIN_LLVM_OPT", str(tmp_path / "selected/opt"))
    monkeypatch.delenv("MERLIN_LLVM_LINK", raising=False)
    assert toolchain.llvm_link() == tmp_path / "selected/llvm-link"
    monkeypatch.setenv("MERLIN_LLVM_LINK", str(tmp_path / "explicit/missing-link"))
    assert toolchain.llvm_link() == tmp_path / "explicit/missing-link"


def test_matching_scalar_extension_attributes_link_and_inline(modules, tmp_path, tools):
    source, helper = modules
    source.write_text(
        SOURCE.replace(
            "declare i32 @merlin_test_helper(i32, ptr)", "declare signext i32 @merlin_test_helper(i32 signext, ptr)"
        )
    )
    helper.write_text(
        HELPER.replace("define i32 @merlin_test_helper(i32 %x", "define signext i32 @merlin_test_helper(i32 signext %x")
    )
    selected = link_and_inline(
        source,
        tmp_path / "extensions",
        helpers=(HelperIR(helper, sha256_file(helper)),),
        required_inlined_symbols=("merlin_test_helper",),
    )
    assert "call i32 @merlin_test_helper" not in selected.read_text()
    assert "call i32 @fallback" in selected.read_text()
