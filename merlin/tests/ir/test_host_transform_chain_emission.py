"""Actual helper linkage plus late legalization: omission must fail on emitted IR."""

from __future__ import annotations

import ctypes
import inspect
import json
import subprocess
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from merlin.common.digest import sha256_file
from merlin.llvmlower import host_llvm_helpers, late_quant_rne, toolchain
from merlin.llvmlower.host_transform_chain import (
    HostLLVMArtifact,
    HostLLVMEmission,
    HostLLVMStageContract,
    HostLLVMTransformChain,
    HostLLVMTransformStage,
    recheck_host_transform_chain,
)
from merlin.runtime.backends.spike_model import _transform_host_ir


def _modules(tmp_path, bits):
    source, helper = tmp_path / "source.ll", tmp_path / "helper.ll"
    source.write_text(f"""declare i{bits} @quantizer(float)
define i{bits} @evaluate(float %x, ptr %visits) {{
  %old = load volatile i32, ptr %visits
  %next = add i32 %old, 1
  store volatile i32 %next, ptr %visits
  %answer = call i{bits} @quantizer(float %x)
  ret i{bits} %answer
}}
""")
    helper.write_text(f"""define i{bits} @quantizer(float %x) alwaysinline {{
  %clo = call float @llvm.maximum.f32(float %x, float {float(-(1 << (bits - 1))):.6e})
  %chi = call float @llvm.minimum.f32(float %clo, float {float((1 << (bits - 1)) - 1):.6e})
  %integer = fptosi float %chi to i{bits}
  %back = sitofp i{bits} %integer to float
  %fraction = fsub float %chi, %back
  %negative = fneg float %fraction
  %magnitude = call float @llvm.maximum.f32(float %fraction, float %negative)
  %greater = fcmp ogt float %magnitude, 5.000000e-01
  %half = fcmp oeq float %magnitude, 5.000000e-01
  %low = and i{bits} %integer, 1
  %odd = icmp ne i{bits} %low, 0
  %tie = and i1 %half, %odd
  %increment = or i1 %greater, %tie
  %sign = fcmp olt float %chi, 0.000000e+00
  %direction = select i1 %sign, i{bits} -1, i{bits} 1
  %delta = select i1 %increment, i{bits} %direction, i{bits} 0
  %answer = add i{bits} %integer, %delta
  ret i{bits} %answer
}}
declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
""")
    return source, helper


@pytest.fixture
def tools():
    for tool in (toolchain.clang(), toolchain.llvm_opt(), toolchain.llvm_link()):
        if not tool.is_file():
            pytest.skip("selected upstream LLVM tools unavailable")


def _stages(helper):
    link = host_llvm_helpers.merlin_host_llvm_transform(
        helpers=(host_llvm_helpers.HelperIR(helper, sha256_file(helper)),), required_inlined_symbols=("quantizer",)
    )
    legalize = late_quant_rne.merlin_host_llvm_transform(toolchain.clang().resolve().parent, host_isa="portable")

    def verify_link(original, selected, work):
        # Inspect actual typed function-body references, not stage names/annotations.
        if host_llvm_helpers._remaining_references(selected.read_text(), ("quantizer",)):
            raise ValueError("required actual helper call/address remains in emission")
        record = json.loads((work / "compilation_recipe.json").read_text())
        if record["status"] != "completed" or record["llvm_ir"]["sha256"] != sha256_file(selected):
            raise ValueError("actual helper compilation is not complete for this emission")
        return HostLLVMEmission(
            HostLLVMArtifact.capture(original),
            HostLLVMArtifact.capture(selected),
            (HostLLVMArtifact.capture(work / "compilation_recipe.json"),),
        )

    def verify_legalization(original, selected, work):
        # Rederive the original bounded source proof and the exact selected rewrite.
        expected, proof = late_quant_rne.rewrite(original.read_text(), host_isa="portable")
        if not proof["routes"] or selected.read_text() != expected:
            raise ValueError("required actual late RNE legalization is missing")
        path = work / "receipt.json"
        record = json.loads(path.read_text())
        if record["source_sha256"] != sha256_file(original) or record["rewritten_sha256"] != sha256_file(selected):
            raise ValueError("late RNE proof is not bound to actual emission")
        return HostLLVMEmission(
            HostLLVMArtifact.capture(original), HostLLVMArtifact.capture(selected), (HostLLVMArtifact.capture(path),)
        )

    def stage(name, transform, verify):
        sources = tuple(
            HostLLVMArtifact.capture(path)
            for path in sorted(
                {
                    Path(inspect.getsourcefile(transform)).resolve(),
                    Path(inspect.getsourcefile(verify)).resolve(),
                    toolchain.llvm_opt().resolve(),
                    toolchain.llvm_link().resolve(),
                    toolchain.clang().resolve(),
                }
            )
        )
        contract = HostLLVMStageContract(name, sources, (HostLLVMArtifact.capture(helper),))
        return HostLLVMTransformStage(contract, transform, verify)

    return stage("inline_helpers", link, verify_link), stage("late_legalization", legalize, verify_legalization)


@pytest.mark.parametrize("bits", [8, 16])
def test_normal_chain_retains_actual_helpers_legalization_and_native_effects(tmp_path, tools, bits):
    source, helper = _modules(tmp_path, bits)
    stages = _stages(helper)
    chain = HostLLVMTransformChain(tuple(stage.contract for stage in stages), stages)
    selected, receipt = _transform_host_ir(source, tmp_path / "chain", None, chain=chain)
    record = recheck_host_transform_chain(Path(receipt["chain"]["path"]))
    assert all(row["status"] == "verified" for row in record["stages"])
    assert "@llvm.roundeven.f32" in selected.read_text()
    assert "call i" + str(bits) + " @quantizer" not in selected.read_text()
    functions = []
    for name, inputs in (("original", (source, helper)), ("composed", (selected,))):
        # Passed tmp_path directories can be reclaimed/reused between parameters;
        # distinct image names keep dlopen from returning the previous width.
        library = tmp_path / (name + "_" + str(bits) + ".so")
        subprocess.run(
            [str(toolchain.clang()), "-O2", "-fPIC", "-shared", *map(str, inputs), "-o", str(library)],
            check=True,
            capture_output=True,
        )
        function = ctypes.CDLL(str(library)).evaluate
        function.argtypes = [ctypes.c_float, ctypes.POINTER(ctypes.c_int32)]
        function.restype = ctypes.c_int8 if bits == 8 else ctypes.c_int16
        functions.append(function)
    lower, upper = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    inputs = np.array(
        [lower - 10, lower, -7.5, -6.5, -0.5, -0.0, 0.0, 0.5, 6.5, 7.5, upper, upper + 10], dtype=np.float32
    )
    inputs = np.concatenate(
        (inputs, np.nextafter(inputs, np.float32(-np.inf)), np.nextafter(inputs, np.float32(np.inf)))
    )
    for value in inputs:
        expected = int(np.rint(np.clip(value, lower, upper)))
        for function in functions:
            visits = ctypes.c_int32(41)
            assert function(float(value), ctypes.byref(visits)) == expected
            assert visits.value == 42


@pytest.mark.parametrize("omitted", [0, 1])
def test_noop_promised_stage_is_rejected_from_actual_emission_not_names(tmp_path, tools, omitted):
    source, helper = _modules(tmp_path, 8)
    stages = list(_stages(helper))

    def skipped(original, _work):
        return original

    # Bind the replacement implementation honestly; retaining the stage's name
    # and a valid source pin must still not discharge its emission obligation.
    extra_source = HostLLVMArtifact.capture(Path(inspect.getsourcefile(skipped)).resolve())
    changed = stages[omitted]
    sources = tuple({artifact.path: artifact for artifact in (*changed.contract.sources, extra_source)}.values())
    contract = replace(changed.contract, sources=sources)
    stages[omitted] = replace(changed, contract=contract, transform=skipped)
    promises = tuple(stage.contract for stage in stages)
    chain = HostLLVMTransformChain(promises, tuple(stages))
    with pytest.raises(ValueError, match="actual helper|late RNE legalization"):
        _transform_host_ir(source, tmp_path / "refused", None, chain=chain)
    assert json.loads((tmp_path / "refused/host_transform_chain.json").read_text())["status"] == "refused"


def test_adding_actual_helper_stage_preserves_previously_selected_late_legalizer(tmp_path, tools):
    source, helper = _modules(tmp_path, 8)
    inline, legalize = _stages(helper)
    chain = HostLLVMTransformChain((inline.contract,), (inline,))
    selected, receipt = _transform_host_ir(source, tmp_path / "composed", legalize.transform, chain=chain)
    record = recheck_host_transform_chain(Path(receipt["chain"]["path"]))
    assert record["legacy_terminal"]["input"] == record["stages"][0]["output"]
    assert b"@llvm.roundeven.f32" in selected.read_bytes()
    proof = json.loads((selected.parent / "receipt.json").read_text())
    assert proof["routes"] and proof["rewritten_sha256"] == receipt["selected_sha256"]
