"""Closed integer publication versus the actual compiled source observer."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest
from test_source_expression_interval_llvm import EFFECTS, compiled

from merlin.llvmlower.source_expression_interval import (
    emit_source_interval_i8_lookup,
    emit_source_interval_lookup,
)
from merlin.llvmlower.source_expression_interval_llvm import (
    rewrite_source_interval_i8_lookup,
    rewrite_source_interval_lookup,
)
from merlin.llvmlower.toolchain import clang


@pytest.fixture(scope="module")
def integer_compiled(compiled, tmp_path_factory):
    old_work, llvm, _, proof, table, _ = compiled
    work = tmp_path_factory.mktemp("closed_integer_lookup")
    selected, report = rewrite_source_interval_i8_lookup(
        llvm, proofs=(proof,), table=table, lookup_symbol="lookup", effects=EFFECTS
    )
    assert len(report["routes"]) == 1
    assert report["routes"][0]["complete_scalar_uses_closed"]
    assert report["routes"][0]["result_type"] == "i8"
    assert "call i8 @lookup" in selected and "declare i8 @lookup" in selected
    (work / "selected.ll").write_text(selected)
    (work / "original.ll").write_text(llvm.replace("@unrelated(", "@original("))
    old_helper = old_work.joinpath("helper.c").read_text()
    old_emitter = emit_source_interval_lookup(
        table_name="table", activation_name="activation", quantizer_name="quant", lookup_name="lookup", leading_bits=12
    )
    new_emitter = emit_source_interval_i8_lookup(
        table_name="table", activation_name="activation", quantizer_name="quant", lookup_name="lookup", leading_bits=12
    )
    assert old_helper.startswith(old_emitter)
    helper = new_emitter + old_helper[len(old_emitter) :]
    # The admission guard is supplied externally. Unsupported modes execute the
    # same original source, without evaluating interval or candidate arithmetic.
    helper += """\n#include <fenv.h>
extern signed char unrelated(float,float), original(float,float);
signed char guarded(float x,float up){if(fegetround()!=FE_TONEAREST)return original(x,up);return unrelated(x,up);}
"""
    (work / "helper.c").write_text(helper)
    subprocess.run(
        [
            str(clang()),
            "-O3",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            str(work / "selected.ll"),
            str(work / "original.ll"),
            str(work / "helper.c"),
            "-lm",
            "-o",
            str(work / "model.so"),
        ],
        check=True,
        capture_output=True,
    )
    library = ctypes.CDLL(str(work / "model.so"))
    functions = [library.guarded, library.original]
    for function in functions:
        function.argtypes = (ctypes.c_float, ctypes.c_float)
        function.restype = ctypes.c_int8
    return library, functions, report


def test_actual_source_integer_result_boundaries_subnormals_zero_tails(integer_compiled):
    _, (candidate, original), _ = integer_compiled
    rng = np.random.default_rng(891)
    raw = np.array([0, 0x80000000, 1, 0x80000001, 0x00800000, 0x80800000, 0x7F7FFFFF, 0xFF7FFFFF], np.uint32)
    x = np.concatenate(
        (
            raw.view(np.float32),
            rng.uniform(-49, 49, 1133).astype(np.float32),
            np.arange(-259, 260, dtype=np.float32) / np.float32(12),
        )
    )
    up = np.concatenate(
        (np.ones(len(raw), np.float32), rng.uniform(-4, 4, 1133).astype(np.float32), np.ones(519, np.float32))
    )
    actual = np.array([candidate(float(a), float(b)) for a, b in zip(x, up)], np.int8)
    expected = np.array([original(float(a), float(b)) for a, b in zip(x, up)], np.int8)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("mode", [0, 0x400, 0x800, 0xC00])
def test_unsupported_mode_source_continuation_and_sticky_flags(integer_compiled, mode):
    _, (candidate, original), _ = integer_compiled
    environment = ctypes.CDLL(None)
    saved = environment.fegetround()
    pairs = [
        (0.0, -1.0),
        (-0.0, 3.0),
        (float(np.float32(1.0 / 12)), 1.0),
        (float(np.nextafter(np.float32(1.0 / 12), np.float32(np.inf))), 1.0),
        (41.25, -3.0),
    ]
    try:
        assert environment.fesetround(mode) == 0
        for flags in (0, 1, 4, 8, 16, 32, 61):
            for x, up in pairs:
                environment.feclearexcept(61)
                environment.feraiseexcept(flags)
                a = original(x, up)
                original_flags = environment.fetestexcept(61)
                environment.feclearexcept(61)
                environment.feraiseexcept(flags)
                b = candidate(x, up)
                candidate_flags = environment.fetestexcept(61)
                assert a == b
                if mode:
                    assert original_flags == candidate_flags
                else:
                    assert candidate_flags & flags == flags
    finally:
        environment.fesetround(saved)


@pytest.mark.parametrize("mutation", ["floating_escape", "observer_escape", "unknown_call", "constant", "strict"])
def test_complete_source_and_compiled_use_refusal(compiled, mutation):
    _, llvm, _, proof, table, _ = compiled
    endpoint = next(line.split(" =")[0].strip() for line in llvm.splitlines() if "fmul float" in line)
    if mutation == "floating_escape":
        changed = llvm.replace("  ret i8", f"  store float {endpoint}, ptr null\n  ret i8")
    elif mutation == "observer_escape":
        scaled = [line.split(" =")[0].strip() for line in llvm.splitlines() if "fmul float" in line][-1]
        changed = llvm.replace("  ret i8", f"  store float {scaled}, ptr null\n  ret i8")
    elif mutation == "unknown_call":
        changed = llvm.replace("  ret i8", "  call void @unknown()\n  ret i8")
    elif mutation == "constant":
        changed = llvm.replace("2.000000e+00", "2.500000e+00")
    else:
        changed = llvm.replace("define i8", "define strictfp i8")
    if mutation == "strict":
        with pytest.raises(ValueError):
            rewrite_source_interval_i8_lookup(
                changed, proofs=(proof,), table=table, lookup_symbol="lookup", effects=EFFECTS
            )
    else:
        after, report = rewrite_source_interval_i8_lookup(
            changed, proofs=(proof,), table=table, lookup_symbol="lookup", effects=EFFECTS
        )
        assert after == changed and not report["routes"]


def test_integer_observation_keeps_additional_integer_uses(compiled):
    _, llvm, _, proof, table, _ = compiled
    result = next(line.strip().split("ret i8 ")[1] for line in llvm.splitlines() if "ret i8 " in line)
    changed = llvm.replace("  ret i8", f"  store i8 {result}, ptr null\n  ret i8")
    after, report = rewrite_source_interval_i8_lookup(
        changed, proofs=(proof,), table=table, lookup_symbol="lookup", effects=EFFECTS
    )
    assert len(report["routes"]) == 1 and f"store i8 {result}" in after


def test_default_and_carrier_route_keep_original_bytes(compiled):
    _, llvm, changed, proof, table, _ = compiled
    assert rewrite_source_interval_i8_lookup("not LLVM")[0] == "not LLVM"
    again, _ = rewrite_source_interval_lookup(
        llvm, proofs=(proof,), table=table, lookup_symbol="lookup", effects=EFFECTS
    )
    assert again == changed


def test_explicit_effect_and_identifier_admission(compiled):
    _, llvm, _, proof, table, _ = compiled
    with pytest.raises(ValueError):
        rewrite_source_interval_i8_lookup(llvm, proofs=(proof,), table=table, lookup_symbol="lookup")
    with pytest.raises(ValueError):
        emit_source_interval_i8_lookup(
            table_name="unproved-name",
            activation_name="activation",
            quantizer_name="quant",
            lookup_name="lookup",
            leading_bits=12,
        )
