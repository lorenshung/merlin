"""Actual typed source-to-LLVM binding and complete escape/dominance refusals."""

from __future__ import annotations

import ctypes
import subprocess

import numpy as np
import pytest
from test_source_expression_interval import source

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    SourceIntervalTable,
    build_source_interval_table,
    close_scalar_i8_observer,
    emit_source_interval_lookup,
)
from merlin.llvmlower.source_expression_interval_llvm import rewrite_source_interval_lookup
from merlin.llvmlower.toolchain import clang

EFFECTS = IntervalEffectContract(True, True, True, True, True)


@pytest.fixture(scope="module")
def compiled(tmp_path_factory):
    work = tmp_path_factory.mktemp("interval_source_binding")
    text = source()
    module = parse_mlir_text(text)
    function = module.body.block.first_op
    endpoint = next(op for op in function.body.block.ops if op.name == "arith.mulf").results[0]
    proof = close_scalar_i8_observer(function.body.block.args[0], endpoint, effects=EFFECTS)
    table = build_source_interval_table(proof.expression, effects=EFFECTS, leading_bits=12, max_table_bytes=32768)
    llvm = lower_to_llvm_ir(text, workdir=work / "lower")
    changed, report = rewrite_source_interval_lookup(
        llvm, proofs=(proof,), table=table, lookup_symbol="lookup", effects=EFFECTS
    )
    assert len(report["routes"]) == 1
    (work / "model.ll").write_text(changed)
    cells = np.frombuffer(table.data, "<f4").reshape(-1, 2)
    constants = ",\n".join(
        "{"
        + ",".join(
            float(x).hex() + "f" if np.isfinite(x) else "__builtin_inff()" if x > 0 else "-__builtin_inff()"
            for x in row
        )
        + "}"
        for row in cells
    )
    helper = emit_source_interval_lookup(
        table_name="table", activation_name="activation", quantizer_name="quant", lookup_name="lookup", leading_bits=12
    )
    c = (
        helper
        + """\nfloat activation(float x){return x*2.0f;}
signed char quant(float x){x=__builtin_fmaxf(x,-128.0f);x=__builtin_fminf(x,127.0f);
return (signed char)__builtin_roundevenf(x);}
"""
        + f"const float table[4096][2]={{{constants}}};\n"
    )
    (work / "helper.c").write_text(c)
    subprocess.run(
        [
            str(clang()),
            "-O3",
            "-ffp-contract=off",
            "-fPIC",
            "-shared",
            str(work / "model.ll"),
            str(work / "helper.c"),
            "-lm",
            "-o",
            str(work / "model.so"),
        ],
        check=True,
        capture_output=True,
    )
    lib = ctypes.CDLL(str(work / "model.so"))
    fn = lib.unrelated
    fn.argtypes = (ctypes.c_float, ctypes.c_float)
    fn.restype = ctypes.c_int8
    return work, llvm, changed, proof, table, fn


def test_actual_compiled_lookup_original_rounding_boundaries_and_tails(compiled):
    _, _, _, _, _, function = compiled
    raw = np.array([0, 0x80000000, 1, 0x80000001, 0x00800000, 0x7F7FFFFF, 0xFF7FFFFF], np.uint32)
    rng = np.random.default_rng(159)
    x = np.concatenate(
        [
            raw.view(np.float32),
            rng.uniform(-40, 40, 1103).astype(np.float32),
            np.arange(-129, 130, dtype=np.float32) / np.float32(6),
        ]
    )
    up = np.concatenate(
        [np.ones(raw.size, np.float32), rng.uniform(-3, 3, 1103).astype(np.float32), np.ones(259, np.float32)]
    )
    with np.errstate(all="ignore"):
        actual_source = np.float32(np.float32(np.float32(x * np.float32(2)) * up) * np.float32(3))
        expected = np.rint(np.clip(actual_source, -128, 127)).astype(np.int8)
    actual = np.array([function(float(a), float(b)) for a, b in zip(x, up)], np.int8)
    np.testing.assert_array_equal(actual, expected)


def test_default_bytes_unknown_module_unchanged():
    text = "deliberately not LLVM: no selected policy"
    assert rewrite_source_interval_lookup(text)[0] == text


@pytest.mark.parametrize("change", ["literal", "fast", "float_escape", "unknown_call", "strict", "dominance"])
def test_actual_compiled_binding_refuses_changed_semantics_or_observer(compiled, change):
    _, llvm, _, proof, table, _ = compiled
    if change == "literal":
        changed = llvm.replace("2.000000e+00", "2.500000e+00")
    elif change == "fast":
        changed = llvm.replace("fmul float", "fmul fast float", 1)
    elif change in ("float_escape", "unknown_call"):
        endpoint = next(line.split(" =")[0].strip() for line in llvm.splitlines() if "fmul float" in line)
        extra = (
            f"  call void @unproved(float {endpoint})"
            if change == "unknown_call"
            else f"  store float {endpoint}, ptr null"
        )
        changed = llvm.replace("  ret i8", extra + "\n  ret i8")
    elif change == "strict":
        changed = llvm.replace("define i8", "define strictfp i8")
    else:
        # A finishing operand defined after the endpoint cannot be passed to a
        # lookup at the endpoint without an independently proved motion.
        lines = llvm.splitlines()
        first = next(i for i, line in enumerate(lines) if "fmul float" in line)
        last = lines[first + 1]
        fields = last.split("fmul float", 1)[1].strip().split(", ")
        up = fields[1]
        lines.insert(first + 1, f"  %newup = fadd float {up}, 0.000000e+00")
        lines[first + 2] = last.replace(up, "%newup")
        changed = "\n".join(lines)
    assert changed != llvm
    if change == "strict":
        with pytest.raises(ValueError):
            rewrite_source_interval_lookup(
                changed, proofs=(proof,), table=table, lookup_symbol="lookup", effects=EFFECTS
            )
    else:
        after, report = rewrite_source_interval_lookup(
            changed, proofs=(proof,), table=table, lookup_symbol="lookup", effects=EFFECTS
        )
        assert not report["routes"] and after == changed


def test_wrong_table_source_identity_and_existing_callback_refuse(compiled):
    _, llvm, _, proof, table, _ = compiled
    bad = SourceIntervalTable("f" * 64, table.leading_bits, table.data, table.valid_cells)
    with pytest.raises(ValueError):
        rewrite_source_interval_lookup(llvm, proofs=(proof,), table=bad, lookup_symbol="lookup", effects=EFFECTS)
    with pytest.raises(ValueError):
        rewrite_source_interval_lookup(
            llvm + "\ndeclare float @lookup(float,float,float)",
            proofs=(proof,),
            table=table,
            lookup_symbol="lookup",
            effects=EFFECTS,
        )
