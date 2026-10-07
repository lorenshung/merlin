"""Independent typed closure, source mutations and full-cell interval contracts."""

from __future__ import annotations

import ctypes
import struct
from fractions import Fraction

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    ScalarExpression,
    ScalarStep,
    build_source_interval_table,
    close_scalar_i8_observer,
    emit_immutable_bytes_llvm,
    emit_source_interval_lookup,
    evaluate_intervals,
    find_closed_scalar_i8_observers,
    validate_closed_scalar_observer,
)

EFFECTS = IntervalEffectContract(True, True, True, True, True)


def source(*, constant="2.0", label="unrelated", escape=False):
    # No exp/SiLU/model-specific grammar is required by the generic analyzer.
    return f"""module {{func.func @{label}(%x:f32,%up:f32)->i8 {{
      %two=arith.constant {constant}:f32
      %act=arith.mulf %x,%two:f32
      {"%escape=arith.addf %act,%x:f32" if escape else ""}
      %prod=arith.mulf %act,%up:f32
      %q=arith.constant 3.0:f32
      %scaled=arith.mulf %prod,%q:f32
      %lo=arith.constant -128.0:f32
      %a=arith.maximumf %scaled,%lo:f32
      %hi=arith.constant 127.0:f32
      %b=arith.minimumf %a,%hi:f32
      %t=arith.fptosi %b:f32 to i8
      %tf=arith.sitofp %t:i8 to f32
      %frac=arith.subf %b,%tf:f32
      %neg=arith.negf %frac:f32
      %abs=arith.maximumf %frac,%neg:f32
      %half=arith.constant 0.5:f32
      %gt=arith.cmpf ogt,%abs,%half:f32
      %eq=arith.cmpf oeq,%abs,%half:f32
      %one=arith.constant 1:i8
      %zero=arith.constant 0:i8
      %odd=arith.andi %t,%one:i8
      %ne=arith.cmpi ne,%odd,%zero:i8
      %tie=arith.andi %eq,%ne:i1
      %bump=arith.ori %gt,%tie:i1
      %zf=arith.constant 0.0:f32
      %negative=arith.cmpf olt,%b,%zf:f32
      %minus=arith.constant -1:i8
      %sign=arith.select %negative,%minus,%one:i8
      %delta=arith.select %bump,%sign,%zero:i8
      %r=arith.addi %t,%delta:i8
      return %r:i8
    }} }}"""


def analyze(text):
    module = parse_mlir_text(text)
    function = next(op for op in module.walk() if op.name == "func.func")
    activation = next(op for op in function.body.block.ops if op.name == "arith.mulf")
    proof = close_scalar_i8_observer(function.body.block.args[0], activation.results[0], effects=EFFECTS)
    return module, proof


def bits(number):
    return struct.unpack("<I", struct.pack("<f", number))[0]


def test_actual_typed_observer_no_workload_or_labels_and_semantic_identity():
    _, first = analyze(source())
    _, renamed = analyze(source(label="other_function"))
    _, changed = analyze(source(constant="2.00000024"))
    validate_closed_scalar_observer(first)
    assert first.expression == renamed.expression
    assert first.expression.canonical_sha256 == renamed.expression.canonical_sha256
    assert first.expression.canonical_sha256 != changed.expression.canonical_sha256
    assert first.quant_factor_bits == bits(3)


def test_semantic_attribute_refusal_and_shared_literal_mutation():
    from xdsl.dialects.builtin import StringAttr

    module = parse_mlir_text(source())
    function = module.body.block.first_op
    activation = next(op for op in function.body.block.ops if op.name == "arith.mulf")
    activation.attributes["unproved_numeric_contract"] = StringAttr("arbitrary")
    with pytest.raises(ValueError):
        close_scalar_i8_observer(function.body.block.args[0], activation.results[0], effects=EFFECTS)
    _, proof = analyze(source())
    factor = proof.observer_operations[1].operands[1].owner
    factor.attributes["unrelated_new_attribute"] = StringAttr("changed")
    with pytest.raises(ValueError):
        validate_closed_scalar_observer(proof)


def test_source_inventory_finds_arithmetic_and_preserves_original_bytes():
    # An independent rational expression, rather than the first measured poly.
    text = source().replace("%act=arith.mulf %x,%two:f32", "%d=arith.divf %two,%x:f32\n%act=arith.mulf %x,%d:f32")
    module = parse_mlir_text(text)
    before = str(module)
    proofs, refusals = find_closed_scalar_i8_observers(module, effects=EFFECTS)
    assert len(proofs) == 1 and not refusals and str(module) == before
    changed = parse_mlir_text(text.replace("%prod=arith.mulf", "%prod=arith.addf"))
    before = str(changed)
    proofs, refusals = find_closed_scalar_i8_observers(changed, effects=EFFECTS)
    assert not proofs and len(refusals) == 1 and str(changed) == before


@pytest.mark.parametrize(
    "change",
    ["escape", "unknown_consumer", "factor_zero", "factor_negative", "factor_infinity", "fastmath", "strictfp"],
)
def test_typed_refusals(change):
    text = source(escape=change == "escape")
    if change == "unknown_consumer":
        text = text.replace("%prod=arith.mulf", "%prod=arith.addf")
    if change == "factor_zero":
        text = text.replace("3.0:f32", "0.0:f32")
    if change == "factor_negative":
        text = text.replace("3.0:f32", "-3.0:f32")
    if change == "factor_infinity":
        text = text.replace("3.0:f32", "0x7f800000:f32")
    if change == "fastmath":
        text = text.replace("mulf %x,%two:f32", "mulf %x,%two fastmath<contract>:f32")
    if change == "strictfp":
        text = text.replace("->i8 {", "->i8 attributes {strictfp} {")
    with pytest.raises(ValueError):
        analyze(text)


@pytest.mark.parametrize("context", ["scalar", "function", "module", "uses"])
def test_complete_source_context_and_use_mutation(context):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import UnitAttr

    module, proof = analyze(source())
    if context == "uses":
        op = arith.AddfOp(proof.endpoint, proof.cut)
        proof.endpoint.owner.parent.insert_op_after(op, proof.endpoint.owner)
    else:
        owner = (
            proof.endpoint.owner
            if context == "scalar"
            else module
            if context == "module"
            else module.body.block.first_op
        )
        owner.attributes["strictfp"] = UnitAttr()
    with pytest.raises(ValueError):
        validate_closed_scalar_observer(proof)


def test_explicit_numeric_effects_not_inferred():
    module = parse_mlir_text(source())
    function = module.body.block.first_op
    activation = next(op for op in function.body.block.ops if op.name == "arith.mulf")
    for field in vars(EFFECTS):
        contract = IntervalEffectContract(**(vars(EFFECTS) | {field: False}))
        with pytest.raises(ValueError):
            close_scalar_i8_observer(function.body.block.args[0], activation.results[0], effects=contract)


@pytest.mark.parametrize("width", [9, 12, 16])
def test_fixed_source_wide_cells_changed_partition_and_resource_budget(width):
    _, proof = analyze(source())
    table = build_source_interval_table(
        proof.expression, effects=EFFECTS, leading_bits=width, max_table_bytes=(1 << width) * 8
    )
    assert len(table.data) == (1 << width) * 8
    cells = np.frombuffer(table.data, "<f4").reshape(-1, 2)
    rng = np.random.default_rng(879)
    words = rng.integers(0, 2**32, size=23017, dtype=np.uint32)
    x = words.view(np.float32)
    lo, hi = cells[words >> np.uint32(32 - width)].T
    admitted = lo <= hi
    with np.errstate(all="ignore"):
        original = np.float32(x * np.float32(2))
    assert np.all((original[admitted] >= lo[admitted]) & (original[admitted] <= hi[admitted]))
    special = np.array([0, 0x80000000, 1, 0x80000001, 0x7F800000, 0xFF800000, 0x7FC00123], np.uint32)
    refused = cells[special >> np.uint32(32 - width)]
    assert np.all(refused[:, 0] > refused[:, 1])
    with pytest.raises(ValueError):
        build_source_interval_table(
            proof.expression, effects=EFFECTS, leading_bits=width, max_table_bytes=len(table.data) - 1
        )


def test_generator_refuses_actual_nonRNE_host_without_mutating_environment():
    lib = ctypes.CDLL(None)
    old = lib.fegetround()
    try:
        assert lib.fesetround(0x400) == 0
        expression = ScalarExpression(
            (ScalarStep("arith.constant", (), "f32", bits(2)), ScalarStep("arith.mulf", (0, 1), "f32"))
        )
        with pytest.raises(ValueError, match="actual host IEEE RNE"):
            build_source_interval_table(expression, effects=EFFECTS, leading_bits=9, max_table_bytes=4096)
        assert lib.fegetround() == 0x400
    finally:
        assert lib.fesetround(old) == 0


def test_exact_rational_single_rounding_FMA_cancellation_and_tails():
    expression = ScalarExpression(
        (
            ScalarStep("arith.constant", (), "f32", bits(1.0000001192092896)),
            ScalarStep("arith.constant", (), "f32", bits(-1.000000238418579)),
            ScalarStep("math.fma", (0, 1, 2), "f32"),
        )
    )
    raw = np.array([bits(1), bits(1) + 1, bits(1) - 1, bits(-1), 1, 0x80000001, bits(1e30)], np.uint32)
    x = raw.view(np.float32)
    lo, hi, valid = evaluate_intervals(expression, x, x)
    for value, lower, upper, admitted in zip(x, lo, hi, valid):
        exact = Fraction(float(value)) * Fraction(1.0000001192092896) - Fraction(1.000000238418579)
        # Exact rational value is bracketed before the original f32 rounding;
        # positive/negative/cancellation inputs are unrelated to any fixture.
        rounded = np.float32(float(exact))
        assert lower <= rounded <= upper
        if admitted:
            assert (lower > 0 and upper > 0) or (lower < 0 and upper < 0)


@pytest.mark.parametrize(
    "steps",
    [
        (),
        (ScalarStep("math.exp", (0,), "f32"),),
        (ScalarStep("arith.mulf", (1, 0), "f32"),),
        (ScalarStep("arith.constant", (), "f32", 0x7F800000),),
        (ScalarStep("arith.constant", (), "i32", 2), ScalarStep("arith.mulf", (0, 1), "f32")),
    ],
)
def test_plan_refuses_unknown_cyclic_nonfinite_or_type_changed(steps):
    with pytest.raises(ValueError):
        ScalarExpression(steps).validate()


def test_portable_emitter_explicit_symbols_and_partition_has_no_ISA():
    c = emit_source_interval_lookup(
        table_name="table", activation_name="source", quantizer_name="quant", lookup_name="lookup", leading_bits=12
    )
    assert "[4096][2]" in c and "w>>20" in c
    assert "asm" not in c and "riscv" not in c
    assert "source(x)" in c and "quant(low_scaled)!=quant(high_scaled)" in c
    with pytest.raises(ValueError):
        emit_source_interval_lookup(
            table_name="bad-name", activation_name="s", quantizer_name="q", lookup_name="l", leading_bits=12
        )


def test_immutable_bytes_cover_quotes_nuls_highwords_and_alignment_refusal():
    data = bytes([0, 0x22, 0x5C, 0xFF, 0x80, 1])
    llvm = emit_immutable_bytes_llvm(data, symbol="readonly_owner", alignment=64)
    assert '[6 x i8] c"\\00\\22\\5C\\FF\\80\\01", align 64' in llvm
    for alignment in (0, 3, -1, True, 2**32):
        with pytest.raises(ValueError):
            emit_immutable_bytes_llvm(data, symbol="readonly_owner", alignment=alignment)


def test_partition_and_storage_budget_require_explicit_caller_choices():
    _, proof = analyze(source())
    with pytest.raises(TypeError):
        build_source_interval_table(proof.expression, effects=EFFECTS)
    with pytest.raises(TypeError):
        build_source_interval_table(proof.expression, effects=EFFECTS, leading_bits=12)
    with pytest.raises(TypeError):
        build_source_interval_table(proof.expression, effects=EFFECTS, max_table_bytes=32768)
    with pytest.raises(TypeError):
        emit_source_interval_lookup(table_name="t", activation_name="a", quantizer_name="q", lookup_name="l")
