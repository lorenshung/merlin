"""Independent source arithmetic, partition proofs and typed helper delivery."""

from __future__ import annotations

import ctypes
import os
import shutil
import struct
import subprocess
from dataclasses import replace

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.source_expression_interval import IntervalEffectContract, close_scalar_i8_observer
from merlin.llvmlower.source_scalar_carrier import (
    ScalarCarrierHelperBinding,
    prepare_source_scalar_carrier,
    reify_source_scalar_carrier_helper_family,
    reify_source_scalar_carrier_helpers,
    validate_source_scalar_carrier,
)
from merlin.llvmlower.source_scalar_carrier_policy import ApproximateScalarCarrierPolicy, ScalarCarrierErrorBudget

EFFECTS = IntervalEffectContract(True, True, True, True, True)


def policy(**changes):
    return replace(
        ApproximateScalarCarrierPolicy(ScalarCarrierErrorBudget(0.5, 0.01), 9, 12, 6144, 32768, True, True, True, True),
        **changes,
    )


def source(expression, factor=3.0):
    return f"""module {{func.func @original(%x:f32,%up:f32)->i8 {{
      %two=arith.constant 2.0:f32
      %minus=arith.constant -2.0:f32
      %three=arith.constant {factor}:f32
      {expression}
      %prod=arith.mulf %act,%up:f32
      %scaled=arith.mulf %prod,%three:f32
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
      %mone=arith.constant -1:i8
      %sign=arith.select %negative,%mone,%one:i8
      %delta=arith.select %bump,%sign,%zero:i8
      %r=arith.addi %t,%delta:i8
      return %r:i8
    }} }}"""


FAMILIES = (
    ("%act=arith.mulf %x,%two:f32", (2.0, 2.0**-22, 0.0)),
    ("%act=arith.mulf %x,%minus:f32", (-2.0, -(2.0**-22), 0.0)),
    ("%act=arith.mulf %x,%x:f32", (1.0, 2.0**-22, 2.0**-46)),
)


def analyze(expression=FAMILIES[0][0], factor=3.0):
    module = parse_mlir_text(source(expression, factor))
    function = module.body.block.first_op
    up = function.body.block.args[1]
    product = next(
        operation
        for operation in function.body.block.ops
        if operation.name == "arith.mulf" and up in operation.operands
    )
    endpoint = next(value for value in product.operands if value is not up)
    return module, close_scalar_i8_observer(function.body.block.args[0], endpoint, effects=EFFECTS)


def word(value):
    return struct.unpack(">I", struct.pack(">f", value))[0]


def coefficients(triple=FAMILIES[0][1]):
    cells = [(0, 0, 0)] * 512
    cells[word(1.0) >> 23] = tuple(word(value) for value in triple)
    return tuple(cells)


def prepare(proof, triple=FAMILIES[0][1], **changes):
    return prepare_source_scalar_carrier(
        [proof], effects=EFFECTS, policy=policy(**changes), coefficient_words=coefficients(triple)
    )


def reify(carrier, proof, **changes):
    names = {
        "table_symbol": "cells",
        "expression_symbol": "expression",
        "observer_symbol": "observer",
        "carrier_symbol": "carrier",
    }
    return reify_source_scalar_carrier_helpers(carrier, proof, **(names | changes))


@pytest.mark.parametrize("expression,triple", FAMILIES)
def test_current_typed_helpers_and_physical_proof_granularity(expression, triple):
    from xdsl.dialects import memref, scf
    from xdsl.dialects.builtin import f32, i1, i8

    module, proof = analyze(expression)
    before = str(module)
    carrier = prepare(proof, triple)
    assert carrier.coefficient_words[127] == coefficients(triple)[127]
    assert carrier.policy.leading_bits < carrier.policy.proof_leading_bits
    helpers = reify(carrier, proof)
    storage, original, observer, selected = tuple(helpers.body.block.ops)
    assert isinstance(storage, memref.GlobalOp) and storage.constant is not None
    assert tuple(storage.type.get_shape()) == (512, 3)
    assert selected.function_type.inputs.data == (f32, f32, i1)
    assert selected.function_type.outputs.data == (i8,)
    branches = [operation for operation in selected.walk() if isinstance(operation, scf.IfOp)]
    assert len(branches) == 1
    assert [operation.name for operation in branches[0].false_region.block.ops] == ["func.call", "scf.yield"]
    assert [operation.name for operation in branches[0].true_region.block.ops].count("math.fma") == 2
    assert selected.attributes["prov.scalar_carrier_sha256"].data == carrier.canonical_sha256
    assert str(module) == before
    validate_source_scalar_carrier(carrier)


@pytest.mark.parametrize(
    "field",
    (
        "approximate_output_permitted",
        "independent_original_output_validation_required",
        "original_source_fallback_retained",
        "target_incoming_rne_predicate_required",
    ),
)
def test_distinct_approximation_permission_is_required(field):
    _, proof = analyze()
    with pytest.raises(ValueError):
        prepare(proof, **{field: False})


@pytest.mark.parametrize("field", tuple(vars(EFFECTS)))
def test_every_source_effect_is_explicit(field):
    _, proof = analyze()
    with pytest.raises(ValueError):
        prepare_source_scalar_carrier(
            [proof], effects=replace(EFFECTS, **{field: False}), policy=policy(), coefficient_words=coefficients()
        )


@pytest.mark.parametrize(
    "changes",
    (
        {"leading_bits": 8},
        {"leading_bits": 13},
        {"proof_leading_bits": 24},
        {"leading_bits": True},
        {"max_table_bytes": 6143},
        {"max_proof_table_bytes": 32767},
        {"budget": ScalarCarrierErrorBudget(0.0, 0.0)},
        {"budget": ScalarCarrierErrorBudget(float("nan"), 0.1)},
    ),
)
def test_extent_budget_and_partition_refusals(changes):
    _, proof = analyze()
    with pytest.raises(ValueError):
        prepare(proof, **changes)


@pytest.mark.parametrize("change", ("words", "budget", "effects", "source", "uses", "context"))
def test_stale_source_and_immutable_certificate_refuse(change):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, UnitAttr, f32

    module, proof = analyze()
    carrier = prepare(proof)
    if change == "words":
        carrier = replace(carrier, coefficient_words=coefficients((3.0, 2.0**-22, 0.0)))
    elif change == "budget":
        carrier = replace(carrier, policy=policy(budget=ScalarCarrierErrorBudget(1.0, 0.01)))
    elif change == "effects":
        carrier = replace(carrier, _effects=replace(EFFECTS, rne=False))
    elif change == "source":
        literal = next(operation for operation in proof.expression_operations if operation.name == "arith.constant")
        literal.properties["value"] = FloatAttr(4.0, f32)
    elif change == "uses":
        proof.endpoint.owner.parent.insert_op_after(arith.AddfOp(proof.endpoint, proof.cut), proof.endpoint.owner)
    else:
        module.attributes["strictfp"] = UnitAttr()
    with pytest.raises(ValueError):
        reify(carrier, proof)


def test_nonfinite_bad_coefficients_and_missing_policy_keep_explicit_refusal():
    _, proof = analyze()
    carrier = prepare(proof, (float("nan"), 0.0, 0.0))
    assert carrier.coefficient_words[127][0] == word(float("nan"))
    wrong = prepare(proof, (100.0, 0.0, 0.0))
    assert wrong.coefficient_words[127][0] == word(float("nan"))
    for values in (list(coefficients()), coefficients()[:-1], ((0, 0, True),) * 512):
        with pytest.raises(ValueError):
            prepare_source_scalar_carrier([proof], effects=EFFECTS, policy=policy(), coefficient_words=values)
    with pytest.raises(ValueError):
        prepare_source_scalar_carrier([proof], effects=EFFECTS, policy=None, coefficient_words=coefficients())


def test_refinement_improves_proof_without_changing_physical_storage():
    _, proof = analyze()
    coarse = prepare(proof, proof_leading_bits=9)
    refined = prepare(proof)
    assert coarse.coefficient_words[127][0] == word(float("nan"))
    assert refined.coefficient_words[127] == coefficients()[127]
    assert len(coarse.coefficient_words) == len(refined.coefficient_words) == 512


def test_carrier_budget_does_not_claim_equal_integer_codes():
    _, proof = analyze()
    carrier = prepare(proof, (2.0625, 2.0**-22, 0.0))
    assert carrier.coefficient_words[127] == coefficients((2.0625, 2.0**-22, 0.0))[127]
    # At x=1/up=8, the admitted carrier moves 48 to 49.5 before
    # the original quantizer. Separate original output validation is mandatory.
    assert abs(2.0625 - 2.0) < carrier.policy.budget.atol
    assert round(2.0625 * 8.0 * 3.0) != round(2.0 * 8.0 * 3.0)


def test_source_trace_is_preserved_without_selecting_policy():
    from xdsl.dialects.builtin import StringAttr

    module = parse_mlir_text(source(FAMILIES[0][0]))
    module.attributes["prov.source_node_ids"] = StringAttr("not-a-policy-selector")
    function = module.body.block.first_op
    endpoint = next(operation.results[0] for operation in function.body.block.ops if operation.name == "arith.mulf")
    proof = close_scalar_i8_observer(function.body.block.args[0], endpoint, effects=EFFECTS)
    carrier = prepare(proof)
    helpers = reify(carrier, proof)
    assert helpers.body.block.last_op.attributes["prov.source_node_ids"] == module.attributes["prov.source_node_ids"]


@pytest.mark.parametrize("names", ({"carrier_symbol": "cells"}, {"table_symbol": "has.dot"}, {"observer_symbol": ""}))
def test_symbols_and_foreign_source_proof_refuse(names):
    _, proof = analyze()
    carrier = prepare(proof)
    with pytest.raises(ValueError):
        reify(carrier, proof, **names)
    _, other = analyze()
    with pytest.raises(ValueError):
        reify(carrier, other)


@pytest.mark.parametrize("expression,triple", FAMILIES)
@pytest.mark.skipif(
    os.environ.get("MERLIN_TEST_SCALAR_CARRIER_NATIVE") != "1", reason="explicit upstream/native qualification"
)
def test_normal_upstream_native_delivery_and_independent_tail_calls(expression, triple, tmp_path):
    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    module, proof = analyze(expression)
    helpers = reify(prepare(proof, triple), proof)
    helpers.body.block.add_op(module.body.block.first_op.clone())
    llvm = lower_to_llvm_ir(str(helpers), workdir=tmp_path / "lower", hoist_static_allocs=False, features=frozenset())
    ir = tmp_path / "helpers.ll"
    ir.write_text(llvm)
    compiler = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    assert compiler is not None
    library = tmp_path / "helpers.so"
    subprocess.run(
        [compiler, "-O2", "-shared", "-fPIC", str(ir), "-lm", "-o", str(library)], check=True, capture_output=True
    )
    native = ctypes.CDLL(str(library))
    native.original.argtypes = [ctypes.c_float, ctypes.c_float]
    native.original.restype = ctypes.c_int8
    native.carrier.argtypes = [ctypes.c_float, ctypes.c_float, ctypes.c_bool]
    native.carrier.restype = ctypes.c_int8
    # Unadmitted cells include zeros/subnormals and finite unrelated intervals.
    # Deliberately avoid undefined nonfinite finishing conversions.
    refused_words = (0, 1, word(-1.25), word(32.0), word(-32.0))
    for length in (1, 17, 65):
        words = np.linspace(word(1.0), word(2.0) - 1, length, dtype=np.uint32)
        for raw in (*words, *refused_words):
            value = struct.unpack(">f", struct.pack(">I", int(raw)))[0]
            for up in (-4.0, -0.25, 0.0, 0.25, 4.0):
                original = native.original(value, up)
                assert native.carrier(value, up, False) == original
                assert native.carrier(value, up, True) == original
    host = ctypes.CDLL(None)
    host.fegetround.restype = ctypes.c_int
    host.fesetround.argtypes = [ctypes.c_int]
    incoming_mode = host.fegetround()
    try:
        # Native qualification explicitly uses Linux libc's four fenv modes.
        # No environment read or ISA predicate is emitted by the shared helper.
        for mode in (0, 0x400, 0x800, 0xC00):
            assert host.fesetround(mode) == 0
            for raw in np.linspace(word(1.0), word(2.0) - 1, 17, dtype=np.uint32):
                value = struct.unpack(">f", struct.pack(">I", int(raw)))[0]
                for up in (-0.3125, 0.3125, 1.0625):
                    assert native.carrier(value, up, False) == native.original(value, up)
    finally:
        assert host.fesetround(incoming_mode) == 0


def family():
    first, left = analyze()
    second, right = analyze(factor=5.0)
    carrier = prepare_source_scalar_carrier(
        (left, right), effects=EFFECTS, policy=policy(), coefficient_words=coefficients()
    )
    bindings = (
        ScalarCarrierHelperBinding(left, "expression_a", "observer_a", "carrier_a"),
        ScalarCarrierHelperBinding(right, "expression_b", "observer_b", "carrier_b"),
    )
    return (first, second), carrier, bindings


def test_family_owns_one_table_and_every_current_finishing_factor():
    from xdsl.dialects import memref

    originals, carrier, bindings = family()
    before = tuple(str(module) for module in originals)
    result = reify_source_scalar_carrier_helper_family(carrier, bindings, table_symbol="shared_cells")
    globals = [operation for operation in result.body.block.ops if isinstance(operation, memref.GlobalOp)]
    assert len(globals) == 1 and globals[0].constant is not None
    assert len(tuple(result.body.block.ops)) == 7
    references = [operation for operation in result.walk() if isinstance(operation, memref.GetGlobalOp)]
    assert len(references) == 2
    assert all(operation.name_.root_reference.data == "shared_cells" for operation in references)
    assert bindings[0].proof.quant_factor_bits == word(3.0)
    assert bindings[1].proof.quant_factor_bits == word(5.0)
    assert tuple(str(module) for module in originals) == before
    assert result.attributes["prov.scalar_carrier_members"].data[0].data == "carrier_a"


@pytest.mark.parametrize("bad", ("mutable", "empty", "namespace", "storage_alias", "foreign", "stale"))
def test_family_composition_refuses_unclosed_members_and_namespace(bad):
    from xdsl.dialects.builtin import FloatAttr, f32

    originals, carrier, bindings = family()
    table = "shared_cells"
    if bad == "mutable":
        bindings = list(bindings)
    elif bad == "empty":
        bindings = ()
    elif bad == "namespace":
        bindings = (bindings[0], replace(bindings[1], carrier_symbol=bindings[0].carrier_symbol))
    elif bad == "storage_alias":
        table = bindings[0].expression_symbol
    elif bad == "foreign":
        _, foreign = analyze(factor=7.0)
        bindings = (bindings[0], replace(bindings[1], proof=foreign))
    else:
        factor = next(
            operation
            for operation in originals[1].body.block.first_op.body.block.ops
            if operation.name == "arith.constant" and operation.value.type == f32 and operation.value.value.data == 5.0
        )
        factor.properties["value"] = FloatAttr(7.0, f32)
    with pytest.raises(ValueError):
        reify_source_scalar_carrier_helper_family(carrier, bindings, table_symbol=table)


@pytest.mark.skipif(
    os.environ.get("MERLIN_TEST_SCALAR_CARRIER_NATIVE") != "1", reason="explicit upstream/native qualification"
)
def test_shared_family_normal_delivery_has_one_physical_table_and_distinct_consumers(tmp_path):
    from xdsl.dialects.builtin import StringAttr

    from merlin.llvmlower.pipeline import lower_to_llvm_ir

    originals, carrier, bindings = family()
    helpers = reify_source_scalar_carrier_helper_family(carrier, bindings, table_symbol="shared_cells")
    for module, name in zip(originals, ("original_a", "original_b")):
        cloned = module.body.block.first_op.clone()
        cloned.properties["sym_name"] = StringAttr(name)
        helpers.body.block.add_op(cloned)
    llvm = lower_to_llvm_ir(str(helpers), workdir=tmp_path / "lower", hoist_static_allocs=False, features=frozenset())
    globals = [line.partition("=")[0].strip() for line in llvm.splitlines() if line.startswith("@")]
    assert globals == ["@shared_cells"]
    ir = tmp_path / "helpers.ll"
    ir.write_text(llvm)
    library = tmp_path / "helpers.so"
    compiler = os.environ.get("MERLIN_CLANG") or shutil.which("clang")
    assert compiler is not None
    subprocess.run(
        [compiler, "-O2", "-shared", "-fPIC", str(ir), "-lm", "-o", str(library)], check=True, capture_output=True
    )
    native = ctypes.CDLL(str(library))
    for suffix in ("a", "b"):
        original = getattr(native, "original_" + suffix)
        observed = getattr(native, "carrier_" + suffix)
        original.argtypes, original.restype = [ctypes.c_float, ctypes.c_float], ctypes.c_int8
        observed.argtypes, observed.restype = [ctypes.c_float, ctypes.c_float, ctypes.c_bool], ctypes.c_int8
        for length in (1, 17, 65):
            for raw in np.linspace(word(1.0), word(2.0) - 1, length, dtype=np.uint32):
                value = struct.unpack(">f", struct.pack(">I", int(raw)))[0]
                for up in (-4.0, -0.25, 0.0, 0.25, 4.0):
                    assert observed(value, up, True) == original(value, up)
                    assert observed(value, up, False) == original(value, up)
    assert native.carrier_a(1.0, 1.0, True) == 6
    assert native.carrier_b(1.0, 1.0, True) == 10
