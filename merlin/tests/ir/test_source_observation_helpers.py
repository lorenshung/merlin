"""Current typed helper source, independently varied expressions and refusals."""

from __future__ import annotations

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.bounded_rne_maps import prove_scalar_bounded_rne
from merlin.llvmlower.source_expression_interval import (
    IntervalEffectContract,
    close_scalar_i8_observer,
    extract_scalar_expression,
    validate_closed_scalar_observer,
)
from merlin.llvmlower.source_observation_helpers import reify_closed_scalar_observer_helpers

EFFECTS = IntervalEffectContract(True, True, True, True, True)


def source(expression="%act=arith.mulf %x,%two:f32"):
    return f"""module {{func.func @independent(%x:f32,%up:f32)->i8 {{
      %two=arith.constant 2.0:f32
      %three=arith.constant 3.0:f32
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
      %minus=arith.constant -1:i8
      %sign=arith.select %negative,%minus,%one:i8
      %delta=arith.select %bump,%sign,%zero:i8
      %r=arith.addi %t,%delta:i8
      return %r:i8
    }} }}"""


def analyze(text=None):
    module = parse_mlir_text(text or source())
    function = module.body.block.first_op
    finishing = next(
        operation
        for operation in function.body.block.ops
        if operation.name == "arith.mulf" and function.body.block.args[1] in operation.operands
    )
    endpoint = next(operand for operand in finishing.operands if operand is not function.body.block.args[1])
    proof = close_scalar_i8_observer(function.body.block.args[0], endpoint, effects=EFFECTS)
    return module, proof


def reify(proof, **overrides):
    return reify_closed_scalar_observer_helpers(
        proof, **{"effects": EFFECTS, "expression_symbol": "expression", "observer_symbol": "observation", **overrides}
    )


@pytest.mark.parametrize(
    "expression",
    [
        "%act=arith.mulf %x,%two:f32",
        "%act=math.fma %x,%two,%three:f32",
        "%d=arith.divf %two,%x:f32\n%act=arith.mulf %x,%d:f32",
    ],
)
def test_fresh_helpers_preserve_actual_arithmetic_and_complete_observation(expression):
    module, proof = analyze(source(expression))
    before = str(module)
    helpers = reify(proof)
    activation, observation = tuple(helpers.body.block.ops)
    fresh, _ = extract_scalar_expression(activation.body.block.args[0], activation.body.block.last_op.operands[0])
    assert fresh == proof.expression
    observed = prove_scalar_bounded_rne(observation.body.block)
    assert observed is not None and observed.raw_input is observation.body.block.args[0]
    assert observed.integer_bits == 8 and observed.bounds == (-128, 127)
    assert str(module) == before
    validate_closed_scalar_observer(proof)


def test_enclosing_trace_preserved_without_label_selection():
    from xdsl.dialects.builtin import StringAttr

    module = parse_mlir_text(source())
    module.attributes["prov.source_node_ids"] = StringAttr("arbitrary trace A")
    module.body.block.first_op.attributes["prov.region_id"] = StringAttr("arbitrary trace B")
    _, proof = analyze(str(module))
    helpers = reify(proof, expression_symbol="renamed_expression", observer_symbol="renamed_observer")
    for function in helpers.body.block.ops:
        assert function.attributes["prov.source_node_ids"] == StringAttr("arbitrary trace A")
        assert function.attributes["prov.region_id"] == StringAttr("arbitrary trace B")
        assert function.attributes["prov.reified_expression_sha256"] == StringAttr(proof.expression.canonical_sha256)
        assert function.attributes["prov.reification"] == StringAttr("closed_scalar_observer_helper")


@pytest.mark.parametrize("context", ["constant", "use", "strictfp"])
def test_stale_current_source_refuses_before_clone(context):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FloatAttr, UnitAttr, f32

    module, proof = analyze()
    if context == "constant":
        constant = next(op for op in proof.expression_operations if op.name == "arith.constant")
        constant.properties["value"] = FloatAttr(4.0, f32)
    elif context == "use":
        addition = arith.AddfOp(proof.endpoint, proof.cut)
        proof.endpoint.owner.parent.insert_op_after(addition, proof.endpoint.owner)
    else:
        module.attributes["strictfp"] = UnitAttr()
    with pytest.raises(ValueError):
        reify(proof)


@pytest.mark.parametrize("field", tuple(vars(EFFECTS)))
def test_effects_are_explicit(field):
    _, proof = analyze()
    effects = IntervalEffectContract(**(vars(EFFECTS) | {field: False}))
    with pytest.raises(ValueError):
        reify(proof, effects=effects)


@pytest.mark.parametrize("name", ["", "has.dot", "4initial", "nonascii_é"])
def test_invalid_abi_names_refuse(name):
    _, proof = analyze()
    with pytest.raises(ValueError):
        reify(proof, expression_symbol=name)


def test_ambiguous_names_missing_effects_and_unproved_object_refuse():
    _, proof = analyze()
    with pytest.raises(ValueError):
        reify(proof, observer_symbol="expression")
    with pytest.raises(ValueError):
        reify(proof, effects=None)
    with pytest.raises(ValueError):
        reify(object())
