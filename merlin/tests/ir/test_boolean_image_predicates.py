"""A comparison over Boolean conversions proves all possible source words."""

import math
import operator

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.boolean_image_predicates import fold_boolean_image_predicates

# Independent of the production predicate table: IEEE cmpf on ordered operands.
ORDERED = {"oeq": operator.eq, "one": operator.ne, "ogt": operator.gt, "olt": operator.lt, "uge": operator.ge}


def source(predicate, threshold, *, typ="f32", conversion="uitofp", element="i1", unknown=False):
    value = "0x7FC00000" if math.isnan(threshold) else str(float(threshold))
    carrier = (
        '%opaque = "fixture.unknown"(%x) : (tensor<2x' + element + ">) -> tensor<2x" + element + ">" if unknown else ""
    )
    return parse_mlir_text(f"""module {{ func.func @forward(%x:tensor<2x{element}>) -> tensor<2xi1> {{
      {carrier}
      %e = tensor.empty() : tensor<2x{typ}>
      %v = linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins({"%opaque" if unknown else "%x"}:tensor<2x{element}>) outs(%e:tensor<2x{typ}>) {{
        ^bb0(%a:{element}, %unused:{typ}):
        %f = arith.{conversion} %a : {element} to {typ}
        linalg.yield %f : {typ}
      }} -> tensor<2x{typ}>
      %c = arith.constant {value} : {typ}
      %t = tensor.splat %c : tensor<2x{typ}>
      %o = tensor.empty() : tensor<2xi1>
      %r = linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,
        affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%v,%t:tensor<2x{typ}>,tensor<2x{typ}>) outs(%o:tensor<2xi1>) {{
        ^bb1(%a:{typ}, %b:{typ}, %unused:i1):
        %p = arith.cmpf {predicate}, %a, %b : {typ}
        linalg.yield %p : i1
      }} -> tensor<2xi1>
      return %r : tensor<2xi1>
    }} }}""")


@pytest.mark.parametrize("typ", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("predicate", sorted(ORDERED))
@pytest.mark.parametrize("threshold", [-1, 0, 0.5, 1, 2])
@pytest.mark.parametrize("conversion", ["uitofp", "sitofp"])
def test_only_identity_or_uniform_images_are_emitted(typ, predicate, threshold, conversion):
    answers = [ORDERED[predicate](v, threshold) for v in (0, -1 if conversion == "sitofp" else 1)]
    module = source(predicate, threshold, typ=typ, conversion=conversion)
    expected = answers[0] == answers[1] or answers == [False, True]
    assert fold_boolean_image_predicates(module) == int(expected)
    module.verify()
    if expected:
        returned = next(op for op in module.walk() if op.name == "func.return").operands[0].owner
        assert returned.name == ("tensor.splat" if answers[0] == answers[1] else "linalg.generic")
        if returned.name == "linalg.generic":
            assert returned.inputs[0] is next(op for op in module.walk() if op.name == "func.func").body.block.args[0]


def test_nan_threshold_is_uniform_for_an_integer_image():
    module = source("oeq", math.nan)
    assert fold_boolean_image_predicates(module) == 1


@pytest.mark.parametrize("options", [dict(element="i8"), dict(unknown=True)])
def test_larger_integer_or_unknown_payload_cannot_gain_a_two_word_proof(options):
    module = source("oeq", 1, **options)
    original = str(module)
    assert fold_boolean_image_predicates(module) == 0
    assert str(module) == original
