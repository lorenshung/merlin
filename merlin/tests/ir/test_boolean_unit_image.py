"""Discrete IEEE identities require the full unsigned Boolean conversion image."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.boolean_unit_image import fold_boolean_unit_image


def source(operation, *, element="f32", integer="i1", literal=None, unknown=False):
    coefficient = f"%c=arith.constant {literal} : {element}" if literal is not None else ""
    carrier = (
        '%opaque="fixture.unknown"(%x) : (tensor<4x' + integer + ">) -> tensor<4x" + integer + ">" if unknown else ""
    )
    args = "%f, %c" if literal is not None else "%f"
    calculation = (
        f'%v="arith.remf"(%f,%c) : ({element},{element}) -> {element}'
        if operation == "arith.remf"
        else f"%v={operation} {args} : {element}"
    )
    return parse_mlir_text(f"""module {{func.func @forward(%x:tensor<4x{integer}>) -> tensor<4x{element}> {{
      {carrier}
      %e=tensor.empty() : tensor<4x{element}>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins({"%opaque" if unknown else "%x"}:tensor<4x{integer}>) outs(%e:tensor<4x{element}>) {{
        ^bb0(%a:{integer},%unused:{element}):
        %f=arith.uitofp %a : {integer} to {element}
        {coefficient}
        {calculation}
        linalg.yield %v : {element}
      }} -> tensor<4x{element}>
      return %r : tensor<4x{element}>
    }} }}""")


@pytest.mark.parametrize("element", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize(
    "operation,literal",
    [
        ("math.sqrt", None),
        ("math.floor", None),
        ("arith.divf", 1.0),
        ("math.powf", 0.5),
        ("arith.remf", 0.5),
        ("arith.remf", 1.25),
        ("arith.remf", -2.0),
        ("arith.subf", 0.0),
    ],
)
def test_every_boolean_word_has_the_proved_ieee_result(element, operation, literal):
    module = source(operation, element=element, literal=literal)
    assert fold_boolean_unit_image(module) == 1
    module.verify()
    assert not any(op.name == operation for op in module.walk())


@pytest.mark.parametrize("options", [dict(integer="i8"), dict(unknown=True)])
def test_other_or_unknown_integer_images_are_not_boolean(options):
    module = source("math.sqrt", **options)
    original = str(module)
    assert fold_boolean_unit_image(module) == 0
    assert str(module) == original


def test_a_rounded_reciprocal_is_not_an_exact_remainder_proof():
    module = source("arith.remf", element="f64", literal="0.3333333333333333")
    original = str(module)
    assert fold_boolean_unit_image(module) == 0
    assert str(module) == original


@pytest.mark.parametrize("zero,identity", [("0.0", True), ("-0.0", False)])
def test_unit_selector_does_not_erase_a_negative_zero_word(zero, identity):
    module = parse_mlir_text(f"""module {{func.func @forward(%x:tensor<4xi1>) -> tensor<4xf32> {{
      %e=tensor.empty() : tensor<4xf32>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%x:tensor<4xi1>) outs(%e:tensor<4xf32>) {{
        ^bb0(%a:i1,%unused:f32):
        %f=arith.uitofp %a : i1 to f32
        %positive=arith.constant 0.0 : f32
        %z=arith.constant {zero} : f32
        %p=arith.cmpf oeq,%f,%positive : f32
        %v=arith.select %p,%z,%f : f32
        linalg.yield %v : f32
      }} -> tensor<4xf32>
      return %r : tensor<4xf32>
    }} }}""")
    original = str(module)
    assert (fold_boolean_unit_image(module) > 0) == identity
    if not identity:
        assert str(module) == original
