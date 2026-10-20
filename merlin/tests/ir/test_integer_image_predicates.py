"""Uniform predicates use the full rounded integer image, never sampled values."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.integer_image_predicates import fold_integer_image_predicates


def source(integer="i64", floating="f32", threshold="0.5", predicate="oeq", conversion="sitofp"):
    return parse_mlir_text(f"""module {{func.func @forward(%x:tensor<4x{integer}>) -> tensor<4xi1> {{
      %e=tensor.empty() : tensor<4x{floating}>
      %f=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%x:tensor<4x{integer}>) outs(%e:tensor<4x{floating}>) {{
        ^bb0(%a:{integer},%unused:{floating}):
          %v=arith.{conversion} %a : {integer} to {floating}
          linalg.yield %v : {floating}
        }} -> tensor<4x{floating}>
      %c=arith.constant {threshold} : {floating}
      %s=tensor.splat %c : tensor<4x{floating}>
      %out=tensor.empty() : tensor<4xi1>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,
        affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%f,%s:tensor<4x{floating}>,tensor<4x{floating}>) outs(%out:tensor<4xi1>) {{
        ^bb1(%a:{floating},%b:{floating},%unused:i1):
          %p=arith.cmpf {predicate},%a,%b : {floating}
          linalg.yield %p : i1
        }} -> tensor<4xi1>
      func.return %r : tensor<4xi1>
    }} }}""")


@pytest.mark.parametrize("integer", ["i8", "i32", "i64"])
@pytest.mark.parametrize("floating", ["f16", "bf16", "f32", "f64"])
@pytest.mark.parametrize("predicate,value", [("oeq", 0), ("une", 1)])
def test_fractional_threshold_is_outside_every_integer_float_image(integer, floating, predicate, value):
    module = source(integer, floating, predicate=predicate)
    assert fold_integer_image_predicates(module) == 1
    module.verify()
    returned = next(op for op in module.walk() if op.name == "func.return").operands[0]
    assert returned.owner.name == "tensor.splat"
    assert bool(returned.owner.operands[0].owner.value.value.data) == bool(value)


@pytest.mark.parametrize(
    "threshold,predicate,value",
    [("300.0", "oeq", False), ("300.0", "olt", True), ("-129.0", "oge", True), ("128.0", "one", True)],
)
def test_typed_endpoint_bound_closes_only_unattainable_integer_values(threshold, predicate, value):
    module = source("i8", threshold=threshold, predicate=predicate)
    assert fold_integer_image_predicates(module) == 1
    module.verify()
    returned = next(op for op in module.walk() if op.name == "func.return").operands[0]
    assert bool(returned.owner.operands[0].owner.value.value.data) == value


@pytest.mark.parametrize("threshold,predicate", [("127.0", "oeq"), ("127.0", "olt"), ("0.0", "one")])
def test_attainable_threshold_does_not_become_a_fill(threshold, predicate):
    module = source("i8", threshold=threshold, predicate=predicate)
    original = str(module)
    assert fold_integer_image_predicates(module) == 0
    assert str(module) == original


def test_real_floating_input_is_not_assumed_integral():
    module = source("f16", "f32", conversion="extf")
    original = str(module)
    assert fold_integer_image_predicates(module) == 0
    assert str(module) == original


def test_promoted_rounded_endpoint_is_still_attainable():
    # INT_MAX rounds upward to 2**31 when compared in binary32.
    module = source("i32", threshold="2147483648.0")
    original = str(module)
    assert fold_integer_image_predicates(module) == 0
    assert str(module) == original
