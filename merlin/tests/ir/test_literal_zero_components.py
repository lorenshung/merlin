# ruff: noqa: E501 -- static MLIR fixtures retain one operation per line
"""Integer zero components are source algebra; floating unknowns remain unknown."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.literal_zero_components import fold_literal_zero_components


def source(literal):
    rhs = "%zero" if literal else "%y"
    return parse_mlir_text(f"""module {{
      func.func @forward(%x:tensor<2xi8>,%y:tensor<2xi8>) -> tensor<2xi32> {{
        %z=arith.constant 0 : i8
        %zero=tensor.splat %z : tensor<2xi8>
        %s=arith.constant 0 : i32
        %seed=tensor.splat %s : tensor<2xi32>
        %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
          ins(%x,{rhs}:tensor<2xi8>,tensor<2xi8>) outs(%seed:tensor<2xi32>) attrs={{test.limb="integer"}} {{
        ^bb0(%a:i8,%b:i8,%acc:i32):
          %x32=arith.extsi %a : i8 to i32
          %y32=arith.extsi %b : i8 to i32
          %p=arith.muli %x32,%y32 : i32
          %v=arith.addi %p,%acc : i32
          linalg.yield %v : i32
        }} -> tensor<2xi32>
        return %r : tensor<2xi32>
      }} }}""")


@pytest.mark.parametrize("literal", [True, False])
def test_only_a_source_literal_zero_removes_a_symbolic_integer_product(literal):
    module = source(literal)
    assert fold_literal_zero_components(module, "test.limb") == int(literal)
    assert any("test.limb" in op.attributes for op in module.walk()) != literal
    module.verify()


def test_unknown_float_times_zero_is_not_an_integer_zero_proof():
    module = parse_mlir_text("""module {
      func.func @forward(%x:tensor<2xf32>) -> tensor<2xi32> {
        %e=tensor.empty() : tensor<2xi8>
        %p=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x:tensor<2xf32>) outs(%e:tensor<2xi8>) {
        ^bb0(%v:f32,%out:i8):
          %z=arith.constant 0.0 : f32
          %m=arith.mulf %v,%z : f32
          %i=arith.fptosi %m : f32 to i8
          linalg.yield %i : i8
        } -> tensor<2xi8>
        %dest=tensor.empty() : tensor<2xi32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%p:tensor<2xi8>) outs(%dest:tensor<2xi32>) attrs={test.limb="integer"} {
        ^bb1(%v:i8,%out:i32):
          %i=arith.extsi %v : i8 to i32
          linalg.yield %i : i32
        } -> tensor<2xi32>
        return %r : tensor<2xi32>
      }}""")
    assert fold_literal_zero_components(module, "test.limb") == 0


def test_zero_result_does_not_erase_an_opaque_call():
    module = parse_mlir_text("""module {
      func.func private @effect() -> i8
      func.func @forward(%x:tensor<2xi8>) -> tensor<2xi32> {
        %e=tensor.empty() : tensor<2xi32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x:tensor<2xi8>) outs(%e:tensor<2xi32>) attrs={test.limb="unproved"} {
        ^bb0(%a:i8,%out:i32):
          %effect=func.call @effect() : () -> i8
          %zero=arith.constant 0 : i32
          linalg.yield %zero : i32
        } -> tensor<2xi32>
        return %r : tensor<2xi32>
      }}""")
    assert fold_literal_zero_components(module, "test.limb") == 0
    assert any(op.name == "func.call" for op in module.walk())
