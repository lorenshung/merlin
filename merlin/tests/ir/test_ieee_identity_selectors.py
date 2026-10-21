"""Dropping an identity selector must preserve the source's NaN-quieting path."""

import pytest
from mlir_fixtures import assert_unchanged


@pytest.mark.parametrize("dtype,bits", [("f16", 16), ("bf16", 16), ("f32", 32), ("f64", 64)])
@pytest.mark.parametrize("correct", [False, True])
def test_zero_sign_selector_is_identity_only_with_the_complete_sign_word(dtype, bits, correct):
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.llvmlower.ieee_identity_selectors import fold_zero_sign_selectors

    mask = -(1 << (bits - 1)) if correct else (1 << (bits - 2))
    module = parse_mlir_text(f"""module {{func.func @forward(%x:{dtype}) -> {dtype} {{
      %zero=arith.constant 0.0 : {dtype}
      %mask=arith.constant {mask} : i{bits}
      %raw=arith.bitcast %x : {dtype} to i{bits}
      %sign=arith.andi %raw,%mask : i{bits}
      %signedzero=arith.bitcast %sign : i{bits} to {dtype}
      %p=arith.cmpf oeq,%x,%zero : {dtype}
      %out=arith.select %p,%signedzero,%x : {dtype}
      return %out : {dtype}
    }} }}""")
    assert fold_zero_sign_selectors(module) == int(correct)
    module.verify()


from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.ieee_identity_selectors import fold_ieee_identity_selectors


def source(predicate="ogt", factor="1.0", *, swapped=False, unknown=False):
    choices = "%m,%x" if swapped else "%x," + ("%opaque" if unknown else "%m")
    return parse_mlir_text(f"""module {{func.func @forward(%x:tensor<4xf32>) -> tensor<4xf32> {{
      %z=arith.constant 0.0 : f32
      %one=arith.constant {factor} : f32
      %zt=tensor.splat %z : tensor<4xf32>
      %ot=tensor.splat %one : tensor<4xf32>
      %e=tensor.empty() : tensor<4xf32>
      %be=tensor.empty() : tensor<4xi1>
      %b=linalg.generic {{indexing_maps=[
          affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]}}
        ins(%x,%zt:tensor<4xf32>,tensor<4xf32>) outs(%be:tensor<4xi1>) {{
        ^bb0(%x0:f32,%z0:f32,%u:i1):
        %p=arith.cmpf {predicate},%x0,%z0 : f32
        linalg.yield %p : i1
      }} -> tensor<4xi1>
      %m=linalg.generic {{indexing_maps=[
          affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]}}
        ins(%x,%ot:tensor<4xf32>,tensor<4xf32>) outs(%e:tensor<4xf32>) {{
        ^bb1(%x1:f32,%one1:f32,%u:f32):
        %p=arith.mulf %x1,%one1 : f32
        linalg.yield %p : f32
      }} -> tensor<4xf32>
      {'%opaque="fixture.unknown"(%m) : (tensor<4xf32>) -> tensor<4xf32>' if unknown else ""}
      %r=linalg.generic {{indexing_maps=[
          affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]}}
        ins(%b,{choices}:tensor<4xi1>,tensor<4xf32>,tensor<4xf32>) outs(%e:tensor<4xf32>) {{
        ^bb2(%condition:i1,%plain:f32,%changed:f32,%out:f32):
        %s=arith.select %condition,%plain,%changed : f32
        linalg.yield %s : f32
      }} -> tensor<4xf32>
      return %r : tensor<4xf32>
    }} }}""")


@pytest.mark.parametrize("predicate,swapped", [("ogt", False), ("ole", False), ("ugt", True), ("uno", True)])
def test_nan_path_keeps_the_original_unit_multiply(predicate, swapped):
    module = source(predicate, swapped=swapped)
    assert fold_ieee_identity_selectors(module) == 1
    assert any(op.name == "arith.mulf" for op in module.walk())
    assert not any(op.name in {"arith.cmpf", "arith.select"} for op in module.walk())


@pytest.mark.parametrize("options", [dict(predicate="ugt"), dict(factor="2.0"), dict(unknown=True)])
def test_different_nan_path_arithmetic_or_unknown_carrier_is_retained(options):
    assert_unchanged(source(**options), fold_ieee_identity_selectors)
