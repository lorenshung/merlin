"""Guard proofs depend on declared code domains, never captured examples."""

from fractions import Fraction

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.quantized_guard_domain import Unproved, canonicalize_quantized_affine, prove_affine


def source(*, subtract=False, scalar=False, live=False, scale="0.03125"):
    sign = "-2" if subtract else "2"
    rhs = "%k=tensor.splat %half : tensor<4xf32>" if scalar else ""
    return parse_mlir_text(
        """module {
      func.func @forward(%x:tensor<4xi8>,%y:tensor<4xi8>) -> tensor<4xi8> {
        %s=arith.constant SCALE : f64
        %st=tensor.splat %s : tensor<f64>
        %z=arith.constant -3 : i64
        %zt=tensor.splat %z : tensor<i64>
        %deqa="quant_ext.dequantize_per_tensor"(%x,%st,%zt) {
          prov.quant.storage_dtype="int8",prov.quant.quant_min=-128:i64,prov.quant.quant_max=127:i64,
          prov.quant.zero_point=-3:i64,prov.quant.scale_numerator=["1"],prov.quant.scale_denominator=["32"]
        } : (tensor<4xi8>,tensor<f64>,tensor<i64>)->tensor<4xf32>
        %deqb="quant_ext.dequantize_per_tensor"(%y,%st,%zt) {
          prov.quant.storage_dtype="int8",prov.quant.quant_min=-128:i64,prov.quant.quant_max=127:i64,
          prov.quant.zero_point=-3:i64,prov.quant.scale_numerator=["1"],prov.quant.scale_denominator=["32"]
        } : (tensor<4xi8>,tensor<f64>,tensor<i64>)->tensor<4xf32>
        %half=arith.constant 0.5 : f32
        RHS
        %e=tensor.empty() : tensor<4xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]} ins(%deqa,RHSARG:tensor<4xf32>,tensor<4xf32>) outs(%e:tensor<4xf32>) {
        ^bb0(%a:f32,%b:f32,%unused:f32):
          %two=arith.constant SIGN : i64
          %alpha=arith.sitofp %two : i64 to f32
          %f=math.fma %b,%alpha,%a : f32
          %bits=arith.constant -4194304 : i32
          %nan=arith.bitcast %bits : i32 to f32
          %isnan=arith.cmpf uno,%f,%f : f32
          %safe=arith.select %isnan,%nan,%f : f32
          %p=arith.mulf %b,%alpha : f32
          %separate=arith.addf %p,%a : f32
          %index=linalg.index 0 : index
          %twoidx=arith.constant 2 : index
          %thread=arith.cmpi ult,%index,%twoidx : index
          ALTERNATIVE
          %v=arith.select %thread,%alternative,%safe : f32
          linalg.yield %v : f32
        } -> tensor<4xf32>
        %q="quant_ext.quantize_per_tensor"(%r,%st,%zt) : (tensor<4xf32>,tensor<f64>,tensor<i64>)->tensor<4xi8>
        return %q : tensor<4xi8>
      }}""".replace("SCALE", scale)
        .replace("RHSARG", "%k" if scalar else "%deqb")
        .replace("RHS", rhs)
        .replace("SIGN", sign)
        .replace(
            "ALTERNATIVE",
            "%one=arith.constant 1.0 : f32\n%alternative=arith.addf %separate,%one : f32"
            if live
            else "%zero=arith.constant 0.0 : f32\n%alternative=arith.addf %separate,%zero : f32",
        )
    )


def generic(module):
    return next(op for op in module.walk() if op.name == "linalg.generic")


@pytest.mark.parametrize(
    "subtract,scalar",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["add.Tensor", "add.Scalar", "sub.Tensor", "sub.Scalar"],
)
def test_quantized_guarded_affine_families(subtract, scalar):
    module = source(subtract=subtract, scalar=scalar)
    proof = prove_affine(generic(module))
    assert proof["status"] == "proven"
    assert list(map(Fraction, proof["coefficients"])) == [1, 0 if scalar else -2 if subtract else 2]
    assert Fraction(proof["constant"]) == (-1 if subtract else 1) if scalar else Fraction(proof["constant"]) == 0
    assert proof["dead_classification_selects"] and proof["equal_rational_branch_selects"]
    assert canonicalize_quantized_affine(module, exact_rational=True)
    body = list(generic(module).body.block.ops)
    assert all(op.name in {"arith.constant", "arith.mulf", "arith.addf", "linalg.yield"} for op in body)


def test_live_guard_with_different_results_refuses():
    module = source(live=True)
    with pytest.raises(Unproved, match="live guard"):
        prove_affine(generic(module))
    original = str(module)
    assert canonicalize_quantized_affine(module, exact_rational=True) == []
    assert str(module) == original


def test_qparam_metadata_must_match_executable_scale():
    module = source(scale="0.0625")
    with pytest.raises(Unproved, match="executable operands"):
        prove_affine(generic(module))


def test_missing_domain_is_conditional_only_and_never_a_rewrite():
    module = source()
    op = generic(module)
    # Removing declared metadata leaves an unresolved domain obligation.
    producer = op.inputs[0].owner
    producer.attributes.pop("prov.quant.scale_numerator")
    assert prove_affine(op, conditional=True)["status"] == "conditional"
    assert canonicalize_quantized_affine(module, exact_rational=True) == []


def test_floating_escape_prevents_rational_signed_zero_canonicalization():
    module = source()
    from xdsl.dialects import tensor

    op = generic(module)
    # A native floating consumer observes signed zeros / intermediate RNE.
    escaped = tensor.CastOp(op.results[0], op.results[0].type)
    op.parent_block().insert_op_after(escaped, op)
    assert canonicalize_quantized_affine(module, exact_rational=True) == []


def test_default_has_no_native_ieee_reassociation_permission():
    module = source()
    original = str(module)
    assert canonicalize_quantized_affine(module) == []
    assert str(module) == original
