"""Integer and IEEE storage boundaries do not inherit unrelated proof gates."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_dtype_domain import component_semantics
from merlin.targetgen.native_unit_policy import construction_evidence


def quotient(divisor, *, unknown=False):
    operand = '%bad="opaque.carrier"(%x) : (tensor<4xi64>) -> tensor<4xi64>' if unknown else ""
    source = "%bad" if unknown else "%x"
    return parse_mlir_text(
        """module {
      func.func @forward(%x:tensor<4xi64>) -> tensor<4xi64> {
        BEFORE
        %d=arith.constant DIVISOR : i64
        %e=tensor.empty() : tensor<4xi64>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]} ins(SOURCE:tensor<4xi64>) outs(%e:tensor<4xi64>) {
        ^bb0(%a:i64,%unused:i64):
          %v=arith.divsi %a,%d:i64
          linalg.yield %v:i64
        } -> tensor<4xi64>
        return %r:tensor<4xi64>
      }}""".replace("BEFORE", operand)
        .replace("DIVISOR", str(divisor))
        .replace("SOURCE", source)
    )


@pytest.mark.parametrize("divisor", [1, 3, 4, 9223372036854775807])
def test_positive_integer_quotient_needs_no_ieee_certificate(divisor, monkeypatch):
    def unavailable():
        raise AssertionError("integer arithmetic must not consult an IEEE proof")

    monkeypatch.setattr("merlin.targetgen.native_ieee_certificate.certificate", unavailable)
    proof = component_semantics(quotient(divisor))
    assert proof["status"] == "proven"
    assert proof["method"] == "typed_storage_and_modular_ring_v1"
    assert construction_evidence(quotient(divisor))["status"] == "eligible"


@pytest.mark.parametrize("divisor", [0, -1])
def test_undefined_quotient_domain_remains_unproved(divisor):
    assert component_semantics(quotient(divisor))["status"] == "refused"
    reasons = {x["reason"] for x in construction_evidence(quotient(divisor))["evidence"]}
    assert "unproved_payload_carrier" in reasons


def test_known_quotient_cannot_authenticate_an_unknown_input_carrier():
    proof = construction_evidence(quotient(4, unknown=True))
    assert proof["status"] == "host"
    assert any(x["reason"] == "unproved_payload_carrier" for x in proof["evidence"])


@pytest.mark.parametrize("target", ["f16", "bf16"])
def test_affine_data_followed_by_rne_storage_conversion_is_not_offset_rounding(target):
    module = parse_mlir_text(
        """module {
      func.func @forward(%x:tensor<4xf32>) -> tensor<4xTARGET> {
        %e=tensor.empty():tensor<4xTARGET>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]} ins(%x:tensor<4xf32>) outs(%e:tensor<4xTARGET>) {
        ^bb0(%a:f32,%unused:TARGET):
          %c=arith.constant 0.5:f32
          %b=arith.addf %a,%c:f32
          %v=arith.truncf %b:f32 to TARGET
          linalg.yield %v:TARGET
        } -> tensor<4xTARGET>
        return %r:tensor<4xTARGET>
      }}""".replace("TARGET", target)
    )
    assert construction_evidence(module, excluded=["readout_offset_rounding"])["status"] == "eligible"


def test_biased_integer_rounding_still_refuses():
    module = parse_mlir_text("""module {
      func.func @forward(%x:f32) -> i32 {
        %c=arith.constant 0.5:f32
        %b=arith.addf %x,%c:f32
        %r=arith.fptosi %b:f32 to i32
        return %r:i32
      }}""")
    assert construction_evidence(module, excluded=["readout_offset_rounding"])["status"] == "host"


def test_boolean_union_reduction_is_a_hack_exclusion():
    module = parse_mlir_text("""module {
      func.func @forward(%x:tensor<4xi1>) -> tensor<i1> {
        %z=arith.constant false
        %s=tensor.splat %z:tensor<i1>
        %r=linalg.reduce ins(%x:tensor<4xi1>) outs(%s:tensor<i1>) dimensions=[0]
          (%a:i1,%b:i1) { %v=arith.ori %a,%b:i1
            linalg.yield %v:i1 }
        return %r:tensor<i1>
      }}""")
    assert construction_evidence(module, excluded=["multi_op_identity"])["status"] == "host"


def test_boolean_storage_carriers_remain_pure_movement():
    module = parse_mlir_text("""module {
      func.func @forward(%x:tensor<2x2xi1>) -> tensor<4xi1> {
        %r=tensor.collapse_shape %x [[0,1]]:tensor<2x2xi1> into tensor<4xi1>
        return %r:tensor<4xi1>
      }}""")
    assert construction_evidence(module)["status"] == "eligible"
