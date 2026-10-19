"""Native roles are derived from source scalar dataflow, never overload names."""

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_unit_policy import construction_evidence


def source(expression, *, literal=False, reduction=False):
    return parse_mlir_text(
        '''module {
      func.func @forward(%x: tensor<3xf32>,%y: tensor<3xf32>) -> tensor<3xf32> {
        %zero=arith.constant 0.0 : f32
        %one=arith.constant 1.0 : f32
        %ones=tensor.splat %one : tensor<3xf32>
        %e=tensor.empty() : tensor<3xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["'''
        + ("reduction" if reduction else "parallel")
        + """"]}
          ins(%x,"""
        + ("%ones" if literal else "%y")
        + """ : tensor<3xf32>,tensor<3xf32>) outs(%e : tensor<3xf32>) {
          ^bb0(%a: f32,%b: f32,%c: f32):
            """
        + expression
        + """
            linalg.yield %v : f32
          } -> tensor<3xf32>
        func.return %r : tensor<3xf32>
      }}"""
    )


def test_parallel_runtime_product_is_not_a_native_contraction():
    result = construction_evidence(source("%v=arith.mulf %a,%b : f32"))
    assert result["status"] == "host"
    assert result["evidence"][0]["reason"] == "parallel_runtime_product_construction"


def test_literal_scale_and_own_linear_math_are_native():
    assert construction_evidence(source("%v=arith.mulf %a,%b : f32", literal=True))["status"] == "eligible"
    assert construction_evidence(source("%v=arith.addf %a,%b : f32"))["status"] == "eligible"


def test_true_reduction_product_retains_contraction_role():
    assert construction_evidence(source("%v=arith.mulf %a,%b : f32", reduction=True))["status"] == "eligible"


def test_rounding_cannot_be_relabelled_as_component_math():
    result = construction_evidence(source("%v=math.floor %a : f32"))
    assert result["status"] == "host"
    assert result["evidence"][0]["reason"] == "nonlinear_rounding_construction"


def test_literal_expression_is_not_mistaken_for_runtime_multiplier():
    assert (
        construction_evidence(source("%n=arith.negf %b : f32\n%v=arith.mulf %a,%n : f32", literal=True))["status"]
        == "eligible"
    )
