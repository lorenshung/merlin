"""General excluded constructions read from IR properties, never from operator names."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.native_unit_policy import construction_evidence


def contraction(table):
    return parse_mlir_text(f"""module {{ func.func @forward(%x: tensor<2x4xf32>) -> tensor<2x3xf32> {{
      %w = arith.constant dense<{table}> : tensor<4x3xf32>
      %z = arith.constant 0.0 : f32
      %e = tensor.splat %z : tensor<2x3xf32>
      %r = linalg.generic {{indexing_maps = [affine_map<(i,j,k)->(i,k)>, affine_map<(i,j,k)->(k,j)>, affine_map<(i,j,k)->(i,j)>],
                            iterator_types = ["parallel", "parallel", "reduction"]}}
        ins(%x, %w : tensor<2x4xf32>, tensor<4x3xf32>) outs(%e : tensor<2x3xf32>) {{
        ^bb0(%a: f32, %b: f32, %acc: f32):
          %p = arith.mulf %a, %b : f32
          %s = arith.addf %acc, %p : f32
          linalg.yield %s : f32
      }} -> tensor<2x3xf32>
      func.return %r : tensor<2x3xf32>
    }} }}""")


@pytest.mark.parametrize(
    "table",
    [
        # DFT twiddles (real part, n = 4): +-1/4 and 0.
        "[[0.25,0.25,0.25],[0.25,0.0,-0.25],[0.25,-0.25,0.25],[0.25,0.0,-0.25]]",
        # Interpolation weights.
        "[[0.75,0.25,0.0],[0.25,0.75,0.0],[0.0,0.75,0.25],[0.0,0.25,0.75]]",
    ],
)
def test_contraction_against_a_multi_valued_literal_table_is_excluded(table):
    result = construction_evidence(contraction(table), excluded=["nonuniform_literal_coefficient_contraction"])
    assert result["status"] == "host"
    assert {e["reason"] for e in result["evidence"]} == {"nonuniform_literal_coefficient_contraction"}


@pytest.mark.parametrize(
    "table",
    [
        "[[1.0,0.0,0.0],[1.0,1.0,0.0],[1.0,1.0,1.0],[0.0,1.0,1.0]]",  # 0/1 selection and prefix sums
        "[[0.5,0.0,0.0],[0.0,0.5,0.0],[0.0,0.0,0.5],[0.0,0.0,0.0]]",  # scaled identity
        "1.0",  # ones (window sums)
    ],
)
def test_selection_sum_and_single_scale_tables_remain_eligible(table):
    result = construction_evidence(contraction(table), excluded=["nonuniform_literal_coefficient_contraction"])
    assert result["status"] == "eligible", result


def division(divisor, op="arith.divf"):
    return parse_mlir_text(f"""module {{ func.func @forward(%x: tensor<4xf32>) -> tensor<4xf32> {{
      %d = arith.constant {divisor} : f32
      %ds = tensor.splat %d : tensor<4xf32>
      %e = tensor.empty() : tensor<4xf32>
      %r = linalg.generic {{indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
                            iterator_types = ["parallel"]}}
        ins(%x, %ds : tensor<4xf32>, tensor<4xf32>) outs(%e : tensor<4xf32>) {{
        ^bb0(%a: f32, %b: f32, %u: f32):
          %q = "{op}"(%a, %b) : (f32, f32) -> f32
          linalg.yield %q : f32
      }} -> tensor<4xf32>
      func.return %r : tensor<4xf32>
    }} }}""")


@pytest.mark.parametrize("op", ["arith.divf", "arith.remf"])
@pytest.mark.parametrize("divisor", ["0.0", "-0.0"])
def test_division_by_a_closed_zero_forces_ieee_specials(op, divisor):
    result = construction_evidence(division(divisor, op), excluded=["forced_ieee_special_division"])
    assert result["status"] == "host"
    assert {e["reason"] for e in result["evidence"]} == {"forced_ieee_special_division"}


@pytest.mark.parametrize("op", ["arith.divf", "arith.remf"])
def test_division_by_a_nonzero_literal_is_not_a_forced_special(op):
    result = construction_evidence(division("2.0", op), excluded=["forced_ieee_special_division"])
    assert result["status"] == "eligible"


def test_zero_divisor_in_a_discarded_branch_is_not_forced():
    # Smith-style complex division by a real literal: c/0 appears only in the
    # alternative a closed |re| >= |im| test discards.
    module = parse_mlir_text("""module { func.func @forward(%x: tensor<4xf32>) -> tensor<4xf32> {
      %e = tensor.empty() : tensor<4xf32>
      %r = linalg.generic {indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>], iterator_types = ["parallel"]}
        ins(%x : tensor<4xf32>) outs(%e : tensor<4xf32>) {
        ^bb0(%a: f32, %u: f32):
          %re = arith.constant -2.0 : f32
          %im = arith.constant 0.0 : f32
          %ar = math.absf %re : f32
          %ai = math.absf %im : f32
          %p = arith.cmpf oge, %ar, %ai : f32
          %t = arith.divf %im, %re : f32
          %s = arith.divf %re, %im : f32
          %d1 = arith.mulf %a, %t : f32
          %d2 = arith.mulf %a, %s : f32
          %v = arith.select %p, %d1, %d2 : f32
          linalg.yield %v : f32
      } -> tensor<4xf32>
      func.return %r : tensor<4xf32>
    } }""")
    assert construction_evidence(module, excluded=["forced_ieee_special_division"])["status"] == "eligible"


def reduction(body, accumulate="arith.mulf"):
    return parse_mlir_text(f"""module {{ func.func @forward(%x: tensor<2x3xf32>) -> tensor<3xf32> {{
      %one = arith.constant 1.0 : f32
      %init = tensor.splat %one : tensor<3xf32>
      %r = linalg.reduce ins(%x : tensor<2x3xf32>) outs(%init : tensor<3xf32>) dimensions = [0]
        (%in: f32, %acc: f32) {{
          %p = {accumulate} %in, %acc : f32
          linalg.yield %p : f32
        }}
      func.return %r : tensor<3xf32>
    }} }}""")


def test_product_reduction_over_runtime_elements_is_excluded():
    result = construction_evidence(reduction(None), excluded=["runtime_product_reduction"])
    assert result["status"] == "host"
    assert {e["reason"] for e in result["evidence"]} == {"runtime_product_reduction"}


def test_additive_reduction_is_not_a_product_reduction():
    result = construction_evidence(reduction(None, "arith.addf"), excluded=["runtime_product_reduction"])
    assert result["status"] == "eligible"
