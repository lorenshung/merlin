"""Static loop analysis derives data polynomials without reading tensor payloads."""

from fractions import Fraction

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.static_polynomials import PolynomialRefusal, analyze_static_polynomial


def source(expression):
    module = parse_mlir_text(
        """module {
      func.func @forward(%x: tensor<3xf16>, %y: tensor<3xf16>) -> tensor<1xf32> {
        %e = tensor.empty() : tensor<1xf32>
        %r = linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          outs(%e : tensor<1xf32>) {
        ^bb0(%unused:f32):
          %lo=arith.constant 0 : index
          %hi=arith.constant 3 : index
          %step=arith.constant 1 : index
          %zero=arith.constant 0.0 : f32
          %sum=scf.for %i=%lo to %hi step %step iter_args(%acc=%zero) -> (f32) {
            %a=tensor.extract %x[%i] : tensor<3xf16>
            %b=tensor.extract %y[%i] : tensor<3xf16>
            %af=arith.extf %a : f16 to f32
            %bf=arith.extf %b : f16 to f32
            EXPR
            %s=arith.addf %acc,%p : f32
            scf.yield %s : f32
          }
          linalg.yield %sum : f32
        } -> tensor<1xf32>
        return %r : tensor<1xf32>
      }}""".replace("EXPR", expression)
    )
    return next(op for op in module.walk() if op.name == "linalg.generic")


def test_static_dot_polynomial_matches_independent_rational_sum():
    op = source("%p=arith.mulf %af,%bf : f32")
    atoms, rows = analyze_static_polynomial(op)
    arrays = [[Fraction(1, 2), Fraction(-3, 4), Fraction(5, 8)], [Fraction(-2), Fraction(3), Fraction(4)]]
    tensor_order = list(dict.fromkeys(value for value, _ in atoms))
    values = [arrays[tensor_order.index(value)][index[0]] for value, index in atoms]
    result = sum(coefficient * __import__("math").prod(values[i] for i in key) for key, coefficient in rows[0].items())
    assert result == sum(a * b for a, b in zip(*arrays)) == Fraction(-3, 4)


def test_cubic_data_work_is_refused():
    op = source("%ab=arith.mulf %af,%bf : f32\n%p=arith.mulf %ab,%af : f32")
    with pytest.raises(PolynomialRefusal, match="degree exceeds two"):
        analyze_static_polynomial(op)


def test_static_analysis_has_a_hard_work_budget():
    with pytest.raises(PolynomialRefusal, match="work budget"):
        analyze_static_polynomial(source("%p=arith.mulf %af,%bf : f32"), work_limit=2)
