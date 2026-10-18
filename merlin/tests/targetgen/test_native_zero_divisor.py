"""Native invalid-result NaN bits require an explicit source policy."""

import json

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.integer_float_boundaries import has_literal_zero_divisor
from merlin.targetgen.core_aten_components import prepare_integer_components


def division_module(literal):
    return parse_mlir_text(
        """builtin.module {
      func.func @forward(%x: tensor<1xi32>) -> tensor<1xf32> {
        %zero=arith.constant """
        + literal
        + """ : f32
        %divisor=tensor.splat %zero : tensor<1xf32>
        %empty=tensor.empty() : tensor<1xf32>
        %r=linalg.generic {
          indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]
        } ins(%x,%divisor : tensor<1xi32>,tensor<1xf32>) outs(%empty : tensor<1xf32>) {
        ^bb0(%a: i32,%b: f32,%unused: f32):
          %v=arith.sitofp %a : i32 to f32
          %r=arith.divf %v,%b : f32
          linalg.yield %r : f32
        } -> tensor<1xf32>
        func.return %r : tensor<1xf32>
      }}"""
    )


@pytest.mark.parametrize("literal,expected", [("0.0", True), ("-0.0", True), ("2.0", False)])
def test_uniform_literal_divisors(literal, expected):
    assert has_literal_zero_divisor(division_module(literal)) is expected


def test_zero_divisor_falls_back_before_provider(tmp_path):
    module = division_module("0.0")
    original = str(module) + "\n"
    (tmp_path / "model.mlir").write_text(original)
    np.savez(tmp_path / "inputs.npz", in0=np.array([0], np.int32))

    class Provider:
        def admit_numeric_schedule(self, *args, **kwargs):
            raise AssertionError("undefined native NaN bits cannot request a device proof")

    assert (
        prepare_integer_components(
            tmp_path,
            dict(cases=[dict(input_abi=[dict(dtype="i32")])]),
            dict(spec={}, host={}),
            Provider(),
            None,
            module,
            limb_attribute="test.limb",
        )
        is None
    )
    assert (tmp_path / "model.mlir").read_text() == original
    assert "NaN-payload policy" in json.loads((tmp_path / "exact-dtype-refusal.json").read_text())["reason"]
