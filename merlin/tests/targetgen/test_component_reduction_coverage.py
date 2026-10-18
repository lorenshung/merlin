"""A native reduction must offload its numerator, not only its input conversion."""

import json

import numpy as np

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_components import prepare_integer_components


def test_integer_conversion_does_not_admit_a_retained_host_float_sum(tmp_path):
    module = parse_mlir_text("""builtin.module {
      func.func @forward(%x: tensor<4xi64>) -> tensor<f32> {
        %e=tensor.empty() : tensor<4xf32>
        %cast=linalg.generic {
          indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]
        } ins(%x : tensor<4xi64>) outs(%e : tensor<4xf32>) {
        ^bb0(%a: i64,%unused: f32):
          %v=arith.sitofp %a : i64 to f32
          linalg.yield %v : f32
        } -> tensor<4xf32>
        %zero=arith.constant 0.0 : f32
        %seed=tensor.splat %zero : tensor<f32>
        %sum=linalg.generic {
          indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]
        } ins(%cast : tensor<4xf32>) outs(%seed : tensor<f32>) {
        ^bb1(%a: f32,%b: f32):
          %v=arith.addf %a,%b : f32
          linalg.yield %v : f32
        } -> tensor<f32>
        func.return %sum : tensor<f32>
      }}""")
    original = str(module) + "\n"
    (tmp_path / "model.mlir").write_text(original)
    np.savez(tmp_path / "inputs.npz", in0=np.array([1, 2, 3, 4], np.int64))

    class Provider:
        def admit_numeric_schedule(self, *args, **kwargs):
            raise AssertionError("input conversion cannot request a reduction placement proof")

    decision = prepare_integer_components(
        tmp_path,
        dict(cases=[dict(input_abi=[dict(dtype="i64")])]),
        dict(spec={}, host={}),
        Provider(),
        None,
        module,
        limb_attribute="test.limb",
    )
    assert decision is None
    assert (tmp_path / "model.mlir").read_text() == original
    assert (
        json.loads((tmp_path / "exact-dtype-refusal.json").read_text())["reason"]
        == "integer conversion alone is not a device reduction"
    )
