"""Uniform input folding preserves iteration bounds and elementwise values."""

import numpy as np

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.splat_inputs import scalarize_splat_inputs

_SOURCE = """module {
func.func @forward(%x: tensor<2x3xf32>, %s: f32) -> tensor<2x3xf32> {
 %splat = tensor.splat %s : tensor<2x3xf32>
 %empty = tensor.empty() : tensor<2x3xf32>
 %y = linalg.generic {
 indexing_maps = [affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>, affine_map<(d0,d1)->(d0,d1)>],
 iterator_types = ["parallel", "parallel"]}
 ins(%x,%splat : tensor<2x3xf32>, tensor<2x3xf32>) outs(%empty : tensor<2x3xf32>) {
 ^bb0(%a: f32,%b: f32,%out: f32):
 %r = arith.mulf %a,%b : f32
 linalg.yield %r : f32
 } -> tensor<2x3xf32>
 return %y : tensor<2x3xf32>
}
}"""


def test_scalarizes_splat_and_keeps_domain_and_region():
    module = parse_mlir_text(_SOURCE)
    assert scalarize_splat_inputs(module) == 1
    module.verify()
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    assert str(generic.inputs[1].type) == "tensor<f32>"
    assert generic.indexing_maps.data[1].data.results == ()
    assert all(len(op.result.type.get_shape()) == 0 for op in module.walk() if op.name == "tensor.splat")
    assert scalarize_splat_inputs(module) == 0
    # Read both operands through the rewritten affine maps at each iteration.
    # This tests actual map semantics, including the empty scalar index tuple.
    x = np.arange(6, dtype=np.float32).reshape(2, 3) - 2
    scale = np.array(0.125, dtype=np.float32)
    actual = np.empty_like(x)
    for index in np.ndindex(x.shape):
        left = x[generic.indexing_maps.data[0].data.eval(index, ())]
        right = scale[generic.indexing_maps.data[1].data.eval(index, ())]
        actual[index] = left * right
    np.testing.assert_array_equal(actual, x * np.full_like(x, scale))


def test_reduction_domain_is_not_removed():
    module = parse_mlir_text(_SOURCE.replace('["parallel", "parallel"]', '["parallel", "reduction"]'))
    assert scalarize_splat_inputs(module) == 0


def test_missing_identity_output_domain_is_refused():
    source = _SOURCE.replace("affine_map<(d0,d1)->(d0,d1)>],", "affine_map<(d0,d1)->(d1,d0)>],")
    assert scalarize_splat_inputs(parse_mlir_text(source)) == 0


def test_build_preparation_keeps_splat_scalar(tmp_path):
    from merlin.runtime.backends.zephyr_model import _prepare_model_mlir

    source = tmp_path / "model.mlir"
    source.write_text(_SOURCE)
    prepared = _prepare_model_mlir(source, tmp_path)
    module = parse_mlir_text(prepared.read_text())
    module.verify()
    assert all(len(op.result.type.get_shape()) == 0 for op in module.walk() if op.name == "tensor.splat")
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    assert str(generic.inputs[1].type) == "tensor<f32>"


def test_native_specialization_and_lowering_accept_uniform_input(tmp_path):
    import pytest

    from merlin.llvmlower import toolchain
    from merlin.llvmlower.lower import lower_model_file
    from merlin.xdsl_dialects._common import text

    if not toolchain.m2m_python().is_file():
        pytest.skip("native MLIR compiler unavailable")
    module = parse_mlir_text(_SOURCE)
    assert scalarize_splat_inputs(module) == 1
    source = tmp_path / "uniform.mlir"
    source.write_text(text(module))
    result = lower_model_file(source, tmp_path / "lower", targets=(), textual=True)
    assert result.ll_path.is_file()
