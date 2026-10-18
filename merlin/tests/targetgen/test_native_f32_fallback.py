"""Finite f32 data need not inherit overflow in a calibration observer."""

import json

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_components import prepare_integer_components


@pytest.mark.parametrize("native", [False, True])
def test_native_f32_is_explicit_and_keeps_the_independent_bit_gate(tmp_path, native):
    module = parse_mlir_text("""module {
      func.func @forward(%x: tensor<3xf32>) -> (tensor<3xf32>,tensor<3xf32>) {
        %e=tensor.empty() : tensor<3xf32>
        %v=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
                          iterator_types=["parallel"]}
          ins(%x : tensor<3xf32>) outs(%e : tensor<3xf32>) {
          ^bb0(%a:f32,%unused:f32):
            %r=arith.negf %a : f32
            linalg.yield %r : f32
        } -> tensor<3xf32>
        return %v,%x : tensor<3xf32>,tensor<3xf32>
      }}""")
    source = str(module) + "\n"
    (tmp_path / "model.mlir").write_text(source)
    x = np.array(
        [np.finfo(np.float32).max, -np.finfo(np.float32).max, np.nextafter(np.float32(0), np.float32(1))], np.float32
    )
    np.savez(tmp_path / "inputs.npz", in0=x)
    record = dict(
        status="bundled",
        input_abi=[dict(dtype="f32")],
        output_indices=[0],
        semantic_boundary=dict(post_indices=[1]),
        native_eager_boundary=dict(
            authority="untransformed_native_cpu_eager",
            output_bytes=[(-x).tobytes().hex()],
            post_input_bytes=[x.tobytes().hex()],
        ),
    )
    called = []

    class Provider:
        def admit_numeric_schedule(self, source, **kwargs):
            called.append(source)
            assert "test.limb" in source
            return dict(status="declined", reason="test deliberately does not grant a device proof")

    decision = prepare_integer_components(
        tmp_path,
        dict(cases=[record]),
        dict(spec={}, host={}),
        Provider(),
        None,
        module,
        limb_attribute="test.limb",
        native_f32=native,
    )
    assert decision is None
    assert bool(called) == native
    if native:
        closure = json.loads((tmp_path / "source-ieee-closure.json").read_text())
        assert closure["output_bytes"] == [(-x).tobytes().hex(), x.tobytes().hex()]
    assert (tmp_path / "model.mlir").read_text() == source
