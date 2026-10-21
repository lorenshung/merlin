"""Unproved native IEEE sources refuse before any device proof or eager readout."""

import json
import os
from pathlib import Path

import numpy as np
import pytest


def test_unproved_native_source_refuses_without_reading_eager_or_inputs(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("declared capture interpreter unavailable")
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.core_aten_components import prepare_integer_components

    source = """module {
      func.func @forward(%x:tensor<1xf16>) -> (tensor<1xf16>,tensor<1xf16>) {
        %z=arith.constant 0.0 : f16
        %zeros=tensor.splat %z : tensor<1xf16>
        %e=tensor.empty() : tensor<1xf16>
        %r=linalg.generic {
          indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]
        } ins(%x,%zeros:tensor<1xf16>,tensor<1xf16>) outs(%e:tensor<1xf16>) {
        ^bb0(%a:f16,%b:f16,%unused:f16):
          %v=arith.subf %a,%b : f16
          linalg.yield %v : f16
        } -> tensor<1xf16>
        return %r,%x : tensor<1xf16>,tensor<1xf16>
      }}"""
    (tmp_path / "model.mlir").write_text(source)
    value = np.array([-0.0], np.float16)
    np.savez(tmp_path / "inputs.npz", in0=value)

    class Provider:
        def admit_numeric_schedule(self, *args, **kwargs):
            raise AssertionError("a source fidelity failure must not ask for a device proof")

    # This independent CPU boundary expects +0; the captured equation gives -0.
    # The reference remains qualification data, never a generated readout value.
    record = dict(
        status="bundled",
        input_abi=[dict(dtype="f16")],
        output_indices=[0],
        semantic_boundary=dict(post_indices=[1]),
        native_eager_boundary=dict(
            authority="untransformed_native_cpu_eager", output_bytes=["0000"], post_input_bytes=["0080"]
        ),
    )
    assert (
        prepare_integer_components(
            tmp_path,
            dict(cases=[record]),
            dict(spec={}, host={}),
            Provider(),
            None,
            parse_mlir_text(source),
            limb_attribute="test.limb",
        )
        is None
    )
    refusal = json.loads((tmp_path / "exact-dtype-refusal.json").read_text())
    assert refusal["code"] == "native_component_transfer_unadmitted"
    assert "source_agreement" not in refusal
    assert (tmp_path / "model.mlir").read_text() == source


def test_native_reduced_storage_requires_complete_source_observation(tmp_path):
    if not os.environ.get("MERLIN_M2M_PYTHON"):
        pytest.skip("declared capture interpreter unavailable")
    from merlin.frontends.linalg_mlir import parse_mlir_text
    from merlin.targetgen.core_aten_components import prepare_integer_components

    source = """module {
      func.func @forward(%x:tensor<1xf16>,%y:tensor<1xf16>,%z:tensor<1xf16>) -> tensor<1xf16> {
        %e=tensor.empty() : tensor<1xf16>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,
          affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%y,%z:tensor<1xf16>,tensor<1xf16>,tensor<1xf16>) outs(%e:tensor<1xf16>) {
          ^bb0(%a:f16,%b:f16,%c:f16,%unused:f16):
            %v=math.fma %a,%b,%c : f16
            linalg.yield %v : f16
        } -> tensor<1xf16>
        return %r : tensor<1xf16>
      }}"""
    (tmp_path / "model.mlir").write_text(source)
    np.savez(tmp_path / "inputs.npz", **{f"in{i}": np.array([1], np.float16) for i in range(3)})

    class Provider:
        def admit_numeric_schedule(self, *args, **kwargs):
            raise AssertionError("unknown source bits cannot request device qualification")

    record = dict(
        status="bundled",
        input_abi=[dict(dtype="f16")] * 3,
        output_indices=[0],
        semantic_boundary=dict(post_indices=[]),
        native_eager_boundary=dict(
            authority="untransformed_native_cpu_eager",
            output_bytes=[np.array([2], np.float16).tobytes().hex()],
            post_input_bytes=[],
        ),
    )
    assert (
        prepare_integer_components(
            tmp_path,
            dict(cases=[record]),
            dict(spec={}, host={}),
            Provider(),
            None,
            parse_mlir_text(source),
            limb_attribute="test.limb",
        )
        is None
    )
    refusal = json.loads((tmp_path / "exact-dtype-refusal.json").read_text())
    assert refusal["code"] == "native_parallel_runtime_product_construction"
    assert (tmp_path / "model.mlir").read_text() == source
