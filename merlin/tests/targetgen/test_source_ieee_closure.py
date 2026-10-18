"""IEEE closure runs in the declared capture interpreter and binds its observations."""

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pytest

from merlin.targetgen.core_aten_numeric import _source_reference


@pytest.mark.parametrize("operation,status", [("arith.addf", "no_source_nan"), ("arith.mulf", "source_nan")])
def test_source_ieee_closure_uses_declared_interpreter(tmp_path, operation, status):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("declared capture interpreter unavailable")
    source = """module {
      func.func @forward(%x:tensor<1xf32>) -> tensor<1xf32> {
        %e=tensor.empty() : tensor<1xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x:tensor<1xf32>) outs(%e:tensor<1xf32>) {
        ^bb0(%a:f32,%unused:f32):
          %zero=arith.constant 0.0 : f32
          %v=OP %a,%zero : f32
          linalg.yield %v : f32
        } -> tensor<1xf32>
        return %r : tensor<1xf32>
      }}""".replace("OP", operation)
    (tmp_path / "model.mlir").write_text(source)
    np.savez(tmp_path / "inputs.npz", in0=np.array([np.inf], np.float32))
    result = _source_reference(tmp_path, "--ieee-closure")
    assert result.returncode == 0, result.stderr[-2000:]
    receipt = json.loads((tmp_path / "source-ieee-closure.json").read_text())
    assert receipt["status"] == status
    assert receipt["source_sha256"] == hashlib.sha256(source.encode()).hexdigest()
    assert receipt["input_sha256"] == hashlib.sha256((tmp_path / "inputs.npz").read_bytes()).hexdigest()
    assert "no_finite_numerical_qualification" in receipt["scope"]


def test_native_source_observation_retains_every_storage_bit(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("declared capture interpreter unavailable")
    source = """module {
      func.func @forward(%x:tensor<3xf16>) -> (tensor<3xf16>,tensor<3xf16>) {
        return %x,%x : tensor<3xf16>,tensor<3xf16>
      }}"""
    (tmp_path / "model.mlir").write_text(source)
    values = np.array([-0.0, 1.0, -1.0], np.float16)
    np.savez(tmp_path / "inputs.npz", in0=values)
    result = _source_reference(tmp_path, "--native-source")
    assert result.returncode == 0, result.stderr[-2000:]
    receipt = json.loads((tmp_path / "source-ieee-closure.json").read_text())
    assert receipt["output_bytes"] == [values.tobytes().hex()] * 2
    assert receipt["scope"] == "typed_source_ieee_storage_bits_no_target_qualification"


def test_native_source_disagreement_refuses_before_provider_selection(tmp_path):
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
    assert refusal["code"] == "native_source_eager_disagreement"
    assert refusal["source_agreement"][0]["results"] == [
        dict(output_index=0, status="mismatch"),
        dict(output_index=1, status="pass"),
    ]
    assert (tmp_path / "model.mlir").read_text() == source


def test_native_source_unsigned_byte_conversion_has_no_sign_extension(tmp_path):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python or not Path(python).is_file():
        pytest.skip("declared capture interpreter unavailable")
    source = """module {
      func.func @forward(%x:tensor<1xi8>) -> tensor<1xf32> {
        %e=tensor.empty() : tensor<1xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x:tensor<1xi8>) outs(%e:tensor<1xf32>) {
        ^bb0(%a:i8,%unused:f32):
          %v=arith.uitofp %a : i8 to f32
          linalg.yield %v : f32
        } -> tensor<1xf32>
        return %r : tensor<1xf32>
      }}"""
    (tmp_path / "model.mlir").write_text(source)
    np.savez(tmp_path / "inputs.npz", in0=np.array([-1], np.int8))
    result = _source_reference(tmp_path, "--native-source")
    assert result.returncode == 0, result.stderr[-2000:]
    receipt = json.loads((tmp_path / "source-ieee-closure.json").read_text())
    assert receipt["output_bytes"] == [np.array([255.0], np.float32).tobytes().hex()]
