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
