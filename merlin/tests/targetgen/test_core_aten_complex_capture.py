"""Complex capture values preserve both components in the physical ABI."""

import subprocess

from merlin.common.paths import repo_root
from merlin.integrations.model2mlir import capture_python


def test_capture_worker_preserves_complex_components():
    source = repo_root() / "src/merlin/targetgen/_m2m_capture_worker.py"
    script = f"""
import importlib.util
import torch
spec=importlib.util.spec_from_file_location("worker", {str(source)!r})
worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
for dtype,spelling in [(torch.complex64,"complex<f32>"),(torch.complex128,"complex<f64>")]:
 value=torch.tensor([[1+2j,-3-4j]],dtype=dtype)
 assert worker._mlir_dtype(dtype)==spelling
 assert worker._to_native(value)==[[dict(kind="complex",real=1.,imag=2.),dict(kind="complex",real=-3.,imag=-4.)]]
"""
    subprocess.run([str(capture_python()), "-c", script], check=True)
