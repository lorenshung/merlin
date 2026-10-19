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


def test_native_conversion_does_not_grow_import_path():
    import sys

    from merlin.targetgen import _m2m_capture_worker as worker

    before = list(sys.path)
    for _ in range(10):
        assert worker._to_native([[1], [2]]) == [[1], [2]]
    assert sys.path == before


def test_scalar_complex_product_keeps_reduction_seed_in_loader():
    """Normalizing the capture must preserve the original eager bits and ABI."""
    from merlin.targetgen.core_aten_capture import loader_source

    tensor = dict(kind='tensor', dtype='complex64', shape=[], stride=[], storage_offset=0,
                  requires_grad=False, values=dict(kind='complex', real=0.0, imag=0.0))
    case = dict(overload='aten.prod.default', arguments=dict(
        args=dict(kind='tuple', items=[tensor]), kwargs={}), expected=tensor)
    source = loader_source(case)
    script = f'''
import torch
scope={{}}
exec({source!r},scope)
model,inputs=scope['get_model_and_inputs']()
for dtype in (torch.complex64,torch.complex128):
 for real in (0.,-0.,1.,-2.,float('inf'),-float('inf'),float('nan')):
  for imag in (0.,-0.,3.,float('inf'),float('nan')):
   x=torch.tensor(complex(real,imag),dtype=dtype)
   expected=torch.ops.aten.prod.default(x)
   actual=model(x)
   assert actual.shape == expected.shape == torch.Size([])
   assert torch.equal(actual.reshape(-1).view(torch.uint8),expected.reshape(-1).view(torch.uint8)), (x,actual,expected)
   assert actual.untyped_storage()._cdata != x.untyped_storage()._cdata
'''
    subprocess.run([str(capture_python()), '-c', script], check=True)
