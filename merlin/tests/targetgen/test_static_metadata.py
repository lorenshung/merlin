"""Static metadata specialization never reads tensor payload values."""

import os
import subprocess

from merlin.common.paths import python_import_roots


def _torch(script):
    result = subprocess.run(
        [os.environ["MERLIN_M2M_PYTHON"], "-c", script],
        env={**os.environ, "PYTHONPATH": os.pathsep.join(str(p) for p in python_import_roots())},
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_static_metadata_queries_use_descriptors_and_preserve_inputs():
    _torch('''
import torch
from merlin.capture.static_metadata import specialize_static_metadata
sources=[torch.arange(12.).reshape(3,4), torch.arange(24.).reshape(3,8)[:,1::2],
         torch.ones(1,4).expand(3,4), torch.empty(0,4)]
queries=[(torch.ops.aten.sym_numel.default,()), (torch.ops.aten.sym_size.int,(-1,)),
         (torch.ops.aten.sym_stride.int,(0,)), (torch.ops.aten.sym_storage_offset.default,()),
         (torch.ops.aten.sym_is_contiguous.default,())]
for query,args in queries:
 class Model(torch.nn.Module):
  def forward(self,x): return torch.ops.aten.scalar_tensor.default(query(x,*args))
 for x in sources:
  model=Model()
  folded,receipt=specialize_static_metadata(model,(x,))
  assert receipt["queries"] == [{"operator":str(query),"value":query(x,*args)}]
  assert receipt["inputs"] == [{"shape":list(x.shape),"stride":list(x.stride()),"storage_offset":x.storage_offset()}]
  assert torch.equal(folded(x),model(x))
  assert len([n for n in folded.graph.nodes if n.op=="placeholder"]) == 1
  assert not any(n.target==query for n in folded.graph.nodes)
class Mixed(torch.nn.Module):
 def forward(self,x): return x+torch.ops.aten.sym_numel.default(x)
original=Mixed()
assert specialize_static_metadata(original,(sources[0],)) == (original,None)
''')
