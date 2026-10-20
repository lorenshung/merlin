"""A literal inside one scalar boundary has the same explicit tensor operand."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.pointwise_primitives import expose_pointwise_primitives


@pytest.mark.parametrize("operation", ["mulf", "divf", "addf", "subf"])
def test_literal_weight_is_exposed_without_changing_the_typed_scalar_boundary(operation):
    module = parse_mlir_text(f"""module {{func.func @forward(%x:tensor<4xf32>) -> tensor<4xf32> {{
      %e=tensor.empty() : tensor<4xf32>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],iterator_types=["parallel"]}}
        ins(%x:tensor<4xf32>) outs(%e:tensor<4xf32>) {{
        ^bb0(%a:f32,%unused:f32):
        %c=arith.constant 1.0 : f32
        %v=arith.{operation} %a,%c : f32
        linalg.yield %v : f32
      }} -> tensor<4xf32>
      return %r : tensor<4xf32>
    }} }}""")
    assert expose_pointwise_primitives(module) == 1
    roots = [op for op in module.walk() if op.name == "linalg.generic"]
    assert len(roots) == 1 and len(roots[0].inputs) == 2
    assert roots[0].inputs[1].owner.name == "tensor.splat"
    assert roots[0].body.block.first_op.name == "arith." + operation
    assert list(roots[0].body.block.first_op.operands) == list(roots[0].body.block.args[:2])
    module.verify()
