"""Shape-one coordinates are literal; other index payloads stay unproved."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.static_unit_indices import fold_static_unit_indices


def source(axis):
    return parse_mlir_text(f"""module {{func.func @forward() -> tensor<2x1x3xi64> {{
      %e=tensor.empty() : tensor<2x1x3xi64>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0,d1,d2)->(d0,d1,d2)>],
        iterator_types=["parallel","parallel","parallel"]}}
        outs(%e:tensor<2x1x3xi64>) {{
        ^bb0(%unused:i64):
        %i=linalg.index {axis} : index
        %v=arith.index_cast %i : index to i64
        linalg.yield %v : i64
      }} -> tensor<2x1x3xi64>
      return %r : tensor<2x1x3xi64>
    }} }}""")


def test_unit_axis_is_a_uniform_zero_fill():
    module = source(1)
    assert fold_static_unit_indices(module) > 0
    module.verify()
    result = next(op for op in module.walk() if op.name == "func.return").operands[0]
    assert result.owner.name == "tensor.splat"
    assert result.owner.operands[0].owner.value.value.data == 0


@pytest.mark.parametrize("axis", [0, 2])
def test_nontrivial_coordinate_is_not_a_literal(axis):
    module = source(axis)
    before = str(module)
    assert fold_static_unit_indices(module) == 0
    assert str(module) == before
