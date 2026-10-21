"""Only zero-iteration domains are folded; nonempty outputs still need DMA."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.structural_folds import fold_empty_reductions


@pytest.mark.parametrize(
    "element,zero",
    [("i1", "0"), ("i8", "0"), ("i32", "0"), ("i64", "0"), ("f16", "0.0"), ("bf16", "0.0"), ("f32", "0.0")],
)
@pytest.mark.parametrize("extent", [0, 1])
def test_empty_reduction_returns_seed_for_every_storage_dtype(element, zero, extent):
    add = "addf" if element in {"f16", "bf16", "f32"} else "addi"
    module = parse_mlir_text(f"""module {{ func.func @forward(%x: tensor<{extent}x{element}>) -> tensor<{element}> {{
      %z=arith.constant {zero} : {element}
      %seed=tensor.splat %z : tensor<{element}>
      %r=linalg.generic {{indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]}}
        ins(%x:tensor<{extent}x{element}>) outs(%seed:tensor<{element}>) {{
        ^bb0(%a:{element},%acc:{element}):
          %v=arith.{add} %a,%acc : {element}
          linalg.yield %v : {element}
        }} -> tensor<{element}>
      func.return %r : tensor<{element}>
    }}}}""")
    assert fold_empty_reductions(module) == int(extent == 0)
    module.verify()
    assert any(op.name == "linalg.generic" for op in module.walk()) == bool(extent)
