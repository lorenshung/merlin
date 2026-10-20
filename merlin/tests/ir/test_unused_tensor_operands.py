"""Unused literals cannot obscure a complete scalar conversion or copy proof."""

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.unused_tensor_operands import prune_unused_tensor_inputs


def test_unused_input_is_removed_with_its_map_and_argument():
    module = parse_mlir_text("""module {func.func @forward(%x:tensor<4xi1>,%c:tensor<4xf32>) -> tensor<4xf32> {
      %e=tensor.empty() : tensor<4xf32>
      %r=linalg.generic {indexing_maps=[
          affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]}
        ins(%x,%c:tensor<4xi1>,tensor<4xf32>) outs(%e:tensor<4xf32>) {
        ^bb0(%a:i1,%unused:f32,%seed:f32):
        %f=arith.uitofp %a : i1 to f32
        linalg.yield %f : f32
      } -> tensor<4xf32>
      return %r : tensor<4xf32>
    }}""")
    assert prune_unused_tensor_inputs(module) == 1
    generic = next(op for op in module.walk() if op.name == "linalg.generic")
    assert len(generic.inputs) == 1 and len(generic.get_indexing_maps()) == 2
    assert len(generic.body.block.args) == 2
    assert prune_unused_tensor_inputs(module) == 0


def test_live_tensor_dependencies_are_not_pruned():
    module = parse_mlir_text("""module {func.func @forward(%x:tensor<4xi32>) -> tensor<4xi32> {
      %e=tensor.empty() : tensor<4xi32>
      %r=linalg.generic {indexing_maps=[
          affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]}
        ins(%x:tensor<4xi32>) outs(%e:tensor<4xi32>) {
        ^bb0(%a:i32,%seed:i32):
        linalg.yield %a : i32
      } -> tensor<4xi32>
      return %r : tensor<4xi32>
    }}""")
    original = str(module)
    assert prune_unused_tensor_inputs(module) == 0 and str(module) == original
