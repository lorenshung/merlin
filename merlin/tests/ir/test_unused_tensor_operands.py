"""Unused literals cannot obscure a complete scalar conversion or copy proof."""

from mlir_fixtures import assert_unchanged, forward, generic, pointwise

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.unused_tensor_operands import prune_unused_tensor_inputs


def test_unused_input_is_removed_with_its_map_and_argument():
    ins = [("%x", "tensor<4xi1>"), ("%c", "tensor<4xf32>")]
    body = generic(ins, "tensor<4xf32>", "%f=arith.uitofp %a : i1 to f32", yields="%f")
    module = forward("%x:tensor<4xi1>,%c:tensor<4xf32>", "tensor<4xf32>", body)
    assert prune_unused_tensor_inputs(module) == 1
    op = next(op for op in module.walk() if op.name == "linalg.generic")
    assert len(op.inputs) == 1 and len(op.get_indexing_maps()) == 2
    assert len(op.body.block.args) == 2
    assert prune_unused_tensor_inputs(module) == 0


def test_live_tensor_dependencies_are_not_pruned():
    assert_unchanged(pointwise(["tensor<4xi32>"], "tensor<4xi32>", "", yields="%a"), prune_unused_tensor_inputs)


def test_unused_scalar_operand_still_defines_a_reduction_iteration_domain():
    module = parse_mlir_text("""module {func.func @forward(%x:tensor<4xf32>) -> tensor<f32> {
      %zero=arith.constant 0.0 : f32
      %seed=tensor.splat %zero : tensor<f32>
      %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->()>],iterator_types=["reduction"]}
        ins(%x:tensor<4xf32>) outs(%seed:tensor<f32>) {
        ^bb0(%unused:f32,%acc:f32):
        linalg.yield %acc : f32
      } -> tensor<f32>
      return %r : tensor<f32>
    }}""")
    assert_unchanged(module, prune_unused_tensor_inputs)
