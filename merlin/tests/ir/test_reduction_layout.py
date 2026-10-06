"""Reduction layout propagation must preserve every scalar accumulation order."""

from merlin.common.mlir_query import parse
from merlin.llvmlower.layout_propagation import rewrite_module

SOURCE = """builtin.module {
 func.func @f(%x: tensor<1x2x3x4xf32>) -> tensor<1x4xf32> {
  %e = tensor.empty() : tensor<1x4x2x3xf32>
  %t = linalg.transpose ins(%x:tensor<1x2x3x4xf32>) outs(%e:tensor<1x4x2x3xf32>) permutation = [0, 3, 1, 2]
  %scale = arith.constant 0.3 : f32
  %s = tensor.splat %scale : tensor<f32>
  %o = tensor.empty() : tensor<1x4x2x3xf32>
  %dq = linalg.generic {indexing_maps = [affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>, affine_map<(d0,d1,d2,d3)->()>, affine_map<(d0,d1,d2,d3)->(d0,d1,d2,d3)>], iterator_types = ["parallel","parallel","parallel","parallel"]} ins(%t,%s:tensor<1x4x2x3xf32>,tensor<f32>) outs(%o:tensor<1x4x2x3xf32>) {
  ^bb0(%v:f32,%scale_arg:f32,%unused:f32):
    %mul = arith.mulf %v,%scale_arg : f32
    %r = math.roundeven %mul : f32
    linalg.yield %r : f32
  } -> tensor<1x4x2x3xf32>
  %z = arith.constant 0.0 : f32
  %init = tensor.splat %z : tensor<1x4xf32>
  %r = linalg.reduce ins(%dq:tensor<1x4x2x3xf32>) outs(%init:tensor<1x4xf32>) dimensions = [2, 3]
  (%v:f32,%acc:f32) {
    %sum = arith.addf %v,%acc : f32
    linalg.yield %sum : f32
  }
  func.return %r : tensor<1x4xf32>
 }
}"""


def returned(module):
    return next(o for o in module.walk() if o.name == "func.return").operands[0].owner


def test_reduce_consumes_physical_layout_without_changing_scalar_chains():
    m = parse(SOURCE)
    report = rewrite_module(m)
    m.verify()
    reduce = returned(m)
    assert report.ordered_reductions == 1
    assert reduce.name == "linalg.generic"
    assert [it.data.value for it in reduce.iterator_types] == ["parallel", "parallel", "reduction", "reduction"]
    assert str(reduce.indexing_maps.data[0].data) == "(d0, d1, d2, d3) -> (d0, d2, d3, d1)"
    assert reduce.inputs[0].type.get_shape() == (1, 2, 3, 4)
    pointwise = reduce.inputs[0].owner
    assert [o.name for o in pointwise.regions[0].block.ops] == ["arith.mulf", "math.roundeven", "linalg.yield"]
    assert pointwise.inputs[0] is next(o for o in m.walk() if o.name == "func.func").body.block.args[0]
    assert [o.name for o in reduce.regions[0].block.ops] == ["arith.addf", "linalg.yield"]
    assert reduce.outputs[0].owner.name == "tensor.splat"
    assert returned(m).results[0].type.get_shape() == (1, 4)


def test_swapping_reduction_axes_is_refused_even_for_equal_shapes():
    source = SOURCE.replace("2x3", "3x3").replace("permutation = [0, 3, 1, 2]", "permutation = [0, 3, 2, 1]")
    m = parse(source)
    report = rewrite_module(m)
    assert report.ordered_reductions == 0
    assert tuple(returned(m).dimensions.get_values()) == (2, 3)


def test_index_observing_pointwise_and_reducer_are_refused():
    indexed = SOURCE.replace("    %mul =", "    %index = linalg.index 1 : index\n    %mul =")
    effect = SOURCE.replace("builtin.module {", "builtin.module {\n func.func private @effect()").replace(
        "    %sum =", "    func.call @effect() : () -> ()\n    %sum ="
    )
    for source in (indexed, effect):
        m = parse(source)
        report = rewrite_module(m)
        assert report.ordered_reductions == 0


def test_unknown_layout_keeps_reduction_unchanged():
    source = SOURCE.replace("%x: tensor<1x2x3x4xf32>", "%x: tensor<1x4x2x3xf32>")
    start = source.index("  %t =")
    end = source.index("\n", start)
    source = source[:start] + source[end:]
    source = source.replace("ins(%t,%s:", "ins(%x,%s:")
    m = parse(source)
    report = rewrite_module(m)
    assert report.ordered_reductions == 0


def test_swapping_parallel_axes_cannot_reinterpret_result_layout():
    source = SOURCE.replace("permutation = [0, 3, 1, 2]", "permutation = [3, 0, 1, 2]")
    source = source.replace("1x4x2x3", "4x1x2x3").replace("tensor<1x4xf32>", "tensor<4x1xf32>")
    m = parse(source)
    report = rewrite_module(m)
    assert report.ordered_reductions == 0
    assert returned(m).results[0].type.get_shape() == (4, 1)


def test_channel_block_schedule_keeps_spatial_reduction_order():
    m = parse(SOURCE)
    report = rewrite_module(m, reduction_channel_block=2)
    m.verify()
    collapse = returned(m)
    assert collapse.name == "tensor.collapse_shape"
    op = collapse.operands[0].owner
    assert report.blocked_reductions == 1
    assert op.inputs[0].type.get_shape() == (1, 2, 3, 2, 2)
    assert [it.data.value for it in op.iterator_types] == ["parallel", "parallel", "reduction", "reduction", "parallel"]
    assert str(op.indexing_maps.data[0].data) == "(d0, d1, d2, d3, d4) -> (d0, d2, d3, d1, d4)"
    assert str(op.indexing_maps.data[1].data) == "(d0, d1, d2, d3, d4) -> (d0, d1, d4)"
    assert [o.name for o in op.regions[0].block.ops] == ["arith.addf", "linalg.yield"]
    assert collapse.results[0].type.get_shape() == (1, 4)


def test_channel_block_nondivisible_shape_preserves_unblocked_schedule():
    m = parse(SOURCE)
    report = rewrite_module(m, reduction_channel_block=3)
    assert report.ordered_reductions == 1 and report.blocked_reductions == 0
    assert returned(m).name == "linalg.generic"


def test_blocked_rewrite_has_no_detached_operations_in_ssa_use_lists():
    m = parse(SOURCE)
    rewrite_module(m, reduction_channel_block=2)
    operations = set(m.walk())
    for op in operations:
        for result in op.results:
            assert all(use.operation in operations for use in result.uses)
