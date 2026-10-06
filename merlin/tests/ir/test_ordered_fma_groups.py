"""Typed endpoint closure must keep live float uses and source numerical nodes."""

import pytest
from xdsl.dialects import arith, builtin, func, math, tensor
from xdsl.dialects.linalg import ops as linalg
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineDimExpr, AffineMap

from merlin.llvmlower.ordered_fma_groups import analyze_ordered_fma_groups, validate_group_source


def contraction(block, a, b, transposed=False):
    ls, rs = (a.type.get_shape(), b.type.get_shape())
    shape = (*ls[:-2], ls[-2], rs[-2] if transposed else rs[-1])
    typ = builtin.TensorType(builtin.f32, shape)
    zero = arith.ConstantOp(builtin.FloatAttr(0.0, builtin.f32))
    seed = tensor.SplatOp(zero.result, [], typ)
    block.add_ops([zero, seed])
    rank = len(shape)
    prefix = tuple((AffineDimExpr(i) for i in range(rank - 2)))
    m, n, k = (AffineDimExpr(i) for i in (rank - 2, rank - 1, rank))
    maps = [
        builtin.AffineMapAttr(AffineMap(rank + 1, 0, x))
        for x in ((*prefix, m, k), (*prefix, n, k) if transposed else (*prefix, k, n), (*prefix, m, n))
    ]
    body = Block(arg_types=[builtin.bf16, builtin.bf16, builtin.f32])
    ae = arith.ExtFOp(body.args[0], builtin.f32)
    be = arith.ExtFOp(body.args[1], builtin.f32)
    fma = math.FmaOp(ae.result, be.result, body.args[2])
    body.add_ops([ae, be, fma, linalg.YieldOp(fma.result)])
    op = linalg.GenericOp(
        inputs=[a, b],
        outputs=[seed.result],
        body=Region(body),
        indexing_maps=builtin.ArrayAttr(maps),
        iterator_types=builtin.ArrayAttr(
            [
                linalg.IteratorTypeAttr(x)
                for x in (*[linalg.IteratorType.PARALLEL] * rank, linalg.IteratorType.REDUCTION)
            ]
        ),
        result_types=[typ],
    )
    block.add_op(op)
    return op


def elementwise(block, inputs, dtype, build, maps=None):
    shape = inputs[0].type.get_shape()
    typ = builtin.TensorType(dtype, shape)
    empty = tensor.EmptyOp((), typ)
    block.add_op(empty)
    body = Block(arg_types=[*(v.type.element_type for v in inputs), dtype])
    ops, value = build(body.args)
    body.add_ops([*ops, linalg.YieldOp(value)])
    identity = AffineMap(len(shape), 0, tuple((AffineDimExpr(i) for i in range(len(shape)))))
    maps = maps or [identity] * (len(inputs) + 1)
    op = linalg.GenericOp(
        inputs=inputs,
        outputs=[empty.tensor],
        body=Region(body),
        indexing_maps=builtin.ArrayAttr([builtin.AffineMapAttr(m) for m in maps]),
        iterator_types=builtin.ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * len(shape)),
        result_types=[typ],
    )
    block.add_op(op)
    return op


def scale(block, value):

    def body(args):
        c = arith.ConstantOp(builtin.FloatAttr(0.125, builtin.f32))
        mul = arith.MulfOp(args[0], c.result)
        return ([c, mul], mul.result)

    return elementwise(block, [value], builtin.f32, body)


def narrow(block, value):

    def body(args):
        cast = arith.TruncFOp(args[0], builtin.bf16)
        return ([cast], cast.result)

    return elementwise(block, [value], builtin.bf16, body)


def module(block, outputs):
    block.add_op(func.ReturnOp(*outputs))
    ir = builtin.ModuleOp(
        [func.FuncOp("forward", ([v.type for v in block.args], [v.type for v in outputs]), Region(block))]
    )
    ir.verify()
    return ir


def simple(escape=False):
    block = Block(arg_types=[builtin.TensorType(builtin.bf16, [2, 3, 4]), builtin.TensorType(builtin.bf16, [2, 4, 5])])
    root = contraction(block, *block.args)
    scaled = scale(block, root.results[0])
    end = narrow(block, scaled.results[0])
    ir = module(block, [end.results[0], *([scaled.results[0]] if escape else [])])
    return (ir, root, scaled, end)


def test_exact_scale_narrow_closure_retains_source_operations():
    ir, root, scaled, end = simple()
    groups = analyze_ordered_fma_groups(ir)
    assert len(groups) == 1
    g = groups[0]
    assert g.closed_bf16_endpoints and g.contractions == (root,) and (g.operations == (root, scaled, end))
    assert g.bf16_outputs == (end.results[0],) and (not g.live_f32_outputs)
    validate_group_source(g)


def test_live_float_return_refuses_narrow_only_replacement():
    ir, root, scaled, end = simple(True)
    g = analyze_ordered_fma_groups(ir)[0]
    assert not g.closed_bf16_endpoints and g.live_f32_outputs == (scaled.results[0],)
    assert g.bf16_outputs == (end.results[0],)


def test_probability_boundary_remains_internal_when_feeding_pv():
    block = Block(
        arg_types=[
            builtin.TensorType(builtin.bf16, [2, 3, 4]),
            builtin.TensorType(builtin.bf16, [2, 5, 4]),
            builtin.TensorType(builtin.bf16, [2, 5, 6]),
        ]
    )
    qk = contraction(block, *block.args[:2], transposed=True)
    scaled = scale(block, qk.results[0])
    prob = narrow(block, scaled.results[0])
    pv = contraction(block, prob.results[0], block.args[2])
    end = narrow(block, pv.results[0])
    ir = module(block, [end.results[0]])
    groups = analyze_ordered_fma_groups(ir)
    assert len(groups) == 1
    g = groups[0]
    assert g.closed_bf16_endpoints and g.contractions == (qk, pv) and (g.bf16_outputs == (end.results[0],))
    assert prob in g.operations


def test_same_shape_roots_merge_through_actual_add_dataflow():
    block = Block(arg_types=[builtin.TensorType(builtin.bf16, [2, 3, 4]), builtin.TensorType(builtin.bf16, [2, 4, 5])])
    a = contraction(block, *block.args)
    b = contraction(block, *block.args)

    def add(args):
        x = arith.AddfOp(args[0], args[1])
        return ([x], x.result)

    summed = elementwise(block, [a.results[0], b.results[0]], builtin.f32, add)
    end = narrow(block, summed.results[0])
    ir = module(block, [end.results[0]])
    groups = analyze_ordered_fma_groups(ir)
    assert len(groups) == 1 and groups[0].contractions == (a, b) and groups[0].closed_bf16_endpoints


def test_external_call_never_grants_purity_from_its_name():
    ir, root, scaled, end = simple()
    block = root.parent
    call = func.CallOp("pure_exp", [scaled.results[0]], [])
    block.insert_op_before(call, end)
    ir.body.block.add_op(func.FuncOp.external("pure_exp", [scaled.results[0].type], []))
    g = analyze_ordered_fma_groups(ir)[0]
    assert not g.closed_bf16_endpoints and any((user is call for value, user, index in g.unsupported_uses))


@pytest.mark.parametrize("mutation", ["constant", "new_use", "new_scalar"])
def test_source_witness_refuses_mutation(mutation):
    ir, root, scaled, end = simple()
    g = analyze_ordered_fma_groups(ir)[0]
    if mutation == "constant":
        constant = next((x for x in scaled.body.block.ops if isinstance(x, arith.ConstantOp)))
        constant.properties["value"] = builtin.FloatAttr(0.25, builtin.f32)
    elif mutation == "new_use":
        root.parent.insert_op_before(func.CallOp("observe", [scaled.results[0]], []), end)
    else:
        scaled.body.block.insert_op_before(
            arith.ConstantOp(builtin.FloatAttr(1, builtin.f32)), scaled.body.block.last_op
        )
    with pytest.raises(ValueError, match="source changed"):
        validate_group_source(g)


def test_equal_shapes_do_not_merge_unrelated_source_calls():
    block = Block(arg_types=[builtin.TensorType(builtin.bf16, [2, 3, 4]), builtin.TensorType(builtin.bf16, [2, 4, 5])])
    a = contraction(block, *block.args)
    b = contraction(block, *block.args)
    ae = narrow(block, a.results[0])
    be = narrow(block, b.results[0])
    ir = module(block, [ae.results[0], be.results[0]])
    groups = analyze_ordered_fma_groups(ir)
    assert len(groups) == 2 and all((g.closed_bf16_endpoints for g in groups))
    assert {g.contractions for g in groups} == {(a,), (b,)}


def test_source_zero_seed_mutation_refuses_old_group():
    ir, root, scaled, end = simple()
    g = analyze_ordered_fma_groups(ir)[0]
    constant = root.outputs[0].owner.operands[0].owner
    constant.properties["value"] = builtin.FloatAttr(-0.0, builtin.f32)
    with pytest.raises(ValueError, match="source changed"):
        validate_group_source(g)


def test_strict_function_scope_refuses_guarded_route():
    ir, root, scaled, end = simple()
    ir.body.block.first_op.attributes["llvm.strictfp"] = builtin.UnitAttr()
    g = analyze_ordered_fma_groups(ir)[0]
    assert not g.closed_bf16_endpoints and g.live_f32_outputs == (root.results[0],)


@pytest.mark.parametrize("mutation", ["function_strictfp", "module_numeric", "block_owner"])
def test_source_witness_refuses_enclosing_context_mutation(mutation):
    ir, root, scaled, end = simple()
    group = analyze_ordered_fma_groups(ir)[0]
    assert group.closed_bf16_endpoints
    function = ir.body.block.first_op
    if mutation == "function_strictfp":
        function.attributes["llvm.strictfp"] = builtin.UnitAttr()
    elif mutation == "module_numeric":
        ir.attributes["numeric.policy"] = builtin.StringAttr("observed_flags")
    else:
        old_block = function.body.block
        function.body.detach_block(old_block)
        new_region = Region(old_block)
        function.add_region(new_region)
    with pytest.raises(ValueError, match="enclosing source context changed"):
        validate_group_source(group)
