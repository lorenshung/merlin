"""Preparation identity follows source coordinates, roots and typed ownership."""

import pytest
from xdsl.dialects import builtin, func, tensor
from xdsl.ir import Block, Region

from merlin.llvmlower.tensor_preparation_identity import (
    TensorPreparationRequest,
    canonical_tensor_read_view,
    find_tensor_preparation_opportunities,
    validate_tensor_preparation_opportunity,
    validate_tensor_read_view,
)


def fixture(shape=(3, 17)):
    block = Block(arg_types=[builtin.TensorType(builtin.bf16, shape)] * 2)
    function = func.FuncOp("ordinary", ([v.type for v in block.args], []), Region(block))
    module = builtin.ModuleOp([function])
    return module, block


def slice_value(block, source, offsets, sizes, strides=None, reduce_rank=False):
    operation = tensor.ExtractSliceOp.from_static_parameters(source, offsets, sizes, strides, reduce_rank)
    block.add_op(operation)
    return operation.results[0]


def request(block, value, pin="1" * 64):
    consumer = func.CallOp("opaque_reader", [value], [])
    block.add_op(consumer)
    return TensorPreparationRequest(value, consumer, pin)


def test_nested_and_direct_strided_slices_identify_exact_same_root_coordinates():
    module, block = fixture()
    first = slice_value(block, block.args[0], [0, 1], [3, 8], [1, 2])
    nested = slice_value(block, first, [1, 2], [2, 3], [1, 2])
    direct = slice_value(block, block.args[0], [1, 5], [2, 3], [1, 4])
    a, b = canonical_tensor_read_view(nested), canonical_tensor_read_view(direct)
    assert nested is not direct and a.identity == b.identity
    assert (a.offsets, a.sizes, a.strides) == ((1, 5), (2, 3), (1, 4))
    opportunities = find_tensor_preparation_opportunities([request(block, nested), request(block, direct)])
    (opportunity,) = opportunities
    assert opportunity.owner_block is block and opportunity.format_sha256 == "1" * 64
    validate_tensor_preparation_opportunity(opportunity)


def test_equal_shape_never_merges_distinct_roots_offsets_or_formats():
    module, block = fixture()
    a = slice_value(block, block.args[0], [0, 0], [3, 4])
    b = slice_value(block, block.args[0], [0, 1], [3, 4])
    c = slice_value(block, block.args[1], [0, 0], [3, 4])
    requests = [request(block, a), request(block, b), request(block, c), request(block, a, "2" * 64)]
    assert find_tensor_preparation_opportunities(requests) == ()


@pytest.mark.parametrize(
    "mutation", ["slice", "root_type", "module_context", "function_context", "consumer", "ownership", "order"]
)
def test_retained_opportunity_refuses_mutation(mutation):
    module, block = fixture()
    a = slice_value(block, block.args[0], [0, 0], [3, 4])
    b = slice_value(block, block.args[0], [0, 0], [3, 4])
    (opportunity,) = find_tensor_preparation_opportunities([request(block, a), request(block, b)])
    if mutation == "slice":
        b.owner.properties["static_offsets"] = builtin.DenseArrayBase.from_list(builtin.i64, [0, 1])
    elif mutation == "root_type":
        block.args[0]._type = builtin.TensorType(builtin.bf16, [3, 18])
    elif mutation == "module_context":
        module.attributes["numeric.policy"] = builtin.StringAttr("changed")
    elif mutation == "function_context":
        module.body.block.first_op.attributes["logical.layout"] = builtin.StringAttr("changed")
    elif mutation == "consumer":
        opportunity.consumers[0].properties["callee"] = builtin.SymbolRefAttr("different")
    elif mutation == "ownership":
        operation = opportunity.consumers[0]
        block.detach_op(operation)
        module.body.block.add_op(operation)
    else:
        operation = opportunity.consumers[0]
        block.detach_op(operation)
        block.insert_op_before(operation, a.owner)
    with pytest.raises(ValueError):
        validate_tensor_preparation_opportunity(opportunity)


@pytest.mark.parametrize(
    "shape,offsets,sizes", [((2, 19), (0, 3), (2, 7)), ((5, 8, 11), (1, 2, 3), (3, 4, 6)), ((3, 0), (0, 0), (3, 0))]
)
def test_independent_extents_and_empty_tensor_views(shape, offsets, sizes):
    module, block = fixture(shape)
    value = slice_value(block, block.args[0], offsets, sizes)
    view = canonical_tensor_read_view(value)
    assert (view.offsets, view.sizes, view.strides) == (offsets, sizes, (1,) * len(shape))
    validate_tensor_read_view(view)


def test_rank_reduction_refuses_instead_of_guessing_axis_correspondence():
    module, block = fixture((1, 9))
    value = slice_value(block, block.args[0], [0, 2], [1, 5], reduce_rank=True)
    with pytest.raises(ValueError, match="same-rank"):
        canonical_tensor_read_view(value)


@pytest.mark.parametrize("bad", ["dynamic", "out_of_bounds", "encoding", "memref"])
def test_unsupported_view_refuses(bad):
    if bad == "memref":
        block = Block(arg_types=[builtin.MemRefType(builtin.bf16, [3, 17])])
        value = block.args[0]
    else:
        module, block = fixture()
        if bad == "encoding":
            block.args[0]._type = builtin.TensorType(builtin.bf16, [3, 17], builtin.StringAttr("encoded"))
            value = block.args[0]
        else:
            value = slice_value(block, block.args[0], [0, 0], [3, 4])
            if bad == "dynamic":
                value.owner.properties["static_offsets"] = builtin.DenseArrayBase.from_list(builtin.i64, [0, -1])
            else:
                value.owner.properties["static_offsets"] = builtin.DenseArrayBase.from_list(builtin.i64, [0, 16])
    with pytest.raises(ValueError):
        canonical_tensor_read_view(value)


def test_census_does_not_grant_opaque_call_purity_or_mutate_source():
    from merlin.xdsl_dialects._common import text

    module, block = fixture()
    requests = [request(block, block.args[0]), request(block, block.args[0])]
    before = text(module, generic=True)
    (opportunity,) = find_tensor_preparation_opportunities(requests)
    assert opportunity.views[0].root is block.args[0]
    assert text(module, generic=True) == before


def test_nonconsuming_request_and_crossblock_producer_refuse():
    module, block = fixture()
    valid = request(block, block.args[0])
    with pytest.raises(ValueError, match="direct consumer"):
        find_tensor_preparation_opportunities([TensorPreparationRequest(block.args[1], valid.consumer, "1" * 64)])
    outer = Block(arg_types=[block.args[0].type])
    invalid = request(block, outer.args[0])
    with pytest.raises(ValueError, match="outside"):
        find_tensor_preparation_opportunities([invalid])
