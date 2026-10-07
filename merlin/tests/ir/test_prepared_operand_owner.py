"""Prepared storage permission cannot outlive typed source/format/epoch evidence."""

import dataclasses

import pytest
from xdsl.dialects import builtin, func
from xdsl.ir import Block, Region

from merlin.llvmlower.prepared_operand_owner import PreparedOperandOwner, PreparedRepresentation, PreparedStorageSpan
from merlin.llvmlower.tensor_preparation_identity import TensorPreparationRequest, find_tensor_preparation_opportunities


def make_owner():
    block = Block(arg_types=[builtin.TensorType(builtin.bf16, [3, 17])])
    module = builtin.ModuleOp([func.FuncOp("source", ([block.args[0].type], []), Region(block))])
    consumers = [func.CallOp("reader", [block.args[0]], []) for _ in range(4)]
    block.add_ops(consumers)
    (opportunity,) = find_tensor_preparation_opportunities(
        [TensorPreparationRequest(block.args[0], op, "1" * 64) for op in consumers]
    )
    representation = PreparedRepresentation(
        "1" * 64,
        "2" * 64,
        "3" * 64,
        (PreparedStorageSpan("planes", 153, 64), PreparedStorageSpan("reconstructed", 408, 8)),
    )

    # Test-only effect validator: actual provider binding must check its proof.
    def checked_effects(opportunity, representation):
        assert len(opportunity.consumers) == 4 and representation.effects_sha256 == "3" * 64

    owner = PreparedOperandOwner(opportunity, representation, validate_effects=checked_effects)
    args = dict(
        epoch=owner.epoch,
        representation=representation,
        capacity=owner.capacity,
        alignment=64,
        spans=owner.layout,
        storage_base=4096,
        live_read_ranges=((8192, 9000),),
    )
    return module, owner, consumers, args


def test_four_consumers_single_owner_then_terminal_epoch():
    module, owner, consumers, args = make_owner()
    assert owner.layout == (("planes", 0, 153), ("reconstructed", 160, 568))
    for consumer in consumers:
        owner.consume(consumer, **args)
    with pytest.raises(ValueError, match="stale"):
        owner.consume(consumers[0], **args)


@pytest.mark.parametrize(
    "bad", ["epoch", "format", "size", "overlap", "alias", "unknown", "reorder", "mutate", "escape"]
)
def test_invalid_or_unknown_proof_refuses(bad):
    module, owner, consumers, args = make_owner()
    consumer = consumers[0]
    if bad == "epoch":
        args["epoch"] = object()
    elif bad == "format":
        args["representation"] = dataclasses.replace(args["representation"], numeric_sha256="4" * 64)
    elif bad == "size":
        args["capacity"] -= 1
    elif bad == "overlap":
        args["spans"] = (("planes", 0, 153), ("reconstructed", 8, 416))
    elif bad == "alias":
        args["live_read_ranges"] = ((4100, 4200),)
    elif bad == "unknown":
        args["live_read_ranges"] = ()
    elif bad == "reorder":
        consumer = consumers[1]
    elif bad == "mutate":
        consumers[2].properties["callee"] = builtin.SymbolRefAttr("mutator")
    else:
        owner.opportunity.owner_block.add_op(func.CallOp("escape", [owner.opportunity.views[0].value], []))
    with pytest.raises(ValueError):
        owner.consume(consumer, **args)


@pytest.mark.parametrize("extent", [True, 0, -1, 1 << 63])
def test_storage_extent_refusal(extent):
    with pytest.raises(ValueError):
        PreparedStorageSpan("x", extent, 8)


def test_storage_and_source_owner_binding_cannot_change_between_consumers():
    module, owner, consumers, args = make_owner()
    owner.consume(consumers[0], **args)
    args["storage_base"] = 16384
    with pytest.raises(ValueError, match="owner changed"):
        owner.consume(consumers[1], **args)
    with pytest.raises(ValueError, match="stale"):
        owner.consume(consumers[1], **args)


def test_empty_range_iterator_refuses_and_invalidates():
    _, owner, consumers, args = make_owner()
    args["live_read_ranges"] = iter(())
    with pytest.raises(ValueError, match="disjointness is unknown"):
        owner.consume(consumers[0], **args)
    args["live_read_ranges"] = ((8192, 9000),)
    with pytest.raises(ValueError, match="stale"):
        owner.consume(consumers[0], **args)


def test_range_iterators_preserve_exact_epoch_binding():
    _, owner, consumers, args = make_owner()
    args["live_read_ranges"] = iter(((8192, 9000),))
    owner.consume(consumers[0], **args)
    args["live_read_ranges"] = iter(((8192, 9000),))
    owner.consume(consumers[1], **args)
    args["live_read_ranges"] = iter(((16384, 17000),))
    with pytest.raises(ValueError, match="owner changed"):
        owner.consume(consumers[2], **args)
