"""Independent shapes/tails and quota overflow for logical row composition."""

import math
from dataclasses import replace

import pytest

from merlin.llvmlower.closed_row_domain import plan_row_partition
from merlin.llvmlower.row_partition_storage import plan_dense_row_scatter, plan_partitioned_consumer_leases


@pytest.mark.parametrize(
    "shape,axis,block",
    [((33,), 0, 16), ((2, 33, 7), 1, 16), ((3, 5, 17), 2, 6), ((17, 4, 3), 0, 8), ((1, 12, 256, 64), 2, 16)],
)
def test_every_dense_element_scattered_once_in_original_order(shape, axis, block):
    p = plan_dense_row_scatter(shape, axis, plan_row_partition(shape[axis], block))
    p.validate()
    observed = {}
    for copy in p.copies:
        for i in range(copy.elements):
            assert copy.destination_offset + i not in observed
            observed[copy.destination_offset + i] = (copy.block, copy.source_offset + i)
    assert set(observed) == set(range(math.prod(shape)))
    assert p.private_scatter_elements == p.output_elements == math.prod(shape)
    assert all(math.prod(p.block_shapes[c.block]) >= c.source_offset + c.elements for c in p.copies)
    with pytest.raises(ValueError, match="contents"):
        replace(p, copies=p.copies[:-1]).validate()


@pytest.mark.parametrize("shape,axis", [((2, 5), -1), ((2, 5), 2), ((2, 0), 1), ((2, 5), True), ((2, 6), 1)])
def test_unrelated_or_invalid_row_storage_refuses(shape, axis):
    with pytest.raises(ValueError):
        plan_dense_row_scatter(shape, axis, plan_row_partition(5, 2))


def test_explicit_quartet_expands_to_exact_inner_lease_order():
    p = plan_partitioned_consumer_leases(4, 16, ordinal_limit=2**64 - 1)
    p.validate()
    assert p.inner_consumers == 64
    assert [p.inner_ordinal(c, b) for c in range(4) for b in range(16)] == list(range(64))
    for c, b in [(4, 0), (0, 16), (-1, 0), (True, 0)]:
        with pytest.raises(ValueError):
            p.inner_ordinal(c, b)
    with pytest.raises(ValueError, match="contents"):
        replace(p, inner_consumers=63).validate()


@pytest.mark.parametrize("consumers,blocks,limit", [(2**60, 16, 2**64 - 1), (0, 16, 99), (4, False, 99), (4, 16, 0)])
def test_lease_overflow_or_unknown_scope_refuses(consumers, blocks, limit):
    with pytest.raises(ValueError):
        plan_partitioned_consumer_leases(consumers, blocks, ordinal_limit=limit)
