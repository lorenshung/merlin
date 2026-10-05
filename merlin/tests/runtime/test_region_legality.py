"""Which consecutive device groups may be offered to a package as ONE region -- a pure dataflow
fact, computed here with lightweight fakes so it is tested apart from real xDSL IR.
"""

from __future__ import annotations

from types import SimpleNamespace

from merlin.llvmlower import region_legality as RL
from merlin.xdsl_dialects.lowering import compute_groups as CG


class _Value:
    """A fake SSA value: only identity matters to `legal_regions`, so nothing else is modeled."""


class _Op:
    def __init__(self, operands=(), results=()):
        self.operands = list(operands)
        self.results = list(results)


def _group(index: int, *, placement: str = "unit0", reads=(), produces: bool = True) -> SimpleNamespace:
    value = _Value() if produces else None
    op = _Op(operands=reads, results=(value,) if value is not None else ())
    return SimpleNamespace(index=index, placement=placement, members=[op]), value


def test_a_chain_of_single_consumer_groups_is_one_legal_region():
    g0, v0 = _group(0, reads=())
    g1, v1 = _group(1, reads=(v0,))
    g2, v2 = _group(2, reads=(v1,))
    assert RL.legal_regions([g0, g1, g2]) == [(0, 2)]


def test_a_value_read_twice_breaks_the_region_at_that_boundary():
    g0, v0 = _group(0, reads=())
    g1, v1 = _group(1, reads=(v0,))
    g2, v2 = _group(2, reads=(v1,))
    # A THIRD reader of g0's own output: the fan-out group, kept OUT of the candidate window by
    # construction (its own read of v0 still counts against g0's consumer count).
    g3 = SimpleNamespace(index=3, placement="unit0", members=[_Op(operands=(v0,))])
    assert RL.legal_regions([g0, g1, g2, g3]) == [(1, 2)]


def test_the_models_own_last_group_may_never_be_a_regions_producing_half():
    """The last device group's product is the model's result -- see `whole_program_buffer`'s own
    convention. A region spanning into it would make that result disappear."""
    g0, v0 = _group(0, reads=())
    g1, v1 = _group(1, reads=(v0,))
    assert RL.legal_regions([g0, g1]) == [(0, 1)]
    # Three groups, chained the same way: still legal, and the LAST one is still excluded from ever
    # being a producing half (there is nothing after it to fold into).
    g2, v2 = _group(2, reads=(v1,))
    assert RL.legal_regions([g0, g1, g2]) == [(0, 2)]


def test_a_host_group_on_either_side_of_a_boundary_is_never_offered():
    g0, v0 = _group(0, placement=CG.HOST, reads=())
    g1, v1 = _group(1, reads=(v0,))
    g2, v2 = _group(2, reads=(v1,))
    # g0 is HOST and excluded outright; only g1/g2 (both device) remain, and they chain legally.
    assert RL.legal_regions([g0, g1, g2]) == [(1, 2)]


def test_no_legal_boundary_at_all_returns_an_empty_list():
    g0, v0 = _group(0, reads=())
    g1, v1 = _group(1, reads=())  # reads nothing of g0's: no dataflow edge between them
    assert RL.legal_regions([g0, g1]) == []


def test_fewer_than_two_device_groups_is_never_a_region():
    g0, _v0 = _group(0, reads=())
    assert RL.legal_regions([g0]) == []
    assert RL.legal_regions([]) == []
