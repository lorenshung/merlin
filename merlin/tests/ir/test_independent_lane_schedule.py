from dataclasses import replace

import pytest

from merlin.llvmlower.independent_lane_schedule import LaneEffects, LaneOperation, independent_lane_schedule

E = LaneEffects(True, True, True, True)
OPS = (
    LaneOperation("p", ("x", "s"), "f64", "rup"),
    LaneOperation("r", ("p", "e"), "f64", "rup"),
    LaneOperation("y", ("x", "r"), "f64", "rdn"),
)


@pytest.mark.parametrize("lanes", [1, 2, 5, 8])
def test_projection(lanes):
    old = independent_lane_schedule(OPS, ("x", "s", "e"), lanes=lanes, effects=E)
    new = independent_lane_schedule(OPS, ("x", "s", "e"), lanes=lanes, effects=E, stage_major=True)
    assert sorted(old) == sorted(new)
    for lane in range(lanes):
        assert [op for l, op in old if l == lane] == [op for l, op in new if l == lane] == list(range(3))


@pytest.mark.parametrize("field", list(vars(E)))
def test_effect_refusal(field):
    with pytest.raises(ValueError):
        independent_lane_schedule(OPS, ("x", "s", "e"), lanes=8, effects=replace(E, **{field: False}), stage_major=True)


@pytest.mark.parametrize("bad", [True, 0, -1, 1.5])
def test_lane_refusal(bad):
    with pytest.raises(ValueError):
        independent_lane_schedule(OPS, ("x", "s", "e"), lanes=bad, effects=E)


@pytest.mark.parametrize(
    "ops",
    [
        (),
        (LaneOperation("p", ("future",), "f64", "rup"),),
        (LaneOperation("x", ("s",), "f64", "rup"),),
        (LaneOperation("p", ("x",), "f64", "unknown"),),
    ],
)
def test_bad_ssa(ops):
    with pytest.raises(ValueError):
        independent_lane_schedule(ops, ("x", "s", "e"), lanes=8, effects=E)
