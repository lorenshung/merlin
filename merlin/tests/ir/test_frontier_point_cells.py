"""Explicit point-observation permission and inert ordinary source emission."""

from dataclasses import replace

import pytest

from merlin.llvmlower.frontier_point_cells import (
    FrontierPointCellsContract,
    c_header,
    prepare_frontier_point_cells,
)
from merlin.llvmlower.source_attention_frontier import SourceAttentionFrontierPlan, emit_source_attention_frontier

CONTRACT = FrontierPointCellsContract(True, True, True, True, True, True, True)
PLAN = SourceAttentionFrontierPlan(
    1,
    2,
    8,
    16,
    6,
    4,
    0.125,
    -87.3365478515625,
    1.4426950216293335,
    (-0.079204238951206207, -0.22433836758136749, 0.30354261398315430, 0.00010703434963943437),
    8388608.0,
    1065353216.0,
    127.0,
    float.fromhex("0x1.5p-17"),
    -127,
    127,
)


@pytest.mark.parametrize("field", tuple(vars(CONTRACT)))
@pytest.mark.parametrize("value", [False, 1, None])
def test_each_effect_and_storage_permission_required(field, value):
    with pytest.raises(ValueError, match="complete pure finite-point"):
        c_header(replace(CONTRACT, **{field: value}))


@pytest.mark.parametrize("bad", [True, False, "point", object()])
def test_implicit_or_untyped_selection_refuses(bad):
    with pytest.raises(ValueError, match="typed finite-point"):
        emit_source_attention_frontier(PLAN, symbol="provider", frontier_point_cells=bad)


def test_default_bytes_and_source_owned_seam():
    ordinary = emit_source_attention_frontier(PLAN, symbol="provider")
    assert ordinary == emit_source_attention_frontier(PLAN, symbol="provider", frontier_point_cells=None)
    selected = emit_source_attention_frontier(PLAN, symbol="provider", frontier_point_cells=CONTRACT)
    assert selected == prepare_frontier_point_cells(ordinary, contract=CONTRACT)
    assert "low[i]==high[i]?0:" in selected
    assert "merlin_frontier_row_finite_points(w->rowlo" in selected
    assert "low[i]==high[i]?0:" not in ordinary
    with pytest.raises(ValueError, match="already selected"):
        prepare_frontier_point_cells(selected, contract=CONTRACT)
    for changed in (
        "",
        ordinary + ordinary,
        ordinary.replace("HEADS*DEPTH,quant_plan,w->pending", "DEPTH,quant_plan,w->pending"),
    ):
        with pytest.raises(ValueError, match="owned source frontier"):
            prepare_frontier_point_cells(changed, contract=CONTRACT)


def test_approximate_policy_and_storage_schedules_do_not_imply_permission():
    from merlin.llvmlower.source_roundoff_policy import ApproximateSourceRoundoffPolicy

    with pytest.raises(ValueError, match="typed finite-point"):
        c_header(ApproximateSourceRoundoffPolicy(*([True] * 8)))
