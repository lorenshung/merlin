"""Only the generated admitted coordinate consumer is rescheduled."""

import pytest

from merlin.llvmlower.independent_lane_schedule import LaneEffects
from merlin.llvmlower.radius_stage_schedule import stage_separable_radius_columns

BODY = (
    "#define MERLIN_ENABLE_EXACT_BOUND_CONVERSION 1\n"
    + "if(radius_plan.valid){if(!merlin_fma_separable_radius_apply(&radius_plan,j,&lo[r*n+j],&hi[r*n+j]))return 0;continue;}"
)
EFFECTS = LaneEffects(True, True, True, True)


def test_exact_consumer_preserves_tail_and_refusal():
    result = stage_separable_radius_columns(BODY, effects=EFFECTS)
    assert "n-j>=8" in result and "j+=7;" in result
    assert "else if(!merlin_fma_separable_radius_apply" in result
    assert result.count("return 0;") == 2


@pytest.mark.parametrize(
    "body",
    [BODY + BODY, BODY.replace("CONVERSION 1", "CONVERSION 0"), BODY.replace("radius_plan,j", "radius_plan,j+1"), ""],
)
def test_unknown_consumer_refuses(body):
    with pytest.raises(ValueError):
        stage_separable_radius_columns(body, effects=EFFECTS)


@pytest.mark.parametrize("field", range(4))
def test_missing_effect_proof_refuses(field):
    fields = [True] * 4
    fields[field] = False
    with pytest.raises(ValueError):
        stage_separable_radius_columns(BODY, effects=LaneEffects(*fields))


def test_normal_emitter_selection_is_explicit_and_dependency_checked():
    from merlin.llvmlower.exact_bound_conversion import ExactBoundConversionContract
    from merlin.llvmlower.source_attention_frontier import SourceAttentionFrontierPlan, emit_source_attention_frontier

    plan = SourceAttentionFrontierPlan(
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
    flags = dict(prepare_product_domain=True, prepare_required_norms=True, separable_source_radius=True)
    exact = ExactBoundConversionContract(True, True, True, True, True)
    control = emit_source_attention_frontier(plan, symbol="provider", **flags)
    assert control == emit_source_attention_frontier(plan, symbol="provider", radius_stage_effects=None, **flags)
    for kw in (
        {},
        {"exact_bound_conversion": exact},
        {"separable_source_radius": True, "prepare_product_domain": True, "prepare_required_norms": True},
    ):
        if "exact_bound_conversion" in kw:
            kw["separable_source_radius"] = False
        with pytest.raises(ValueError, match="radius staging"):
            emit_source_attention_frontier(plan, symbol="provider", radius_stage_effects=EFFECTS, **kw)
    selected = emit_source_attention_frontier(
        plan, symbol="provider", exact_bound_conversion=exact, radius_stage_effects=EFFECTS, **flags
    )
    plain = emit_source_attention_frontier(plan, symbol="provider", exact_bound_conversion=exact, **flags)
    assert selected == stage_separable_radius_columns(plain, effects=EFFECTS)
