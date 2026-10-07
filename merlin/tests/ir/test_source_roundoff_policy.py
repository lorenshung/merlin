"""Approximate numerical permission is separate from source/effect identity."""

from dataclasses import replace

import pytest

from merlin.llvmlower.closed_group_writer import ClosedGroupWriterContract
from merlin.llvmlower.consumer_observed_group_writer import ConsumerObservedGroupPreparation
from merlin.llvmlower.source_attention_frontier import SourceAttentionFrontierPlan, emit_source_attention_frontier
from merlin.llvmlower.source_roundoff_policy import ApproximateSourceRoundoffPolicy, prepare_source_roundoff_estimates

POLICY = ApproximateSourceRoundoffPolicy(True, True, True, True, True, True, True, True)
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
FLAGS = dict(
    prepare_product_domain=True,
    prepare_required_norms=True,
    separable_source_radius=True,
    prepare_softmax_domain=True,
    prepare_probability_bins=True,
    prepare_encoded_rows=True,
    prepare_softmax_spans=True,
    prepare_probability_points=True,
)


@pytest.mark.parametrize("field", list(vars(POLICY)))
@pytest.mark.parametrize("value", [False, 1, None])
def test_each_explicit_permission_required(field, value):
    with pytest.raises(ValueError, match="explicit approximate"):
        replace(POLICY, **{field: value}).validate()


def test_default_bytes_original_fallback_and_typed_normal_selection():
    ordinary = emit_source_attention_frontier(PLAN, symbol="provider", **FLAGS)
    assert ordinary == emit_source_attention_frontier(PLAN, symbol="provider", source_roundoff_estimate=None, **FLAGS)
    selected = emit_source_attention_frontier(PLAN, symbol="provider", source_roundoff_estimate=POLICY, **FLAGS)
    assert selected == prepare_source_roundoff_estimates(ordinary, policy=POLICY)
    assert "no rigorous source or exact-observer certificate" in selected
    assert (
        ordinary[ordinary.index(" merlin_fma_bound eligibility") : ordinary.index("static int endpoint_intervals")]
        in selected
    )
    assert "source_rms_point_products.h" not in ordinary
    assert "merlin_source_rms4_point_product_estimates" in selected


@pytest.mark.parametrize("bad", [True, False, "rms4", object()])
def test_implicit_or_untyped_selection_refuses(bad):
    with pytest.raises(ValueError, match="typed approximate"):
        emit_source_attention_frontier(PLAN, symbol="provider", source_roundoff_estimate=bad, **FLAGS)


def test_unknown_producer_and_missing_dependencies_refuse():
    with pytest.raises(ValueError, match="complete point"):
        emit_source_attention_frontier(PLAN, symbol="provider", source_roundoff_estimate=POLICY)
    ordinary = emit_source_attention_frontier(PLAN, symbol="provider", **FLAGS)
    for source in (
        "",
        ordinary + ordinary,
        ordinary.replace("const merlin_encoded_row_equality *aproof", "const void *aproof"),
    ):
        with pytest.raises(ValueError, match="owned encoded-row"):
            prepare_source_roundoff_estimates(source, policy=POLICY)
    selected = prepare_source_roundoff_estimates(ordinary, policy=POLICY)
    with pytest.raises(ValueError, match="already selected"):
        prepare_source_roundoff_estimates(selected, policy=POLICY)


def test_approximate_route_cannot_claim_exact_consumer_permission():
    contract = ClosedGroupWriterContract(
        "1" * 64,
        "provider",
        "2" * 64,
        "3" * 64,
        POLICY.numerical_policy,
        "4" * 64,
        8,
        True,
        True,
        True,
        True,
        "rne_returned_values",
        "independent_original_output_gate_required",
    )
    with pytest.raises(ValueError, match="complete producer observation"):
        ConsumerObservedGroupPreparation([contract], [])
