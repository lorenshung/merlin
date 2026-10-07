"""Owned producer facts preserve defaults and refuse incomplete source closure."""

from dataclasses import replace

import pytest

from merlin.llvmlower.produced_bf16_row_facts import (
    ProducedBF16RowFactsContract,
    prepare_produced_bf16_row_facts,
)
from merlin.llvmlower.source_attention_frontier import SourceAttentionFrontierPlan, emit_source_attention_frontier
from merlin.llvmlower.source_product_family import SourceProductFamilyContract
from merlin.llvmlower.source_roundoff_policy import ApproximateSourceRoundoffPolicy

CONTRACT = ProducedBF16RowFactsContract(*([True] * 12))
POLICY = ApproximateSourceRoundoffPolicy(*([True] * 8))
PLAN = SourceAttentionFrontierPlan(
    2,
    5,
    4,
    12,
    5,
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
    prepare_readonly_rhs=True,
    fuse_encoded_witness=True,
)


@pytest.mark.parametrize("field", list(vars(CONTRACT)))
@pytest.mark.parametrize("value", [False, 1, None])
def test_complete_source_effect_lifetime_permission_required(field, value):
    with pytest.raises(ValueError, match="complete owned BF16"):
        emit_source_attention_frontier(
            PLAN, symbol="provider", produced_bf16_row_facts=replace(CONTRACT, **{field: value}), **FLAGS
        )


@pytest.mark.parametrize("bad", [True, False, "reuse", object()])
def test_no_implicit_or_untyped_permission(bad):
    with pytest.raises(ValueError, match="typed owned BF16"):
        emit_source_attention_frontier(PLAN, symbol="provider", produced_bf16_row_facts=bad, **FLAGS)


def test_default_original_policy_and_scalar_fallback_preserved():
    ordinary = emit_source_attention_frontier(PLAN, symbol="provider", **FLAGS)
    assert ordinary == emit_source_attention_frontier(PLAN, symbol="provider", produced_bf16_row_facts=None, **FLAGS)
    selected = emit_source_attention_frontier(PLAN, symbol="provider", produced_bf16_row_facts=CONTRACT, **FLAGS)
    assert selected == prepare_produced_bf16_row_facts(ordinary, plan=PLAN, contract=CONTRACT)
    assert "source_rms_produced_maximum.h" not in selected
    assert "merlin_bf16_radix_row_widen_produced" in selected
    assert "else ok=merlin_bf16_radix_row_widen" in selected
    assert "rhs->epoch!=epoch||rhs->next!=consumer||consumer>=rhs->uses" in selected
    assert "merlin_source_point_span_matches(points,w->a,(size_t)m*k,epoch)" in selected
    assert "*proof=(merlin_encoded_row_equality)" in selected
    # All nonlinear/source observer stages remain byte identical.
    start, end = "static int endpoint_intervals(", "/* Descriptors are semantic host views"
    assert ordinary[ordinary.index(start) : ordinary.index(end)] in selected


def test_original_approximate_rms_policy_is_separate_and_unchanged():
    ordinary = emit_source_attention_frontier(PLAN, symbol="provider", source_roundoff_estimate=POLICY, **FLAGS)
    selected = emit_source_attention_frontier(
        PLAN, symbol="provider", source_roundoff_estimate=POLICY, produced_bf16_row_facts=CONTRACT, **FLAGS
    )
    assert "no rigorous source or exact-observer certificate" in selected
    assert '#include "source_rms_produced_maximum.h"' in selected
    assert "bmetadata?merlin_source_rms4_point_product_estimates_produced_max" in selected
    assert ":merlin_source_rms4_point_product_estimates(" in selected
    original_fallback = ordinary[
        ordinary.index(" merlin_fma_bound eligibility") : ordinary.index("static int soft_details")
    ]
    assert original_fallback in selected


@pytest.mark.parametrize("missing", ["prepare_probability_points", "prepare_readonly_rhs", "fuse_encoded_witness"])
def test_missing_owned_dependency_refuses(missing):
    flags = dict(FLAGS)
    flags[missing] = False
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="provider", produced_bf16_row_facts=CONTRACT, **flags)


@pytest.mark.parametrize("chunk,segment,depth", [(5, 2, 1), (7, 3, 1)])
def test_metadata_must_fit_proven_dead_storage(chunk, segment, depth):
    plan = replace(PLAN, chunk=chunk, segment=segment, depth=depth, denominator_lanes=1)
    with pytest.raises(ValueError, match="dead private producer storage"):
        emit_source_attention_frontier(plan, symbol="provider", produced_bf16_row_facts=CONTRACT, **FLAGS)


def test_unknown_duplicate_or_changed_producer_grammar_refuses():
    ordinary = emit_source_attention_frontier(PLAN, symbol="provider", **FLAGS)
    changed = [
        "",
        ordinary + ordinary,
        ordinary.replace(
            "if(!MERLIN_SOURCE_ISFINITE(value))return 0;x->source[r*k+z]=value;", "x->source[r*k+z]=value;"
        ),
        ordinary.replace("p[off+j]=point.value;", "p[off+j]=point.value+1;"),
        ordinary.replace("w->a[r*length+z]=h->p[ix];", "w->a[r*length+z]=h->p[ix]+1;"),
    ]
    for text in changed:
        with pytest.raises(ValueError, match="owned"):
            prepare_produced_bf16_row_facts(text, plan=PLAN, contract=CONTRACT)
    selected = prepare_produced_bf16_row_facts(ordinary, plan=PLAN, contract=CONTRACT)
    with pytest.raises(ValueError, match="already selected"):
        prepare_produced_bf16_row_facts(selected, plan=PLAN, contract=CONTRACT)


def test_complete_integer_family_composition_preserves_all_planes():
    family = SourceProductFamilyContract(*([True] * 6))
    selected = emit_source_attention_frontier(
        PLAN,
        symbol="provider",
        produced_bf16_row_facts=CONTRACT,
        source_product_family=family,
        integer_reconstruction=True,
        fuse_integer_reconstruction=True,
        **FLAGS,
    )
    assert "merlin_attention_complete_product_family" in selected
    assert "merlin_bf16_radix_row_widen_produced" in selected
    assert "MERLIN_RADIX_FUSED_INTEGER_GROUPS*ROWS*CHUNK" in selected
