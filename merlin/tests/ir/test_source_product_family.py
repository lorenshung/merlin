"""Exact shared product callback is an explicit source/storage choice."""

from dataclasses import replace

import pytest

from merlin.llvmlower.source_attention_frontier import SourceAttentionFrontierPlan, emit_source_attention_frontier
from merlin.llvmlower.source_product_family import (
    SourceProductFamilyContract,
    prepare_source_product_family,
    source_product_families,
)

CONTRACT = SourceProductFamilyContract(*([True] * 6))
PLAN = SourceAttentionFrontierPlan(
    2,
    3,
    4,
    8,
    3,
    2,
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
FUSED = dict(integer_reconstruction=True, fuse_integer_reconstruction=True)


@pytest.mark.parametrize("field", tuple(vars(CONTRACT)))
@pytest.mark.parametrize("bad", [False, 1, None])
def test_each_completion_domain_and_storage_contract_required(field, bad):
    with pytest.raises(ValueError, match="complete exact source"):
        emit_source_attention_frontier(
            PLAN, symbol="provider", source_product_family=replace(CONTRACT, **{field: bad}), **FUSED
        )


@pytest.mark.parametrize("bad", [True, False, object(), "family"])
def test_implicit_or_untyped_selection_refuses(bad):
    with pytest.raises(ValueError, match="typed source product-family"):
        emit_source_attention_frontier(PLAN, symbol="provider", source_product_family=bad, **FUSED)


def test_default_bytes_and_complete_readout_preserved():
    ordinary = emit_source_attention_frontier(PLAN, symbol="provider", **FUSED)
    assert ordinary == emit_source_attention_frontier(PLAN, symbol="provider", source_product_family=None, **FUSED)
    selected = emit_source_attention_frontier(PLAN, symbol="provider", source_product_family=CONTRACT, **FUSED)
    assert selected == prepare_source_product_family(ordinary, plan=PLAN, contract=CONTRACT)
    assert "w->readout,sizeof(w->readout)/sizeof(int32_t),m,n,k" in selected
    assert "int32_t readout[MERLIN_RADIX_FUSED_INTEGER_GROUPS*ROWS*CHUNK];" in selected
    assert "planes[degree]=w->readout+(size_t)degree*ROWS*CHUNK;" in selected
    assert "merlin_radix_integer_fused_exact_f64(w->center,planes,(size_t)m*n);" in selected
    assert "if(!product(opaque,w->ap" not in selected
    assert "merlin_integer_product_family_callback product" in selected
    with pytest.raises(ValueError, match="already selected"):
        prepare_source_product_family(selected, plan=PLAN, contract=CONTRACT)


def test_all_product_shapes_have_complete_nine_pairs_and_owned_stride():
    families = source_product_families(PLAN)
    assert [(f.rows, f.columns, f.reduction_length) for f in families] == [(3, 8, 4), (3, 4, 3), (3, 4, 2)]
    for family in families:
        assert family.plane_stride == 24
        assert [len(g.pairs) for g in family.groups] == [1, 2, 3, 2, 1]
        assert sum(len(g.pairs) for g in family.groups) == 9
    assert len(source_product_families(replace(PLAN, chunk=6, segment=2))) == 2
    with pytest.raises(ValueError, match="typed complete"):
        source_product_families(object())


def test_missing_storage_and_unknown_source_seams_refuse():
    for flags in ({}, dict(integer_reconstruction=True)):
        with pytest.raises(ValueError, match="complete fused integer readout"):
            emit_source_attention_frontier(PLAN, symbol="provider", source_product_family=CONTRACT, **flags)
    ordinary = emit_source_attention_frontier(PLAN, symbol="provider", **FUSED)
    for source in (
        "",
        ordinary + ordinary,
        ordinary.replace("w->readout[degree],m,n,k,degree", "w->readout[0],m,n,k,degree"),
    ):
        with pytest.raises(ValueError, match="owned complete fused"):
            prepare_source_product_family(source, plan=PLAN, contract=CONTRACT)
