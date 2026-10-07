"""The source observation/effects contract cannot be inferred from tensor shapes."""

from dataclasses import replace

import pytest

from merlin.llvmlower.bf16_integer_observer import (
    BF16IntegerObserverContract,
    c_header,
    prepare_bf16_integer_observer,
)
from merlin.llvmlower.source_attention_frontier import SourceAttentionFrontierPlan, emit_source_attention_frontier

CONTRACT = BF16IntegerObserverContract(*([True] * 7))
PLAN = SourceAttentionFrontierPlan(
    2,
    3,
    2,
    8,
    3,
    2,
    0.125,
    -87.0,
    1.442695,
    (-0.079, -0.224, 0.303, 0.0001),
    8388608.0,
    1065353216.0,
    127.0,
    0.00001,
    -127,
    127,
)


@pytest.mark.parametrize("field", list(CONTRACT.__dataclass_fields__))
@pytest.mark.parametrize("value", [False, 1, None])
def test_every_observation_and_effect_permission_required(field, value):
    with pytest.raises(ValueError):
        c_header(replace(CONTRACT, **{field: value}))


def test_unknown_or_repeated_source_refuses():
    with pytest.raises(ValueError):
        prepare_bf16_integer_observer("", contract=CONTRACT)
    with pytest.raises(ValueError):
        c_header(True)
    source = prepare_bf16_integer_observer('#include "bf16_quant_frontier.h"', contract=CONTRACT)
    assert "merlin_bf16_rne_clamped_integer_word" in source
    assert " float product=merlin_frontier_bf16(x*inverse);" in source
    assert "nearbyintf(" not in source
    with pytest.raises(ValueError):
        prepare_bf16_integer_observer(source, contract=CONTRACT)
    repeated = prepare_bf16_integer_observer(
        '#include "bf16_quant_frontier.h"\n#include "bf16_quant_frontier.h"', contract=CONTRACT
    )
    assert repeated.count('#include "bf16_quant_frontier.h"') == 1
    assert repeated.count("static inline int merlin_bf16_rne_clamped_integer_word") == 1


def test_normal_emitter_is_explicit_and_default_identity():
    original = emit_source_attention_frontier(PLAN, symbol="p")
    assert original == emit_source_attention_frontier(PLAN, symbol="p", bf16_integer_observer=None)
    assert "merlin_bf16_rne_clamped_integer_word" not in original
    selected = emit_source_attention_frontier(PLAN, symbol="p", bf16_integer_observer=CONTRACT)
    assert "merlin_bf16_rne_clamped_integer_word" in selected
    assert " float product=merlin_frontier_bf16(x*inverse);" in selected
    with pytest.raises(ValueError):
        emit_source_attention_frontier(PLAN, symbol="p", bf16_integer_observer=True)
