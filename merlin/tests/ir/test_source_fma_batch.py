"""No implicit precision, effect or finite-domain permission for FMA batching."""

from dataclasses import fields, replace

import pytest

from merlin.llvmlower.source_fma_batch import (
    SourceFmaBatchContract,
    emit_source_fma_batch_permission,
)

GOOD = SourceFmaBatchContract(8, *([True] * 9))


def test_complete_private_source_permission():
    permission = emit_source_fma_batch_permission(GOOD)
    assert "MERLIN_ENABLE_SOURCE_FMA_BATCH_8" in permission
    assert "explicit independent source FMA batch provider required" in permission


@pytest.mark.parametrize("lanes", [True, 0, 4, 16, 8.0])
def test_unknown_width_refuses(lanes):
    with pytest.raises(ValueError):
        emit_source_fma_batch_permission(replace(GOOD, lanes=lanes))


@pytest.mark.parametrize("field", [f.name for f in fields(GOOD) if f.name != "lanes"])
@pytest.mark.parametrize("value", [False, None, 1])
def test_missing_numeric_or_observation_obligation_refuses(field, value):
    with pytest.raises(ValueError):
        emit_source_fma_batch_permission(replace(GOOD, **{field: value}))


def test_untyped_permission_refuses():
    with pytest.raises(ValueError):
        emit_source_fma_batch_permission({"lanes": 8})
