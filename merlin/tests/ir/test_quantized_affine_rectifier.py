import copy

import numpy as np
import pytest

from merlin.llvmlower import quantized_affine_pair as pair
from merlin.llvmlower import quantized_affine_rectifier as rectifier


def _sparse_proof():
    return pair.derive(
        0.011258588172495365,
        0.00940733402967453,
        0.011643771082162857,
        p=298,
        q=249,
        scale=0.0032446938566863537,
        relu=True,
    )


def test_complete_original_ordered_source_relation():
    proof = _sparse_proof()
    plan = rectifier.derive(proof, max_pairs=1)
    assert plan["pairs"] == 65536
    assert plan["corrected_table_sha256"] == proof["source_table_sha256"]
    assert len(plan["relation"]) == 1
    row = plan["relation"][0]
    assert (row["lhs"], row["rhs"], row["correction"]) == (107, 5, 1)
    assert row["indicator_pairs"] == 1
    assert row["indicator"]["range"] == [0, 1]
    assert rectifier.validate(plan) is plan


@pytest.mark.parametrize("relu", [False, True])
def test_exact_predictor_needs_no_correction(relu):
    proof = pair.derive(1, 0.5, 1, p=2, q=1, scale=0.5, relu=relu)
    plan = rectifier.derive(proof, max_pairs=0)
    assert plan["relation"] == []
    assert plan["corrected_table_sha256"] == proof["source_table_sha256"]


@pytest.mark.parametrize("lhs,rhs", [(-128, -128), (-128, 127), (127, -128), (127, 127), (0, 0), (-1, 0), (0, -1)])
def test_key_and_clipped_rectifiers_exhaustive_boundaries(lhs, rhs):
    a = np.arange(-128, 128, dtype=np.int64)[:, None]
    b = np.arange(-128, 128, dtype=np.int64)[None, :]
    key, u, v, indicator = rectifier._rectifier(a, b, lhs, rhs)
    assert np.array_equal(indicator, (a == lhs) & (b == rhs))
    assert key.min() >= -65535 and key.max() <= 65535
    assert u.min() == v.min() == 0
    assert u.max() <= 127 and v.max() <= 127


def test_multiple_relation_and_negative_corrections():
    proof = pair.derive(0.2, 0.3, 1, p=2, q=3, scale=0.1, relu=False)
    plan = rectifier.derive(proof, max_pairs=1000)
    assert len(plan["relation"]) == proof["mismatched_pairs"] > 1
    assert {row["correction"] for row in plan["relation"]} == {-1, 1}
    assert plan["corrected_table_sha256"] == proof["source_table_sha256"]


def test_complete_cardinality_refusal_and_changed_source():
    proof = _sparse_proof()
    with pytest.raises(ValueError, match="cardinality"):
        rectifier.derive(proof, max_pairs=0)
    changed = copy.deepcopy(proof)
    changed["source"]["lhs_scale"] *= 2
    with pytest.raises(ValueError, match="unchanged"):
        rectifier.derive(changed, max_pairs=1)


@pytest.mark.parametrize("field,value", [("corrected_table_sha256", "bad"), ("key_range", [0, 0]), ("correction", 2)])
def test_rederive_refuses_changed_network_witness(field, value):
    plan = rectifier.derive(_sparse_proof(), max_pairs=1)
    if field in ("key_range", "correction"):
        plan["relation"][0][field] = value
    else:
        plan[field] = value
    with pytest.raises(ValueError, match="changed"):
        rectifier.validate(plan)


@pytest.mark.parametrize("lhs,rhs", [(-128, -128), (-128, 127), (127, -128), (127, 127), (0, 0), (-1, 0), (0, -1)])
def test_axis_offset_indicators_exact_over_complete_domain(lhs, rhs):
    a = np.arange(-128, 128, dtype=np.int64)[:, None]
    b = np.arange(-128, 128, dtype=np.int64)[None, :]
    offsets, parts, indicator, stages = rectifier._axis_rectifier(a, b, lhs, rhs)
    assert np.array_equal(indicator, (a == lhs) & (b == rhs))
    assert all(-255 <= int(value.min()) <= int(value.max()) <= 255 for value in offsets)
    assert all(0 <= int(value.min()) <= int(value.max()) <= 127 for value in parts)
    assert all(0 <= stage["output_range"][0] <= stage["output_range"][1] <= 1 for stage in stages)


def test_axis_fallback_binds_the_same_original_observation():
    original = rectifier.derive(_sparse_proof(), max_pairs=1)
    fallback = rectifier.derive(_sparse_proof(), max_pairs=1, indicator_family="axis_offsets")
    assert fallback["source_table_sha256"] == original["source_table_sha256"]
    assert fallback["corrected_table_sha256"] == original["corrected_table_sha256"]
    assert fallback["relation"][0]["indicator_pairs"] == 1
    assert [row["seed"] for row in fallback["relation"][0]["offsets"]] == [-107, 107, -5, 5]
    assert rectifier.validate(fallback) is fallback
    fallback["relation"][0]["offsets"][0]["coefficient"] = -1
    with pytest.raises(ValueError, match="changed"):
        rectifier.validate(fallback)
