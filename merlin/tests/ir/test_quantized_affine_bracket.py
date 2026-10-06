"""Complete collisions, signed transitions and interval optimality witnesses."""

import numpy as np
import pytest

from merlin.llvmlower.quantized_affine_bracket import _readout, derive_bracket
from merlin.llvmlower.quantized_affine_pair import source_table

SOURCE = {
    "lhs_scale": 0.011258588172495365,
    "rhs_scale": 0.00940733402967453,
    "output_scale": 0.011643771082162857,
    "relu": True,
}


def assert_interval_witnesses(proof):
    disjoint = proof["disjoint_interval_witnesses"]
    assert len(disjoint) == proof["minimum_scales_in_corresponding_threshold_family"]
    for earlier, later in zip(disjoint, disjoint[1:]):
        assert earlier["scale_bits_max"] < later["scale_bits_min"]
    selected = [int(np.float32(row["scale"]).view(np.uint32)) for row in proof["predictors"]]
    for row in proof["intervals"]:
        assert any(row["scale_bits_min"] <= bits <= row["scale_bits_max"] for bits in selected)
        for bits in (row["scale_bits_min"], row["scale_bits_max"]):
            assert _readout(row["last_sum"], bits, proof["source"]["relu"]) <= row["output"]
            assert _readout(row["first_sum"], bits, proof["source"]["relu"]) >= row["output"] + 1
        for bits in (row["scale_bits_min"] - 1, row["scale_bits_max"] + 1):
            if 1 <= bits <= 0x7F7FFFFF:
                assert not (
                    _readout(row["last_sum"], bits, proof["source"]["relu"]) <= row["output"]
                    and _readout(row["first_sum"], bits, proof["source"]["relu"]) >= row["output"] + 1
                )


def test_raw_source_collisions_refuse_even_two_readouts_of_same_sum():
    proof = derive_bracket(**SOURCE, p=73, q=61)
    assert proof["raw_conflicting_sums"] == 53
    assert not proof["two_readouts_exact"] and not proof["predictors"]
    expected = source_table(**SOURCE)
    for witness in proof["raw_collision_witnesses"]:
        left, right = witness["pairs"]
        assert left["output"] != right["output"]
        for pair in (left, right):
            assert 73 * pair["lhs"] + 61 * pair["rhs"] == witness["sum"]
            assert expected[pair["lhs"] + 128, pair["rhs"] + 128] == pair["output"]


def test_four_disjoint_threshold_intervals_refuse_two_corresponding_scales():
    proof = derive_bracket(**SOURCE, p=225, q=188)
    assert proof["source_output_monotone"] and proof["raw_conflicting_sums"] == 0
    assert proof["minimum_scales_in_corresponding_threshold_family"] == 4
    assert not proof["two_readouts_exact"] and "more than two" in proof["refusal"]
    assert_interval_witnesses(proof)


def test_complete_injective_affine_domain_two_scale_joint_certificate():
    proof = derive_bracket(**SOURCE, p=298, q=249)
    assert proof["reachable_sums"] == 65536
    assert proof["two_readouts_exact"] and proof["refusal"] is None
    assert proof["minimum_scales_in_corresponding_threshold_family"] == 2
    assert proof["joint_certificate"]["conflicting_tuples"] == 0
    assert proof["joint_certificate"]["different_prediction_pairs"] == 44
    assert_interval_witnesses(proof)


def test_independent_signed_source_single_scale_suffices():
    proof = derive_bracket(0.0625, 0.03125, 0.125, p=2, q=1, relu=False)
    assert proof["single_readout_exact"] and not proof["two_readouts_exact"]
    assert proof["predictors"][0]["scale"] == 0.25
    assert_interval_witnesses(proof)


def test_signed_brackets_cover_negative_and_positive_transitions():
    proof = derive_bracket(**dict(SOURCE, relu=False), p=298, q=249)
    assert proof["source_output_monotone"] and proof["raw_conflicting_sums"] == 0
    assert min(row["output"] for row in proof["intervals"]) == -128
    assert_interval_witnesses(proof)


def test_constant_source_needs_no_threshold_but_one_optional_readout_is_exact():
    proof = derive_bracket(0.0001, 0.0001, 1000.0, p=1, q=1)
    assert proof["minimum_scales_in_corresponding_threshold_family"] == 0
    assert not proof["intervals"] and not proof["disjoint_interval_witnesses"]
    assert proof["single_readout_exact"]


def test_nonfinite_source_domain_refuses():
    with pytest.raises(ValueError, match="finite ordered"):
        derive_bracket(1e38, 1e38, 1.0, p=1, q=1)
