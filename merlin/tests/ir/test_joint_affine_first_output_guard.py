"""Complete first-output rejection proof and portable packed decoder checks."""

import hashlib
from copy import deepcopy

import numpy as np
import pytest
from test_quantized_affine_joint import PREDICTORS, SOURCE, run_decoder

from merlin.llvmlower.quantized_affine_joint import derive_first_output_guard, derive_joint, emit_joint_decoder
from merlin.llvmlower.quantized_affine_pair import predictor_table, source_table

BRACKET = [
    dict(p=298, q=249, scale=0.0032446938566863537, relu=True),
    dict(p=298, q=249, scale=0.0032447930425405502, relu=True),
]


def test_complete_first_value_guard_covers_every_source_mismatch():
    proof = derive_joint(**SOURCE, predictors=BRACKET)
    guard = derive_first_output_guard(proof)
    assert guard["pairs"] == 65536 and not guard["first_exact"]
    assert guard["mask"] == 255 and guard["value"] == 107
    assert guard["inexact_first_raw_byte_values"] == [107]
    assert guard["accepted_raw_byte_values"] == [107]
    first = predictor_table(**proof["predictors"][0])
    expected = source_table(**proof["source"])
    rejected = first.view(np.uint8) & guard["mask"] != guard["value"]
    assert np.array_equal(first[rejected], expected[rejected])
    assert np.sum(first != expected) == 1


@pytest.mark.parametrize("predictors", [BRACKET, PREDICTORS])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_actual_guarded_decoder_all_pairs_masks_tails_and_immutable_second(tmp_path, predictors, packed, offset):
    run_decoder(
        tmp_path, derive_joint(**SOURCE, predictors=predictors), packed=packed, offset=offset, first_output_guard=True
    )


@pytest.mark.parametrize("predictors", [BRACKET, PREDICTORS])
def test_zero_byte_word_predicate_all_lane_masks_and_values(predictors):
    proof = derive_joint(**SOURCE, predictors=predictors)
    guard = derive_first_output_guard(proof)
    mask, value = guard["mask"], guard["value"]
    repeat, high, bounded = 0x0101010101010101, 0x8080808080808080, (1 << 64) - 1
    # Every position/value, every possible acceptance lane mask, and adjacent
    # zero-byte borrow cases. Flag positions can be conservative; rejecting a
    # whole word must never hide any accepted byte. Byte order is irrelevant.
    for other in range(256):
        for lanes in range(256):
            values = [value if lanes & (1 << i) else other for i in range(8)]
            word = sum(byte << (8 * i) for i, byte in enumerate(values))
            difference = (word & (mask * repeat)) ^ (value * repeat)
            accepted = bool(((difference - repeat) & bounded) & (difference ^ bounded) & high)
            assert accepted == any(byte & mask == value for byte in values)


def test_unselective_bit_guard_retains_existing_decoder():
    source = dict(lhs_scale=0.003, rhs_scale=0.004, output_scale=0.002, relu=False)
    predictors = [dict(p=1, q=0, scale=1.0, relu=False), dict(p=0, q=1, scale=1.0, relu=False)]
    proof = derive_joint(**source, predictors=predictors)
    assert derive_first_output_guard(proof)["mask"] == 0
    assert emit_joint_decoder(proof, "joint", first_output_guard=True) == emit_joint_decoder(proof, "joint")


def test_exact_first_predictor_needs_no_decoder_reads(tmp_path):
    source = dict(lhs_scale=0.5, rhs_scale=0.5, output_scale=1.0, relu=False)
    predictor = dict(p=1, q=1, scale=0.5, relu=False)
    proof = derive_joint(**source, predictors=[predictor, predictor])
    guard = derive_first_output_guard(proof)
    assert guard["first_exact"] and not guard["accepted_raw_byte_values"]
    code = emit_joint_decoder(proof, "joint", first_output_guard=True)
    assert "first[i]" not in code and "second[i]" not in code
    run_decoder(tmp_path, proof, packed=True, offset=1, first_output_guard=True)


def test_default_emitted_decoder_bytes_unchanged_and_mutation_refuses():
    proof = derive_joint(**SOURCE, predictors=PREDICTORS)
    code = emit_joint_decoder(proof, "joint")
    assert (
        hashlib.sha256(code.encode()).hexdigest() == "87a1a1aca463d73d965926a320d97303ab9fd6fa74d5e0c993c7504a062f4d22"
    )
    assert code == emit_joint_decoder(proof, "joint", first_output_guard=False)
    changed = deepcopy(proof)
    changed["source_table_sha256"] = "00" * 32
    with pytest.raises(ValueError, match="unchanged complete"):
        derive_first_output_guard(changed)
    with pytest.raises(ValueError, match="boolean"):
        emit_joint_decoder(proof, "joint", first_output_guard=1)
