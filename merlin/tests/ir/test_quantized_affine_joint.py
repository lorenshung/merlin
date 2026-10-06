"""No sampled tuple agreement can grant eligibility to a joint decoder."""

import ctypes
import hashlib
import subprocess
from copy import deepcopy

import numpy as np
import pytest

from merlin.llvmlower.quantized_affine_joint import derive_joint, emit_joint_decoder, nearby_candidates
from merlin.llvmlower.quantized_affine_pair import predictor_table, source_table

SOURCE = dict(
    lhs_scale=0.011258588172495365, rhs_scale=0.00940733402967453, output_scale=0.011643771082162857, relu=True
)
PREDICTORS = [dict(p=73, q=61, scale=0.013245166279375553), dict(p=523, q=437, scale=0.0018487992929294705)]


def test_two_individually_inexact_predictions_jointly_close_entire_domain():
    proof = derive_joint(**SOURCE, predictors=PREDICTORS)
    assert proof["pairs"] == 65536 and proof["conflicting_tuples"] == 0
    assert proof["decoder_exact_for_all_pairs"] and proof["equal_tuple_identity"]
    assert proof["individual_mismatched_pairs"] == [53, 10]
    assert proof["different_prediction_pairs"] == 63
    assert proof["maximum_predictor_byte_delta"] == 1
    assert proof["decoder_format"] == "adjacent_byte_tuple"
    assert len(bytes.fromhex(proof["decoder_hex"])) == 512
    assert proof["backend_predictor_implementation"].startswith("UNKNOWN")
    assert proof["runtime_different_prediction_rate"].startswith("UNKNOWN")


def test_exact_complete_collision_witness_refuses_emission():
    predictors = [PREDICTORS[0], dict(p=79, q=66, scale=0.01224024873226881)]
    proof = derive_joint(**SOURCE, predictors=predictors)
    assert proof["conflicting_tuples"] == 24 and not proof["decoder_exact_for_all_pairs"]
    expected = source_table(**SOURCE)
    tables = [predictor_table(**row, relu=True) for row in predictors]
    for witness in proof["collision_witnesses"]:
        pair = witness["source_pairs"]
        assert pair[0]["output"] != pair[1]["output"]
        for row in pair:
            index = row["lhs"] + 128, row["rhs"] + 128
            assert int(expected[index]) == row["output"]
            assert [int(table[index]) for table in tables] == witness["predictions"]
    with pytest.raises(ValueError, match="conflicting predictor tuple"):
        emit_joint_decoder(proof, "joint")


def run_decoder(tmp_path, proof, *, packed, offset, first_output_guard=False):
    source = emit_joint_decoder(proof, "joint", packed=packed, first_output_guard=first_output_guard)
    c = tmp_path / "decoder.c"
    c.write_text(source)
    library = tmp_path / ("decoder_" + hashlib.sha256(c.read_bytes()).hexdigest() + ".so")
    subprocess.run(["cc", "-O2", "-fPIC", "-shared", str(c), "-o", str(library)], check=True)
    fn = ctypes.CDLL(str(library)).joint
    fn.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    predictions = [predictor_table(**row).ravel() for row in proof["predictors"]]
    expected = source_table(**proof["source"]).ravel()
    first, second = [np.full(65536 + 64, 37, np.int8) for _ in range(2)]
    for a, b in zip((first, second), predictions, strict=True):
        a[offset : offset + 65536] = b
    original_second = second.copy()
    fn(first.ctypes.data + offset, second.ctypes.data + offset, 65536)
    assert np.array_equal(first[offset : offset + 65536], expected)
    assert (first[:offset] == 37).all() and (first[offset + 65536 :] == 37).all()
    assert np.array_equal(second, original_second)
    indices = np.random.default_rng(12).permutation(65536)[:1013]
    first[:] = 37
    second[:] = 37
    for a, b in zip((first, second), predictions, strict=True):
        a[offset : offset + 1013] = b[indices]
    original_second = second.copy()
    fn(first.ctypes.data + offset, second.ctypes.data + offset, 1013)
    assert np.array_equal(first[offset : offset + 1013], expected[indices])
    assert (first[offset + 1013 :] == 37).all() and (first[:offset] == 37).all()
    assert np.array_equal(second, original_second)
    if proof["equal_tuple_identity"] and (predictions[0] != expected).any():
        different = int(np.flatnonzero(predictions[0] != expected)[0])
        equal = int(np.flatnonzero(predictions[0] == predictions[1])[0])
        patterns = np.array([different if mask & (1 << lane) else equal for mask in range(256) for lane in range(8)])
        for a, b in zip((first, second), predictions, strict=True):
            a[offset : offset + 2048] = b[patterns]
        original_second = second.copy()
        fn(first.ctypes.data + offset, second.ctypes.data + offset, 2048)
        assert np.array_equal(first[offset : offset + 2048], expected[patterns])
        assert np.array_equal(second, original_second)


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("offset", [0, 1])
def test_actual_c_all_pairs_word_masks_unaligned_buffers_tails_and_guards(tmp_path, packed, offset):
    run_decoder(tmp_path, derive_joint(**SOURCE, predictors=PREDICTORS), packed=packed, offset=offset)


def test_full_joint_table_when_equal_predictions_do_not_mean_source_identity(tmp_path):
    source = dict(lhs_scale=0.003, rhs_scale=0.004, output_scale=0.002, relu=False)
    predictors = [dict(p=1, q=0, scale=1.0, relu=False), dict(p=0, q=1, scale=1.0, relu=False)]
    proof = derive_joint(**source, predictors=predictors)
    assert proof["decoder_exact_for_all_pairs"] and not proof["equal_tuple_identity"]
    assert proof["decoder_format"] == "full_byte_tuple"
    assert len(bytes.fromhex(proof["decoder_hex"])) == 65536
    run_decoder(tmp_path, proof, packed=True, offset=1)


def test_candidate_guesses_are_source_derived_and_require_separate_eligibility():
    candidates = list(nearby_candidates(**SOURCE, max_denominator=66))
    assert candidates and any((row["p"], row["q"]) == (79, 66) for row in candidates)
    with pytest.raises(ValueError, match="bounded denominator"):
        list(nearby_candidates(**SOURCE, max_denominator=0))


def test_mutated_proof_or_invalid_symbol_refuses():
    proof = derive_joint(**SOURCE, predictors=PREDICTORS)
    changed = deepcopy(proof)
    changed["decoder_hex"] = "01" + changed["decoder_hex"][2:]
    with pytest.raises(ValueError, match="unchanged complete"):
        emit_joint_decoder(changed, "joint")
    with pytest.raises(ValueError, match="identifier"):
        emit_joint_decoder(proof, "while")
    with pytest.raises(ValueError, match="two explicit"):
        derive_joint(**SOURCE, predictors=PREDICTORS[:1])
