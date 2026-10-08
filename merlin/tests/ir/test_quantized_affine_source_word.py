"""Complete relation and independent byte-borrow qualification of source scans."""

import ctypes
from copy import deepcopy

import numpy as np
import pytest
from test_quantized_affine_sparse_predicate import CONTRACTS, library

from merlin.llvmlower.quantized_affine_pair import (
    _source_word_hit_expression,
    derive,
    derive_source_word_guard,
    emit_correction,
    predictor_table,
    source_table,
)

SOURCE_CONTRACTS = [
    *CONTRACTS,
    # Four exceptional lhs bytes share one rhs byte. This exercises choosing
    # and scanning the other axis without a forged certificate or sample bound.
    (dict(lhs_scale=1 / 246, rhs_scale=128, output_scale=1, relu=True), dict(p=0, q=1, scale=128), 4),
]


@pytest.mark.parametrize("source,predictor,count", SOURCE_CONTRACTS)
def test_guard_covers_complete_pair_relation(source, predictor, count):
    proof = derive(**source, **predictor)
    guard = derive_source_word_guard(proof, limit=16)
    assert guard["sparse_cardinality"] == count and guard["pairs"] == 65536
    mismatch = source_table(**source) != predictor_table(**predictor, relu=source["relu"])
    axis = 0 if guard["axis"] == "lhs" else 1
    positions = np.argwhere(mismatch)
    raw = (positions[:, axis] - 128) & 255
    assert sorted(set(map(int, raw))) == guard["raw_byte_values"]
    assert guard["runtime_hit_rate"] == "UNKNOWN"


@pytest.mark.parametrize("values", [[], [0], [1], [128], [255], [0, 1, 255], [2, 129]])
def test_actual_compiled_boolean_all_masks_values_lanes_and_adjacent_borrows(tmp_path, values):
    expression = _source_word_hit_expression(values)
    lib = library(tmp_path, "#include <stdint.h>\nint hit(uint64_t raw){return !!(" + expression + ");}\n")
    hit = lib.hit
    hit.argtypes = [ctypes.c_uint64]
    miss = next(byte for byte in range(256) if byte not in values)
    target = values[0] if values else miss

    def check(lanes):
        raw = int.from_bytes(bytes(lanes), "little")
        assert bool(hit(raw)) == any(byte in values for byte in lanes)

    # Every hit/no-hit mask, including adjacent zero/one subtract borrows.
    for mask in range(256):
        check([target if mask & (1 << lane) else miss for lane in range(8)])
    # Every individual byte in every lane; other lanes deliberately lack a hit.
    for lane in range(8):
        for byte in range(256):
            lanes = [miss] * 8
            lanes[lane] = byte
            check(lanes)
    # All individual adjacent byte pairs, at each possible adjacency. This
    # covers unsigned whole-word borrow chains independently of source scales.
    for first in range(256):
        for second in range(256):
            lanes = [miss] * 8
            lanes[0:2] = [first, second]
            check(lanes)
    for lane in range(7):
        for pair in [(0, 1), (1, 0), (255, 0), (0, 255), (128, 129), (129, 128)]:
            lanes = [miss] * 8
            lanes[lane : lane + 2] = pair
            check(lanes)


@pytest.mark.parametrize("source,predictor,count", SOURCE_CONTRACTS)
@pytest.mark.parametrize("guard", [False, True])
@pytest.mark.parametrize("offset", range(8))
def test_full_source_domain_aligned_unaligned_dirty_tails_guards_and_immutable_inputs(
    tmp_path, source, predictor, count, guard, offset
):
    proof = derive(**source, **predictor)
    code = emit_correction(proof, "correct", output_value_guard=guard, sparse_pair_limit=16, source_word_guard=True)
    correct = library(tmp_path, code).correct
    correct.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t]
    lhs = np.repeat(np.arange(-128, 128, dtype=np.int8), 256)
    rhs = np.tile(np.arange(-128, 128, dtype=np.int8), 256)
    expected, predicted = source_table(**source).ravel(), predictor_table(**predictor, relu=source["relu"]).ravel()
    a, b, c = [np.full(65536 + 32, -99, np.int8) for _ in range(3)]
    a[offset : offset + 65536], b[offset : offset + 65536] = lhs, rhs
    before_a, before_b = a.copy(), b.copy()
    c[offset : offset + 65536] = predicted
    correct(a.ctypes.data + offset, b.ctypes.data + offset, c.ctypes.data + offset, 65536)
    assert np.array_equal(c[offset : offset + 65536], expected)
    assert (c[:offset] == -99).all() and (c[offset + 65536 :] == -99).all()
    assert np.array_equal(a, before_a) and np.array_equal(b, before_b)
    mismatch = np.flatnonzero(expected != predicted)
    for n in range(16):
        indices = np.arange(n)
        if mismatch.size and n:
            indices[-min(mismatch.size, n) :] = mismatch[:n]
        a[offset : offset + n], b[offset : offset + n] = lhs[indices], rhs[indices]
        c.fill(-99)
        c[offset : offset + n] = predicted[indices]
        before_a, before_b = a.copy(), b.copy()
        correct(a.ctypes.data + offset, b.ctypes.data + offset, c.ctypes.data + offset, n)
        assert np.array_equal(c[offset : offset + n], expected[indices])
        assert (c[:offset] == -99).all() and (c[offset + n :] == -99).all()
        assert np.array_equal(a, before_a) and np.array_equal(b, before_b)


@pytest.mark.parametrize("source,predictor,count", [SOURCE_CONTRACTS[0], SOURCE_CONTRACTS[-1]])
def test_no_hit_word_reads_neither_other_source_nor_output(tmp_path, source, predictor, count):
    proof = derive(**source, **predictor)
    guard = derive_source_word_guard(proof, limit=16)
    miss = next(byte for byte in range(256) if byte not in guard["raw_byte_values"])
    code = emit_correction(proof, "correct", sparse_pair_limit=16, source_word_guard=True)
    arguments = "a,(const int8_t*)0" if guard["axis"] == "lhs" else "(const int8_t*)0,a"
    code += f"""
int no_hit(void){{int8_t a[16] __attribute__((aligned(8)));
for(unsigned i=0;i<16;i++)a[i]={miss};
correct({arguments},(int8_t*)0,16);return 1;}}
"""
    assert library(tmp_path, code).no_hit() == 1


def test_guard_refuses_changed_proof_and_invalid_modes():
    source, predictor, _ = CONTRACTS[0]
    proof = derive(**source, **predictor)
    changed = deepcopy(proof)
    changed["correction_bitmap_hex"] = "00" * 8192
    for value in [None, {}, changed]:
        with pytest.raises(ValueError, match="unchanged complete"):
            derive_source_word_guard(value, limit=16)
    for kwargs in [
        dict(source_word_guard=1),
        dict(source_word_guard=True),
        dict(source_word_guard=True, sparse_pair_limit=1, packed_prefix=False),
    ]:
        with pytest.raises(ValueError, match="valid identifier"):
            emit_correction(proof, "correct", **kwargs)
