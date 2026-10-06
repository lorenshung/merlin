"""Exhaustive emitted predicate/correction, bounded source replay and old bytes."""

import ctypes
import hashlib
import subprocess
from copy import deepcopy

import numpy as np
import pytest

from merlin.llvmlower.quantized_affine_pair import (
    derive,
    derive_sparse_pair_predicate,
    emit_correction,
    predictor_table,
    source_table,
)

SOURCE = dict(lhs_scale=0.011258588172495365, rhs_scale=0.00940733402967453, output_scale=0.011643771082162857)
PREDICTOR = dict(p=298, q=249, scale=0.0032446938566863537)
CONTRACTS = [
    (dict(SOURCE, relu=True), PREDICTOR, 1),
    (dict(SOURCE, relu=False), PREDICTOR, 2),
    (dict(lhs_scale=0.125, rhs_scale=0.03125, output_scale=0.25, relu=False), dict(p=4, q=1, scale=0.125), 0),
]


def library(tmp_path, code):
    source = tmp_path / "sparse.c"
    source.write_text(code)
    target = tmp_path / (hashlib.sha256(code.encode()).hexdigest() + ".so")
    subprocess.run(
        [
            "cc",
            "-O2",
            "-fno-fast-math",
            "-ffp-contract=off",
            "-frounding-math",
            "-fPIC",
            "-shared",
            str(source),
            "-o",
            str(target),
        ],
        check=True,
    )
    return ctypes.CDLL(str(target))


@pytest.mark.parametrize("source,predictor,count", CONTRACTS)
def test_actual_c_predicate_all_65536_raw_pairs(tmp_path, source, predictor, count):
    proof = derive(**source, **predictor)
    sparse = derive_sparse_pair_predicate(proof, limit=16)
    assert sparse["cardinality"] == count and sparse["pairs"] == 65536
    code = """#include <stdint.h>
#include <stddef.h>
void enumerate(uint8_t *out){int8_t a[1],b[1];const size_t i=0;
for(unsigned lhs=0;lhs<256;lhs++)for(unsigned rhs=0;rhs<256;rhs++){
a[0]=(int8_t)lhs;b[0]=(int8_t)rhs;out[(lhs<<8)|rhs]=EXPR;}}
""".replace("EXPR", sparse["predicate_c"])
    enum = library(tmp_path, code).enumerate
    enum.argtypes = [ctypes.c_void_p]
    actual = np.zeros(65536, np.uint8)
    enum(actual.ctypes.data)
    mismatch = source_table(**source) != predictor_table(**predictor, relu=source["relu"])
    expected = np.roll(mismatch, 128, axis=(0, 1)).ravel()
    assert np.array_equal(actual, expected)


@pytest.mark.parametrize("source,predictor,count", CONTRACTS)
@pytest.mark.parametrize(
    "packed,guard,offset", [(False, False, 0), (True, False, 1), (False, True, 3), (True, True, 7)]
)
def test_source_exact_correction_all_pairs_dirty_tails_guards_inputs(
    tmp_path, source, predictor, count, packed, guard, offset
):
    proof = derive(**source, **predictor)
    code = emit_correction(proof, "correct", packed_prefix=packed, output_value_guard=guard, sparse_pair_limit=16)
    assert "_ambiguous[8192]" not in code
    correct = library(tmp_path, code).correct
    correct.argtypes = [ctypes.c_void_p] * 3 + [ctypes.c_size_t]
    lhs = np.repeat(np.arange(-128, 128, dtype=np.int8), 256)
    rhs = np.tile(np.arange(-128, 128, dtype=np.int8), 256)
    a, b, c = [np.full(65536 + 32, -99, np.int8) for _ in range(3)]
    a[offset : offset + 65536], b[offset : offset + 65536] = lhs, rhs
    original_a, original_b = a.copy(), b.copy()
    expected = source_table(**source).ravel()
    predicted = predictor_table(**predictor, relu=source["relu"]).ravel()
    c[offset : offset + 65536] = predicted
    correct(a.ctypes.data + offset, b.ctypes.data + offset, c.ctypes.data + offset, 65536)
    assert np.array_equal(c[offset : offset + 65536], expected)
    assert (c[:offset] == -99).all() and (c[offset + 65536 :] == -99).all()
    assert np.array_equal(a, original_a) and np.array_equal(b, original_b)
    # Put every exceptional pair in a scalar tail and repeat on dirty storage.
    mismatch = np.flatnonzero(expected != predicted)
    for tail in range(8):
        n = 24 + tail
        indices = np.arange(n)
        if mismatch.size:
            indices[-min(mismatch.size, n) :] = mismatch[:n]
        a[offset : offset + n], b[offset : offset + n] = lhs[indices], rhs[indices]
        c.fill(-99)
        c[offset : offset + n] = predicted[indices]
        before_a, before_b = a.copy(), b.copy()
        correct(a.ctypes.data + offset, b.ctypes.data + offset, c.ctypes.data + offset, n)
        assert np.array_equal(c[offset : offset + n], expected[indices])
        assert (c[:offset] == -99).all() and (c[offset + n :] == -99).all()
        assert np.array_equal(a, before_a) and np.array_equal(b, before_b)


def test_grouped_predicate_short_circuits_rhs_access(tmp_path):
    proof = derive(**dict(SOURCE, relu=True), **PREDICTOR)
    code = emit_correction(proof, "correct", output_value_guard=True, sparse_pair_limit=1)
    code += """
int reject_lhs_without_rhs(void){int8_t a=0,c=107;
correct_pair(&a,(const int8_t*)0,&c,0);return c==107;}
"""
    assert library(tmp_path, code).reject_lhs_without_rhs() == 1


def test_explicit_budget_and_changed_certificate_refuse():
    proof = derive(**dict(SOURCE, relu=False), **PREDICTOR)
    with pytest.raises(ValueError, match="exceeds explicit"):
        derive_sparse_pair_predicate(proof, limit=1)
    for limit in [0, -1, 17, True, 1.0]:
        with pytest.raises(ValueError, match="limit1..16"):
            derive_sparse_pair_predicate(proof, limit=limit)
    changed = deepcopy(proof)
    changed["correction_bitmap_hex"] = "00" * 8192
    for value in [None, {}, changed]:
        with pytest.raises(ValueError, match="unchanged complete"):
            derive_sparse_pair_predicate(value, limit=16)
    for limit in [-1, 17, True, 1.0]:
        with pytest.raises(ValueError, match="valid identifier"):
            emit_correction(proof, "correct", sparse_pair_limit=limit)


@pytest.mark.parametrize(
    "packed,guard,digest",
    [
        (False, False, "aeb24e69f62c8d84e86c2712512169e05eaee639f797fad77d412329fdca6c05"),
        (True, False, "7a202a24057acc1da6542d778dae45d7852d8bd6fc74f5edbbe26d51dbeea575"),
        (False, True, "0f49cfc6c64b544e424c159efda8c5a800d7f60a21471832b06c8f1f468287a0"),
        (True, True, "f9b4bbfe017e3bf156c4ca1e5ebd1f2bcc13a3218dfc9a5840747f58dd5e00fc"),
    ],
)
def test_default_emitted_bytes_unchanged(packed, guard, digest):
    proof = derive(**dict(SOURCE, relu=True), **PREDICTOR)
    code = emit_correction(proof, "correct", packed_prefix=packed, output_value_guard=guard)
    assert hashlib.sha256(code.encode()).hexdigest() == digest
