"""A non-finite native expectation never creates an unbounded error floor."""

import struct
from fractions import Fraction

import numpy as np
import pytest

from merlin.targetgen.native_rational_reference import compare_reduction_floor


@pytest.mark.parametrize(
    "dtype,word,width",
    [
        ("f16", 0x7C00, 2),
        ("bf16", 0x7F80, 2),
        ("f32", 0x7F800000, 4),
        ("f64", 0x7FF0000000000000, 8),
        ("complex<f32>", 0x7F800000, 4),
    ],
)
@pytest.mark.parametrize("negative", [False, True])
def test_infinity_requires_same_class_and_sign(monkeypatch, dtype, word, width, negative):
    sign = 1 << (8 * width - 1)
    expected = word | (sign if negative else 0)
    check(monkeypatch, dtype, width, expected, expected, "pass")
    for actual in (0, sign, word ^ (sign if not negative else 0), word | 1):
        check(monkeypatch, dtype, width, expected, actual, "mismatch")


def check(monkeypatch, dtype, width, expected, actual, status):
    from merlin.targetgen.native_rational_reference import Gaussian

    is_complex = dtype.startswith("complex")
    monkeypatch.setattr(
        "merlin.targetgen.native_rational_reference.rational_outputs",
        lambda *args, **kw: [np.array([Gaussian(Fraction(3)) if is_complex else Fraction(3)], dtype=object)],
    )
    suffix = bytes(width) if is_complex else b""

    def raw(value):
        return value.to_bytes(width, "little") + suffix

    record = dict(
        case={},
        output_indices=[0],
        output_abi=[dict(dtype=dtype)],
        native_eager_boundary=dict(output_bytes=[raw(expected).hex()]),
    )
    assert compare_reduction_floor(record, [raw(actual)])["status"] == status


@pytest.mark.parametrize(
    "actual,status",
    [
        (0x7FC00011, "pass"),
        (0x7FC00012, "mismatch"),
        (0xFFC00011, "mismatch"),
        (0x7F800000, "mismatch"),
        (0, "mismatch"),
    ],
)
def test_nan_retains_defined_native_payload(monkeypatch, actual, status):
    check(monkeypatch, "f32", 4, 0x7FC00011, actual, status)


def test_finite_expectation_still_uses_exact_rational_floor(monkeypatch):
    def word(x):
        return struct.unpack("<I", struct.pack("<f", x))[0]

    check(monkeypatch, "f32", 4, word(4), word(3), "pass")
    check(monkeypatch, "f32", 4, word(4), word(1), "mismatch")
    check(monkeypatch, "f32", 4, word(4), 0x7F800000, "mismatch")
