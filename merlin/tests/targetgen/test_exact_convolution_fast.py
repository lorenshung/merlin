"""Integer-grid convolution agrees exactly with independent scalar Fractions."""

from fractions import Fraction
from itertools import product

import numpy as np
import pytest

from merlin.targetgen.exact_quantized_reference import _convolution, _convolution_fraction, _integer_grid


def test_grid_recovers_integer_codes_without_qparam_metadata():
    codes = np.array([-128, 127, 0, 1, -1], dtype=object)
    scale = Fraction(float(np.float32(0.03)))
    integer, factor = _integer_grid(codes * scale)
    assert factor == scale
    np.testing.assert_array_equal(integer, codes)
    np.testing.assert_array_equal(integer * factor, codes * scale)


@pytest.mark.parametrize(
    "dims,groups,stride,padding,dilation",
    [
        (1, 1, 1, 0, 1),
        (1, 2, 2, 1, 2),
        (2, 1, 1, 1, 1),
        (2, 2, 2, 1, 1),
        (2, 3, 1, 2, 2),
        (3, 1, 1, 1, 1),
        (3, 2, 2, 1, 2),
    ],
)
@pytest.mark.parametrize("bias", [False, True])
def test_randomized_grouped_dilated_convolution(dims, groups, stride, padding, dilation, bias):
    rng = np.random.default_rng(166)
    shape = (2, groups * 2, *([5] * dims))
    weight_shape = (groups * 3, 2, *([2] * dims))
    source = rng.integers(-128, 128, shape).astype(object) * Fraction(float(np.float32(0.03)))
    weight = rng.integers(-128, 128, weight_shape).astype(object)
    for co in range(weight.shape[0]):
        weight[co] *= Fraction(float(np.float32(0.02 + co * 0.001)))
    bias_value = np.array([Fraction(co - 3, 37) for co in range(weight.shape[0])], dtype=object) if bias else None
    fast = _convolution(source, weight, bias_value, stride, padding, dilation, groups)
    slow = _convolution_fraction(source, weight, bias_value, stride, padding, dilation, groups)
    np.testing.assert_array_equal(fast, slow)
    # The final quantize must also preserve every asymmetric code and tie.
    for zp in (0, -13, 127):
        scale = Fraction(float(np.float32(0.01)))

        def code(a):
            return np.array([min(127, max(-128, round(v / scale) + zp)) for v in a.flat])

        np.testing.assert_array_equal(code(fast), code(slow))


@pytest.mark.parametrize(
    "source,weight",
    [
        ([-128, 127], [-128, 127]),
        ([-(1 << 31), (1 << 31) - 1], [1, 1]),
        ([(1 << 62), 1], [2, 1]),
        ([1 << 80, 1], [-(1 << 70), 1]),
        ([1 << 80, 1], [0, 0]),
        ([0, 0], [1 << 80, 1]),
        ([Fraction(1, 3), Fraction(1, 7)], [Fraction(1, 11), Fraction(-1, 17)]),
        ([Fraction(2) ** -149, Fraction(2) ** 100], [Fraction(2) ** -149, Fraction(2) ** 100]),
    ],
)
def test_boundaries_and_python_integer_fallback(source, weight):
    x = np.asarray(source, dtype=object).reshape(1, 1, 2)
    w = np.asarray(weight, dtype=object).reshape(1, 1, 2)
    np.testing.assert_array_equal(_convolution(x, w, None, 1, 1, 1, 1), _convolution_fraction(x, w, None, 1, 1, 1, 1))


def test_overflow_bound_selects_python_integer_macs(monkeypatch):
    original = np.matmul
    types = []

    def matmul(a, b):
        types.append((a.dtype, b.dtype))
        return original(a, b)

    monkeypatch.setattr(np, "matmul", matmul)
    x = np.array([1 << 62, 1], dtype=object).reshape(1, 1, 2)
    w = np.array([2, 1], dtype=object).reshape(1, 1, 2)
    actual = _convolution(x, w, None, 1, 0, 1, 1)
    assert types == [(np.dtype(object), np.dtype(object))]
    assert actual.item() == (1 << 63) + 1


def test_int64_prefix_bound_and_near_limit(monkeypatch):
    original = np.matmul
    types = []

    def matmul(a, b):
        types.append((a.dtype, b.dtype))
        return original(a, b)

    monkeypatch.setattr(np, "matmul", matmul)
    x = np.array([(1 << 61) - 1, 1], dtype=object).reshape(1, 1, 2)
    w = np.array([2, 1], dtype=object).reshape(1, 1, 2)
    assert _convolution(x, w, None, 1, 0, 1, 1).item() == (1 << 62) - 1
    assert types == [(np.dtype(np.int64), np.dtype(np.int64))]


def test_exhaustive_small_integer_and_half_tie_domain():
    for values in product(range(-2, 3), repeat=4):
        x = np.asarray(values[:2], dtype=object).reshape(1, 1, 2) * Fraction(1, 2)
        w = np.asarray(values[2:], dtype=object).reshape(1, 1, 2) * Fraction(1, 4)
        fast = _convolution(x, w, None, 1, 0, 1, 1)
        slow = _convolution_fraction(x, w, None, 1, 0, 1, 1)
        assert fast.item() == slow.item()
        assert round(fast.item()) == round(slow.item())
