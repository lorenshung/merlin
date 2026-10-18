"""Rational tensor arithmetic and correctly rounded quantized boundaries.

These reusable arithmetic primitives contain no graph evaluator or device answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from fractions import Fraction

import numpy as np


def rational(value):
    if isinstance(value, Fraction):
        return value
    if isinstance(value, (int, np.integer, bool)):
        return Fraction(int(value))
    value = float(value)
    if not math.isfinite(value):
        raise ValueError("exact reference requires finite rational values")
    return Fraction.from_float(value)


def exact_array(value):
    if isinstance(value, ExactTensor):
        return value.values
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return np.vectorize(rational, otypes=[object])(np.asarray(value, dtype=object))


def round_binary(value, bits=32):
    """Correct RNE to IEEE binary32/64, without a double rounding via binary64."""
    value = rational(value)
    if bits == 64:
        return float(value)
    if bits not in (16, 32):
        raise ValueError("unsupported floating boundary precision")
    if not value:
        return 0.0
    fraction_bits, minimum_exp, maximum_exp = (23, -126, 127) if bits == 32 else (10, -14, 15)
    magnitude = abs(value)
    exponent = magnitude.numerator.bit_length() - magnitude.denominator.bit_length()
    if magnitude < Fraction(2) ** exponent:
        exponent -= 1
    unit = Fraction(2) ** (max(minimum_exp, exponent) - fraction_bits)
    rounded = round(value / unit) * unit
    if abs(rounded) >= Fraction(2) ** (maximum_exp + 1):
        raise ValueError("exact floating boundary overflows")
    return -0.0 if not rounded and value < 0 else float(rounded)


def _parameters(scale, zero_point, axis, ndim):
    scales, zeros = exact_array(scale), exact_array(zero_point)
    if any(value <= 0 for value in scales.flat):
        raise ValueError("quantization scales must be finite and positive")
    if any(value.denominator != 1 for value in zeros.flat):
        raise ValueError("quantization zero points must be integers")
    if axis is not None:
        axis %= ndim
        shape = [1] * ndim
        shape[axis] = scales.size
        scales, zeros = scales.reshape(shape), zeros.reshape(shape)
    elif scales.size != 1 or zeros.size != 1:
        raise ValueError("per-tensor quantization requires scalar parameters")
    return scales, zeros


def quantize(value, scale, zero_point=0, quant_min=-128, quant_max=127, *, axis=None):
    """Infinite precision divide, RNE, integer zero point, then clamp (no tolerance)."""
    if type(quant_min) is not int or type(quant_max) is not int or quant_min > quant_max:
        raise ValueError("invalid integer quantization range")
    values = exact_array(value)
    scales, zeros = _parameters(scale, zero_point, axis, values.ndim)
    return np.vectorize(lambda x, s, z: min(quant_max, max(quant_min, round(x / s) + int(z))), otypes=[np.int64])(
        values, scales, zeros
    )


@dataclass
class ExactTensor:
    values: np.ndarray
    dtype: object
    codes: np.ndarray | None = None
    scale: object = None


def dequantize(value, scale, zero_point=0, quant_min=-128, quant_max=127, *, axis=None, dtype=None):
    codes = exact_array(value)
    if any(x.denominator != 1 or not quant_min <= x <= quant_max for x in codes.flat):
        raise ValueError("integer codes outside the declared quantization range")
    scales, zeros = _parameters(scale, zero_point, axis, codes.ndim)
    centered = codes - zeros
    integers = np.vectorize(int, otypes=[object])(centered)
    return ExactTensor(centered * scales, dtype, integers, scale)


def materialize(value):
    import torch

    if isinstance(value, ExactTensor):
        dtype = value.dtype or torch.float32
        if dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
            return torch.tensor(value.values.tolist(), dtype=dtype)
        if dtype == torch.bfloat16:
            raise ValueError("exact reference does not yet implement a bfloat16 result boundary")
        bits = {torch.float16: 16, torch.float32: 32, torch.float64: 64}[dtype]
        values = np.vectorize(lambda x: round_binary(x, bits), otypes=[float])(value.values)
        return torch.tensor(values.tolist(), dtype=dtype).reshape(value.values.shape)
    if isinstance(value, tuple):
        return tuple(materialize(x) for x in value)
    if isinstance(value, list):
        return [materialize(x) for x in value]
    if isinstance(value, dict):
        return {key: materialize(x) for key, x in value.items()}
    return value
