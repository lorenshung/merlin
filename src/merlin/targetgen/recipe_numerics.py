"""Exact integer parameters for the non-contraction int8 recipes, and their bit-level semantics.

A per-tensor symmetric int8 graph carries a floating scale on every edge. A non-contraction operation
between two such edges is still integer work on an accumulate-into-int32, read-out-once datapath, but
only if its floating coefficients are restated as INTEGERS applied in the int32 accumulator plus ONE
rounding readout. Rounding each operand onto the int8 load path first (the obvious recipe) rounds twice
and saturates whenever a coefficient exceeds one; measured on realistic data that is 50-95 % bit-exact
where the single-rounding form is 99.99-100 %.

This module is the target-agnostic arithmetic of those forms. It computes, from the edge scales:

- :func:`linear_combination` -- ``y = a*s_a/s_o + b*s_b/s_o`` (add.Tensor / sub.Tensor) as signed
  integer weights ``Wa, Wb`` (a best rational approximation of ``s_b/s_a``) and one readout scale;
- :func:`affine` -- ``y = (x*s_x*k + c)/s_o`` (add/sub/mul/div.Scalar, neg, relu) as an integer gain
  ``G`` (chosen so ``G*c/s_x`` is nearly an integer), an int32 accumulator constant ``B`` and one
  readout scale, negative for a negation;
- :func:`requantize` -- a movement (copy, cat input, pad interior) between two scales;
- :func:`constant_code` -- a fill or pad value on the output grid;
- :func:`window_mean_readout` -- the readout scale of an average pool or mean over a count.

and :func:`readout` / :meth:`IntegerRecipe.evaluate` execute a form exactly as a readout does: int32
accumulator, optional ReLU on the accumulator, ONE binary32 multiply, round half to even, saturate.
Which hardware realises the integer weights (a constant ``c*I`` contraction, a scaled load, an
accumulator seed) is a lowering decision and is not made here. Nothing here knows a target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

#: The largest integer weight the forms use, as three signed int8 parts of at most 127 each. A weight
#: is applied as ``sum(part * I)`` contractions into the int32 accumulator, so it never passes an int8
#: load path. 381 gives a ratio error near 1e-5 for the scale pairs measured.
MAX_WEIGHT = 381
MAX_PART = 127
INT32 = (-(2**31), 2**31 - 1)


class RecipeNumericsError(ValueError):
    """A form cannot be stated exactly with the given scales (names the obstacle)."""


def _positive(name: str, value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value <= 0:
        raise RecipeNumericsError(f"{name} must be a finite positive scale, got {value!r}")
    return value


def _f32(value: float) -> float:
    return float(np.float32(value))


def split_weight(weight: int, part_max: int = MAX_PART) -> tuple[int, ...]:
    """``weight`` as signed parts of at most ``part_max`` each (the int8 ``c*I`` constants)."""
    if type(weight) is not int:
        raise RecipeNumericsError("an integer weight is required")
    if weight == 0:
        return ()
    sign, remaining, parts = (1 if weight > 0 else -1), abs(weight), []
    while remaining:
        part = min(remaining, part_max)
        parts.append(sign * part)
        remaining -= part
    return tuple(parts)


def best_rational(ratio: float, *, max_weight: int = MAX_WEIGHT) -> tuple[int, int]:
    """``(p, q)`` with ``q > 0``, ``|p|, q <= max_weight`` and ``p/q`` closest to ``ratio``.

    Ties prefer the smaller ``q`` (fewer and smaller constant parts).
    """
    if not math.isfinite(ratio):
        raise RecipeNumericsError("ratio must be finite")
    best = None
    for q in range(1, max_weight + 1):
        p = round(ratio * q)
        if abs(p) > max_weight:
            continue
        error = abs(p / q - ratio)
        if best is None or error < best[0] - 1e-18:
            best = (error, p, q)
    if best is None:
        raise RecipeNumericsError(f"no weight pair within {max_weight} approximates {ratio!r}")
    return best[1], best[2]


@dataclass(frozen=True)
class IntegerRecipe:
    """``y = sat(RNE(f32(act(sum_i w_i * x_i + constant)) * f32(readout_scale)))``.

    ``weights`` apply to the int8 operand codes in order; ``constant`` is the int32 accumulator seed;
    ``activation`` is ``none`` or ``relu`` and acts on the accumulator BEFORE the scale, as a readout
    does. ``ratio_error`` records how far the integer form is from the floating coefficients.
    """

    weights: tuple[int, ...]
    constant: int
    readout_scale: float
    activation: str = "none"
    qmin: int = -128
    qmax: int = 127
    ratio_error: float = 0.0
    form: str = ""

    def __post_init__(self) -> None:
        if self.activation not in {"none", "relu"}:
            raise RecipeNumericsError(f"activation must be none or relu, got {self.activation!r}")
        if self.activation == "relu" and self.readout_scale < 0:
            # The readout applies ReLU before the scale: relu(x)*s with s < 0 is never relu(x*s).
            raise RecipeNumericsError("a readout ReLU cannot pair with a negative readout scale")
        if not INT32[0] <= self.constant <= INT32[1]:
            raise RecipeNumericsError("the accumulator constant exceeds int32")
        if any(abs(weight) > MAX_WEIGHT for weight in self.weights):
            raise RecipeNumericsError(f"an integer weight exceeds {MAX_WEIGHT}")

    def parts(self) -> tuple[tuple[int, ...], ...]:
        """Each weight as its signed int8 ``c*I`` parts."""
        return tuple(split_weight(weight) for weight in self.weights)

    def accumulator_bound(self, operand_bound: int = 128) -> int:
        """The largest |accumulator| the form can produce for int8 operands."""
        return sum(abs(weight) for weight in self.weights) * operand_bound + abs(self.constant)

    def evaluate(self, *codes) -> np.ndarray:
        """Execute the form on int8 operand codes (arrays broadcast), exactly as specified."""
        if len(codes) != len(self.weights):
            raise RecipeNumericsError(f"the form takes {len(self.weights)} operand(s), got {len(codes)}")
        accumulator = np.full(np.broadcast_shapes(*(np.shape(c) for c in codes)) if codes else (), self.constant)
        accumulator = accumulator.astype(np.int64)
        for weight, code in zip(self.weights, codes, strict=True):
            accumulator = accumulator + weight * np.asarray(code, dtype=np.int64)
        if np.any(accumulator < INT32[0]) or np.any(accumulator > INT32[1]):
            raise RecipeNumericsError("the int32 accumulator overflowed")
        return readout(accumulator, self.readout_scale, relu=self.activation == "relu", qmin=self.qmin, qmax=self.qmax)

    def to_dict(self) -> dict:
        return {
            "form": self.form,
            "weights": list(self.weights),
            "weight_parts": [list(parts) for parts in self.parts()],
            "constant": self.constant,
            "readout_scale": self.readout_scale,
            "activation": self.activation,
            "quant_min": self.qmin,
            "quant_max": self.qmax,
            "ratio_error": self.ratio_error,
        }


def readout(accumulator, scale: float, *, relu: bool = False, qmin: int = -128, qmax: int = 127) -> np.ndarray:
    """Accumulator -> int8 code: optional ReLU, one binary32 multiply, round half to even, saturate."""
    values = np.asarray(accumulator, dtype=np.int64)
    if relu:
        values = np.maximum(values, 0)
    if np.any(np.abs(values) >= 2**24):
        # binary32 holds every integer below 2^24 exactly; beyond it the conversion itself rounds.
        raise RecipeNumericsError("accumulator values at or beyond 2^24 are not exact in binary32")
    product = values.astype(np.float32) * np.float32(scale)
    return np.clip(np.rint(product), qmin, qmax).astype(np.int64)


def linear_combination(
    s_a: float,
    s_b: float,
    s_out: float,
    *,
    sign_b: int = 1,
    activation: str = "none",
    qmin: int = -128,
    qmax: int = 127,
    max_weight: int = MAX_WEIGHT,
) -> IntegerRecipe:
    """``y = (s_a*a + sign_b*s_b*b)/s_out`` (add.Tensor, sub.Tensor with ``sign_b=-1``).

    ``Wa/Wb`` is the best rational approximation of ``s_a/s_b`` within ``max_weight``; the minus of a
    subtraction is the sign of ``Wb``, applied in the int32 accumulator and never to an int8 operand
    (negating -128 on a load saturates). The readout scale is ``s_a/(s_out*Wa)``.
    """
    s_a, s_b, s_out = _positive("s_a", s_a), _positive("s_b", s_b), _positive("s_out", s_out)
    if sign_b not in (1, -1):
        raise RecipeNumericsError("sign_b must be +1 or -1")
    wb, wa = best_rational(s_b / s_a, max_weight=max_weight)
    if wb == 0:
        raise RecipeNumericsError(f"s_b/s_a = {s_b / s_a!r} rounds to a zero weight within {max_weight}")
    return IntegerRecipe(
        weights=(wa, sign_b * wb),
        constant=0,
        readout_scale=_f32(s_a / (s_out * wa)),
        activation=activation,
        qmin=qmin,
        qmax=qmax,
        ratio_error=abs(wb / wa - s_b / s_a) / (s_b / s_a),
        form="linear_combination",
    )


def affine(
    s_x: float,
    s_out: float,
    *,
    gain: float = 1.0,
    offset: float = 0.0,
    activation: str = "none",
    qmin: int = -128,
    qmax: int = 127,
    max_weight: int = MAX_WEIGHT,
    tolerance: float = 3e-5,
) -> IntegerRecipe:
    """``y = (gain*s_x*x + offset)/s_out`` (add.Scalar, sub.Scalar, mul.Scalar, div.Scalar, neg, relu).

    With an offset the accumulator is ``G*x + B``: ``G`` is the smallest integer gain up to
    ``max_weight`` for which ``G*offset/(gain*s_x)`` lies within ``tolerance`` input codes of an
    integer (else the closest one), ``B`` is that integer as an int32 seed, and the readout scale is
    ``gain*s_x/(s_out*G)``. Without an offset ``G = 1`` and the whole coefficient is the readout scale,
    negative for a negation. A relu with a negative overall gain is refused (see IntegerRecipe).
    """
    s_x, s_out = _positive("s_x", s_x), _positive("s_out", s_out)
    if not math.isfinite(gain) or gain == 0 or not math.isfinite(offset):
        raise RecipeNumericsError("gain must be finite and nonzero, and offset finite")
    shift = offset / (gain * s_x)  # the offset in input codes
    if shift == 0:
        return IntegerRecipe(
            weights=(1,),
            constant=0,
            readout_scale=_f32(gain * s_x / s_out),
            activation=activation,
            qmin=qmin,
            qmax=qmax,
            form="affine",
        )
    best = None
    for g in range(1, max_weight + 1):
        seed = round(g * shift)
        error = abs(g * shift - seed) / g
        if best is None or error < best[0] - 1e-18:
            best = (error, g, seed)
        if error <= tolerance:
            best = (error, g, seed)
            break
    error, g, seed = best
    return IntegerRecipe(
        weights=(g,),
        constant=int(seed),
        readout_scale=_f32(gain * s_x / (s_out * g)),
        activation=activation,
        qmin=qmin,
        qmax=qmax,
        ratio_error=error,
        form="affine",
    )


def requantize(s_in: float, s_out: float) -> dict:
    """A movement between two scales: an exact copy when equal, else one rounding at ``s_in/s_out``.

    A load-path scale rounds once and saturates, which matches the reference requantization exactly
    (cat with independent input scales measured 100 %), so a movement never needs the accumulator.
    """
    s_in, s_out = _positive("s_in", s_in), _positive("s_out", s_out)
    if s_in == s_out:
        return {"form": "copy", "scale": 1.0}
    return {"form": "scaled_copy", "scale": _f32(s_in / s_out)}


def constant_code(value: float, s_out: float, *, qmin: int = -128, qmax: int = 127) -> int:
    """A fill or pad value on the output grid: ``sat(RNE(value/s_out))``, computed at compile time."""
    s_out = _positive("s_out", s_out)
    if not math.isfinite(value):
        # -inf/+inf fills (masks) saturate to the rails, as the reference requantization does.
        return qmin if value < 0 else qmax
    return int(min(qmax, max(qmin, round(value / s_out))))


def window_mean_readout(s_in: float, divisor: int, s_out: float) -> float:
    """Readout scale of an average pool / mean over ``divisor`` summed elements: ``s_in/(divisor*s_out)``."""
    if type(divisor) is not int or divisor < 1:
        raise RecipeNumericsError("divisor must be a positive integer count")
    return _f32(_positive("s_in", s_in) / (divisor * _positive("s_out", s_out)))


def narrow_range(qmax: int = 127) -> tuple[int, int]:
    """Symmetric narrow int8 range for the producer of a negation, so ``-(-128)`` never occurs."""
    return -qmax, qmax
