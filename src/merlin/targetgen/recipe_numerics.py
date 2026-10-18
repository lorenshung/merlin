"""Prove integer recipes against exact rational quantized arithmetic.

Hardware readout is modeled explicitly as a binary32 multiply. A candidate
recipe is admitted only after every input code agrees with the rational oracle;
coefficient approximation alone never establishes exactness.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from merlin.targetgen.exact_arithmetic import quantize, rational

#: The largest integer weight the forms use, as three signed int8 parts of at most 127 each. A weight
#: is applied as ``sum(part * I)`` contractions into the int32 accumulator, so it never passes an int8
#: load path. Every proposed form still requires exhaustive exact code equality.
MAX_WEIGHT = 381
MAX_PART = 127
INT32 = (-(2**31), 2**31 - 1)


class RecipeNumericsError(ValueError):
    """A form cannot be stated exactly with the given scales (names the obstacle)."""


def _positive(name: str, value: float) -> float:
    value = rational(value)
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
        if best is None or error < best[0]:
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
        if not math.isfinite(self.readout_scale):
            raise RecipeNumericsError("readout scale must be finite")
        if self.activation not in {"none", "relu"}:
            raise RecipeNumericsError(f"activation must be none or relu, got {self.activation!r}")
        if self.activation == "relu" and self.readout_scale < 0:
            # The readout applies ReLU before the scale: relu(x)*s with s < 0 is never relu(x*s).
            raise RecipeNumericsError("a readout ReLU cannot pair with a negative readout scale")
        if type(self.constant) is not int or not INT32[0] <= self.constant <= INT32[1]:
            raise RecipeNumericsError("the accumulator constant exceeds int32")
        if any(type(weight) is not int or abs(weight) > MAX_WEIGHT for weight in self.weights):
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
            "ratio_error": float(self.ratio_error),
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
    s_a, s_b, s_out = (
        rational(_positive(name, value)) for name, value in (("s_a", s_a), ("s_b", s_b), ("s_out", s_out))
    )
    if sign_b not in (1, -1):
        raise RecipeNumericsError("sign_b must be +1 or -1")
    wb, wa = best_rational(float(s_b / s_a), max_weight=max_weight)
    if wb == 0:
        raise RecipeNumericsError(f"s_b/s_a = {s_b / s_a!r} rounds to a zero weight within {max_weight}")
    recipe = IntegerRecipe(
        weights=(wa, sign_b * wb),
        constant=0,
        readout_scale=_f32(s_a / (s_out * wa)),
        activation=activation,
        qmin=qmin,
        qmax=qmax,
        ratio_error=abs(wb / wa - s_b / s_a) / (s_b / s_a),
        form="linear_combination",
    )
    return prove_recipe(recipe, (s_a / s_out, sign_b * s_b / s_out))


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
    tolerance: float = 0,
) -> IntegerRecipe:
    """Propose integer weights/seed, then prove the complete code domain exactly.

    A small coefficient error is insufficient: every rounded/clamped output
    must equal rational evaluation of the original gain and offset.
    """
    s_x, s_out = rational(_positive("s_x", s_x)), rational(_positive("s_out", s_out))
    gain, offset = rational(gain), rational(offset)
    if tolerance != 0:
        raise RecipeNumericsError("exact recipes do not permit an approximation tolerance")
    if not math.isfinite(gain) or gain == 0 or not math.isfinite(offset):
        raise RecipeNumericsError("gain must be finite and nonzero, and offset finite")
    shift = offset / (gain * s_x)  # the offset in input codes
    if shift == 0:
        recipe = IntegerRecipe(
            weights=(1,),
            constant=0,
            readout_scale=_f32(gain * s_x / s_out),
            activation=activation,
            qmin=qmin,
            qmax=qmax,
            form="affine",
        )
        return prove_recipe(recipe, (gain * s_x / s_out,), offset / s_out)
    best = None
    for g in range(1, max_weight + 1):
        seed = round(g * shift)
        error = abs(g * shift - seed) / g
        if best is None or error < best[0]:
            best = (error, g, seed)
        if error <= tolerance:
            best = (error, g, seed)
            break
    error, g, seed = best
    recipe = IntegerRecipe(
        weights=(g,),
        constant=int(seed),
        readout_scale=_f32(gain * s_x / (s_out * g)),
        activation=activation,
        qmin=qmin,
        qmax=qmax,
        ratio_error=error,
        form="affine",
    )
    return prove_recipe(recipe, (gain * s_x / s_out,), offset / s_out)


def prove_recipe(recipe, coefficients, offset=0):
    """Exhaust the full signed-i8 Cartesian domain; a single mismatch refuses."""
    codes = np.arange(-128, 128)
    inputs = np.meshgrid(*([codes] * len(coefficients)), indexing="ij")
    values = np.full(inputs[0].shape, rational(offset), dtype=object)
    for coefficient, code in zip(coefficients, inputs, strict=True):
        values += rational(coefficient) * code
    if recipe.activation == "relu":
        values = np.maximum(values, 0)
    expected = quantize(values, 1, quant_min=recipe.qmin, quant_max=recipe.qmax)
    actual = recipe.evaluate(*inputs)
    if not np.array_equal(actual, expected):
        count = int(np.count_nonzero(actual != expected))
        raise RecipeNumericsError(f"exact-arithmetic proof failed for {count} input codes")
    return recipe


def requantize(s_in: float, s_out: float, *, qmin=-128, qmax=127) -> dict:
    """A movement between two scales: an exact copy when equal, else one rounding at ``s_in/s_out``.

    A load-path scale rounds once and saturates, which matches the reference requantization exactly
    (cat with independent input scales measured 100 %), so a movement never needs the accumulator.
    """
    s_in, s_out = _positive("s_in", s_in), _positive("s_out", s_out)
    if s_in == s_out:
        return {"form": "copy", "scale": 1.0}
    scale = _f32(s_in / s_out)
    codes = np.arange(-128, 128)
    expected = quantize(codes * rational(s_in), s_out, quant_min=qmin, quant_max=qmax)
    if not np.array_equal(readout(codes, scale, qmin=qmin, qmax=qmax), expected):
        raise RecipeNumericsError("scaled copy fails exact-arithmetic proof")
    return {"form": "scaled_copy", "scale": scale}


def constant_code(value: float, s_out: float, *, qmin: int = -128, qmax: int = 127) -> int:
    """A fill or pad value on the output grid: ``sat(RNE(value/s_out))``, computed at compile time."""
    s_out = _positive("s_out", s_out)
    if math.isnan(value):
        raise RecipeNumericsError("a NaN fill has no exact rational code")
    if not math.isfinite(value):
        # -inf/+inf fills (masks) saturate to the rails, as the reference requantization does.
        return qmin if value < 0 else qmax
    return int(min(qmax, max(qmin, round(rational(value) / rational(s_out)))))


def window_mean_readout(s_in: float, divisor: int, s_out: float, *, qmin=-128, qmax=127) -> float:
    """Readout scale of an average pool / mean over ``divisor`` summed elements: ``s_in/(divisor*s_out)``."""
    if type(divisor) is not int or divisor < 1:
        raise RecipeNumericsError("divisor must be a positive integer count")
    ratio = rational(_positive("s_in", s_in)) / (divisor * rational(_positive("s_out", s_out)))
    if divisor * 256 > 262144:
        raise RecipeNumericsError("window sum domain exceeds the exhaustive proof limit")
    scale = _f32(ratio)
    sums = np.arange(-128 * divisor, 127 * divisor + 1)
    expected = quantize(sums * ratio, 1, quant_min=qmin, quant_max=qmax)
    if not np.array_equal(readout(sums, scale, qmin=qmin, qmax=qmax), expected):
        raise RecipeNumericsError("window readout fails exact-arithmetic proof")
    return scale


def narrow_range(qmax: int = 127) -> tuple[int, int]:
    """Symmetric narrow int8 range for the producer of a negation, so ``-(-128)`` never occurs."""
    return -qmax, qmax
