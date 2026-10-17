"""The integer forms of the non-contraction int8 recipes round once, and say exactly how.

WHY THIS EXISTS. Gate 3 (realistic per-tensor int8 data) found that the obvious recipes for the
non-contraction support set round twice: each operand is rounded onto the int8 load path and then
summed (add/sub.Tensor, 75-95 % bit-exact), a scalar offset is rounded separately from the input
(add/sub.Scalar, 50-88 %, and saturating whenever the input coefficient exceeds one), and some
readouts hard-code a scale (weight gradient at 1, average pool at 1/divisor). The fixed forms restate
every coefficient as integers applied in the int32 accumulator plus ONE readout. These tests hold the
forms to that: at most one output step from the real-valued reference, almost always zero, and
better than the double-rounding form on the same data.
"""

from __future__ import annotations

import numpy as np
import pytest

from merlin.targetgen import recipe_numerics as RN

_A, _B = np.meshgrid(np.arange(-128, 128), np.arange(-128, 128))
_X = np.arange(-128, 128)
_RNG = np.random.default_rng(20261017)


def _reference(values):
    return np.clip(np.rint(values), -128, 127)


@pytest.mark.parametrize("sign", [1, -1])
@pytest.mark.parametrize("trial", range(6))
def test_two_input_sum_and_difference_round_once(sign, trial):
    s_a, s_b, s_out = (float(v) for v in _RNG.uniform(0.005, 0.1, size=3))
    recipe = RN.linear_combination(s_a, s_b, s_out, sign_b=sign)
    assert max(abs(w) for w in recipe.weights) <= RN.MAX_WEIGHT and recipe.constant == 0
    assert all(abs(part) <= RN.MAX_PART for parts in recipe.parts() for part in parts)
    assert (recipe.weights[1] < 0) == (sign < 0), "the minus lives in an integer weight, not an int8 operand"
    expected = _reference((s_a * _A + sign * s_b * _B) / s_out)
    error = np.abs(recipe.evaluate(_A, _B) - expected)
    assert error.max() <= 1 and (error == 0).mean() >= 0.995
    naive = np.clip(
        np.clip(np.rint(s_a / s_out * _A), -128, 127) + np.clip(np.rint(sign * s_b / s_out * _B), -128, 127),
        -128,
        127,
    )
    assert (error == 0).mean() >= (np.abs(naive - expected) == 0).mean()


def test_the_scalar_offset_becomes_an_int32_seed_with_no_saturation():
    """Pixel centring p - 0.5 with an input coefficient of 2: the load-path form clips half the codes."""
    s_x, s_out = 1 / 127, 1 / 254
    recipe = RN.affine(s_x, s_out, offset=-0.5)
    assert recipe.weights == (2,) and recipe.constant == -127 and recipe.ratio_error == 0
    assert np.array_equal(recipe.evaluate(_X), _reference((s_x * _X - 0.5) / s_out))


@pytest.mark.parametrize("trial", range(8))
def test_affine_forms_round_once(trial):
    s_x, s_out = (float(v) for v in _RNG.uniform(0.005, 0.1, size=2))
    offset = float(_RNG.uniform(-3, 3))
    gain = float(_RNG.choice([1.0, -1.0, 0.37, 2.5]))
    recipe = RN.affine(s_x, s_out, gain=gain, offset=offset)
    expected = _reference((gain * s_x * _X + offset) / s_out)
    error = np.abs(recipe.evaluate(_X) - expected)
    assert error.max() <= 1 and (error == 0).mean() >= 0.99
    assert recipe.accumulator_bound() < 2**24


def test_negation_is_a_negative_readout_and_narrow_range_keeps_it_exact():
    recipe = RN.affine(0.04, 0.04, gain=-1.0)
    assert recipe.weights == (1,) and recipe.readout_scale == -1.0
    low, high = RN.narrow_range()
    codes = np.arange(low, high + 1)
    assert np.array_equal(recipe.evaluate(codes), -codes)
    assert recipe.evaluate(np.array([-128]))[0] == 127  # why the producer should use the narrow range


def test_relu_runs_on_the_accumulator_before_the_scale():
    recipe = RN.affine(0.02, 0.03, activation="relu")
    assert np.array_equal(recipe.evaluate(_X), _reference(np.maximum(0.02 * _X, 0) / 0.03))
    with pytest.raises(RN.RecipeNumericsError, match="negative readout scale"):
        RN.affine(0.02, 0.03, gain=-1.0, activation="relu")


def test_readout_is_one_binary32_multiply_rounded_half_to_even():
    assert list(RN.readout(np.array([1, 3, 5, -1, -3]), 0.5)) == [0, 2, 2, 0, -2]
    assert list(RN.readout(np.array([-5, 5]), 1.0, relu=True)) == [0, 5]
    assert list(RN.readout(np.array([1000, -1000]), 1.0)) == [127, -128]
    with pytest.raises(RN.RecipeNumericsError):
        RN.readout(np.array([2**24]), 1.0)


def test_movement_constants_and_readout_scales():
    assert RN.requantize(0.1, 0.1) == {"form": "copy", "scale": 1.0}
    assert RN.requantize(0.1, 0.2)["form"] == "scaled_copy"
    assert RN.constant_code(0.5, 0.25) == 2 and RN.constant_code(-1.7, 0.01) == -128
    assert RN.constant_code(float("-inf"), 0.1) == -128 and RN.constant_code(0.0, 0.1) == 0
    assert RN.window_mean_readout(0.1, 4, 0.05) == pytest.approx(0.5)
    assert RN.split_weight(381) == (127, 127, 127) and RN.split_weight(-130) == (-127, -3)


def test_inexpressible_inputs_are_refused_with_the_reason():
    with pytest.raises(RN.RecipeNumericsError, match="positive"):
        RN.linear_combination(0.0, 0.1, 0.1)
    with pytest.raises(RN.RecipeNumericsError, match="zero weight"):
        RN.linear_combination(1.0, 1e-6, 0.1)
    with pytest.raises(RN.RecipeNumericsError, match="divisor"):
        RN.window_mean_readout(0.1, 0, 0.1)
