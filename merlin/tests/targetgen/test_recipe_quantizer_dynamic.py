"""The ``quantize_`` configuration a dynamic recipe builds carries the recipe's own formats.

WHY THIS EXISTS. ``_recipe_quantizer.quantize_config`` built
``Float8DynamicActivationFloat8WeightConfig(granularity=...)`` and nothing else. TorchAO defaults both
``activation_dtype`` and ``weight_dtype`` to ``float8_e4m3fn``, so a recipe stating ``fp8_e5m2`` was
realised as e4m3 without an error -- a different datapath under the recipe's name. The same function
read the WEIGHT's granularity for both tensors, so a recipe asking for one activation scale per tensor
got a per-row activation scale while the capture stats reported the recipe's word for it.

CHECKED WITHOUT TORCH. The ordinary test interpreter deliberately has no torch (the capture venv owns
it), so ``torch`` and ``torchao.quantization`` are replaced by recording stand-ins: the test asserts on
the arguments the constructor RECEIVED, which is exactly what the defect dropped. The real classes'
argument names (``activation_dtype``, ``weight_dtype``, ``granularity`` as an ``(activation, weight)``
pair) were read from torchao 0.17.0 and 0.18.0.
"""

from __future__ import annotations

import sys
import types

import pytest

from merlin.targetgen import _recipe_quantizer as RQ


class _Recorded:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class _PerTensor:
    pass


class _PerRow:
    pass


@pytest.fixture
def stub_framework(monkeypatch):
    torch = types.ModuleType("torch")
    torch.float8_e4m3fn = "float8_e4m3fn"
    torch.float8_e5m2 = "float8_e5m2"
    torch.int8 = "int8"
    quantization = types.ModuleType("torchao.quantization")
    quantization.Float8DynamicActivationFloat8WeightConfig = type("Float8Dyn", (_Recorded,), {})
    quantization.Int8DynamicActivationInt8WeightConfig = type("Int8Dyn", (_Recorded,), {})
    quantization.PerTensor = _PerTensor
    quantization.PerRow = _PerRow
    torchao = types.ModuleType("torchao")
    torchao.quantization = quantization
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torchao", torchao)
    monkeypatch.setitem(sys.modules, "torchao.quantization", quantization)


def _recipe(*, w_dtype, a_dtype, w_gran, a_gran):
    return {
        "schema": RQ.RECIPE_SCHEMA,
        "status": "derived",
        "weight": {"dtype": w_dtype, "granularity": w_gran, "symmetric": True},
        "activation": {"dtype": a_dtype, "granularity": a_gran, "symmetric": True, "mode": "dynamic"},
    }


def _granularity_kinds(config):
    activation, weight = config.kwargs["granularity"]
    return type(activation), type(weight)


@pytest.mark.usefixtures("stub_framework")
class TestFloat8FormatsAreExplicit:
    @pytest.mark.parametrize("fmt, torch_name", [("fp8_e4m3", "float8_e4m3fn"), ("fp8_e5m2", "float8_e5m2")])
    def test_the_recipe_format_reaches_both_dtypes(self, fmt, torch_name):
        config = RQ.quantize_config(_recipe(w_dtype=fmt, a_dtype=fmt, w_gran="channel", a_gran="token"))
        assert config.kwargs["activation_dtype"] == torch_name
        assert config.kwargs["weight_dtype"] == torch_name

    def test_mixed_formats_are_refused_not_collapsed(self):
        with pytest.raises(RQ.RecipeError, match="activations and 'fp8_e4m3' weights"):
            RQ.quantize_config(_recipe(w_dtype="fp8_e4m3", a_dtype="fp8_e5m2", w_gran="tensor", a_gran="tensor"))


@pytest.mark.usefixtures("stub_framework")
class TestEachTensorUsesItsOwnGranularity:
    def test_per_tensor_activations_and_weights(self):
        config = RQ.quantize_config(_recipe(w_dtype="fp8_e4m3", a_dtype="fp8_e4m3", w_gran="tensor", a_gran="tensor"))
        assert _granularity_kinds(config) == (_PerTensor, _PerTensor)

    def test_per_token_activations_with_per_channel_weights(self):
        config = RQ.quantize_config(_recipe(w_dtype="fp8_e4m3", a_dtype="fp8_e4m3", w_gran="channel", a_gran="token"))
        assert _granularity_kinds(config) == (_PerRow, _PerRow)

    def test_a_mixed_float8_granularity_is_refused_by_name(self):
        # Formerly built PerRow for both: the activation's "tensor" was never read.
        with pytest.raises(RQ.RecipeError, match="activations per 'tensor' and weights per 'channel'"):
            RQ.quantize_config(_recipe(w_dtype="fp8_e4m3", a_dtype="fp8_e4m3", w_gran="channel", a_gran="tensor"))

    def test_an_unknown_activation_granularity_is_refused(self):
        with pytest.raises(RQ.RecipeError, match="no configuration in this build"):
            RQ.quantize_config(_recipe(w_dtype="fp8_e4m3", a_dtype="fp8_e4m3", w_gran="tensor", a_gran="block"))


@pytest.mark.usefixtures("stub_framework")
class TestInt8KeepsItsPerTokenActivation:
    def test_per_token_int8_builds_with_the_weight_granularity(self):
        config = RQ.quantize_config(_recipe(w_dtype="int8", a_dtype="int8", w_gran="channel", a_gran="token"))
        assert isinstance(config.kwargs["granularity"], _PerRow)

    def test_per_tensor_int8_activations_are_refused(self):
        with pytest.raises(RQ.RecipeError, match="per token"):
            RQ.quantize_config(_recipe(w_dtype="int8", a_dtype="int8", w_gran="tensor", a_gran="tensor"))
