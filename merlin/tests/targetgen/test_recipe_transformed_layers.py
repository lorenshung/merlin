"""A planned layer counts as quantized by any observable transform, not one storage convention.

TorchAO's quantize_ swaps a Linear's weight for a tensor subclass, but a quantizing transform can
also swap the module for its own quantized class storing the weight as float8 in a plain buffer, or
cast the weight in place. Counting only weights that stopped being plain tensors reported such a
layer as untransformed, so a fully quantized model could be refused as unquantized.
"""

from __future__ import annotations

import pytest

from merlin.targetgen import _recipe_quantizer as RQ

torch = pytest.importorskip("torch")


class _Model(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.kept = torch.nn.Linear(4, 4)
        self.swapped = torch.nn.Linear(4, 4)
        self.subclassed = torch.nn.Linear(4, 4)
        self.cast = torch.nn.Linear(4, 4)
        self.removed = torch.nn.Linear(4, 4)


class _QuantLinear(torch.nn.Module):
    """A swapped-in quantized layer whose weight is a plain buffer in a one-byte float format."""

    def __init__(self, weight):
        super().__init__()
        self.register_buffer("weight", weight.detach().to(torch.float8_e4m3fn))
        self.register_buffer("scale", torch.ones(()))


class _Wrapped(torch.Tensor):
    pass


def _transform(model):
    model.swapped = _QuantLinear(model.swapped.weight)
    model.subclassed.weight = torch.nn.Parameter(
        model.subclassed.weight.detach().as_subclass(_Wrapped), requires_grad=False
    )
    model.cast.weight = torch.nn.Parameter(model.cast.weight.detach().to(torch.bfloat16), requires_grad=False)
    del model.removed
    return model


def test_swap_subclass_and_storage_dtype_each_count_as_transformed():
    model = _Model()
    before = RQ.layer_state(model)
    changed = RQ.transformed(before, _transform(model))
    assert changed == {"swapped", "subclassed", "cast"}


def test_untouched_model_has_no_transformed_layer():
    model = _Model()
    assert RQ.transformed(RQ.layer_state(model), model) == set()


def test_a_weight_that_was_already_a_subclass_or_float8_is_not_counted_again():
    model = _Model()
    _transform(model)
    assert RQ.transformed(RQ.layer_state(model), model) == set()


def test_swapped_module_storing_float8_in_a_plain_buffer_is_counted():
    model = torch.nn.Sequential(torch.nn.Linear(2, 2))
    before = RQ.layer_state(model)
    model[0] = _QuantLinear(model[0].weight)
    assert type(model[0].weight) is torch.Tensor  # the convention the earlier check could not see
    assert RQ.transformed(before, model) == {"0"}
