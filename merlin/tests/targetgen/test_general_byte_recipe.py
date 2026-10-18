"""Byte Q/DQ admission retains centering and range instead of assuming zero."""

from types import SimpleNamespace

import pytest

from merlin.targetgen._recipe_integer import _per_tensor_scale, integer_recipes


class Node:
    def __init__(self, target, args, **meta):
        self.op, self.target, self.args, self.kwargs = "call_function", target, args, {}
        self.meta, self.users, self.name = meta, [], "test_node"


@pytest.mark.parametrize(
    "token,zero,low,high", [("int8", -11, -128, 127), ("uint8", 131, 0, 255), ("int8", -2, -8, 7), ("uint8", 7, 0, 15)]
)
def test_byte_recipe_records_the_complete_source_qparams(token, zero, low, high):
    torch = pytest.importorskip("torch")
    dtype = getattr(torch, token)
    source = Node("quantized_decomposed.dequantize_per_tensor.default", (None, 0.25, zero, low, high, dtype))
    output = Node("quantized_decomposed.quantize_per_tensor.default", (None, 0.5, zero, low, high, dtype))
    node = Node("aten.clone.default", (source,), merlin_recipe_family="movement")
    node.users = [output]
    rows = integer_recipes(SimpleNamespace(graph=SimpleNamespace(nodes=[node])))
    assert len(rows) == 1 and "refused" not in rows[0]
    assert rows[0]["input_qparams"][0]["zero_point"] == zero
    assert rows[0]["integer_recipe"]["centering"] == "retained_source_integer"
    assert rows[0]["readout"] == "full_i32"


def test_invalid_byte_parameters_still_refuse():
    torch = pytest.importorskip("torch")
    for zero, low, high, dtype in [(256, 0, 255, torch.uint8), (0, -128, 255, torch.uint8), (0, -129, 127, torch.int8)]:
        source = Node("quantized_decomposed.dequantize_per_tensor.default", (None, 0.25, zero, low, high, dtype))
        assert _per_tensor_scale(source, "quantized_decomposed.dequantize_per_tensor") is None
