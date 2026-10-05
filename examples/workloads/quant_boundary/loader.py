"""Target-independent PyTorch probe for a per-tensor int8-to-float boundary."""

import torch
from torch.ao.quantization import quantize_pt2e as _register_quantized_decomposed  # noqa: F401


class DequantizePerTensor(torch.nn.Module):
    def forward(self, values):
        return torch.ops.quantized_decomposed.dequantize_per_tensor.default(values, 0.25, 0, -128, 127, torch.int8)


def get_model_and_inputs():
    values = torch.tensor([[[-128, -17, -1, 0], [1, 13, 37, 127]]], dtype=torch.int8)
    return DequantizePerTensor().eval(), (values,)
