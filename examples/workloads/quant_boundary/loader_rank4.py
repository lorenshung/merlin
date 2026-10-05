"""Rank-4 companion to the per-tensor int8-to-float host-boundary probe."""

import torch
from torch.ao.quantization import quantize_pt2e as _register_quantized_decomposed  # noqa: F401


class DequantizePerTensor(torch.nn.Module):
    def forward(self, values):
        return torch.ops.quantized_decomposed.dequantize_per_tensor.default(
            values, 0.3, 0, -128, 127, torch.int8
        )


def get_model_and_inputs():
    values = torch.arange(-128, 128, dtype=torch.int16).to(torch.int8).reshape(1, 1, 16, 16)
    return DequantizePerTensor().eval(), (values,)
