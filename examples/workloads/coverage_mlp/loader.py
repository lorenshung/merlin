"""Small mixed-operation workload for inspecting frontend coverage and precision."""

import torch


def get_model_and_inputs():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(8),
        torch.nn.Linear(8, 4),
    ).eval()
    inputs = (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16,)
    return model, inputs
