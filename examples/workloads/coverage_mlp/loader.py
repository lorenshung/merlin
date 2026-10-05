"""Small mixed-operation workload for inspecting frontend coverage and precision."""

import torch


def get_model_and_inputs():
    model = torch.nn.Sequential(
        torch.nn.Linear(8, 8),
        torch.nn.ReLU(),
        torch.nn.LayerNorm(8),
        torch.nn.Linear(8, 4),
    ).eval()
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    values = torch.arange(parameter.numel(), dtype=torch.int64).reshape(parameter.shape)
                    parameter.copy_(
                        (((values * 37 + ordinal * 101 + field * 53) % 251) - 125).to(parameter.dtype) / 512
                    )
    inputs = (torch.arange(16, dtype=torch.float32).reshape(2, 8) / 16,)
    return model, inputs
