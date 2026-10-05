"""Independent small causal decoder for iteration, not TinyLlama or its checkpoint."""

import torch


class CausalDecoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(64, 32)
        self.norm = torch.nn.RMSNorm(32)
        self.qkv = torch.nn.Linear(32, 96, bias=False)
        self.projection = torch.nn.Linear(32, 32, bias=False)
        self.gate = torch.nn.Linear(32, 64, bias=False)
        self.up = torch.nn.Linear(32, 64, bias=False)
        self.down = torch.nn.Linear(64, 32, bias=False)
        self.head = torch.nn.Linear(32, 64, bias=False)
        mask = torch.full((8, 8), float("-inf")).triu(diagonal=1)
        self.register_buffer("mask", mask)

    def forward(self, tokens):
        hidden = self.embedding(tokens)
        query, key, value = self.qkv(self.norm(hidden)).chunk(3, dim=-1)
        query = query.reshape(1, 8, 4, 8).transpose(1, 2)
        key = key.reshape(1, 8, 4, 8).transpose(1, 2)
        value = value.reshape(1, 8, 4, 8).transpose(1, 2)
        scores = (query @ key.transpose(-2, -1)) / (8**0.5)
        attention = torch.softmax(scores + self.mask, dim=-1) @ value
        hidden = hidden + self.projection(attention.transpose(1, 2).reshape(1, 8, 32))
        normalized = self.norm(hidden)
        hidden = hidden + self.down(torch.nn.functional.silu(self.gate(normalized)) * self.up(normalized))
        return self.head(hidden)


def get_model_and_inputs():
    model = CausalDecoder().eval()
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    values = torch.arange(parameter.numel(), dtype=torch.int64).reshape(parameter.shape)
                    parameter.copy_(
                        (((values * 37 + ordinal * 101 + field * 53) % 251) - 125).to(parameter.dtype) / 512
                    )
    return model, (torch.arange(8, dtype=torch.int64).reshape(1, 8),)
