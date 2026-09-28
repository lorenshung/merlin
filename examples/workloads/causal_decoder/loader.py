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
    return CausalDecoder().eval(), (torch.arange(8, dtype=torch.int64).reshape(1, 8),)
