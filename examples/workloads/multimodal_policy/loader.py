"""Independent vision/text/state policy pattern, not a SmolVLA checkpoint or session."""

import torch


class MultimodalPolicy(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vision = torch.nn.Conv2d(3, 16, 4, stride=4)
        self.text = torch.nn.Embedding(64, 16)
        self.state = torch.nn.Linear(4, 16)
        self.norm = torch.nn.LayerNorm(16)
        self.query = torch.nn.Linear(16, 16)
        self.key = torch.nn.Linear(16, 16)
        self.value = torch.nn.Linear(16, 16)
        self.action = torch.nn.Linear(16, 4)

    def forward(self, image, tokens, state):
        vision = self.vision(image).flatten(2).transpose(1, 2)
        context = self.norm(torch.cat((vision, self.text(tokens)), dim=1))
        query = self.query(self.state(state).unsqueeze(1))
        scores = (query @ self.key(context).transpose(-2, -1)) / 4
        attended = torch.softmax(scores, dim=-1) @ self.value(context)
        return state + self.action(attended.squeeze(1))


def get_model_and_inputs():
    image = torch.arange(192, dtype=torch.float32).reshape(1, 3, 8, 8) / 192
    tokens = torch.arange(4, dtype=torch.int64).reshape(1, 4)
    state = torch.linspace(-1, 1, 4).reshape(1, 4)
    model = MultimodalPolicy().eval()
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    values = torch.arange(parameter.numel(), dtype=torch.int64).reshape(parameter.shape)
                    parameter.copy_(
                        (((values * 37 + ordinal * 101 + field * 53) % 251) - 125).to(parameter.dtype) / 512
                    )
    return model, (image, tokens, state)
