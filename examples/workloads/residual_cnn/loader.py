"""Independent residual-CNN iteration model, not a ResNet50 implementation."""

import torch


class ResidualCNN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = torch.nn.Conv2d(3, 8, 3, padding=1)
        self.conv1 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(8, 8, 3, padding=1)
        self.head = torch.nn.Linear(8, 4)

    def forward(self, image):
        features = torch.relu(self.stem(image))
        residual = self.conv2(torch.relu(self.conv1(features)))
        pooled = torch.relu(features + residual).mean(dim=(-2, -1))
        return self.head(pooled)


def get_model_and_inputs():
    image = torch.arange(3 * 16 * 16, dtype=torch.float32).reshape(1, 3, 16, 16) / 768
    model = ResidualCNN().eval()
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    values = torch.arange(parameter.numel(), dtype=torch.int64).reshape(parameter.shape)
                    parameter.copy_(
                        (((values * 37 + ordinal * 101 + field * 53) % 251) - 125).to(parameter.dtype) / 512
                    )
    return model, (image,)
