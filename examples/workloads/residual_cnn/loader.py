"""Independent residual-CNN iteration model, not a ResNet50 implementation.

An optional adjacent ``profile.json`` (written by ``merlin experiment corpus variants``) is a
selected capture input: the sealed capture inventories its bytes with this loader before the
model is built. Without one, the model and its inputs are exactly the original single-block,
16x16 development workload.
"""

import hashlib
import json
from pathlib import Path

import torch


class ResidualCNN(torch.nn.Module):
    def __init__(self, channels=8, blocks=1):
        super().__init__()
        self.stem = torch.nn.Conv2d(3, channels, 3, padding=1)
        self.conv1 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.conv2 = torch.nn.Conv2d(channels, channels, 3, padding=1)
        self.head = torch.nn.Linear(channels, 4)
        # Registered after the original modules so the single-block model keeps its module order,
        # state keys and seeded parameters.
        self.residuals = torch.nn.ModuleList(
            torch.nn.ModuleList(
                (
                    torch.nn.Conv2d(channels, channels, 3, padding=1),
                    torch.nn.Conv2d(channels, channels, 3, padding=1),
                )
            )
            for _ in range(blocks - 1)
        )

    def forward(self, image):
        features = torch.relu(self.stem(image))
        residual = self.conv2(torch.relu(self.conv1(features)))
        features = torch.relu(features + residual)
        for conv1, conv2 in self.residuals:
            residual = conv2(torch.relu(conv1(features)))
            features = torch.relu(features + residual)
        pooled = features.mean(dim=(-2, -1))
        return self.head(pooled)


def _selected_profile():
    path = Path(__file__).with_name("profile.json")
    if not path.exists():
        return {"channels": 8, "spatial_side": 16, "blocks": 1}, None
    if path.is_symlink() or not path.is_file():
        raise ValueError("residual CNN profile must be an adjacent ordinary file")
    raw = path.read_bytes()
    profile = json.loads(raw)
    if not isinstance(profile, dict) or set(profile) != {"schema", "workload_id", "channels", "spatial_side", "blocks"}:
        raise ValueError("residual CNN profile has an unsupported shape")
    if profile["schema"] != "merlin.iteration_workload_profile.v1" or profile["workload_id"] != "residual_cnn":
        raise ValueError("residual CNN profile has an unsupported identity")
    for key, maximum in (("channels", 256), ("spatial_side", 512), ("blocks", 16)):
        if type(profile[key]) is not int or not 1 <= profile[key] <= maximum:
            raise ValueError(f"residual CNN profile has an invalid {key}")
    return profile, hashlib.sha256(raw).hexdigest()


def get_model_and_inputs():
    profile, profile_sha256 = _selected_profile()
    side = profile["spatial_side"]
    elements = 3 * side * side
    image = torch.arange(elements, dtype=torch.float32).reshape(1, 3, side, side) / elements
    model = ResidualCNN(profile["channels"], profile["blocks"]).eval()
    with torch.no_grad():
        for ordinal, module in enumerate(model.modules()):
            if isinstance(module, (torch.nn.Linear, torch.nn.Conv2d, torch.nn.Embedding)):
                for field, parameter in enumerate(module.parameters(recurse=False)):
                    values = torch.arange(parameter.numel(), dtype=torch.int64).reshape(parameter.shape)
                    parameter.copy_(
                        (((values * 37 + ordinal * 101 + field * 53) % 251) - 125).to(parameter.dtype) / 512
                    )
    if profile_sha256 is not None:
        model.session_provenance = {
            "workload_id": "residual_cnn",
            "workload_role": "iteration",
            "input_source": "seeded_synthetic",
            "profile_sha256": profile_sha256,
        }
    return model, (image,)
