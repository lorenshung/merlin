"""Independent host-control/math probes; not a model or a support declaration.

Select a case with M2M_HOST_PROBE_CASE before capture. Each case returns one
finite FP32 tensor so the existing saved-capture scalar-host checker can inspect
every output. Integer addition exposes all result bits as four exact 16-bit words;
this does not claim exhaustive input coverage or exact arbitrary int64-to-FP32 conversion.
"""

import os

import torch

CASES = (
    "negate",
    "select",
    "sine",
    "cosine",
    "integer_to_float",
    "integer_add",
    "integer_compare",
    "integer_tensor_compare",
    "position_mask",
    "boolean_prefix_sum",
    "boolean_sum",
    "bucketize_left",
    "bucketize_right",
    "identity_alias",
)


class HostControlMath(torch.nn.Module):
    def __init__(self, case):
        super().__init__()
        if case not in CASES:
            raise ValueError(f"unknown host-control/math case: {case!r}")
        self.case = case

    def forward(self, x, y=None):
        if self.case == "identity_alias":
            return torch.ops.aten.alias.default(x)
        if self.case == "negate":
            return -x.unsqueeze(0)
        if self.case == "select":
            value = x.unsqueeze(0)
            return torch.where(value >= 0, value, -value)
        if self.case == "sine":
            return torch.sin(x)
        if self.case == "cosine":
            return torch.cos(x)
        if self.case == "integer_to_float":
            return x.reshape(1, 1, -1).to(torch.float32)
        if self.case == "integer_add":
            summed = x + 1
            words = [((summed >> shift) & 65535).to(torch.float32) for shift in (0, 16, 32, 48)]
            return torch.stack(words).flatten()
        if self.case == "integer_compare":
            return torch.cat(((x == 0).to(torch.float32), (x <= 0).to(torch.float32)))
        if self.case == "integer_tensor_compare":
            return torch.cat(((x == y).to(torch.float32), (x <= y).to(torch.float32)))
        if self.case == "boolean_prefix_sum":
            return torch.cumsum(x, dim=1).to(torch.float32)
        if self.case == "boolean_sum":
            return torch.sum(x, dim=1).to(torch.float32)
        if self.case.startswith("bucketize_"):
            return torch.bucketize(x, y, right=self.case == "bucketize_right").to(torch.float32)
        positions = torch.arange(x.shape[0], dtype=torch.int64, device=x.device) + x
        rows = positions.reshape(1, 1, -1, 1)
        columns = positions.reshape(1, 1, 1, -1)
        return (rows <= columns).to(torch.float32)


def get_model_and_inputs():
    case = os.environ.get("M2M_HOST_PROBE_CASE", "negate")
    model = HostControlMath(case).eval()
    if case.startswith("integer_"):
        values = [-(2**63), -(2**24) - 1, -3, -1, 0, 1, 3, 2**24 + 1, 2**63 - 1]
        sample = torch.tensor(values, dtype=torch.int64)
        if case == "integer_tensor_compare":
            other = torch.tensor([-(2**63), -(2**24), -4, 0, -1, 1, 2, 2**24, 2**63 - 1], dtype=torch.int64)
            return model, (sample, other)
    elif case == "position_mask":
        sample = torch.tensor([0, -1, 2, -3, 4, -5, 6], dtype=torch.int64)
    elif case.startswith("boolean_"):
        sample = torch.tensor(
            [[False, False, False, False, False], [True, True, True, True, True], [True, False, True, False, True]],
            dtype=torch.bool,
        )
    elif case.startswith("bucketize_"):
        sample = torch.tensor([[-3.0, -2.0, -1.0, -0.5, 0.0], [1.0, 1.25, 2.0, 4.0, 5.0]], dtype=torch.float32)
        boundaries = torch.tensor([-2.0, -0.5, 1.25, 4.0], dtype=torch.float32)
        return model, (sample, boundaries)
    else:
        values = [-3.25, -1.5, -0.125, -0.0, 0.125, 0.5, 1.5, 3.25]
        sample = torch.tensor(values, dtype=torch.float32).reshape(1, 2, 4)
    return model, (sample,)
