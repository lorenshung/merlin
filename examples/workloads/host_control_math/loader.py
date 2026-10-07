"""Independent host-control/math probes; not a model or a support declaration.

Select a case with M2M_HOST_PROBE_CASE before capture. Each case returns one
tensor; typed pointwise cases retain raw Boolean or integer outputs. Integer
addition exposes all result bits as four exact 16-bit words;
this does not claim exhaustive input coverage or exact arbitrary int64-to-FP32 conversion.
"""

import os

import torch

CASES = (
    "negate",
    "select",
    "sine",
    "cosine",
    "sine_rank4",
    "cosine_rank4",
    "arange_f32_exact",
    "arange_f32_fractional",
    "integer_to_float",
    "integer_add",
    "integer_compare",
    "integer_tensor_compare",
    "position_mask",
    "boolean_prefix_sum",
    "boolean_sum",
    "boolean_not",
    "boolean_and",
    "boolean_select",
    "boolean_mul_lhs_singleton",
    "boolean_mul_rhs_singleton",
    "tensor_not_equal",
    "tensor_bitwise_xor",
    "min_values",
    "f32_cumsum_rows",
    "f32_cumsum_columns",
    "f32_cumsum_signed_zero",
    "bucketize_left",
    "bucketize_right",
    "bucketize_left_nan",
    "bucketize_right_nan",
    "identity_alias",
)

_MATRIX_OPERATIONS = (
    "f32_add",
    "f32_sub",
    "f32_mul",
    "f32_neg",
    "f32_le",
    "f32_select",
    "f32_nonzero",
    "i64_add",
    "i64_sub",
    "i64_mul",
    "i64_le",
    "i64_to_f32",
    "i1_and",
    "i1_xor",
    "i1_not",
    "i1_to_f32",
    "i1_to_i64",
    "i1_mul_lhs_projected",
    "i1_mul_rhs_projected",
)
_MATRIX_SHAPE = (2, 2, 3, 7)
MATRIX_CASES = tuple(f"matrix_{operation}_r{rank}" for operation in _MATRIX_OPERATIONS for rank in range(1, 5))


def _matrix_operation(case):
    prefix, separator, rank = case.rpartition("_r")
    if (
        not separator
        or not prefix.startswith("matrix_")
        or prefix.removeprefix("matrix_") not in _MATRIX_OPERATIONS
        or rank not in {"1", "2", "3", "4"}
    ):
        raise ValueError(f"unknown pointwise matrix case: {case!r}")
    return prefix.removeprefix("matrix_"), int(rank)


def _periodic_tensor(values, shape, dtype):
    count = torch.Size(shape).numel()
    return torch.tensor([values[index % len(values)] for index in range(count)], dtype=dtype).reshape(shape)


def _matrix_inputs(operation, rank):
    shape = _MATRIX_SHAPE[-rank:]
    floats = (-0.0, 0.0, -3.25, 1.5, -0.125, 0.5, 2.0, -1.5, 1e8)
    other_floats = (0.0, -0.0, -3.25, -2.0, 0.25, 0.5, -1.0, 1.5, -1e8)
    integers = (-(2**63), 2**63 - 1, -7, -1, 0, 1, 7, 2**24 + 1)
    other_integers = (1, 2, -3, 2**63 - 1, -(2**63), -1, 3, 2**24)
    booleans = (False, True, False, True, True, False, True)
    other_booleans = (True, False, False, True, False, True, True)
    if operation == "f32_select":
        return (
            _periodic_tensor(booleans, shape, torch.bool),
            _periodic_tensor(floats, shape, torch.float32),
            _periodic_tensor(other_floats, shape, torch.float32),
        )
    if operation.startswith("f32_"):
        x = _periodic_tensor(floats, shape, torch.float32)
        y = _periodic_tensor(other_floats, shape, torch.float32)
    elif operation.startswith("i64_"):
        x = _periodic_tensor(integers, shape, torch.int64)
        y = _periodic_tensor(other_integers, shape, torch.int64)
    else:
        x_shape = (1, *shape[1:]) if operation.endswith("lhs_projected") else shape
        y_shape = (1, *shape[1:]) if operation.endswith("rhs_projected") else shape
        x = _periodic_tensor(booleans, x_shape, torch.bool)
        y = _periodic_tensor(other_booleans, y_shape, torch.bool)
    if operation in {"f32_neg", "f32_nonzero", "i64_to_f32", "i1_not", "i1_to_f32", "i1_to_i64"}:
        return (x,)
    return x, y


class HostControlMath(torch.nn.Module):
    def __init__(self, case):
        super().__init__()
        if case not in CASES and case not in MATRIX_CASES:
            raise ValueError(f"unknown host-control/math case: {case!r}")
        self.case = case
        self.matrix_operation = _matrix_operation(case)[0] if case in MATRIX_CASES else None

    def forward(self, x, y=None, z=None):
        if self.matrix_operation is not None:
            operation = self.matrix_operation
            if operation in {"f32_add", "i64_add"}:
                return x + y
            if operation in {"f32_sub", "i64_sub"}:
                return x - y
            if operation in {"f32_mul", "i64_mul", "i1_mul_lhs_projected", "i1_mul_rhs_projected"}:
                return x * y
            if operation in {"f32_le", "i64_le"}:
                return x <= y
            if operation == "f32_select":
                return torch.where(x, y, z)
            if operation == "f32_nonzero":
                return x.to(torch.bool)
            if operation in {"i64_to_f32", "i1_to_f32"}:
                return x.to(torch.float32)
            if operation == "i1_to_i64":
                return x.to(torch.int64)
            if operation == "f32_neg":
                return -x
            if operation == "i1_and":
                return x & y
            if operation == "i1_xor":
                return x ^ y
            if operation == "i1_not":
                return ~x
            raise AssertionError("validated matrix operation is not implemented")
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
        if self.case == "sine_rank4":
            return torch.sin(x.unsqueeze(0))
        if self.case == "cosine_rank4":
            return torch.cos(x.unsqueeze(0))
        if self.case == "arange_f32_exact":
            return torch.arange(-0.75, x.shape[0] - 0.75, 0.5, dtype=torch.float32, device=x.device)
        if self.case == "arange_f32_fractional":
            return torch.arange(-0.3, x.shape[0] - 0.3, 0.2, dtype=torch.float32, device=x.device)
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
        if self.case == "boolean_not":
            return torch.bitwise_not(x).to(torch.float32)
        if self.case == "boolean_and":
            return torch.bitwise_and(x, y).to(torch.float32)
        if self.case == "boolean_select":
            return torch.where(x, y, torch.bitwise_not(y)).to(torch.float32)
        if self.case in {"boolean_mul_lhs_singleton", "boolean_mul_rhs_singleton"}:
            return torch.mul(x, y)
        if self.case == "tensor_not_equal":
            return torch.ne(x, y)
        if self.case == "tensor_bitwise_xor":
            return torch.bitwise_xor(x, y)
        if self.case == "min_values":
            return torch.min(x, dim=1).values
        if self.case == "f32_cumsum_columns":
            return torch.cumsum(x, dim=0)
        if self.case in {"f32_cumsum_rows", "f32_cumsum_signed_zero"}:
            return torch.cumsum(x, dim=-1)
        if self.case.startswith("bucketize_"):
            return torch.bucketize(x, y, right=self.case.startswith("bucketize_right")).to(torch.float32)
        if self.case != "position_mask":
            raise AssertionError("validated host-control/math case is not implemented")
        positions = torch.arange(x.shape[0], dtype=torch.int64, device=x.device) + x
        rows = positions.reshape(1, 1, -1, 1)
        columns = positions.reshape(1, 1, 1, -1)
        return (rows <= columns).to(torch.float32)


def get_model_and_inputs():
    case = os.environ.get("M2M_HOST_PROBE_CASE", "negate")
    model = HostControlMath(case).eval()
    if case in MATRIX_CASES:
        operation, rank = _matrix_operation(case)
        return model, _matrix_inputs(operation, rank)
    if case.startswith("integer_"):
        values = [-(2**63), -(2**24) - 1, -3, -1, 0, 1, 3, 2**24 + 1, 2**63 - 1]
        sample = torch.tensor(values, dtype=torch.int64)
        if case == "integer_tensor_compare":
            other = torch.tensor([-(2**63), -(2**24), -4, 0, -1, 1, 2, 2**24, 2**63 - 1], dtype=torch.int64)
            return model, (sample, other)
    elif case == "position_mask":
        sample = torch.tensor([0, -1, 2, -3, 4, -5, 6], dtype=torch.int64)
    elif case == "tensor_not_equal":
        sample = torch.tensor([[-(2**63), -(2**24) - 1, -1, 0, 2**63 - 1], [3, 4, 5, -7, 2**24 + 1]], dtype=torch.int64)
        other = torch.tensor([[-(2**63), -(2**24), 0, 0, 2**63 - 2], [3, -4, 6, -7, 2**24]], dtype=torch.int64)
        return model, (sample, other)
    elif case == "tensor_bitwise_xor":
        sample = torch.tensor([False, True, False, True, True, False, True, False, True], dtype=torch.bool)
        other = torch.tensor([True, True, False, False, True, True, False, False, False], dtype=torch.bool)
        return model, (sample, other)
    elif case in {"boolean_mul_lhs_singleton", "boolean_mul_rhs_singleton"}:
        dense = torch.tensor(
            [
                [True, False, True, False, True, False, True],
                [False, True, True, False, False, True, True],
                [True, True, False, False, True, True, False],
            ],
            dtype=torch.bool,
        )
        if case == "boolean_mul_lhs_singleton":
            row = torch.tensor([[True, False, True, True, False, True, False]], dtype=torch.bool)
            return model, (row, dense)
        column = torch.tensor([[True], [False], [True]], dtype=torch.bool)
        return model, (dense, column)
    elif case.startswith("boolean_"):
        sample = torch.tensor(
            [[False, False, False, False, False], [True, True, True, True, True], [True, False, True, False, True]],
            dtype=torch.bool,
        )
        if case in {"boolean_and", "boolean_select"}:
            other = torch.tensor(
                [[True, False, True, False, True], [False, True, False, True, False], [True, True, False, False, True]],
                dtype=torch.bool,
            )
            return model, (sample, other)
    elif case == "min_values":
        sample = torch.tensor(
            [[1.0, 0.0, -0.0, 2.0, 2.0], [4.0, -1.0, -1.0, 5.0, 3.0], [7.0, 5.0, 6.0, 5.0, 8.0]], dtype=torch.float32
        )
    elif case == "f32_cumsum_rows":
        sample = torch.tensor([[1e8, 1.0, -1e8, 0.0, -0.0], [2e8, 2.0, -2e8, -0.0, 0.0]], dtype=torch.float32)
    elif case == "f32_cumsum_columns":
        sample = torch.tensor([[1e8, -0.0], [1.0, 0.0], [-1e8, -0.0]], dtype=torch.float32)
    elif case == "f32_cumsum_signed_zero":
        sample = torch.tensor([-0.0, 0.0, -0.0, 0.0, 1.0, -1.0], dtype=torch.float32)
    elif case.startswith("arange_f32_"):
        sample = torch.tensor([0.0, 0.25, 0.5, 0.75], dtype=torch.float32)
    elif case.startswith("bucketize_"):
        sample = torch.tensor([[-3.0, -2.0, -1.0, -0.5, 0.0], [1.0, 1.25, 2.0, 4.0, 5.0]], dtype=torch.float32)
        if case.endswith("_nan"):
            sample = torch.tensor(
                [[float("nan"), -float("inf"), -1.0, -0.0, 0.0], [float("inf"), 1.0, 1.25, 2.0, 5.0]],
                dtype=torch.float32,
            )
        boundaries = torch.tensor([-2.0, -0.5, 1.25, 4.0], dtype=torch.float32)
        return model, (sample, boundaries)
    else:
        values = [-3.25, -1.5, -0.125, -0.0, 0.125, 0.5, 1.5, 3.25]
        sample = torch.tensor(values, dtype=torch.float32).reshape(1, 2, 4)
    return model, (sample,)
