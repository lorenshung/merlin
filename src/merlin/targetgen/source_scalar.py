"""Storage-width scalar primitives for the independent typed-source observer."""

from __future__ import annotations

import ctypes
import ctypes.util
from functools import lru_cache


@lru_cache
def _fma(bits):
    # C99 fma/fmaf specifies one correctly rounded operation, including signed
    # zero, subnormals, infinities and invalid results. Python multiplication
    # followed by addition (even in binary64 for f32) is not a fused oracle.
    library = ctypes.CDLL(ctypes.util.find_library("m"))
    function = getattr(library, "fmaf" if bits == 32 else "fma")
    scalar = ctypes.c_float if bits == 32 else ctypes.c_double
    function.argtypes = [scalar, scalar, scalar]
    function.restype = scalar
    return function


def fused_multiply_add(a, b, c):
    import torch

    if a.dtype not in (torch.float32, torch.float64) or b.dtype != a.dtype or c.dtype != a.dtype:
        raise ValueError("source fma requires matching f32/f64 operands")
    function = _fma(32 if a.dtype == torch.float32 else 64)
    a, b, c = torch.broadcast_tensors(a, b, c)
    values = [
        function(x, y, z)
        for x, y, z in zip(a.flatten().tolist(), b.flatten().tolist(), c.flatten().tolist(), strict=True)
    ]
    return torch.tensor(values, dtype=a.dtype).reshape(a.shape)


def unsigned_to_float(value, width, target):
    import torch

    from merlin.targetgen.exact_arithmetic import round_binary

    if width not in (1, 8, 16, 32, 64) or target not in (torch.float32, torch.float64):
        raise ValueError("source unsigned conversion requires a supported integer width and f32/f64 result")
    bits = 32 if target == torch.float32 else 64
    values = [round_binary(int(x) & ((1 << width) - 1), bits) for x in value.flatten().tolist()]
    return torch.tensor(values, dtype=target).reshape(value.shape)


def logical_shift(value, amount, width, *, left=False):
    import torch

    value, amount = torch.broadcast_tensors(value, amount)
    result = []
    for x, shift in zip(value.flatten().tolist(), amount.flatten().tolist(), strict=True):
        if not 0 <= shift < width:
            raise ValueError("source shift amount is outside its integer width")
        word = int(x) & ((1 << width) - 1)
        word = ((word << shift) if left else (word >> shift)) & ((1 << width) - 1)
        result.append(word - (1 << width) if word >= (1 << (width - 1)) else word)
    return torch.tensor(result, dtype=value.dtype).reshape(value.shape)
