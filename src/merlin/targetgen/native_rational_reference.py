"""Independent finite real/Gaussian-rational oracle for native reduction floors."""

from dataclasses import dataclass
from fractions import Fraction

import numpy as np


@dataclass(frozen=True)
class Gaussian:
    real: Fraction
    imag: Fraction = Fraction(0)

    @staticmethod
    def of(value):
        return value if isinstance(value, Gaussian) else Gaussian(Fraction(value))

    def __add__(self, other):
        other = self.of(other)
        return Gaussian(self.real + other.real, self.imag + other.imag)

    __radd__ = __add__

    def __neg__(self):
        return Gaussian(-self.real, -self.imag)

    def __sub__(self, other):
        return self + -self.of(other)

    def __rsub__(self, other):
        return self.of(other) + -self

    def __mul__(self, other):
        other = self.of(other)
        return Gaussian(
            self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.of(other)
        norm = other.real**2 + other.imag**2
        return Gaussian(
            (self.real * other.real + self.imag * other.imag) / norm,
            (self.imag * other.real - self.real * other.imag) / norm,
        )


def _decode(value, *, finite_subproblem=False):
    def decode(child):
        return _decode(child, finite_subproblem=finite_subproblem)

    if isinstance(value, list):
        return [decode(v) for v in value]
    if not isinstance(value, dict):
        if finite_subproblem and isinstance(value, float) and not np.isfinite(value):
            return Fraction(0)
        return Fraction(value) if isinstance(value, float) else value
    kind = value.get("kind")
    if kind == "float":
        return decode(float(value["value"]))
    if kind == "complex":
        return Gaussian(decode(value["real"]), decode(value["imag"]))
    if kind == "tensor":
        return np.asarray(decode(value["values"]), dtype=object).reshape(value["shape"])
    if kind == "tuple":
        return tuple(decode(v) for v in value["items"])
    return {key: decode(v) for key, v in value.items()}


def rational_outputs(case, *, finite_subproblem=False):
    """Evaluate native math, without eager, emitted IR, or device answers."""
    from merlin.targetgen.exact_quantized_reference import _convolution, _fold, _pool, _pool_backward

    def scalar_proof(value):
        if isinstance(value, dict):
            if value.get("kind") == "tensor":
                return
            if value.get("kind") == "float":
                scalar_proof(float(value["value"]))
                return
            for child in value.values():
                scalar_proof(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                scalar_proof(child)
        elif isinstance(value, float) and float(np.float32(value)) != value:
            raise ValueError("native rational scalar lacks an exact opmath representation proof")

    scalar_proof(case["arguments"])
    arguments = _decode(case["arguments"], finite_subproblem=finite_subproblem)
    a, kwargs = arguments["args"], arguments["kwargs"]
    name = case["overload"]
    op = name.split(".")[1]
    if op in {"mm", "bmm", "matmul"}:
        result = np.matmul(a[0], a[1])
    elif op == "addmm":
        # Keep arrays on the left so NumPy broadcasts Gaussian scalars through
        # individual exact elements instead of coercing a whole array to Fraction.
        result = a[0] * kwargs.get("beta", 1) + np.matmul(a[1], a[2]) * kwargs.get("alpha", 1)
    elif op == "convolution":
        if a[6] or any(a[7]):
            raise ValueError("native rational oracle refuses transposed convolution")
        result = _convolution(a[0], a[1], a[2], a[3], a[4], a[5], a[8])
    elif op == "mean":
        axes = a[1] if len(a) > 1 else kwargs.get("dim")
        axis = None if axes is None or axes == [] else tuple(axes) if isinstance(axes, (list, tuple)) else axes
        if a[0].ndim == 0 and axis in (0, -1, (0,), (-1,)):
            axis = None
        result = np.mean(a[0], axis=axis, keepdims=a[2] if len(a) > 2 else kwargs.get("keepdim", False))
    elif op.startswith(("avg_pool", "adaptive_avg_pool", "_adaptive_avg_pool")) and "backward" not in op:
        result = _pool(a[0], name, a, kwargs)
    elif op.startswith(("avg_pool", "adaptive_avg_pool", "_adaptive_avg_pool")) and op.endswith("backward"):
        result = _pool_backward(a[0], a[1].shape, name, a, kwargs)
    elif op == "col2im":
        result = _fold(a[0], *a[1:])
    elif op == "convolution_backward":
        grad, source, weight = a[:3]
        stride, padding, dilation, transposed, output_padding, groups, mask = a[4:11]
        if transposed or any(output_padding) or source.ndim != 4:
            raise ValueError("native rational backward oracle requires non-transposed 2-D geometry")
        dx, dw = np.full(source.shape, Fraction(0), object), np.full(weight.shape, Fraction(0), object)
        db = np.sum(grad, axis=(0, 2, 3))
        cg, kh, kw = weight.shape[1:]
        fg = weight.shape[0] // groups
        for batch, co, oy, ox in np.ndindex(grad.shape):
            for ci, ky, kx in np.ndindex((cg, kh, kw)):
                iy, ix = oy * stride[0] - padding[0] + ky * dilation[0], ox * stride[1] - padding[1] + kx * dilation[1]
                channel = co // fg * cg + ci
                if 0 <= iy < source.shape[2] and 0 <= ix < source.shape[3]:
                    dx[batch, channel, iy, ix] += grad[batch, co, oy, ox] * weight[co, ci, ky, kx]
                    dw[co, ci, ky, kx] += grad[batch, co, oy, ox] * source[batch, channel, iy, ix]
        return [value for value, wanted in zip((dx, dw, db), mask) if wanted]
    else:
        raise ValueError("native rational oracle has no reduction semantics for " + name)
    return [np.asarray(result, dtype=object)]


def compare_reduction_floor(record, outputs):
    """Every real/imag error is bounded by native eager error, exactly."""
    import math
    import struct

    exact = rational_outputs(
        record["case"],
        finite_subproblem=record.get("numeric_admission", {})
        .get("host_floor", {})
        .get("finite_subproblem_overlay", False),
    )
    if len(exact) != len(record["output_indices"]):
        raise ValueError("rational native reference does not cover every user result")
    reference = record["native_eager_boundary"]["output_bytes"]

    def decode(raw, dtype):
        if dtype == "bf16":
            return [struct.unpack("<f", bytes(2) + raw[i : i + 2])[0] for i in range(0, len(raw), 2)]
        token = {"f16": "e", "f32": "f", "f64": "d", "complex<f32>": "f", "complex<f64>": "d"}.get(dtype)
        if token is None:
            raise ValueError("native reduction floor requires a floating result")
        return [value[0] for value in struct.iter_unpack("<" + token, raw)]

    results = []
    for ordinal, (index, abi, value) in enumerate(
        zip(record["output_indices"], record["output_abi"], exact, strict=True)
    ):
        actual = decode(outputs[index], abi["dtype"])
        host = decode(bytes.fromhex(reference[ordinal]), abi["dtype"])
        rational = []
        for item in value.flat:
            rational.extend([item.real, item.imag] if isinstance(item, Gaussian) else [Fraction(item)])
        if len(actual) != len(host) or len(actual) != len(rational):
            raise ValueError("rational native reference shape differs from result bytes")
        failures = []
        width = len(outputs[index]) // len(actual) if actual else 1
        eager_raw = bytes.fromhex(reference[ordinal])
        for i, (a, h, e) in enumerate(zip(actual, host, rational, strict=True)):
            if outputs[index][i * width : (i + 1) * width] == eager_raw[i * width : (i + 1) * width]:
                continue
            # A non-finite eager value has no rational error radius. Infinity
            # must retain its sign; NaNs retain the native storage contract's
            # sign, quiet bit and payload (the equal-byte case above).
            if not math.isfinite(h):
                if not (math.isinf(h) and a == h):
                    failures.append(i)
                continue
            if a == h and (a != 0 or math.copysign(1, a) == math.copysign(1, h)):
                continue
            if not math.isfinite(a):
                failures.append(i)
            elif a == h == 0 or abs(Fraction(a) - e) > abs(Fraction(h) - e):
                failures.append(i)
        results.append(
            dict(
                output_index=index,
                status="mismatch" if failures else "pass",
                components=len(actual),
                mismatched_components=len(failures),
                first_mismatch=failures[0] if failures else None,
            )
        )
    return dict(
        status="pass" if all(row["status"] == "pass" for row in results) else "mismatch",
        comparison="exact_rational_error_not_greater_than_native_per_component",
        results=results,
    )
