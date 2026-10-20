"""Independent exact-rational oracle for source quantized FX graphs.

Unknown operations refuse qualification. Eager floating divergence is diagnostic.
"""

from __future__ import annotations

import math
import operator
from fractions import Fraction
from itertools import chain, product
from types import SimpleNamespace

import numpy as np

from merlin.targetgen.exact_arithmetic import (
    ExactTensor,
    exact_array,
    materialize,
)
from merlin.targetgen.exact_arithmetic import (
    dequantize as dequantize,
)
from merlin.targetgen.exact_arithmetic import (
    quantize as quantize,
)
from merlin.targetgen.exact_arithmetic import (
    rational as rational,
)
from merlin.targetgen.exact_arithmetic import (
    round_binary as round_binary,
)


def mac_observation(lhs, rhs, accumulator, scales):
    """Integer diagnostics only; rounded scale products never feed the oracle."""
    maximum = max((abs(int(x)) for x in accumulator.flat), default=0)
    factors = None
    if all(scale is not None for scale in scales):
        factors = exact_array(scales[0]) * exact_array(scales[1])

        def diagnostic_float(value):
            try:
                result = float(value)
            except OverflowError:
                return None
            return result if math.isfinite(result) else None

        factors = np.vectorize(diagnostic_float, otypes=[object])(factors).tolist()
    return dict(
        lhs=scales[0],
        rhs=scales[1],
        product=factors,
        operands=[dict(quantized_max_abs=max((abs(int(x)) for x in value.flat), default=0)) for value in (lhs, rhs)],
        accumulator=accumulator.tolist(),
        accumulator_max_abs=maximum,
    )


def _convolution_fraction(x, w, bias, stride, padding, dilation, groups):
    dims = x.ndim - 2

    def expand(value):
        return [value] * dims if isinstance(value, int) else list(value)

    stride, padding, dilation = map(expand, (stride, padding, dilation))
    if isinstance(padding, str):
        raise ValueError("exact convolution requires explicit numeric padding")
    kernel = w.shape[2:]
    shape = tuple(
        (x.shape[2 + i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1 for i in range(dims)
    )
    if any(size < 0 for size in shape):
        raise ValueError("invalid convolution geometry")
    output = np.full((x.shape[0], w.shape[0], *shape), Fraction(0), dtype=object)
    per_group = w.shape[0] // groups
    for n, co, *point in np.ndindex(output.shape):
        total = Fraction(0) if bias is None else bias[co]
        for ci in range(w.shape[1]):
            for tap in np.ndindex(kernel):
                source = tuple(point[d] * stride[d] - padding[d] + tap[d] * dilation[d] for d in range(dims))
                if all(0 <= source[d] < x.shape[2 + d] for d in range(dims)):
                    total += x[(n, co // per_group * w.shape[1] + ci, *source)] * w[(co, ci, *tap)]
        output[(n, co, *point)] = total
    return output


def _integer_grid(values):
    """Factor rational values into exact integer codes times one common scale.

    Taking the gcd on a common denominator also recovers the small integer
    codes from already-dequantized Fraction inputs, without trusting metadata.
    """
    values = np.asarray(values, dtype=object)
    fractions = [rational(v) for v in values.flat]
    denominator = math.lcm(*(v.denominator for v in fractions))
    numerators = [v.numerator * (denominator // v.denominator) for v in fractions]
    divisor = math.gcd(*numerators) or 1
    codes = np.asarray([v // divisor for v in numerators], dtype=object).reshape(values.shape)
    return codes, Fraction(divisor, denominator)


def _convolution(x, w, bias, stride, padding, dilation, groups):
    """Exact integer convolution followed by one rational scaling per output.

    If K*max(abs(X))*max(abs(W)) <= INT64_MAX, every product and every
    prefix sum fit int64, independent of summation order. Otherwise numpy's
    object matmul uses Python integers. No floating arithmetic or rounding
    participates in either path. Per-output weight grids preserve per-channel
    scales; bias remains a separate exact rational addition.
    """
    x, w = np.asarray(x, dtype=object), np.asarray(w, dtype=object)
    dims = x.ndim - 2
    if dims < 1 or isinstance(padding, str):
        return _convolution_fraction(x, w, bias, stride, padding, dilation, groups)

    def expand(value):
        return [value] * dims if isinstance(value, int) else list(value)

    stride, padding, dilation = map(expand, (stride, padding, dilation))
    kernel = w.shape[2:]
    shape = tuple(
        (x.shape[2 + i] + 2 * padding[i] - dilation[i] * (kernel[i] - 1) - 1) // stride[i] + 1 for i in range(dims)
    )
    if any(size < 0 for size in shape):
        raise ValueError("invalid convolution geometry")
    if (
        not groups
        or x.shape[1] != groups * w.shape[1]
        or w.shape[0] % groups
        or any(v <= 0 for v in (*stride, *dilation, *kernel))
        or any(v < 0 for v in padding)
    ):
        raise ValueError("invalid convolution geometry")
    result = np.empty((x.shape[0], w.shape[0], *shape), dtype=object)
    if not result.size:
        return result
    source, source_scale = _integer_grid(x)
    weight_codes, scales = [], []
    for channel in w:
        codes, scale = _integer_grid(channel)
        weight_codes.append(codes)
        scales.append(scale * source_scale)
    weights = np.asarray(weight_codes, dtype=object)
    bound = (
        math.prod(w.shape[1:])
        * max((abs(v) for v in source.flat), default=0)
        * max((abs(v) for v in weights.flat), default=0)
    )
    dtype = np.int64 if bound <= (1 << 63) - 1 else object
    # The product bound alone does not constrain a factor when the other is
    # zero. Check storage bounds too, before converting either array.
    if any(not -(1 << 63) <= v < 1 << 63 for v in chain(source.flat, weights.flat)):
        dtype = object
    source, weights = source.astype(dtype), weights.astype(dtype)
    padded = np.pad(source, [(0, 0), (0, 0), *[(p, p) for p in padding]], mode="constant")
    if dtype is object:
        # np.pad inserts numpy.int64 zeros even into an object array. Such a
        # scalar can coerce a large Python integer back to int64 during a MAC.
        padded = np.vectorize(int, otypes=[object])(padded)
    effective_kernel = tuple(d * (k - 1) + 1 for d, k in zip(dilation, kernel, strict=True))
    windows = np.lib.stride_tricks.sliding_window_view(padded, effective_kernel, axis=tuple(range(2, x.ndim)))
    windows = windows[
        (slice(None), slice(None), *[slice(None, None, s) for s in stride], *[slice(None, None, d) for d in dilation])
    ]
    channels = w.shape[1]
    per_group = w.shape[0] // groups
    axes = (0, *range(2, 2 + dims), 1, *range(2 + dims, 2 + 2 * dims))
    for group in range(groups):
        columns = (
            windows[:, group * channels : (group + 1) * channels].transpose(axes).reshape(-1, math.prod(w.shape[1:]))
        )
        first, last = group * per_group, (group + 1) * per_group
        accumulators = np.matmul(columns, weights[first:last].reshape(per_group, -1).T)
        accumulators = accumulators.reshape(x.shape[0], *shape, per_group)
        accumulators = np.moveaxis(accumulators, -1, 1)
        for offset, channel in enumerate(range(first, last)):
            total = accumulators[:, offset].astype(object) * scales[channel]
            if bias is not None:
                total += rational(bias[channel])
            result[:, channel] = total
    return result


def _pool(x, name, args, kwargs, *, gradient=None):
    dims = int(name.split("pool", 1)[1][0])
    prefix = x.ndim - dims
    if "adaptive" in name:
        output_size = args[1]
        shape = (output_size,) * dims if isinstance(output_size, int) else tuple(output_size)

        def windows(point):
            return [
                range(
                    point[d] * x.shape[prefix + d] // shape[d],
                    ((point[d] + 1) * x.shape[prefix + d] + shape[d] - 1) // shape[d],
                )
                for d in range(dims)
            ]

        divisor_override = None
    else:

        def seq(value, default):
            if value is None or value == []:
                value = default
            return [value] * dims if isinstance(value, int) else list(value)

        kernel = seq(args[1], 1)
        stride = seq(args[2] if len(args) > 2 else kwargs.get("stride"), kernel)
        pad = seq(args[3] if len(args) > 3 else kwargs.get("padding", 0), 0)
        ceil = args[4] if len(args) > 4 else kwargs.get("ceil_mode", False)
        include = args[5] if len(args) > 5 else kwargs.get("count_include_pad", True)
        divisor_override = args[6] if len(args) > 6 else kwargs.get("divisor_override")
        shape = []
        for size, k, s, p in zip(x.shape[prefix:], kernel, stride, pad, strict=True):
            out = (size + 2 * p - k + (s - 1 if ceil else 0)) // s + 1
            if ceil and (out - 1) * s >= size + p:
                out -= 1
            shape.append(out)

        def windows(point):
            return [
                range(
                    point[d] * stride[d] - pad[d],
                    min(point[d] * stride[d] - pad[d] + kernel[d], x.shape[prefix + d] + pad[d]),
                )
                for d in range(dims)
            ]

    out = np.empty((*x.shape[:prefix], *shape), dtype=object)
    adjoint = np.full(x.shape, Fraction(0), dtype=object) if gradient is not None else None
    if gradient is not None and gradient.shape != out.shape:
        raise ValueError("pool gradient shape does not match forward geometry")
    for index in np.ndindex(out.shape):
        ranges = windows(index[prefix:])
        valid = [p for p in product(*ranges) if all(0 <= p[d] < x.shape[prefix + d] for d in range(dims))]
        count = divisor_override or (
            math.prod(len(r) for r in ranges) if "adaptive" not in name and include else len(valid)
        )
        if not count:
            raise ValueError("exact average pool has an empty divisor")
        if gradient is None:
            out[index] = sum((x[(*index[:prefix], *p)] for p in valid), Fraction(0)) / count
        else:
            for point in valid:
                adjoint[(*index[:prefix], *point)] += gradient[index] / count
    return out if gradient is None else adjoint


def _pool_backward(gradient, shape, name, args, kwargs):
    source = np.empty(shape, dtype=object)
    dims = int(name.split("pool", 1)[1][0])
    forward_args = (source, gradient.shape[-dims:]) if "adaptive" in name else (source, *args[2:])
    return _pool(source, name, forward_args, kwargs, gradient=gradient)


def _fold(columns, output_size, kernel_size, dilation, padding, stride):
    """Adjoint of spatial patch extraction, accumulating rational entries unchanged."""
    unbatched = columns.ndim == 2
    if unbatched:
        columns = columns[None]
    taps = math.prod(kernel_size)
    channels, remainder = divmod(columns.shape[1], taps)
    blocks = tuple(
        (s + 2 * p - d * (k - 1) - 1) // t + 1
        for s, p, d, k, t in zip(output_size, padding, dilation, kernel_size, stride, strict=True)
    )
    if remainder or math.prod(blocks) != columns.shape[2] or any(b <= 0 for b in blocks):
        raise ValueError("fold columns do not match spatial geometry")
    out = np.full((columns.shape[0], channels, *output_size), Fraction(0), dtype=object)
    for batch, channel in np.ndindex(out.shape[:2]):
        for tap_index, tap in enumerate(np.ndindex(tuple(kernel_size))):
            for block_index, block in enumerate(np.ndindex(blocks)):
                point = tuple(
                    b * s - p + k * d for b, s, p, k, d in zip(block, stride, padding, tap, dilation, strict=True)
                )
                if all(0 <= i < s for i, s in zip(point, output_size, strict=True)):
                    out[(batch, channel, *point)] += columns[batch, channel * taps + tap_index, block_index]
    return out[0] if unbatched else out


def convolution_backward(operands, args, node_name, sites):
    """Exact spatial adjoints, plus independent code-domain GEMM observations."""
    grad, source, weight = [exact_array(x) for x in operands]
    stride, padding, dilation = map(list, (args[4], args[5], args[6]))
    transposed, output_padding, groups, mask = args[7:11]
    if transposed or any(output_padding) or source.ndim != 4:
        raise ValueError("exact reference requires non-transposed 2-D convolution_backward")
    dx, dw = np.full(source.shape, Fraction(0), dtype=object), np.full(weight.shape, Fraction(0), dtype=object)
    db = np.sum(grad, axis=(0, 2, 3))
    n, cin, h, width = source.shape
    cout, cg, kh, kw = weight.shape
    fg, oh, ow = cout // groups, grad.shape[2], grad.shape[3]
    for batch, co, oy, ox in np.ndindex(grad.shape):
        for ci, ky, kx in np.ndindex((cg, kh, kw)):
            iy, ix = oy * stride[0] - padding[0] + ky * dilation[0], ox * stride[1] - padding[1] + kx * dilation[1]
            if 0 <= iy < h and 0 <= ix < width:
                channel = co // fg * cg + ci
                if mask[0]:
                    dx[batch, channel, iy, ix] += grad[batch, co, oy, ox] * weight[co, ci, ky, kx]
                if mask[1]:
                    dw[co, ci, ky, kx] += grad[batch, co, oy, ox] * source[batch, channel, iy, ix]
    if all(isinstance(x, ExactTensor) and x.codes is not None for x in operands):
        g, x, w = [value.codes for value in operands]
        positions = n * oh * ow
        rows = [g[:, i * fg : (i + 1) * fg].transpose(1, 0, 2, 3).reshape(fg, positions) for i in range(groups)]
        for which in (0, 1):
            if not mask[which]:
                continue
            for i in range(groups):
                if which == 0:
                    left, right = w[i * fg : (i + 1) * fg].reshape(fg, cg * kh * kw).T, rows[i]
                else:
                    padded = np.pad(
                        x[:, i * cg : (i + 1) * cg],
                        ((0, 0), (0, 0), (padding[0], padding[0]), (padding[1], padding[1])),
                    )
                    patches = [
                        padded[
                            :,
                            :,
                            ky * dilation[0] : ky * dilation[0] + oh * stride[0] : stride[0],
                            kx * dilation[1] : kx * dilation[1] + ow * stride[1] : stride[1],
                        ]
                        .transpose(0, 2, 3, 1)
                        .reshape(positions, cg)
                        for ky, kx in np.ndindex((kh, kw))
                    ]
                    left, right = rows[i], np.concatenate(patches, axis=1)
                scale_order = (2, 0) if which == 0 else (0, 1)
                scales = [operands[j].scale for j in scale_order]
                scales = [value.detach().cpu().tolist() if hasattr(value, "detach") else value for value in scales]
                sites.append(
                    dict(
                        node=f"{node_name}.{'grad_input' if which == 0 else 'grad_weight'}{i}",
                        kind="conv2d",
                        **mac_observation(left, right, np.matmul(left, right), scales),
                        readout=None,
                        readout_order="exact rational; one boundary RNE",
                        arithmetic="exact_rational",
                    )
                )
    dtype = operands[0].dtype
    return tuple(
        ExactTensor(value, dtype) if wanted else None for value, wanted in zip((dx, dw, db), mask, strict=True)
    )


def evaluate_operation(target, args, kwargs):
    """Closed rational operations. Unknown semantics cannot acquire an exactness claim."""
    import torch

    name = str(target)
    flat = torch.utils._pytree.tree_leaves((args, kwargs))
    exact = next((x for x in flat if isinstance(x, ExactTensor)), None)
    source = exact if exact is not None else next((x for x in flat if isinstance(x, torch.Tensor)), None)
    if "out" in kwargs or name.endswith(".out"):
        raise ValueError("exact reference refuses output-buffer mutation")
    dtype = kwargs.get("dtype") or getattr(source, "dtype", torch.get_default_dtype())
    if dtype not in {None, torch.float16, torch.float32, torch.float64, torch.bfloat16}:
        raise ValueError("exact reference refuses an implicit integer cast")
    op = name.split(".")[1] if name.startswith("aten.") else name
    if op in {"fill", "full", "full_like", "zeros", "zeros_like", "ones", "ones_like"}:
        value = 0 if op.startswith("zeros") else 1 if op.startswith("ones") else args[1]
        shape = args[0] if op in {"full", "zeros", "ones"} else args[0].shape
        return ExactTensor(np.full(shape, rational(value), dtype=object), dtype)
    a = [exact_array(x) if isinstance(x, (ExactTensor, torch.Tensor)) else x for x in args]
    if op in {"add", "sub", "mul", "div", "true_divide"}:
        alpha = rational(args[2] if len(args) > 2 and op in {"add", "sub"} else kwargs.get("alpha", 1))
        left, right = exact_array(a[0]), exact_array(a[1])
        if op in {"add", "sub"}:
            right = right * alpha
        if kwargs.get("rounding_mode") is not None:
            raise ValueError("exact reference refuses rounded division")
        out = {
            "add": operator.add,
            "sub": operator.sub,
            "mul": operator.mul,
            "div": operator.truediv,
            "true_divide": operator.truediv,
        }[op](left, right)
    elif op in {"neg", "relu", "abs"}:
        out = -a[0] if op == "neg" else np.maximum(a[0], 0) if op == "relu" else abs(a[0])
    elif op in {"clamp", "clamp_min", "clamp_max", "hardtanh"}:
        low = args[1] if len(args) > 1 else kwargs.get("min_val", -1) if op == "hardtanh" else kwargs.get("min")
        high = args[2] if len(args) > 2 else kwargs.get("max_val", 1) if op == "hardtanh" else kwargs.get("max")
        if op == "clamp_max":
            low, high = None, low
        out = a[0]
        if low is not None:
            out = np.maximum(out, rational(low))
        if high is not None:
            out = np.minimum(out, rational(high))
    elif op in {"mm", "bmm", "matmul", "linear", "addmm", "baddbmm"}:
        if op in {"addmm", "baddbmm"}:
            out = rational(kwargs.get("beta", 1)) * a[0] + rational(kwargs.get("alpha", 1)) * np.matmul(a[1], a[2])
        elif op == "linear":
            out = np.matmul(a[0], a[1].swapaxes(-1, -2))
            if len(a) > 2 and a[2] is not None:
                out += a[2]
        else:
            out = np.matmul(a[0], a[1])
    elif op in {"sum", "mean", "amax", "amin"}:
        axis = args[1] if len(args) > 1 else kwargs.get("dim")
        axis = None if axis is None or axis == [] else tuple(axis) if isinstance(axis, (list, tuple)) else axis
        keep = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        out = {"sum": np.sum, "mean": np.mean, "amax": np.max, "amin": np.min}[op](a[0], axis=axis, keepdims=keep)
    elif op in {"conv1d", "conv2d", "conv3d", "convolution"}:
        if op == "convolution" and (args[6] or any(args[7])):
            raise ValueError("exact reference refuses transposed convolution")
        stride = args[3] if len(args) > 3 else kwargs.get("stride", 1)
        pad = args[4] if len(args) > 4 else kwargs.get("padding", 0)
        dilation = args[5] if len(args) > 5 else kwargs.get("dilation", 1)
        groups = args[8] if op == "convolution" else args[6] if len(args) > 6 else kwargs.get("groups", 1)
        out = _convolution(a[0], a[1], a[2] if len(a) > 2 else None, stride, pad, dilation, groups)
    elif op.startswith(("avg_pool", "adaptive_avg_pool", "_adaptive_avg_pool")) and "backward" not in op:
        out = _pool(a[0], name, args, kwargs)
    elif op.startswith(("avg_pool", "adaptive_avg_pool", "_adaptive_avg_pool")) and op.endswith("backward"):
        out = _pool_backward(a[0], a[1].shape, name, args, kwargs)
    elif op == "col2im":
        out = _fold(a[0], *args[1:])
    elif op in {"view", "reshape", "_unsafe_view", "flatten"}:
        if op == "flatten":
            start = (args[1] if len(args) > 1 else 0) % a[0].ndim
            end = (args[2] if len(args) > 2 else -1) % a[0].ndim
            if end < start:
                raise ValueError("invalid flatten dimensions")
            shape = (*a[0].shape[:start], math.prod(a[0].shape[start : end + 1]), *a[0].shape[end + 1 :])
        else:
            shape = args[1]
        out = a[0].reshape(shape)
    elif op in {"clone", "detach", "contiguous", "alias"}:
        out = a[0].copy()
    elif op in {"_to_copy", "to"}:
        dtype = kwargs.get("dtype", dtype)
        if dtype not in {torch.float16, torch.float32, torch.float64}:
            raise ValueError("exact reference refuses an intermediate integer cast")
        out = a[0].copy()
    elif op in {"permute", "transpose", "t"}:
        out = (
            a[0].transpose(args[1])
            if op == "permute"
            else a[0].swapaxes(args[1], args[2])
            if op == "transpose"
            else a[0].T
        )
    elif op in {"squeeze", "unsqueeze"}:
        out = (
            np.expand_dims(a[0], args[1])
            if op == "unsqueeze"
            else np.squeeze(a[0], axis=args[1] if len(args) > 1 else None)
        )
    elif op in {"expand", "broadcast_to"}:
        shape = tuple(
            old if new == -1 else new
            for old, new in zip((1,) * (len(args[1]) - a[0].ndim) + a[0].shape, args[1], strict=True)
        )
        out = np.broadcast_to(a[0], shape)
    elif op == "repeat":
        out = np.tile(a[0], args[1])
    elif op in {"slice", "select"}:
        indices = [slice(None)] * a[0].ndim
        indices[args[1]] = slice(args[2], args[3], args[4] if len(args) > 4 else 1) if op == "slice" else args[2]
        out = a[0][tuple(indices)]
    elif op in {"cat", "stack"}:
        out = (np.concatenate if op == "cat" else np.stack)(
            [exact_array(x) for x in args[0]], axis=args[1] if len(args) > 1 else 0
        )
    elif op == "constant_pad_nd" or op.startswith(("reflection_pad", "replication_pad")):
        widths = [(0, 0)] * a[0].ndim
        for i in range(len(args[1]) // 2):
            widths[-1 - i] = tuple(args[1][2 * i : 2 * i + 2])
        if op == "constant_pad_nd":
            cropped = a[0][
                tuple(slice(max(-lo, 0), size - max(-hi, 0)) for size, (lo, hi) in zip(a[0].shape, widths, strict=True))
            ]
            widths = [(max(lo, 0), max(hi, 0)) for lo, hi in widths]
            out = np.pad(cropped, widths, constant_values=rational(args[2] if len(args) > 2 else 0))
        else:
            # Reflect about the ORIGINAL endpoints: cropping before reflection
            # would change the mirror when one side crops and the other pads.
            out = a[0]
            for axis, (low, high) in enumerate(widths):
                size = a[0].shape[axis]
                if size <= 0 or size + low + high <= 0:
                    raise ValueError("exact padding requires nonempty dimensions")
                positions = np.arange(size + low + high) - low
                if op.startswith("reflection"):
                    if low >= size or high >= size:
                        raise ValueError("reflection padding must be smaller than the input dimension")
                    positions = np.where(positions < 0, -positions, positions)
                    positions = np.where(positions >= size, 2 * size - 2 - positions, positions)
                else:
                    positions = np.clip(positions, 0, size - 1)
                out = np.take(out, positions, axis=axis)
    elif op in {"slice_scatter", "select_scatter"}:
        out = a[0].copy()
        indices = [slice(None)] * out.ndim
        indices[args[2]] = (
            slice(args[3], args[4], args[5] if len(args) > 5 else 1) if op == "slice_scatter" else args[3]
        )
        out[tuple(indices)] = a[1]
    elif op in {"fill", "full", "full_like", "zeros", "zeros_like", "ones", "ones_like"}:
        value = 0 if op.startswith("zeros") else 1 if op.startswith("ones") else args[1]
        shape = args[0] if op in {"full", "zeros", "ones"} else a[0].shape
        out = np.full(shape, rational(value), dtype=object)
    else:
        raise ValueError(f"exact reference refuses rational operation {name}")
    return ExactTensor(np.asarray(out, dtype=object), dtype)


def run(model, inputs):
    """Evaluate source Q/DQ graph exactly; eager differences are informational only."""
    import copy
    import logging

    import torch

    sites = []
    counts = {"linear": 0, "conv2d": 0, "matmul": 0}
    boundary_policy = getattr(model, "_recipe_quantization_stats", {}).get("framework_capture_policy", {}).get("quantization_nonfinite")

    class Reference(torch.fx.Interpreter):
        def run_node(self, node):
            if node.op != "call_function":
                return super().run_node(node)
            args, kwargs = self.fetch_args_kwargs_from_env(node)
            name = str(node.target)
            if name.startswith("quantized_decomposed."):
                channel = "per_channel" in name
                if "quantize_per_" not in name:
                    raise ValueError(f"exact reference refuses quantizer {name}")
                value, scale, zp = args[:3]
                axis, low, high, dtype = args[3:7] if channel else (None, *args[3:6])
                if dtype not in {torch.int8, torch.uint8}:
                    raise ValueError("exact quantization requires int8 or uint8 storage")
                storage = torch.iinfo(dtype)
                if not storage.min <= low <= high <= storage.max:
                    raise ValueError("quantization range exceeds storage")
                if ".dequantize_" in name:
                    position = 7 if channel else 6
                    out_dtype = kwargs.get("out_dtype", args[position] if len(args) > position else torch.float32)
                    return dequantize(value, scale, zp, low, high, axis=axis, dtype=out_dtype)
                codes = quantize(value, scale, zp, low, high, axis=axis, nonfinite=boundary_policy)
                return torch.tensor(codes.tolist(), dtype=dtype).reshape(codes.shape)
            leaves = torch.utils._pytree.tree_leaves((args, kwargs))
            if name == "aten.isnan.default" and isinstance(args[0], ExactTensor):
                values = np.vectorize(lambda x: not isinstance(x, Fraction) and math.isnan(float(x)), otypes=[bool])(args[0].values)
                return torch.tensor(values, dtype=torch.bool)
            if name == "aten.where.self" and any(isinstance(x, ExactTensor) for x in leaves):
                def storage(value):
                    return value.values if isinstance(value, ExactTensor) else value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value

                selected = np.where(storage(args[0]), storage(args[1]), storage(args[2]))
                # A selector observes only selected elements. An unselected NaN
                # does not enter rational arithmetic or invalidate that branch.
                dtype = next(x.dtype for x in args[1:] if isinstance(x, (ExactTensor, torch.Tensor)))
                if boundary_policy == "byte_storage_zero_nan_saturating_infinity":
                    values = np.vectorize(lambda x: x if not isinstance(x, Fraction) and not math.isfinite(float(x)) else rational(x), otypes=[object])(selected)
                    return ExactTensor(values, dtype)
                return ExactTensor(exact_array(selected), dtype)
            constant = name.split(".")[1] if name.startswith("aten.") else None
            if not any(isinstance(x, ExactTensor) for x in leaves) and constant not in {
                "fill",
                "full",
                "full_like",
                "zeros",
                "zeros_like",
                "ones",
                "ones_like",
            }:
                return super().run_node(node)
            if node.target == operator.getitem or getattr(node.target, "_m2m_tuple_selection", False):
                value = args[0]
                return (
                    ExactTensor(value.values[args[1]], value.dtype)
                    if isinstance(value, ExactTensor)
                    else value[args[1]]
                )
            if name == "aten.convolution_backward.default":
                counts["conv2d"] += 1
                return convolution_backward(args[:3], args, node.name, sites)
            result = evaluate_operation(node.target, args, kwargs)
            op = name.split(".")[1] if name.startswith("aten.") else name
            if op in {"mm", "bmm", "matmul", "linear", "addmm", "baddbmm", "conv1d", "conv2d", "conv3d", "convolution"}:
                kind = "linear" if op == "linear" else "conv2d" if "conv" in op else "matmul"
                counts[kind] += 1
                operands = args[1:3] if op in {"addmm", "baddbmm"} else args[:2]
                scales = [x.scale if isinstance(x, ExactTensor) else None for x in operands]
                scales = [scale.detach().cpu().tolist() if hasattr(scale, "detach") else scale for scale in scales]
                acc = None
                if all(isinstance(x, ExactTensor) and x.codes is not None for x in operands) and op in {
                    "mm",
                    "bmm",
                    "matmul",
                    "linear",
                    "addmm",
                    "baddbmm",
                }:
                    lhs, rhs = [x.codes for x in operands]
                    if op == "linear":
                        rhs = rhs.swapaxes(-1, -2)
                    acc = np.matmul(lhs, rhs)
                elif all(isinstance(x, ExactTensor) and x.codes is not None for x in operands) and "conv" in op:
                    lhs, rhs = [x.codes for x in operands]
                    stride = args[3] if len(args) > 3 else kwargs.get("stride", 1)
                    padding = args[4] if len(args) > 4 else kwargs.get("padding", 0)
                    dilation = args[5] if len(args) > 5 else kwargs.get("dilation", 1)
                    groups = args[8] if op == "convolution" else args[6] if len(args) > 6 else kwargs.get("groups", 1)
                    acc = _convolution(lhs, rhs, None, stride, padding, dilation, groups)
                    acc = np.moveaxis(acc, 1, -1).reshape(-1, rhs.shape[0])
                    acc = np.vectorize(int, otypes=[object])(acc)
                sites.append(
                    dict(
                        node=node.name,
                        kind=kind,
                        **(
                            mac_observation(operands[0].codes, operands[1].codes, acc, scales)
                            if acc is not None
                            else dict(lhs=scales[0], rhs=scales[1], accumulator=None)
                        ),
                        readout=None,
                        readout_order="exact rational; one boundary RNE",
                        arithmetic="exact_rational",
                    )
                )
            return result

    saved = copy.deepcopy(tuple(inputs))
    reference = Reference(model, garbage_collect_values=False)
    exact = reference.run(*inputs)
    output = materialize(exact)
    divergence = {"status": "not_checked", "informational": True}
    try:
        eager = model(*saved)
        eager_leaves, eager_tree = torch.utils._pytree.tree_flatten(eager)
        actual_leaves, actual_tree = torch.utils._pytree.tree_flatten(output)
        different = 0
        if eager_tree != actual_tree:
            raise ValueError("eager and exact output structures differ")
        for left, right in zip(eager_leaves, actual_leaves, strict=True):
            different += int((left != right).sum()) if isinstance(left, torch.Tensor) else int(left != right)
        divergence = {
            "status": "diverged" if different else "equal",
            "different_elements": different,
            "informational": True,
        }
        if different:
            logging.getLogger(__name__).info("eager-float divergence: %s", divergence)
    except Exception as exc:
        divergence["reason"] = str(exc)
    return SimpleNamespace(
        output=output,
        post_inputs=inputs,
        sites=sites,
        contraction_count=sum(counts.values()),
        linear_count=counts["linear"],
        conv2d_count=counts["conv2d"],
        matmul_count=counts["matmul"],
        arithmetic="exact_rational",
        eager_float_divergence=divergence,
    )
