"""Exact definitions of the affine-map and data-movement command-buffer opcodes.

``AFFINE_MAP``, the movement forms of ``MOVEMENT``, ``CONCAT``, ``INSERT_SLICE`` and ``FILL`` are defined
ONCE, here, and the reference engine, the command-buffer simulator and the capsule golden all call these
functions. Three engines that each restated "round half to even after one binary32 multiply" would be
three chances to disagree about the one thing an exact-integer grade checks.

Every attribute an opcode needs is read and validated here, and an absent required attribute or an
unknown one is refused by name (:class:`IntegerOpError`) rather than defaulted: an engine that applied a
default the buffer never stated would agree with another engine that applied the same default, and the
capsule would pass having proved nothing about the attribute it omitted.

The numeric form of ``AFFINE_MAP`` is :class:`merlin.targetgen.recipe_numerics.IntegerRecipe`: integer
weights and an int32 constant in the accumulator, ReLU on the accumulator, then ONE readout.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from .tensor import Tensor


class IntegerOpError(ValueError):
    """A command's operands or attributes do not state a well-defined integer operation."""


#: Integer container ranges an output may saturate into.
_RANGES = {"i8": (-128, 127), "i16": (-(2**15), 2**15 - 1), "i32": (-(2**31), 2**31 - 1)}
_AFFINE_ATTRS = frozenset(
    {"weights", "constant", "activation", "readout_scale", "output_dtype", "quant_min", "quant_max"}
)
_MOVE_FORMS = {
    "identity": frozenset(),
    "permute": frozenset({"perm"}),
    "slice": frozenset({"offsets", "sizes", "strides"}),
    "pad": frozenset({"low", "high", "mode"}),
    "broadcast": frozenset(),
    "repeat": frozenset({"repeats"}),
}
_MOVE_COMMON = frozenset({"form", "output_dtype", "semantic", "scale", "quant_min", "quant_max", "value_code"})


def _array(t: Tensor) -> np.ndarray:
    return np.asarray(t.data, dtype=np.int64).reshape(t.shape)


def _tensor(values: np.ndarray, dtype: str) -> Tensor:
    return Tensor(tuple(int(d) for d in values.shape), [int(v) for v in values.reshape(-1)], dtype)


def _int_list(attrs: dict[str, Any], key: str, *, op: str, length: int | None = None) -> list[int]:
    value = attrs.get(key)
    if not isinstance(value, list) or any(type(v) is not int for v in value):
        raise IntegerOpError(f"{op}: {key} must be a list of integers, got {value!r}")
    if length is not None and len(value) != length:
        raise IntegerOpError(f"{op}: {key} needs {length} entries (one per dimension), got {len(value)}")
    return value


def _range(attrs: dict[str, Any], dtype: str, *, op: str) -> tuple[int, int]:
    if dtype not in _RANGES:
        raise IntegerOpError(f"{op}: output_dtype must be one of {sorted(_RANGES)}, got {dtype!r}")
    low, high = _RANGES[dtype]
    qmin, qmax = attrs.get("quant_min", low), attrs.get("quant_max", high)
    if type(qmin) is not int or type(qmax) is not int or not low <= qmin <= qmax <= high:
        raise IntegerOpError(f"{op}: quant_min/quant_max must be integers within {dtype}'s range")
    return qmin, qmax


def _requantize(values: np.ndarray, attrs: dict[str, Any], dtype: str, *, op: str, scale: Any = None) -> np.ndarray:
    """One load-path rounding at ``scale`` (when stated), saturated into the output range."""
    scale = attrs.get("scale") if scale is None else scale
    if scale is None:
        return values
    if not isinstance(scale, (int, float)) or not np.isfinite(scale):
        raise IntegerOpError(f"{op}: scale must be a finite number")
    qmin, qmax = _range(attrs, dtype, op=op)
    product = values.astype(np.float32) * np.float32(scale)
    return np.clip(np.rint(product), qmin, qmax).astype(np.int64)


def affine_map(x: Tensor, y: Tensor | None, attrs: dict[str, Any], *, op: str = "AFFINE_MAP") -> Tensor:
    """``dst = sat(RNE(f32(act(w0*x + w1*y + constant)) * f32(readout_scale)))``; see the ABI."""
    from merlin.targetgen.recipe_numerics import IntegerRecipe, RecipeNumericsError

    unknown = sorted(set(attrs) - _AFFINE_ATTRS - {"name"})
    if unknown:
        raise IntegerOpError(f"{op}: unknown attribute(s) {unknown}")
    missing = sorted({"weights", "constant", "activation", "readout_scale", "output_dtype"} - set(attrs))
    if missing:
        raise IntegerOpError(f"{op}: required attribute(s) {missing} absent; no default is applied")
    weights = _int_list(attrs, "weights", op=op)
    if len(weights) != (1 if y is None else 2):
        raise IntegerOpError(f"{op}: {len(weights)} weight(s) for {1 if y is None else 2} operand(s)")
    constant, scale, dtype = attrs["constant"], attrs["readout_scale"], str(attrs["output_dtype"])
    if type(constant) is not int or not isinstance(scale, (int, float)):
        raise IntegerOpError(f"{op}: constant must be an integer and readout_scale a number")
    qmin, qmax = _range(attrs, dtype, op=op)
    a = _array(x)
    operands = [a]
    if y is not None:
        b = _array(y)
        if b.ndim > a.ndim or any(bd not in (1, ad) for bd, ad in zip(b.shape[::-1], a.shape[::-1], strict=False)):
            raise IntegerOpError(f"{op}: y{list(b.shape)} does not broadcast to x{list(a.shape)}")
        if a.ndim and (b.ndim == 0 or b.shape[-1] != a.shape[-1]):
            raise IntegerOpError(f"{op}: y may broadcast along leading dims only, not the innermost one")
        operands.append(np.broadcast_to(b, a.shape))
    try:
        if dtype == "i32":
            if float(scale) != 1.0:
                raise IntegerOpError(f"{op}: an i32 result has no readout; readout_scale must be 1.0")
            accumulator = sum(w * v for w, v in zip(weights, operands, strict=True)) + constant
            if attrs["activation"] == "relu":
                accumulator = np.maximum(accumulator, 0)
            elif attrs["activation"] != "none":
                raise IntegerOpError(f"{op}: activation must be none or relu")
            if np.any(accumulator < qmin) or np.any(accumulator > qmax):
                raise IntegerOpError(f"{op}: the int32 accumulator overflowed its declared range")
            return _tensor(np.asarray(accumulator), dtype)
        recipe = IntegerRecipe(
            weights=tuple(weights),
            constant=constant,
            readout_scale=float(scale),
            activation=str(attrs["activation"]),
            qmin=qmin,
            qmax=qmax,
        )
        return _tensor(recipe.evaluate(*operands), dtype)
    except RecipeNumericsError as exc:
        raise IntegerOpError(f"{op}: {exc}") from exc


def move(src: Tensor, attrs: dict[str, Any], dst_shape: list[int] | None, *, op: str = "MOVEMENT") -> Tensor:
    """A single-source ``MOVEMENT``: identity, permute, slice, pad, broadcast or repeat (+ optional scale)."""
    form = str(attrs.get("form", "identity"))
    if form not in _MOVE_FORMS:
        raise IntegerOpError(f"{op}: form must be one of {sorted(_MOVE_FORMS)}, got {form!r}")
    unknown = sorted(set(attrs) - _MOVE_COMMON - _MOVE_FORMS[form] - {"name"})
    if unknown:
        raise IntegerOpError(f"{op}: attribute(s) {unknown} do not belong to form {form!r}")
    missing = sorted(_MOVE_FORMS[form] - set(attrs))
    if missing:
        raise IntegerOpError(f"{op}: form {form!r} requires {missing}; no default is applied")
    if "value_code" in attrs and not (form == "pad" and attrs.get("mode") == "constant"):
        raise IntegerOpError(f"{op}: value_code belongs to a constant pad only")
    dtype = str(attrs.get("output_dtype", src.dtype))
    a = _array(src)
    rank = a.ndim
    if form == "identity":
        out = a
    elif form == "permute":
        perm = _int_list(attrs, "perm", op=op, length=rank)
        if sorted(perm) != list(range(rank)):
            raise IntegerOpError(f"{op}: perm {perm} is not a permutation of {rank} dims")
        out = np.transpose(a, perm)
    elif form == "slice":
        offsets = _int_list(attrs, "offsets", op=op, length=rank)
        sizes = _int_list(attrs, "sizes", op=op, length=rank)
        strides = _int_list(attrs, "strides", op=op, length=rank)
        if any(s < 1 for s in strides) or any(
            o < 0 or n < 0 or o + (n - 1) * s >= d
            for o, n, s, d in zip(offsets, sizes, strides, a.shape, strict=True)
            if n
        ):
            raise IntegerOpError(f"{op}: slice is out of bounds or has a nonpositive stride")
        out = a[tuple(slice(o, o + n * s, s) for o, n, s in zip(offsets, sizes, strides, strict=True))]
    elif form == "pad":
        low = _int_list(attrs, "low", op=op, length=rank)
        high = _int_list(attrs, "high", op=op, length=rank)
        if any(v < 0 for v in low + high):
            raise IntegerOpError(f"{op}: pad widths must be nonnegative")
        mode = attrs["mode"]
        widths = list(zip(low, high, strict=True))
        if mode == "constant":
            code = attrs.get("value_code")
            if type(code) is not int:
                raise IntegerOpError(f"{op}: a constant pad requires an integer value_code (the quantized pad value)")
            out = np.pad(a, widths, mode="constant", constant_values=code)
        elif mode == "reflect":
            if any(lo >= d or hi >= d for (lo, hi), d in zip(widths, a.shape, strict=True)):
                raise IntegerOpError(f"{op}: a reflect pad must be narrower than the dimension it reflects")
            out = np.pad(a, widths, mode="reflect")
        elif mode == "replicate":
            out = np.pad(a, widths, mode="edge")
        else:
            raise IntegerOpError(f"{op}: pad mode must be constant, reflect or replicate, got {mode!r}")
    elif form == "broadcast":
        if dst_shape is None:
            raise IntegerOpError(f"{op}: a broadcast needs its declared destination shape")
        target = tuple(int(d) for d in dst_shape)
        if rank and (len(target) < rank or target[-1] != a.shape[-1]):
            raise IntegerOpError(f"{op}: a broadcast replays rows along leading dims only, not the innermost one")
        try:
            out = np.broadcast_to(a, target)
        except ValueError as exc:
            raise IntegerOpError(f"{op}: {list(a.shape)} does not broadcast to {list(target)}") from exc
    else:  # repeat
        repeats = attrs["repeats"]
        if not isinstance(repeats, list) or len(repeats) < rank or any(type(r) is not int or r < 1 for r in repeats):
            raise IntegerOpError(f"{op}: repeats must list a positive count per dim (at least the source rank)")
        out = np.tile(a, repeats)
    out = _requantize(np.asarray(out), attrs, dtype, op=op)
    if dst_shape is not None and tuple(out.shape) != tuple(int(d) for d in dst_shape):
        raise IntegerOpError(f"{op}: the {form} result {list(out.shape)} differs from the declared {list(dst_shape)}")
    return _tensor(np.ascontiguousarray(out), dtype)


def concat(srcs: list[Tensor], attrs: dict[str, Any], *, op: str = "CONCAT") -> Tensor:
    """``dst = cat([requant(src_i, scales[i])], axis)``; ``scales`` absent means shared qparams."""
    unknown = sorted(set(attrs) - {"axis", "output_dtype", "scales", "quant_min", "quant_max", "name"})
    if unknown:
        raise IntegerOpError(f"{op}: unknown attribute(s) {unknown}")
    if len(srcs) < 2:
        raise IntegerOpError(f"{op}: needs at least two sources")
    axis = attrs.get("axis")
    if type(axis) is not int:
        raise IntegerOpError(f"{op}: axis is required (an integer); no default is applied")
    dtype = str(attrs.get("output_dtype", srcs[0].dtype))
    scales = attrs.get("scales")
    if scales is not None and (not isinstance(scales, list) or len(scales) != len(srcs)):
        raise IntegerOpError(f"{op}: scales must list one scale per source")
    parts = [
        _requantize(_array(s), attrs, dtype, op=op, scale=None if scales is None else float(scales[i]))
        for i, s in enumerate(srcs)
    ]
    try:
        return _tensor(np.concatenate(parts, axis=axis), dtype)
    except (ValueError, np.exceptions.AxisError) as exc:
        raise IntegerOpError(f"{op}: {exc}") from exc


def insert_slice(dest: Tensor, src: Tensor, attrs: dict[str, Any], *, op: str = "INSERT_SLICE") -> Tensor:
    """``dst = dest`` with ``dst[offsets + i*strides] = src`` (slice_scatter / select_scatter)."""
    unknown = sorted(set(attrs) - {"offsets", "strides", "output_dtype", "name"})
    if unknown:
        raise IntegerOpError(f"{op}: unknown attribute(s) {unknown}")
    base, part = _array(dest), _array(src)
    if base.ndim != part.ndim:
        raise IntegerOpError(f"{op}: src rank {part.ndim} differs from dest rank {base.ndim}; keep the size-1 dim")
    offsets = _int_list(attrs, "offsets", op=op, length=base.ndim)
    strides = _int_list(attrs, "strides", op=op, length=base.ndim)
    if any(s < 1 for s in strides) or any(
        o < 0 or (n and o + (n - 1) * s >= d)
        for o, n, s, d in zip(offsets, part.shape, strides, base.shape, strict=True)
    ):
        raise IntegerOpError(f"{op}: the slice lies outside dest or has a nonpositive stride")
    out = base.copy()
    out[tuple(slice(o, o + n * s, s) for o, n, s in zip(offsets, part.shape, strides, strict=True))] = part
    return _tensor(out, str(attrs.get("output_dtype", dest.dtype)))


def fill(shape: list[int], attrs: dict[str, Any], *, op: str = "FILL") -> Tensor:
    """Every element of the declared destination is ``value_code`` (an already-quantized code)."""
    unknown = sorted(set(attrs) - {"value_code", "output_dtype", "name"})
    if unknown:
        raise IntegerOpError(f"{op}: unknown attribute(s) {unknown}")
    code, dtype = attrs.get("value_code"), attrs.get("output_dtype")
    if type(code) is not int or not isinstance(dtype, str):
        raise IntegerOpError(f"{op}: value_code (an integer code) and output_dtype are required")
    qmin, qmax = _range({}, dtype, op=op)
    if not qmin <= code <= qmax:
        raise IntegerOpError(f"{op}: value_code {code} does not fit {dtype}")
    return _tensor(np.full(tuple(int(d) for d in shape), code, dtype=np.int64), dtype)


#: The opcodes :func:`execute` defines (``MOVEMENT`` forms go through :func:`move`).
OPCODES = ("AFFINE_MAP", "CONCAT", "INSERT_SLICE", "FILL")


def declared_shape(cb: dict[str, Any], name: str) -> list[int] | None:
    """A tensor's declared shape in a command buffer, or None when it is not declared."""
    spec = (cb.get("tensors") or {}).get(name)
    return list(spec["shape"]) if isinstance(spec, dict) and isinstance(spec.get("shape"), list) else None


def source_keys(ops: dict[str, Any]) -> list[str]:
    """A CONCAT's ``src0, src1, ...`` operand keys in source order."""
    keys = [key for key in ops if key.startswith("src") and key[3:].isdigit()]
    return sorted(keys, key=lambda key: int(key[3:]))


def execute(op: str, ops: dict[str, Any], attrs: dict[str, Any], env: dict[str, Tensor], shape) -> Tensor:
    """Evaluate one AFFINE_MAP / CONCAT / INSERT_SLICE / FILL command against ``env``."""
    where = f"{op} {ops.get('dst')!r}"
    try:
        if op == "AFFINE_MAP":
            return affine_map(env[ops["x"]], env[ops["y"]] if "y" in ops else None, attrs, op=where)
        if op == "CONCAT":
            return concat([env[ops[key]] for key in source_keys(ops)], attrs, op=where)
        if op == "INSERT_SLICE":
            return insert_slice(env[ops["dest"]], env[ops["src"]], attrs, op=where)
    except KeyError as exc:
        raise IntegerOpError(f"{where}: operand {exc} is missing or undefined") from exc
    if op != "FILL":
        raise IntegerOpError(f"{op} is not an integer-op opcode")
    if shape is None:
        raise IntegerOpError(f"{where}: a fill needs its declared destination shape")
    return fill(shape, attrs, op=where)
