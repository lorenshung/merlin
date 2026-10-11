"""Captured input storage specialization shared by bufferization and callers."""

from __future__ import annotations


def _apply_input_layout(module, contract):
    from torch_mlir import ir

    functions = [
        op
        for op in module.body.operations
        if op.operation.name == "func.func" and ir.StringAttr(op.attributes["sym_name"]).value == "forward"
    ]
    if len(functions) != 1:
        raise ValueError("input layout requires one forward definition")
    function = functions[0]
    arguments = function.regions[0].blocks[0].arguments
    records = contract["inputs"]
    if len(arguments) != len(records):
        raise ValueError("input layout ABI count mismatch")
    existing = function.attributes.get("arg_attrs")
    attrs = list(ir.ArrayAttr(existing)) if existing is not None else [ir.DictAttr.get({}) for _ in arguments]
    for i, (argument, record) in enumerate(zip(arguments, records)):
        if record is None:
            continue
        shape, strides, offset = record["shape"], record["stride"], record["storage_offset"]
        typ = ir.RankedTensorType(argument.type)
        if list(typ.shape) != shape or len(strides) != typ.rank:
            raise ValueError("captured input shape differs from the compiled input ABI")
        if any(type(v) is not int or v < 0 for v in [*strides, offset]):
            raise ValueError("captured input ABI requires static nonnegative strides and offset")
        updated = {entry.name: entry.attr for entry in ir.DictAttr(attrs[i])}
        updated["bufferization.buffer_layout"] = ir.StridedLayoutAttr.get(offset, strides)
        attrs[i] = ir.DictAttr.get(updated)
    function.attributes["arg_attrs"] = ir.ArrayAttr.get(attrs)


def is_identity_layout(record):
    if record is None:
        return True
    stride = 1
    expected = []
    for size in reversed(record["shape"]):
        expected.append(stride)
        stride *= size
    return record["storage_offset"] == 0 and record["stride"] == list(reversed(expected))


def storage_array(logical, record):
    """Scatter transport values into captured storage; refuse inconsistent overlap."""
    import numpy as np

    if record is None:
        return logical
    shape, strides, offset = record["shape"], record["stride"], record["storage_offset"]
    if list(logical.shape) != shape or len(strides) != len(shape):
        raise ValueError("runtime input shape differs from captured input ABI")
    if any(type(v) is not int or v < 0 for v in [*shape, *strides, offset]):
        raise ValueError("runtime input ABI requires static nonnegative layout")
    if is_identity_layout(record):
        return logical
    capacity = offset if 0 in shape else offset + 1 + sum((n - 1) * s for n, s in zip(shape, strides))
    physical = np.zeros(max(1, capacity), dtype=logical.dtype)
    seen = {}
    for coordinate in np.ndindex(*shape):
        address = offset + sum(i * s for i, s in zip(coordinate, strides))
        value = logical[coordinate]
        raw = value.tobytes()
        if address in seen and seen[address] != raw:
            raise ValueError("runtime input values disagree at overlapping captured storage addresses")
        seen[address] = raw
        physical[address] = value
    return physical


def descriptor_lines(records):
    """Bind rank-exact descriptors to the same storage ABI used by the compiler."""
    lines = []
    for i, record in enumerate(records):
        if record is None:
            continue
        rank = len(record["shape"])
        lines.append(f"((int64_t *)d[{i}])[2] = {record['storage_offset']};")
        for axis, stride in enumerate(record["stride"]):
            lines.append(f"((int64_t *)d[{i}])[{3 + rank + axis}] = {stride};")
    return lines
