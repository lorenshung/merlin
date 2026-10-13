"""Source-only coverage checks for raw CPU reads of portable tensor payloads."""

from math import prod

from merlin.targetgen.capture_refusals import InputStorageUnavailable


def _geometry(tensor):
    if not isinstance(tensor, dict) or tensor.get("kind") != "tensor":
        raise ValueError("raw input storage requires a canonical tensor document")
    shape, stride, offset = tensor["shape"], tensor["stride"], tensor["storage_offset"]
    if (
        len(shape) != len(stride)
        or any(type(v) is not int or v < 0 for v in [*shape, *stride, offset])
        or "values" not in tensor
    ):
        raise ValueError("raw input storage requires recorded static nonnegative tensor geometry")
    return shape, stride, offset


def _raw_coverage(tensor, rows):
    shape, stride, offset = _geometry(tensor)
    span = offset + (1 + sum((n - 1) * s for n, s in zip(shape, stride)) if all(shape) else 0)
    # Nonnegative strides cannot fill a gap below the next sorted stride. Until
    # that gap, shifted copies of the covered interval overlap or touch exactly.
    prefix = 1 if all(shape) else 0
    dimensions = sorted(zip(stride, shape)) if prefix else ()
    for step, extent in dimensions:
        if prefix >= rows or (step > prefix and extent > 1):
            break
        prefix += (extent - 1) * step
    if prefix >= rows:
        return None
    missing = offset + prefix
    return {
        "shape": shape,
        "stride": stride,
        "storage_offset": offset,
        "required_raw_range": [offset, offset + rows],
        "first_unrecorded_address": missing,
        "recorded_storage_span": span,
        "cause": "raw_read_out_of_bounds" if missing >= span else "unrecorded_storage_hole",
    }


def require_native_input_storage(overload, arguments):
    """Require every raw mean/rstd row read to have a recorded logical address.

    The portable loader constructs CPU tensors, including when unconsumed device
    metadata is present. CPU native_layer_norm_backward reads statistics from
    data_ptr[row], ignoring their logical strides. Padding bytes and storage
    outside the recorded logical addresses have no canonical payload authority.
    """
    if overload != "aten.native_layer_norm_backward.default":
        return

    def argument(index, name):
        return positional[index] if index < len(positional) else arguments["kwargs"][name]

    try:
        positional = arguments["args"]["items"]
        source = argument(1, "input")
        shape, _, _ = _geometry(source)
        normalized = argument(2, "normalized_shape")
        if isinstance(normalized, dict) and normalized.get("kind") == "tuple":
            normalized = normalized["items"]
        if (
            not normalized
            or any(type(v) is not int or v <= 0 for v in normalized)
            or len(normalized) > len(shape)
            or shape[-len(normalized) :] != normalized
        ):
            raise ValueError("raw statistic row count requires a matching positive normalized shape")
        rows = prod(shape) // prod(normalized)
        unavailable = []
        for index, name in ((3, "mean"), (4, "rstd")):
            missing = _raw_coverage(argument(index, name), rows)
            if missing is not None:
                unavailable.append({"argument": name, "argument_index": index, **missing})
    except (KeyError, TypeError, ValueError) as exc:
        raise InputStorageUnavailable(
            str(exc), details={"overload": overload, "cause": "unrepresentable_input_storage"}
        ) from exc
    if unavailable:
        raise InputStorageUnavailable(
            "CPU layer-norm backward raw statistic reads lack recorded input storage",
            details={"overload": overload, "row_count": rows, "unavailable": unavailable},
        )
