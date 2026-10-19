"""Declared tensor layouts for certified CPU scalar/vector storage forms.

Only metadata keys are copied. Payloads, expected results, operation names and
case identities cannot choose a path. Unknown compound layouts stay unproved.
"""


def input_layouts(report, arguments):
    result = {}

    def tensors(value):
        if isinstance(value, dict):
            if value.get("kind") == "tensor":
                return [{key: value[key] for key in ("shape", "stride") if key in value}]
            return [item for child in value.values() for item in tensors(child)]
        if isinstance(value, list):
            return [item for child in value for item in tensors(child)]
        return []

    for record in report["cases"]:
        if record.get("status") not in {None, "bundled"}:
            continue
        layouts = tensors((record.get("case") or {}).get("arguments", {}))
        indices = record.get("input_indices", [])
        if len(layouts) != len(indices):
            continue
        for index, layout in zip(indices, layouts):
            if index < len(arguments):
                result[arguments[index]] = layout
    return result


def profile(layout, *, block, grain):
    from math import prod

    shape, stride = layout.get("shape"), layout.get("stride")
    if shape is None or stride is None or len(shape) != len(stride) or any(s < 0 for s in (*shape, *stride)):
        return None
    total = prod(shape)
    if total > grain:
        return None  # Parallel range boundaries need their explicitly captured source profile.
    axes = sorted((i for i, n in enumerate(shape) if n > 1), key=lambda i: stride[i])
    size = 1
    dense = True
    for axis in axes:
        if stride[axis] != size:
            dense = False
        size *= shape[axis]
    if dense:
        row, vector = total, True
    elif len(shape) <= 1 or all(s == 0 for s in stride):
        axes = list(reversed(range(len(shape))))
        row, vector = total, not stride or stride[0] in (0, 1)
    else:
        # A non-dense multidimensional tensor can reorder/coalesce rows in
        # ways not established by this closed layout theorem.
        return None
    return dict(
        shape=list(shape),
        stride=list(stride),
        axes=axes,
        row=row,
        vector_prefix=row // block * block if vector else 0,
        vector_width=block,
        dense=dense,
        theorem="dense_physical_order_or_rank_one_tensor_iterator_v1",
    )


def vector_predicate(record, emit):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IndexType, IntegerAttr, i1
    from xdsl.dialects.linalg import ops as L

    if not record["vector_prefix"]:
        return emit(arith.ConstantOp(IntegerAttr(0, i1)))

    def c(value):
        return emit(arith.ConstantOp(IntegerAttr(value, IndexType())))

    position, factor = c(0), 1
    for axis in record["axes"]:
        index = emit(L.IndexOp(axis))
        position = emit(arith.AddiOp(position, emit(arith.MuliOp(index, c(factor)))))
        factor *= record["shape"][axis]
    local = emit(arith.RemUIOp(position, c(max(1, record["row"]))))
    return emit(arith.CmpiOp(local, c(record["vector_prefix"]), "ult"))
