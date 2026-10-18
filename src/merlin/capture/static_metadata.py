"""Specialize pure metadata queries under the static capture input ABI."""

STATIC_METADATA_QUERIES = frozenset(
    f"aten.{name}" for name in (
        "sym_numel.default", "sym_size.int", "sym_stride.int",
        "sym_storage_offset.default", "sym_is_contiguous.default",
    )
)


def specialize_static_metadata(model, inputs):
    """Fold a metadata-only FX graph from input descriptors, never tensor values.

    Capture fixes shapes, strides and offsets. Keep every original input in the
    graph so normal boundary observations and alias checks still apply. Mixed
    data/metadata programs retain the ordinary frontend path.
    """
    import copy

    import torch
    from torch.fx.node import map_arg

    queries = {getattr(getattr(torch.ops.aten, name.split(".")[1]), name.split(".")[2])
               for name in STATIC_METADATA_QUERIES}
    try:
        graph = torch.fx.symbolic_trace(copy.deepcopy(model))
    except (RuntimeError, TypeError, ValueError, torch.fx.proxy.TraceError):
        return model, None
    nodes = list(graph.graph.nodes)
    calls = [node for node in nodes if node.op == "call_function"]
    if (
        not any(node.target in queries for node in calls)
        or any(node.op not in {"placeholder", "call_function", "output"} for node in nodes)
        or any(node.target not in queries | {torch.ops.aten.scalar_tensor.default} for node in calls)
    ):
        return model, None
    placeholders = [node for node in nodes if node.op == "placeholder"]
    if len(placeholders) != len(inputs) or not all(isinstance(value, torch.Tensor) for value in inputs):
        return model, None
    arguments = dict(zip(placeholders, inputs, strict=True))
    folded = []
    for node in calls:
        if node.target not in queries:
            continue
        # Only direct input descriptors have authority here. No tensor producer
        # is evaluated or inferred from its sealed expected result.
        if not node.args or node.args[0] not in arguments:
            return model, None
        args = map_arg(node.args, arguments.__getitem__)
        kwargs = map_arg(node.kwargs, arguments.__getitem__)
        value = node.target(*args, **kwargs)
        if type(value) not in {int, bool}:
            return model, None
        folded.append({"operator": str(node.target), "value": value})
        for user in list(node.users):
            user.args = map_arg(user.args, lambda arg: value if arg is node else arg)
            user.kwargs = map_arg(user.kwargs, lambda arg: value if arg is node else arg)
        graph.graph.erase_node(node)
    graph.graph.lint()
    graph.recompile()
    return graph, {
        "authority": "static_capture_input_descriptors",
        "inputs": [
            {"shape": list(value.shape), "stride": list(value.stride()), "storage_offset": value.storage_offset()}
            for value in inputs
        ],
        "queries": folded,
    }
