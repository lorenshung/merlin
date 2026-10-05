"""Report source-to-device format obligations without treating routing as code generation."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Any

from merlin.common import mlir_query as mq


def input_format_changes(groups: Sequence[Any]) -> dict[str, int]:
    """Count device groups whose source operand types differ from their routed types."""
    from .compute_groups import HOST, _dtype_token

    changes: Counter[str] = Counter()
    for group in groups:
        if group.placement == HOST or group.root is None or not group.root.operands:
            continue
        observed = [_dtype_token(mq.type_shape_dtype(value.type)[1]) for value in group.root.operands[:2]]
        routed = [group.in_dtype, group.weight_dtype or group.in_dtype][: len(observed)]
        if any(src is not None and dst is not None and src != dst for src, dst in zip(observed, routed)):
            changes[f"{'+'.join(str(value) for value in observed)} -> {'+'.join(str(value) for value in routed)}"] += 1
    return dict(sorted(changes.items()))
