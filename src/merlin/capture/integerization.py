"""Validate a complete integer/preserved-floating PT2E contraction partition.

This is frontend evidence, not accelerator or host admission. Preserving a
non-FP32 Q/DQ contraction never makes it an integer contraction or grants a
target implementation for that floating operation.
"""

from __future__ import annotations


def contraction_partition(receipt: dict) -> dict[str, int]:
    """Reject missing work, unresolved rewrites, or unaccounted precision refusals."""
    if not isinstance(receipt, dict) or receipt.get("schema") != "m2m.pt2e-integerize.v1":
        raise ValueError("unsupported integerization receipt")

    def count(value: object) -> int:
        if type(value) is not int or value < 0:
            raise ValueError("invalid integerization contraction census")
        return value

    seen = count(receipt.get("quantized_contractions_seen"))
    integerized = count(receipt.get("quantized_contractions_integerized"))
    remaining = count(receipt.get("quantized_contractions_remaining"))
    if seen < 1 or integerized < 1 or seen != integerized + remaining:
        raise ValueError("incomplete integerization contraction census")
    by_kind = receipt.get("quantized_by_kind")
    kinds = {"linear", "conv2d", "matmul", "unsupported"}
    if not isinstance(by_kind, dict) or set(by_kind) != kinds:
        raise ValueError("missing per-kind integerization contraction census")
    totals = [0, 0, 0]
    for kind in kinds:
        row = by_kind[kind]
        if not isinstance(row, dict):
            raise ValueError("invalid per-kind integerization contraction census")
        values = [count(row.get(key)) for key in ("seen", "integerized", "remaining")]
        if values[0] != values[1] + values[2]:
            raise ValueError("inconsistent per-kind integerization contraction census")
        totals = [total + value for total, value in zip(totals, values)]
    if totals != [seen, integerized, remaining] or by_kind["unsupported"]["seen"]:
        raise ValueError("unresolved or inconsistent integerization contraction census")

    refusals = receipt.get("refusals")
    if not isinstance(refusals, list):
        raise ValueError("missing integerization refusal inventory")
    decisions = receipt.get("precision_decisions")
    # Historical complete-integer receipts did not carry a precision ledger.
    # They can never authorize a preserved floating region through this path.
    if decisions is None:
        if remaining or refusals or receipt.get("precision_decision_counts") is not None:
            raise ValueError("remaining contractions lack an exact preserved-precision ledger")
        return {"seen": seen, "integerized": integerized, "preserved": 0}
    if not isinstance(decisions, list) or len(decisions) != seen:
        raise ValueError("incomplete integerization precision ledger")
    counts = {"integerized_i32": 0, "preserve_float_qdq": 0, "unresolved": 0}
    identities: set[tuple[str, str]] = set()
    preserved: set[tuple[str, str]] = set()
    decision_kinds = {kind: [0, 0, 0] for kind in kinds}
    for row in decisions:
        if not isinstance(row, dict):
            raise ValueError("invalid integerization precision decision")
        kind, node, decision = row.get("kind"), row.get("node"), row.get("decision")
        if (
            not isinstance(kind, str)
            or kind not in kinds
            or not isinstance(node, str)
            or not node
            or not isinstance(decision, str)
            or decision not in counts
        ):
            raise ValueError("invalid integerization precision decision")
        identity = (kind, node)
        if identity in identities:
            raise ValueError("duplicate integerization precision decision")
        identities.add(identity)
        counts[decision] += 1
        decision_kinds[kind][0] += 1
        if decision == "integerized_i32":
            decision_kinds[kind][1] += 1
        elif decision == "preserve_float_qdq":
            if (
                not isinstance(row.get("source_dtype"), str)
                or row["source_dtype"] not in {"torch.bfloat16", "torch.float16"}
                or row.get("required_numeric_semantics") != "dequantize-before-floating-contraction"
            ):
                raise ValueError("preserved Q/DQ lacks exact non-FP32 floating semantics")
            preserved.add(identity)
            decision_kinds[kind][2] += 1
    declared_counts = receipt.get("precision_decision_counts")
    if not isinstance(declared_counts, dict) or set(declared_counts) != set(counts):
        raise ValueError("missing integerization precision partition counts")
    declared_counts = {key: count(value) for key, value in declared_counts.items()}
    if (
        counts != {"integerized_i32": integerized, "preserve_float_qdq": remaining, "unresolved": 0}
        or declared_counts != counts
        or any(
            decision_kinds[kind] != [by_kind[kind][key] for key in ("seen", "integerized", "remaining")]
            for kind in kinds
        )
    ):
        raise ValueError("unresolved or inconsistent integerization precision partition")
    refused: set[tuple[str, str]] = set()
    for row in refusals:
        if not isinstance(row, dict) or not all(isinstance(row.get(key), str) for key in ("kind", "node")):
            raise ValueError("invalid integerization refusal")
        identity = (row.get("kind"), row.get("node"))
        if (
            identity not in preserved
            or identity in refused
            or not isinstance(row.get("reason"), str)
            or not row["reason"]
        ):
            raise ValueError("integerization refusal is not an exact preserved-precision decision")
        refused.add(identity)
    if refused != preserved:
        raise ValueError("preserved-precision decisions and refusal inventory differ")
    return {"seen": seen, "integerized": integerized, "preserved": remaining}


def integerize_static_pt2e(model, inputs):
    """Reuse frontend integerization while preserving each source result layout.

    The frontend's conv rewrite computes its i32 matrix in patch order and
    exposes an NHWC-backed NCHW view. Materialize the source's own dense output
    layout after the readout; its rank and arithmetic are unchanged.
    """
    import torch
    from m2m.capture.pt2e_integerize import integerize_pt2e

    layouts = {}
    for node in model.graph.nodes:
        value = node.meta.get("val")
        if node.target == torch.ops.aten.conv2d.default and isinstance(value, torch.Tensor):
            if value.is_contiguous():
                layout = torch.contiguous_format
            elif value.is_contiguous(memory_format=torch.channels_last):
                layout = torch.channels_last
            else:
                raise ValueError("integer convolution source has no supported dense result layout")
            layouts[node.name] = (layout, list(value.stride()))
    model, receipt = integerize_pt2e(model, inputs)
    restorations = []
    graph = model.graph
    for node in list(graph.nodes):
        source = node.meta.get("pt2e_integerized_from")
        if source not in layouts:
            continue
        layout, stride = layouts[source]
        with graph.inserting_after(node):
            restored = graph.call_function(torch.ops.aten.clone.default, args=(node,), kwargs={"memory_format": layout})
        node.replace_all_uses_with(restored)
        restored.args = (node,)
        restored.meta = dict(node.meta)
        node.meta.pop("pt2e_integerized_from")
        restorations.append({"source": source, "stride": stride, "memory_format": str(layout)})
    graph.lint()
    model.recompile()
    receipt["result_layout_restorations"] = restorations
    return model, receipt
