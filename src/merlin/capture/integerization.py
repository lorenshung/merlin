"""Validate a complete integer/preserved-floating PT2E contraction partition.

This is frontend evidence, not accelerator or host admission. Preserving a
non-FP32 Q/DQ contraction never makes it an integer contraction or grants a
target implementation for that floating operation.
"""

from __future__ import annotations

#: Recipe families whose quantized work is not a contraction (merlin.targetgen.quant_recipe).
NON_CONTRACTION_ANNOTATIONS = ("annotated_sums", "annotated_means", "annotated_affine_maps", "annotated_movements")


def non_contraction_annotations(quantization_stats: dict | None) -> dict[str, int] | None:
    """The PT2E quantizer's non-contraction annotation census, when it annotated any; else None."""
    if not isinstance(quantization_stats, dict):
        return None
    counts = {key: quantization_stats.get(key, 0) for key in NON_CONTRACTION_ANNOTATIONS}
    if any(type(value) is not int or value < 0 for value in counts.values()) or not sum(counts.values()):
        return None
    return counts


def non_contraction_program(receipt: dict | None) -> bool:
    """A receipt for a quantized program with NO contraction but annotated non-contraction work.

    The frontend integerizes contractions only, so such a program's receipt has an all-zero
    contraction census; the capture worker marks it ``integer_program: non_contraction`` with the
    quantizer's annotation census. Every contraction-census check that requires at least one
    contraction accepts exactly this marked, zero, refusal-free receipt instead.
    """
    if not isinstance(receipt, dict) or receipt.get("integer_program") != "non_contraction":
        return False
    annotations = receipt.get("non_contraction_annotations")
    by_kind = receipt.get("quantized_by_kind")
    return (
        isinstance(annotations, dict)
        and set(annotations) == set(NON_CONTRACTION_ANNOTATIONS)
        and all(type(value) is int and value >= 0 for value in annotations.values())
        and sum(annotations.values()) > 0
        and all(
            receipt.get(key) == 0
            for key in (
                "quantized_contractions_seen",
                "quantized_contractions_integerized",
                "quantized_contractions_remaining",
            )
        )
        and isinstance(by_kind, dict)
        and all(isinstance(row, dict) and not any(row.values()) for row in by_kind.values())
        and receipt.get("refusals") == []
    )


def contraction_partition(receipt: dict) -> dict[str, int]:
    """Reject missing work, unresolved rewrites, or unaccounted precision refusals."""
    if not isinstance(receipt, dict) or receipt.get("schema") != "m2m.pt2e-integerize.v1":
        raise ValueError("unsupported integerization receipt")
    if non_contraction_program(receipt):
        return {"seen": 0, "integerized": 0, "preserved": 0}

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


#: The frontend's integer readouts: two f32 multiplies on the host (PT2E-portable), or the int8
#: accelerator readout -- an int32 bias seed and ONE RNE requantization (gate-3 validated).
READOUT_NUMERICS = ("float_readout", "integer_readout")


def accelerator_readout_numerics(recipe=None):
    """Exact integer semantics are mandatory for accelerator capture."""
    numerics = (recipe or {}).get("readout_numerics", "integer_readout")
    if numerics != "integer_readout":
        raise ValueError("accelerator capture requires integer_readout; float_readout is host-only")
    return numerics


def integerize_static_pt2e(model, inputs, *, numerics: str = "integer_readout"):
    """Reuse frontend integerization while preserving each source result layout.

    The frontend's conv rewrite computes its i32 matrix in patch order and
    exposes an NHWC-backed NCHW view. Materialize the source's own dense output
    layout after the readout; its rank and arithmetic are unchanged.

    ``numerics`` selects the frontend's readout (:data:`READOUT_NUMERICS`). A frontend build that
    predates the selection can only give the float readout, so asking it for the integer one is refused
    rather than silently answered with the other arithmetic.
    """
    if numerics not in READOUT_NUMERICS:
        raise ValueError(f"unknown integer readout numerics {numerics!r}; expected one of {READOUT_NUMERICS}")
    import inspect

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
    if "numerics" in inspect.signature(integerize_pt2e).parameters:
        model, receipt = integerize_pt2e(model, inputs, numerics=numerics)
    elif numerics == "float_readout":
        model, receipt = integerize_pt2e(model, inputs)
    else:
        raise ValueError("the selected frontend integerizer cannot produce the integer_readout numerics")
    receipt.setdefault("readout_numerics", numerics)
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
