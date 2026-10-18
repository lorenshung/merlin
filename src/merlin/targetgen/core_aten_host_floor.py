"""Exact native dtype gate, independent of tolerance-based corpus comparisons."""

IEEE_WORDS = {
    "f16": (2, 10, 5),
    "bf16": (2, 7, 8),
    "f32": (4, 23, 8),
    "f64": (8, 52, 11),
    "complex<f32>": (4, 23, 8),
    "complex<f64>": (8, 52, 11),
}


def grade_special_bits(record, outputs):
    """Qualify propagation without imposing a source reduction order on finite data."""
    boundary = record.get("native_eager_boundary", {})
    if boundary.get("authority") != "untransformed_native_cpu_eager":
        return dict(status="ungradable", reason="independent native special boundary unavailable")
    indices = record.get("native_eager_indices", {"outputs": record["output_indices"]})["outputs"]
    comparisons = []
    for index, abi, encoded in zip(indices, record["output_abi"], boundary["output_bytes"], strict=True):
        if abi["dtype"] not in IEEE_WORDS:
            continue
        width, fraction, exponent = IEEE_WORDS[abi["dtype"]]
        mask = ((1 << exponent) - 1) << fraction
        expected = bytes.fromhex(encoded)
        if index >= len(outputs) or len(outputs[index]) != len(expected):
            return dict(status="mismatch", reason="source special boundary has a different size")
        for offset in range(0, len(expected), width):
            a, b = outputs[index][offset : offset + width], expected[offset : offset + width]
            if any(int.from_bytes(raw, "little") & mask == mask for raw in (a, b)):
                comparisons.append(a == b)
    return dict(status="pass" if all(comparisons) else "mismatch", comparison="native_nonfinite_storage_bits")


def grade_native_bits(record, outputs, *, execution_error=None):
    boundary = record.get("native_eager_boundary")
    if not boundary or boundary.get("authority") != "untransformed_native_cpu_eager":
        return dict(status="ungradable", reason="native dtype execution lacks an independent eager byte boundary")
    if outputs is None or execution_error is not None:
        return dict(status="execution_failed", reason="native dtype execution produced no complete byte boundary")
    indices = record.get(
        "native_eager_indices",
        {"outputs": record["output_indices"], "posts": record.get("semantic_boundary", {}).get("post_indices", [])},
    )
    expected = {}
    for boundary_indices, values in (
        (indices["outputs"], boundary["output_bytes"]),
        (indices["posts"], boundary["post_input_bytes"]),
    ):
        if len(boundary_indices) != len(values):
            return dict(status="ungradable", reason="native eager reference does not cover the exported boundary")
        for index, raw in zip(boundary_indices, values, strict=True):
            if index in expected and expected[index] != raw:
                return dict(status="ungradable", reason="native eager aliases disagree in their storage bits")
            expected[index] = raw
    comparisons = [
        dict(output_index=i, status="pass" if i < len(outputs) and outputs[i].hex() == raw else "mismatch")
        for i, raw in sorted(expected.items())
    ]
    passed = all(row["status"] == "pass" for row in comparisons)
    if not passed and record.get("numeric_admission", {}).get("host_floor", {}).get("order_dependent_reduction"):
        post_indices = set(indices["posts"])
        if all(row["status"] == "pass" for row in comparisons if row["output_index"] in post_indices):
            from merlin.targetgen.native_rational_reference import compare_reduction_floor

            try:
                floor = compare_reduction_floor(record, outputs)
            except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
                floor = dict(status="ungradable", reason=str(exc))
            if floor["status"] == "pass":
                return {**floor, "reason": "", "post_input_bits": "exact"}
    return dict(
        status="pass" if passed else "mismatch",
        comparison="exact_native_eager_storage_bits",
        signed_zero="distinct",
        nan_payload="preserved",
        results=comparisons,
        reason="" if passed else "native dtype output bits differ from independent CPU eager",
    )


def native_boundary_has_nan(report):
    """Qualification only: a native NaN has no ordered rational-error floor.

    These independently captured CPU bytes can refuse an unproved candidate;
    they never supply a result, NaN sign/payload, packing value or epilogue.
    """
    formats = IEEE_WORDS
    for record in report["cases"]:
        boundary = record.get("native_eager_boundary", {})
        if boundary.get("authority") != "untransformed_native_cpu_eager":
            continue
        for abi, encoded in zip(record.get("output_abi", []), boundary.get("output_bytes", []), strict=True):
            if abi["dtype"] not in formats:
                continue
            width, fraction, exponent = formats[abi["dtype"]]
            payload = bytes.fromhex(encoded)
            if len(payload) % width:
                raise ValueError("native boundary word is incomplete")
            exponent_mask, fraction_mask = ((1 << exponent) - 1) << fraction, (1 << fraction) - 1
            for offset in range(0, len(payload), width):
                word = int.from_bytes(payload[offset : offset + width], "little")
                if word & exponent_mask == exponent_mask and word & fraction_mask:
                    return True
    return False
