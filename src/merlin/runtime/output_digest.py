"""Verify compact target output evidence against an independently validated array."""

from __future__ import annotations

import hashlib

import numpy as np


def verify_output_sha256(text: str, reference: np.ndarray) -> dict:
    """Require one canonical f32le digest matching all reference elements.

    The caller must separately establish the reference's numerical validity and
    provenance; a matching digest alone does not validate model semantics.
    """
    records = [line.split() for line in text.splitlines() if line.startswith("OUT_SHA256")]
    if len(records) != 1:
        raise ValueError("expected exactly one OUT_SHA256 f32le record")
    record = records[0]
    if (
        len(record) != 5
        or record[:2] != ["OUT_SHA256", "f32le"]
        or any(not field or any(c not in "0123456789" for c in field) for field in record[2:4])
        or len(record[4]) != 64
        or any(c not in "0123456789abcdef" for c in record[4])
    ):
        raise ValueError("invalid OUT_SHA256 f32le record")
    array = np.asarray(reference)
    if array.dtype.kind != "f" or array.dtype.itemsize != 4:
        raise ValueError("reference must contain f32 values")
    raw = array.astype("<f4", copy=False).tobytes(order="C")
    elements, byte_count, observed = record[2:]
    expected = hashlib.sha256(raw).hexdigest()
    if int(elements) != array.size or int(byte_count) != len(raw):
        raise ValueError("output digest element or byte count mismatch")
    if observed != expected:
        raise ValueError("full output SHA256 mismatch")
    return {"encoding": "f32le", "elements": array.size, "bytes": len(raw), "sha256": expected}
