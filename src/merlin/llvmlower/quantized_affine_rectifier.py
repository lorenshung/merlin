"""Exact finite-domain correction through integer affine maps and rectifiers.

This optional synthesis does not establish a target implementation or a source
rewrite. Its input is the complete ordered-binary32 pair certificate; providers
must independently close every intermediate arithmetic, rounding, clipping,
buffer, resource and execution obligation. No runtime sample grants eligibility.
"""

from __future__ import annotations

import hashlib

import numpy as np

from . import quantized_affine_pair as pair


def _range(values):
    return [int(values.min()), int(values.max())]


def _rectifier(a, b, lhs, rhs):
    # The positional base is derived from the complete signed-byte alphabet.
    # It is an injective key because differences in b have magnitude < 256.
    key = 256 * a + b - (256 * lhs + rhs)
    positive = np.clip(key, 0, 127)
    negative = np.clip(-key, 0, 127)
    indicator = np.clip(1 - positive - negative, 0, 127)
    return key, positive, negative, indicator


def _axis_rectifier(a, b, lhs, rhs):
    offsets = (a - lhs, lhs - a, b - rhs, rhs - b)
    parts = [np.clip(offset, 0, 127) for offset in offsets]
    indicator = np.ones((256, 256), dtype=np.int64)
    stages = []
    for part in parts:
        wide = indicator - part
        indicator = np.clip(wide, 0, 127)
        stages.append(dict(wide_range=_range(wide), output_range=_range(indicator)))
    return offsets, parts, indicator, stages


def derive(proof, *, max_pairs, indicator_family="positional_key"):
    """Synthesize a bounded exact sparse correction from a complete certificate.

    Each mismatch receives an injective affine key, its positive and negative
    clipped rectifiers, and a singleton indicator. Integer correction additions
    remain wide until the final source signed-byte clamp. This permits multiple
    disjoint mismatch pairs without depending on their captured frequency.
    """
    if type(max_pairs) is not int or max_pairs < 0:
        raise ValueError("explicit nonnegative correction cardinality limit required")
    if not isinstance(proof, dict) or proof != pair.derive(**proof["source"], **proof["predictor"]):
        raise ValueError("unchanged complete source/predictor pair certificate required")
    if indicator_family not in ("positional_key", "axis_offsets"):
        raise ValueError("unknown exact singleton indicator family")
    expected = pair.source_table(**proof["source"])

    predicted = pair.predictor_table(**proof["predictor"], relu=proof["source"]["relu"])
    locations = np.argwhere(expected != predicted)
    if len(locations) > max_pairs:
        raise ValueError("complete mismatch relation exceeds explicit correction cardinality limit")
    a = np.arange(-128, 128, dtype=np.int64)[:, None]
    b = np.arange(-128, 128, dtype=np.int64)[None, :]
    corrected = predicted.astype(np.int64)
    relation = []
    for left, right in locations:
        lhs, rhs = int(left) - 128, int(right) - 128
        if indicator_family == "positional_key":
            key, positive, negative, indicator = _rectifier(a, b, lhs, rhs)
            witness = dict(
                key=dict(lhs_coefficient=256, rhs_coefficient=1, seed=-(256 * lhs + rhs)),
                positive=dict(scale=1, clip=[0, 127], range=_range(positive)),
                negative=dict(scale=-1, clip=[0, 127], range=_range(negative)),
                indicator=dict(
                    seed=1, positive_coefficient=-1, negative_coefficient=-1, clip=[0, 127], range=_range(indicator)
                ),
                key_range=_range(key),
            )
        else:
            offsets, parts, indicator, stages = _axis_rectifier(a, b, lhs, rhs)
            witness = dict(
                offsets=[
                    dict(
                        axis=axis,
                        coefficient=coefficient,
                        seed=seed,
                        wide_range=_range(offset),
                        output_range=_range(part),
                    )
                    for axis, coefficient, seed, offset, part in zip(
                        ("lhs", "lhs", "rhs", "rhs"),
                        (1, -1, 1, -1),
                        (-lhs, lhs, -rhs, rhs),
                        offsets,
                        parts,
                        strict=True,
                    )
                ],
                indicator=dict(seed=1, part_coefficient=-1, clip=[0, 127], range=_range(indicator), stages=stages),
            )

        exact_indicator = (a == lhs) & (b == rhs)
        if not np.array_equal(indicator, exact_indicator):
            raise ValueError("rectifier does not identify its complete singleton relation")
        delta = int(expected[left, right]) - int(predicted[left, right])
        corrected = corrected + delta * indicator
        relation.append(
            dict(
                lhs=lhs,
                rhs=rhs,
                correction=delta,
                **witness,
                indicator_pairs=int(indicator.sum()),
                corrected_prefix_range=_range(corrected),
            )
        )
    if not np.array_equal(corrected, expected):
        raise ValueError("derived sparse correction differs from complete original source observation")
    widest = max([128, *(abs(value) for row in relation for value in row.get("key_range", [0]))])
    if widest >= 1 << 31:
        raise ValueError("rectifier intermediate exceeds signed-i32 representation")
    return dict(
        schema="quantized_affine_rectifier_certificate_v1",
        source=proof["source"],
        predictor=proof["predictor"],
        pairs=65536,
        max_pairs=max_pairs,
        indicator_family=indicator_family,
        relation=relation,
        source_table_sha256=proof["source_table_sha256"],
        predictor_table_sha256=proof["predictor_table_sha256"],
        correction_bitmap_sha256=proof["correction_bitmap_sha256"],
        corrected_table_sha256=hashlib.sha256(corrected.astype(np.int8).tobytes()).hexdigest(),
        predictor_range=_range(predicted),
        corrected_range=_range(corrected),
        source_arithmetic=proof["source_arithmetic"],
        indicator_semantics=(
            "Exact integer affine key; clip(key,0,127), clip(-key,0,127), clip(1-u-v,0,127)"
            if indicator_family == "positional_key"
            else "Four clipped source-coordinate offsets; successive clip(e-offset,0,127) from e=1"
        ),
        correction_semantics="Exact integer prediction plus each disjoint singleton correction; final source clamp",
        exact_for_all_source_pairs=True,
        obligations=[
            "Provider independently qualifies the original predictor conversion/product/RNE/clamp",
            "Every integer key, seed, prefix and correction remains exact before the declared clipping",
            "Scaled integer reads implement exact scales +1 and -1 and the declared rectifiers",
            "Source inputs remain immutable and intermediate/output storage owners are disjoint",
            "Provider closes intermediate storage lifetimes, instruction capabilities and full transfer costs",
            "Original source floating environment and observation/effect contract remain caller obligations",
        ],
        performance="UNKNOWN: synthesis is not a cycle, traffic or profitability proof",
    )


def validate(certificate):
    """Recreate the complete source proof and every circuit witness."""
    if not isinstance(certificate, dict):
        raise ValueError("rectifier certificate dictionary required")
    proof = pair.derive(**certificate["source"], **certificate["predictor"])
    if certificate != derive(
        proof, max_pairs=certificate["max_pairs"], indicator_family=certificate["indicator_family"]
    ):
        raise ValueError("rectifier certificate changed after complete-domain synthesis")
    return certificate
