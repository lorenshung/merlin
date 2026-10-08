"""Gemmini provider of the additive Core ATen overlay.

This is the target edge of the generic overlay mechanism in
:mod:`merlin.targetgen.core_aten_overlay`.  Its cases may add to the portable bounded suite, but
may never satisfy or replace a portable obligation.  Most cases invoke Core ATen overloads.  The
single ``aten._int_mm.default`` case is an explicitly labelled non-Core bridge to Gemmini's widening
``i8 x i8 -> i32`` datapath; it is never counted in the Core ATen denominator.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from merlin.common.paths import repo_root
from merlin.targetgen.core_aten_overlay import (
    CORE,
    NON_CORE_BRIDGE,
    build_overlay,
    overlay_digest,
    profile_digest,
    validate_overlay,
)

_ABI_HEADER = Path("examples/gemmini/phase1/contracts/harness_curated/gemmini-rocc-tests/include/gemmini_params.h")
_TARGET_CONTRACT = Path("examples/gemmini/target/contracts/target_contract.yaml")
_CONFORMANCE = Path("experiments/reference-data/phase0/conformance/gemmini.yaml")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _macro(text: str, name: str) -> int:
    """First ``#define NAME <integer>`` line; any other spelling of the line does not match."""

    for line in text.split("\n"):
        directive, separator, rest = line.partition("#define")
        if directive or not separator or not rest[:1].isspace():
            continue
        fields = rest.split()
        if len(fields) == 2 and fields[0] == name and fields[1].isdecimal():
            return int(fields[1])
    raise RuntimeError(f"Gemmini ABI header does not define integer macro {name}")


def gemmini_profile() -> dict[str, Any]:
    """Read and validate the pinned Gemmini configuration used to derive the overlay."""

    header_path = repo_root() / _ABI_HEADER
    contract_path = repo_root() / _TARGET_CONTRACT
    conformance_path = repo_root() / _CONFORMANCE
    header = header_path.read_text(encoding="utf-8")
    facts = {
        "tile_edge": _macro(header, "DIM"),
        "scratchpad_banks": _macro(header, "BANK_NUM"),
        "scratchpad_rows_per_bank": _macro(header, "BANK_ROWS"),
        "accumulator_rows": _macro(header, "ACC_ROWS"),
        "dma_max_bytes": _macro(header, "MAX_BYTES"),
        "operand_dtype": "int8",
        "accumulator_dtype": "int32",
        "readout_dtypes": ["int8", "int32"],
    }
    if "typedef int8_t elem_t;" not in header or "typedef int32_t acc_t;" not in header:
        raise RuntimeError("Gemmini ABI header no longer declares the expected i8/i32 datapath")
    if facts != {
        "tile_edge": 16,
        "scratchpad_banks": 4,
        "scratchpad_rows_per_bank": 4096,
        "accumulator_rows": 1024,
        "dma_max_bytes": 64,
        "operand_dtype": "int8",
        "accumulator_dtype": "int32",
        "readout_dtypes": ["int8", "int32"],
    }:
        raise RuntimeError(f"Gemmini configuration changed; re-derive the overlay instead of guessing: {facts}")
    return {
        "schema_version": 1,
        "name": "gemmini-core-aten-additive-v1",
        "target": "gemmini",
        "hardware_config": "GemminiRocketConfig",
        "facts": facts,
        "sources": [
            {"path": str(_ABI_HEADER), "sha256": _sha256(header_path)},
            {"path": str(_TARGET_CONTRACT), "sha256": _sha256(contract_path)},
            {"path": str(_CONFORMANCE), "sha256": _sha256(conformance_path)},
        ],
        "scope": (
            "additive Gemmini contraction boundary witnesses; never a replacement for portable Core ATen obligations"
        ),
    }


def _cycle(shape: tuple[int, ...], values: tuple[int, ...]):
    import torch

    count = math.prod(shape)
    base = torch.tensor(values, dtype=torch.int8)
    return base.repeat((count + len(values) - 1) // len(values))[:count].reshape(shape)


def _mm_args(m: int, k: int, n: int, *, transpose_rhs: bool = False, extrema: bool = False):
    values = (-128, -1, 0, 1, 127) if extrema else (-4, -3, -2, -1, 0, 1, 2, 3)
    left = _cycle((m, k), values)
    if transpose_rhs:
        right = _cycle((n, k), tuple(reversed(values))).transpose(0, 1)
    else:
        right = _cycle((k, n), tuple(reversed(values)))
    return (left, right), {}


def _bmm_args(tile: int):
    return (
        _cycle((2, tile, 2 * tile), (-4, -3, -2, -1, 0, 1, 2, 3)),
        _cycle((2, 2 * tile, tile), (3, 2, 1, 0, -1, -2, -3, -4)),
    ), {}


def _conv_args(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: tuple[int, int],
    padding: tuple[int, int],
):
    return (
        _cycle(input_shape, (-4, -3, -2, -1, 0, 1, 2, 3)),
        _cycle(weight_shape, (3, 2, 1, 0, -1, -2, -3, -4)),
        None,
        stride,
        padding,
        (1, 1),
        False,
        (0, 0),
        1,
    ), {}


def _candidate_specs(tile: int):
    return [
        (
            "mm_aligned_contiguous",
            "aten.mm.default",
            lambda: _mm_args(tile, 2 * tile, tile),
            {"gemmini::cell::contraction/i8/aligned", "gemmini::rank::contraction/2"},
            CORE,
        ),
        (
            "mm_aligned_rhs_transposed",
            "aten.mm.default",
            lambda: _mm_args(tile, 2 * tile, tile, transpose_rhs=True),
            {
                "gemmini::cell::contraction/i8/aligned",
                "gemmini::layout::contraction/rhs_transposed",
                "gemmini::rank::contraction/2",
                "gemmini::readout::int8",
            },
            CORE,
        ),
        (
            "mm_partial",
            "aten.mm.default",
            lambda: _mm_args(tile, 2 * tile - 1, tile - 1),
            {"gemmini::cell::contraction/i8/partial", "gemmini::rank::contraction/2"},
            CORE,
        ),
        (
            "mm_sub_tile",
            "aten.mm.default",
            lambda: _mm_args(max(1, tile // 8), max(1, tile // 4), max(1, tile // 4)),
            {"gemmini::cell::contraction/i8/sub_tile", "gemmini::rank::contraction/2"},
            CORE,
        ),
        (
            "bmm_batched",
            "aten.bmm.default",
            lambda: _bmm_args(tile),
            {"gemmini::rank::contraction/3"},
            CORE,
        ),
        (
            "conv_3x3_valid",
            "aten.convolution.default",
            lambda: _conv_args((1, 4, 8, 8), (8, 4, 3, 3), (1, 1), (0, 0)),
            {"gemmini::conv::3x3_valid", "gemmini::rank::contraction/4"},
            CORE,
        ),
        (
            "conv_3x3_pad1",
            "aten.convolution.default",
            lambda: _conv_args((1, 4, 8, 8), (8, 4, 3, 3), (1, 1), (1, 1)),
            {"gemmini::conv::3x3_pad1", "gemmini::rank::contraction/4"},
            CORE,
        ),
        (
            "conv_3x3_stride2",
            "aten.convolution.default",
            lambda: _conv_args((1, 4, 8, 8), (8, 4, 3, 3), (2, 2), (0, 0)),
            {"gemmini::conv::3x3_stride2", "gemmini::rank::contraction/4"},
            CORE,
        ),
        (
            "conv_1x1_pointwise",
            "aten.convolution.default",
            lambda: _conv_args((1, tile, 6, 6), (tile, tile, 1, 1), (1, 1), (0, 0)),
            {"gemmini::conv::1x1_pointwise", "gemmini::rank::contraction/4"},
            CORE,
        ),
        (
            "int_mm_widening_bridge",
            "aten._int_mm.default",
            lambda: _mm_args(tile, 2 * tile, tile, extrema=True),
            {"gemmini::numeric::i8xi8_i32", "gemmini::readout::int32"},
            NON_CORE_BRIDGE,
        ),
    ]


class GemminiOverlayProvider:
    """Supplies Gemmini's pinned configuration, candidate pool and descriptive fields."""

    def profile(self) -> dict[str, Any]:
        return gemmini_profile()

    def candidate_specs(self, profile: Mapping[str, Any]):
        return _candidate_specs(int(profile["facts"]["tile_edge"]))

    def document_fields(self) -> dict[str, Any]:
        return {
            "scope": "additive Gemmini constraints layered on the portable bounded Core ATen suite",
            "claim": "exact minimum over the finite evidence-derived Gemmini candidate pool",
            "non_core_bridge_overloads": ["aten._int_mm.default"],
            "deferred_to_existing_target_conformance": [
                "DMA payload boundaries at 63/64/65 i8 and 15/16/17 i32 elements",
                "fits_double/fits_single/spills operand-store regimes and K=4096/4112/8192/8208",
                "bias_add/acc_scale/relu/maxpool fused epilogues and carried configuration state",
                "round-to-nearest-even and saturating i8 accumulator-scale readout",
                "legal convolution/pooling parameter combinations beyond the four evidenced source geometries",
            ],
        }

    def summary(self, document: Mapping[str, Any]) -> str:
        return gemmini_overlay_summary(document)


PROVIDER = GemminiOverlayProvider()


def gemmini_profile_digest(profile: Mapping[str, Any]) -> str:
    return profile_digest(profile)


def gemmini_overlay_digest(document: Mapping[str, Any]) -> str:
    return overlay_digest(document)


def build_gemmini_overlay(*, pytorch_version: str) -> dict[str, Any]:
    """Build, duplicate-run, and exactly minimize the additive Gemmini candidate pool."""

    return build_overlay(PROVIDER, pytorch_version=pytorch_version)


def validate_gemmini_overlay(
    document: Mapping[str, Any],
    *,
    validate_eager_oracles: bool = True,
    validate_solver_certificate: bool = True,
) -> list[dict[str, Any]]:
    """Fail closed unless an overlay still matches its sources, solver proof, and eager oracles."""

    return validate_overlay(
        PROVIDER,
        document,
        validate_eager_oracles=validate_eager_oracles,
        validate_solver_certificate=validate_solver_certificate,
    )


def gemmini_overlay_summary(document: Mapping[str, Any]) -> str:
    facts = document["profile"]["facts"]
    selected = [case["overlay_case_name"] for case in document["selected_cases"]]
    return "\n".join(
        [
            "# Additive Gemmini PyTorch coverage overlay",
            "",
            f"- Configuration: `{document['profile']['hardware_config']}`",
            f"- Datapath: `{facts['operand_dtype']} x {facts['operand_dtype']} -> {facts['accumulator_dtype']}`",
            f"- Tile edge: {facts['tile_edge']}",
            f"- Candidates: {document['candidate_count']}",
            f"- Exact minimum selected: {document['selected_count']}",
            f"- Evidence-derived obligations: {document['obligation_count']}",
            "- Portable Core ATen cases replaced: 0",
            "",
            "The `_int_mm` case is an explicitly non-Core bridge to widening accumulation and does not "
            "change the 193-overload denominator. DMA, capacity, epilogue, and detailed convolution "
            "constraints remain additive requirements of the existing Gemmini target-conformance corpus.",
            "",
            "## Selected cases",
            "",
            *[f"- `{name}`" for name in selected],
            "",
        ]
    )
