#!/usr/bin/env python3
"""Build the additive, evidence-derived Gemmini overlay for the Core ATen suite."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path

from merlin.common.paths import artifacts_dir
from merlin.targetgen._aten_opset_worker import core_opset
from merlin.targetgen.core_aten_cover import json_bytes


def _load_provider():
    """Load the sibling provider by path; examples are not an installed package."""

    path = Path(__file__).resolve().with_name("overlay_provider.py")
    spec = importlib.util.spec_from_file_location("gemmini_core_aten_overlay_provider", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args(argv)
    provider = _load_provider()
    output = args.output_dir or artifacts_dir() / "verification" / "core-aten" / "gemmini-overlay"
    output.mkdir(parents=True, exist_ok=True)
    document = provider.build_gemmini_overlay(pytorch_version=core_opset()["torch"])
    (output / "gemmini_core_aten_overlay.json").write_bytes(json_bytes(document))
    (output / "gemmini_core_aten_overlay.md").write_text(provider.gemmini_overlay_summary(document), encoding="utf-8")
    print(
        f"{document['selected_count']}/{document['candidate_count']} cases cover "
        f"{document['obligation_count']} additive Gemmini obligations"
    )
    print(f"artifacts: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
