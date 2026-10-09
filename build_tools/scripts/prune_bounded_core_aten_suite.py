#!/usr/bin/env python3
"""Apply the recorded cut ledger to the bounded Core ATen suite and report the set after each cut."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from merlin.common.paths import artifacts_dir, repo_root
from merlin.targetgen.core_aten_prune import apply_ledger, load_ledger


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--suite",
        type=Path,
        default=repo_root()
        / "merlin-tests/PyTorchFullCoverageKernelSet/artifacts/bounded/bounded_core_aten_suite.json",
    )
    p.add_argument("--ledger", type=Path, default=repo_root() / "experiments/reference-data/core_aten_prune/cuts.yaml")
    p.add_argument("--output", type=Path, default=artifacts_dir() / "verification/core-aten/pruned")
    args = p.parse_args()
    raw = args.suite.read_bytes()
    ledger = load_ledger(args.ledger)
    pruned, steps = apply_ledger(json.loads(raw), ledger, hashlib.sha256(raw).hexdigest())
    out = args.output / steps[-1]["step"]
    out.mkdir(parents=True, exist_ok=True)
    suite_bytes = json.dumps(pruned, sort_keys=True).encode()
    (out / "suite.json").write_bytes(suite_bytes)
    (out / "state.json").write_text(
        json.dumps(
            {
                "parent_sha256": pruned["pruning"]["parent_sha256"],
                "ledger_sha256": pruned["pruning"]["ledger_sha256"],
                "suite_sha256": hashlib.sha256(suite_bytes).hexdigest(),
                "steps": steps,
            },
            indent=2,
        )
        + "\n"
    )
    for step in steps:
        line = f"{step['step']}: {step['case_count']} cases, {step['obligations_covered_count']} obligations covered"
        if step["step"] != "parent":
            line += (
                f" (-{step['cases_removed']} cases, -{step['obligations_lost_by_design']} by design,"
                f" -{len(step['obligations_lost_collateral'])} collateral,"
                f" {len(step['overloads_emptied'])} overloads emptied)"
            )
        print(line)
    print(out)


if __name__ == "__main__":
    main()
