"""Execute the public prepared cohort through an explicitly selected reference package."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
from pathlib import Path
from types import SimpleNamespace

from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.core_aten_capsules import run_capsule
from merlin.targetgen.core_aten_device import load_execution_provider


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--release", type=Path, required=True)
    parser.add_argument("--provider", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    args.output.mkdir(parents=True, mode=0o700)
    os.environ["MERLIN_TARGET_CONTRACT"] = str(args.release / "payload/experiment/contracts/target_contract.yaml")
    os.environ["MERLIN_TARGET_EXPERIMENT"] = str(args.release / "payload/experiment/target_experiment.yaml")
    provider = load_execution_provider("gemmini", args.provider / "merlin-support/core_aten/execution_provider.py")
    # Use the same native full-call adapter and capsule runner as candidate grading.
    adapter = provider.full_call_oracle("gemmini")
    public = args.release / "payload/corpus/public"
    capsules = [load_capsule(p.parent) for p in public.rglob("capsule.yaml")]
    scored = [c for c in capsules if c.get("scored", True)]

    def execute(capsule):
        result = run_capsule(
            capsule,
            args.provider,
            SimpleNamespace(run_path=args.output / capsule["name"]),
            SimpleNamespace(target="gemmini", rtl_tiers=()),
            {"L2": adapter},
            None,
            None,
            600,
            False,
        )
        result["source_reference"] = capsule["source_reference"]
        result["numeric_flow"] = capsule["operation"]["attributes"].get("numeric_flow")
        return result

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(execute, scored))
    from collections import Counter

    by_op = {}
    for result in results:
        tally = by_op.setdefault(result["source_reference"], dict(total=0, passed=0, executed_instructions=0))
        tally["total"] += 1
        tally["passed"] += result["status"] == "pass"
        tally["executed_instructions"] += result["executed_instructions"]
    summary = dict(
        scope="public scored cohort only; functional Spike/libgemmini, no RTL timing claim",
        total=len(results),
        passed=sum(r["status"] == "pass" for r in results),
        statuses=dict(Counter(r["status"] for r in results)),
        by_op=by_op,
        failures=[
            dict(capsule=r["capsule"], op=r["source_reference"], failure=r["failure"])
            for r in results
            if r["status"] != "pass"
        ],
        results=results,
    )
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "results"}, indent=2))


if __name__ == "__main__":
    main()
