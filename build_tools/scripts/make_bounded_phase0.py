#!/usr/bin/env python3
"""Define a strict retained-call Phase 0 experiment from a bounded suite.

No captures or cases are omitted because of placement results. Host exceptions
are explicit reasoned inputs, never inferred from a failed device preparation.
Use the printed CLI commands after generation; Phase 0 packaging is not an
execution/numerical verdict on Spike or RTL.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def write_yaml(path, document):
    path.write_text(yaml.safe_dump(document, sort_keys=False))
    path.chmod(0o600)


def generate(args):
    suite = json.loads(args.suite.read_bytes())
    cases = suite.get("selected_cases", suite.get("cases"))
    if not cases:
        raise ValueError("suite must contain cases or selected_cases")
    ids = [case.get("case_id", case["overload"]) for case in cases]
    identities = set(ids)
    if len(identities) != len(ids):
        raise ValueError("duplicate case identities")
    support = set(args.support.read_text().split())
    if not support or support - {case["overload"] for case in cases}:
        raise ValueError("support roster is empty or absent from suite")
    exceptions = json.loads(args.host_exceptions.read_bytes()) if args.host_exceptions else []
    by_exception = {}
    for row in exceptions:
        if (
            set(row) != {"case_id", "category", "reason", "evidence"}
            or not all(isinstance(value, str) and value.strip() for value in row.values())
            or row["case_id"] in by_exception
            or row["case_id"] not in identities
        ):
            raise ValueError("host exception requires a unique suite case_id, category, reason and evidence")
        by_exception[row["case_id"]] = row
    support_cases = [case for case in cases if case["overload"] in support]
    other_cases = [case for case in cases if case["overload"] not in support]
    if args.other_count is not None:
        # Round robin across overloads, with stable ordering inside each group.
        from collections import defaultdict

        groups = defaultdict(list)
        for case in sorted(other_cases, key=lambda c: c.get("case_id", c["overload"])):
            groups[case["overload"]].append(case)
        other_cases = []
        for index in range(max(map(len, groups.values()))):
            other_cases.extend(group[index] for group in groups.values() if index < len(group))
        other_cases = other_cases[: args.other_count]
    selected = sorted(support_cases + other_cases, key=lambda c: c.get("case_id", c["overload"]))
    public = [
        c.get("case_id", c["overload"]) for c in support_cases if c.get("case_id", c["overload"]) not in by_exception
    ]
    public_ids = set(public)
    selected_ids = {c.get("case_id", c["overload"]) for c in selected}
    guards = [c.get("case_id", c["overload"]) for c in selected if c.get("case_id", c["overload"]) not in public_ids]
    if not public or len(guards) < 2:
        raise ValueError("strict experiment requires device cases and two disjoint host calls")
    output = args.output
    output.mkdir(parents=True, exist_ok=False, mode=0o700)
    # A hidden host call is taken from the actual suite, never duplicated or invented.
    hidden = guards.pop(0)
    selected_suite = output / "cases.json"
    selected_suite.write_text(
        json.dumps({"cases": selected, "denominator_sha256": suite.get("denominator_sha256")}, sort_keys=True) + "\n"
    )
    selected_suite.chmod(0o600)
    descriptor = yaml.safe_load(args.descriptor.read_bytes())
    target = descriptor["target"]
    (descriptor.get("workload_spec") or {}).pop("applications", None)
    write_yaml(output / "descriptor.yaml", descriptor)
    flow = {
        "policy": "spec_int8",
        "software_spec": str(args.software_spec),
        "host_capabilities": str(args.host_capabilities),
        "execution_provider": str(args.execution_provider),
        "rtl_facts": str(args.facts),
        "device_package": str(args.device_package),
    }

    def row(select, lane):
        return {"cases": str(selected_suite), "captures": str(args.captures), "select": sorted(select), "lane": lane}

    recipe = {
        "capsule_policy": "derived_only",
        "capsules": [],
        "performance_oracles": {"L2": "spike", "L3": "verilator"},
        "datapath": {
            "semantic_defaults": {"must_accelerate": True},
            "compare": "exact_int",
            "requant_output_dtype": "i8",
            "requant_shift": 3,
            "required_oracle_tiers": ["L0", "L1", "L2", "L3"],
        },
        "core_aten": {
            "workers": args.workers,
            "overlay": str(args.overlay),
            "numeric_flow": flow,
            "cohorts": {
                "public": [row(public, "device")],
                "hidden": [row([hidden], "host-guard")],
                "host_guard": [row(guards, "host-guard")],
            },
        },
    }
    write_yaml(output / "recipe.yaml", recipe)
    derive = output / "derive"
    for name, recipe_path in (
        ("experiment.yaml", output / "recipe.yaml"),
        ("experiment-run.yaml", derive / "recipe.yaml"),
    ):
        write_yaml(
            output / name,
            {
                "schema_version": 1,
                "id": f"{target}-bounded-core-aten-phase0",
                "target": target,
                "description": "Strict bounded calls; support cases require device unless explicitly excepted.",
                "inputs": {"target": str(output / "descriptor.yaml")},
                "phases": {
                    0: {
                        "adapter": "capsule_derivation",
                        "config": {
                            "recipe": str(recipe_path),
                            "descriptor": str(output / "descriptor.yaml"),
                            "capability_contract": str(args.capability_contract),
                            "evidence_mode": "diagnostic",
                            "performance_template": str(args.performance_template),
                        },
                    }
                },
            },
        )
    receipt = {
        "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        "selected_count": len(selected),
        "support_count": len(support_cases),
        "device_count": len(public),
        "hidden_count": 1,
        "host_guard_count": len(guards),
        "host_exceptions": [by_exception[k] for k in sorted(by_exception) if k in selected_ids],
        "scope": "definition only; no execution claim",
    }
    (output / "definition.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (output / "definition.json").chmod(0o600)
    print(json.dumps({k: v for k, v in receipt.items() if k != "host_exceptions"}, indent=2))
    import shlex

    def quote(path):
        return shlex.quote(str(path))

    print(
        f"python -m merlin.cli experiment corpus derive {quote(output / 'experiment.yaml')} "
        f"--rtl-facts {quote(args.facts)} --output {quote(derive)}"
    )
    print(
        f"python -m merlin.cli experiment run {quote(output / 'experiment-run.yaml')} --phase 0 "
        f"--phase0-rtl-facts {quote(args.facts)} --phase0-conformance-spec {quote(derive / 'requirements.yaml')} "
        f"--phase0-synth-profile {quote(derive / 'synth.yaml')} --phase0-evidence-mode diagnostic "
        f"--run-dir {quote(output / 'run')}"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "suite",
        "captures",
        "support",
        "output",
        "descriptor",
        "facts",
        "device-package",
        "overlay",
        "software-spec",
        "host-capabilities",
        "execution-provider",
        "capability-contract",
        "performance-template",
    ):
        parser.add_argument("--" + name, type=lambda p: Path(p).absolute(), required=True)
    parser.add_argument("--host-exceptions", type=lambda p: Path(p).absolute())
    parser.add_argument("--other-count", type=int, help="Keep all support cases plus this many other cases")
    parser.add_argument("--workers", type=int, choices=range(1, 9), default=8)
    args = parser.parse_args()
    if args.other_count is not None and args.other_count < 2:
        parser.error("--other-count must be at least 2")
    generate(args)


if __name__ == "__main__":
    main()
