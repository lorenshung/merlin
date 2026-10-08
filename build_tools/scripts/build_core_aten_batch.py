#!/usr/bin/env python3
"""Build one merged Core ATen MLIR bundle from canonical case captures."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from merlin.common.paths import artifacts_dir
from merlin.targetgen.core_aten_batch import build_core_aten_batch
from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch


def _write_json(path: Path, document: object) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _run_spike_bundle(directory: Path, report: dict, *, arena_mb: int, timeout: int) -> tuple[dict, dict]:
    from merlin.runtime.backends import spike_model

    output_bytes = None
    try:
        build = spike_model.build(
            directory,
            directory / "spike-build",
            inputs_npz=directory / "inputs.npz",
            arena_mb=arena_mb,
            dump_all_outputs=True,
        )
        run = spike_model.run(
            build["elf"],
            mem_bytes=build["mem_bytes"],
            timeout=timeout,
            vlen=build.get("vlen"),
        )
        output_bytes = run["output_bytes"]
        if len(output_bytes) != report["output_count"]:
            raise RuntimeError(f"hardware emitted {len(output_bytes)} outputs; bundle expects {report['output_count']}")
        (directory / "spike-console.txt").write_text(run["console"], encoding="utf-8")
        _write_json(directory / "spike-output-bytes.json", [item.hex() for item in output_bytes])
        execution = {
            "status": "ran",
            "elf": str(build["elf"]),
            "build_hash": build.get("build_hash"),
            "output_count": len(output_bytes),
        }
    except Exception as exc:  # each stage failure remains an explicit artifact
        execution = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    _write_json(directory / "spike-execution.json", execution)
    verdict = grade_core_aten_batch(
        report,
        output_bytes,
        execution_error=execution.get("error") if execution["status"] != "ran" else None,
    )
    _write_json(directory / "core_aten_batch_verdict.json", verdict)
    return execution, verdict


def _regrade_existing(output_dir: Path, report: dict) -> dict:
    """Reapply the current comparison policy to retained raw hardware output bytes."""
    aggregate = grade_core_aten_batch(report, execution_error="no recorded hardware execution")
    ledgers = [output_dir / "shard_executions.json"]
    ledgers += sorted(
        output_dir.glob("recovery*_executions.json"),
        key=lambda path: (
            0 if path.stem == "recovery_executions" else int(path.stem.removeprefix("recovery").split("_", 1)[0])
        ),
    )
    for ledger in ledgers:
        if not ledger.is_file():
            continue
        for item in json.loads(ledger.read_text(encoding="utf-8")):
            directory = Path(item["directory"])
            shard = json.loads((directory / "core_aten_batch_map.json").read_text(encoding="utf-8"))
            execution = item.get("execution", {})
            output_path = directory / "spike-output-bytes.json"
            output_bytes = (
                [bytes.fromhex(value) for value in json.loads(output_path.read_text(encoding="utf-8"))]
                if execution.get("status") == "ran" and output_path.is_file()
                else None
            )
            verdict = grade_core_aten_batch(
                shard,
                output_bytes,
                execution_error=execution.get("error") if output_bytes is None else None,
            )
            _write_json(directory / "core_aten_batch_verdict.json", verdict)
            for overload in item["overloads"]:
                aggregate["cases"][overload] = verdict["cases"][overload]
    counts: dict[str, int] = {}
    for case_verdict in aggregate["cases"].values():
        status = case_verdict["status"]
        counts[status] = counts.get(status, 0) + 1
    aggregate["status_counts"] = dict(sorted(counts.items()))
    aggregate["passed_count"] = counts.get("pass", 0)
    _write_json(output_dir / "core_aten_batch_verdict.json", aggregate)
    return aggregate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=None)
    parser.add_argument("--captures", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-spike", action="store_true", help="build one ELF and run every result through Spike")
    parser.add_argument("--arena-mb", type=int, default=256)
    parser.add_argument("--timeout", type=int, default=3600)
    parser.add_argument(
        "--max-cases-per-shard",
        type=int,
        default=0,
        help="partition clean captures into deterministic batches of at most N cases",
    )
    parser.add_argument(
        "--recover-failed-shards",
        action="store_true",
        help="split failed shards from an earlier run and merge their verdicts",
    )
    parser.add_argument(
        "--recovery-source",
        type=Path,
        default=None,
        help="execution ledger to split (defaults to shard_executions.json)",
    )
    parser.add_argument(
        "--recovery-label", default="recovery", help="subdirectory and ledger stem for this recovery round"
    )
    parser.add_argument(
        "--regrade-existing", action="store_true", help="reapply current comparison policy to retained raw output bytes"
    )
    args = parser.parse_args(argv)
    corpus_path = args.corpus or artifacts_dir() / "verification" / "core-aten" / "core_aten_test_cases.json"
    corpus = json.loads(corpus_path.read_text(encoding="utf-8"))
    report = build_core_aten_batch(corpus, args.captures, args.output_dir)
    print(
        f"Bundled {report['bundled_count']}/{report['case_count']} cases, "
        f"{report['input_count']} inputs, {report['output_count']} outputs"
    )
    print(args.output_dir / "core_aten_batch_map.json")
    if args.max_cases_per_shard < 0:
        parser.error("--max-cases-per-shard must be nonnegative")
    if args.regrade_existing:
        verdict = _regrade_existing(args.output_dir, report)
        print(verdict["status_counts"])
        return 0
    if args.recover_failed_shards:
        if not args.max_cases_per_shard or not args.run_spike:
            parser.error("recovery requires --run-spike and --max-cases-per-shard")
        source = args.recovery_source or args.output_dir / "shard_executions.json"
        prior = json.loads(source.read_text(encoding="utf-8"))
        aggregate = json.loads((args.output_dir / "core_aten_batch_verdict.json").read_text(encoding="utf-8"))
        recoveries = []
        for shard_record in prior:
            if shard_record.get("execution", {}).get("status") == "ran":
                continue
            names = shard_record["overloads"]
            for offset in range(0, len(names), args.max_cases_per_shard):
                selected = set(names[offset : offset + args.max_cases_per_shard])
                parent_name = Path(shard_record["directory"]).name
                directory = args.output_dir / args.recovery_label / f"{parent_name}-{offset:03d}"
                shard = build_core_aten_batch(corpus, args.captures, directory, selected_overloads=selected)
                execution, verdict = _run_spike_bundle(directory, shard, arena_mb=args.arena_mb, timeout=args.timeout)
                recovery = {"directory": str(directory), "overloads": sorted(selected), "execution": execution}
                recoveries.append(recovery)
                for overload in selected:
                    aggregate["cases"][overload] = verdict["cases"][overload]
                counts: dict[str, int] = {}
                for case_verdict in aggregate["cases"].values():
                    status = case_verdict["status"]
                    counts[status] = counts.get(status, 0) + 1
                aggregate["status_counts"] = dict(sorted(counts.items()))
                aggregate["passed_count"] = counts.get("pass", 0)
                _write_json(args.output_dir / "core_aten_batch_verdict.json", aggregate)
                _write_json(args.output_dir / f"{args.recovery_label}_executions.json", recoveries)
                print(f"recovery {directory.name}: {execution} {verdict['status_counts']}", flush=True)
        return 0 if all(item["execution"]["status"] == "ran" for item in recoveries) else 1
    if args.max_cases_per_shard:
        names = [item["overload"] for item in report["cases"] if item["status"] == "bundled"]
        aggregate = grade_core_aten_batch(report, execution_error="batch shard has not run")
        execution_records = []
        for offset in range(0, len(names), args.max_cases_per_shard):
            selected = set(names[offset : offset + args.max_cases_per_shard])
            shard_number = offset // args.max_cases_per_shard
            directory = args.output_dir / "shards" / f"shard-{shard_number:03d}"
            shard = build_core_aten_batch(corpus, args.captures, directory, selected_overloads=selected)
            shard_record = {
                "shard": shard_number,
                "directory": str(directory),
                "overloads": sorted(selected),
                "output_count": shard["output_count"],
            }
            if args.run_spike:
                execution, verdict = _run_spike_bundle(
                    directory,
                    shard,
                    arena_mb=args.arena_mb,
                    timeout=args.timeout,
                )
                shard_record["execution"] = execution
                for overload in selected:
                    aggregate["cases"][overload] = verdict["cases"][overload]
                counts: dict[str, int] = {}
                for case_verdict in aggregate["cases"].values():
                    status = case_verdict["status"]
                    counts[status] = counts.get(status, 0) + 1
                aggregate["status_counts"] = dict(sorted(counts.items()))
                aggregate["passed_count"] = counts.get("pass", 0)
                _write_json(args.output_dir / "core_aten_batch_verdict.json", aggregate)
                print(f"shard {shard_number}: {execution} {verdict['status_counts']}", flush=True)
            execution_records.append(shard_record)
            _write_json(args.output_dir / "shard_executions.json", execution_records)
        print(f"Wrote {len(execution_records)} deterministic batch shards", flush=True)
        return (
            0
            if all(item.get("execution", {}).get("status") == "ran" for item in execution_records)
            else (1 if args.run_spike else 0)
        )
    if args.run_spike:
        execution, verdict = _run_spike_bundle(args.output_dir, report, arena_mb=args.arena_mb, timeout=args.timeout)
        print(execution)
        print(verdict["status_counts"])
        return 0 if execution["status"] == "ran" else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
