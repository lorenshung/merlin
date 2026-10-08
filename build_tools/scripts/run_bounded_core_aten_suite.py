#!/usr/bin/env python3
"""Capture, execute, and grade every bounded case with resumable process workers."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from merlin.common import provenance
from merlin.targetgen.core_aten_bounded_runner import (
    SavedResponseAdapter,
    atomic_json,
    bounded_loader_source,
    bundle_admission_reason,
    case_digest,
    grade_portable_outputs,
    semantic_observation,
)
from merlin.targetgen.core_aten_capture import case_capture_name


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def stamp(args):
    artifacts = {"spike": Path(os.environ["MERLIN_SPIKE"])}
    if args.extension_library:
        artifacts["extension_library"] = args.extension_library
    sources = [Path(__file__), Path(__import__("merlin.targetgen.core_aten_bounded_runner", fromlist=["x"]).__file__)]
    if args.facts:
        sources.append(args.facts)
    pins = {name: provenance.verify(name) for name in args.pin}
    diff = subprocess.check_output(["git", "diff", "HEAD"], cwd=args.repo)
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=args.repo, text=True
    ).splitlines()
    for relative in sorted(untracked):
        candidate = args.repo / relative
        if candidate.is_file():
            diff += subprocess.run(
                ["git", "diff", "--no-index", "/dev/null", str(candidate)], capture_output=True
            ).stdout
    return provenance.record(
        pins=pins,
        sources=sources,
        artifacts=artifacts,
        extra={
            "merlin_commit": git(args.repo, "rev-parse", "HEAD"),
            "merlin_diff_sha256": hashlib.sha256(diff).hexdigest(),
            "model2mlir_commit": git(args.m2m_dir, "rev-parse", "HEAD"),
            "device_package_commit": git(args.device_package, "rev-parse", "HEAD") if args.device_package else None,
            "target": args.target,
            "execution_kind": "functional_simulator",
            "suite_sha256": hashlib.sha256(args.suite.read_bytes()).hexdigest(),
        },
    )


def capture(args, case, block):
    from merlin.targetgen.capsule_source import PytorchRefSource
    from merlin.targetgen.core_aten_cover import _provenance_ops

    directory = args.output / "captures" / case_capture_name(case)
    directory.mkdir(parents=True, exist_ok=True)
    loader = directory / "capsule.pytorch.py"
    loader.write_text(bounded_loader_source(case))
    source = PytorchRefSource(m2m_dir=args.m2m_dir, python=args.m2m_python, timeout=args.capture_timeout)
    os.environ["PYTHONPATH"] = str(args.m2m_dir)
    try:
        artifact = source.capture_loader(loader, "fp32", workdir=directory / "capture-work")
        mlir = directory / "capsule.linalg.mlir"
        mlir.write_text(artifact.linalg_mlir)
        actual = sorted(_provenance_ops(mlir))
        record = {
            "status": "captured_exact" if case["overload"] in actual else "captured_lost_exact_provenance",
            "capture_meta": artifact.meta,
            "captured_overloads": actual,
        }
        atomic_json(directory / "inputs.json", artifact.inputs)
        atomic_json(directory / "golden.json", artifact.golden)
    except Exception as exc:
        (directory / "capsule.linalg.mlir").unlink(missing_ok=True)
        record = {"status": "capture_failed", "reason": f"{type(exc).__name__}: {exc}"}
    record.update(case_id=case["case_id"], overload=case["overload"], case_sha256=case_digest(case), provenance=block)
    atomic_json(directory / "capture.json", record)


def execute(args, case, block):
    from merlin.targetgen.core_aten_batch import build_core_aten_batch
    from merlin.targetgen.core_aten_device import run_spike_bundle

    directory = args.output / args.label / "shards" / case_capture_name(case)
    directory.mkdir(parents=True, exist_ok=True)
    captured = json.loads((args.output / "captures" / case_capture_name(case) / "capture.json").read_text())
    started = time.time()
    if captured["status"] == "capture_failed":
        observation = {
            "status": "capture_unavailable",
            "reason": captured["reason"],
            "evidence": {"lane": "unassigned", "capture": captured["status"]},
        }
    else:
        try:
            refusal = bundle_admission_reason(
                captured, (args.output / "captures" / case_capture_name(case) / "capsule.linalg.mlir").read_text()
            )
            if refusal:
                raise ValueError(refusal)
            batch = build_core_aten_batch({"cases": [case]}, args.output / "captures", directory)
            execution, grades = run_spike_bundle(
                directory,
                batch,
                arena_mb=args.arena_mb,
                timeout=args.execution_timeout,
                target=args.target,
                device_package=args.device_package,
                rtl_facts=args.facts,
            )
            grade = grades["cases"][case["case_id"]]
            original_grade = grade
            if execution["status"] == "ran" and grade["status"] == "ungradable":
                outputs = [
                    bytes.fromhex(item) for item in json.loads((directory / "spike-output-bytes.json").read_text())
                ]
                grade = grade_portable_outputs(batch, outputs)["cases"][case["case_id"]]
            if execution["status"] == "failed" and not (directory / "spike-build" / "model.elf").exists():
                grade["status"] = "compile_lowering_failed"
            evidence = {
                "lane": grade.get("lane", "host"),
                "numeric_verdict": grade,
                "execution": execution,
                "directory": str(directory),
            }
            if original_grade != grade:
                evidence.update(
                    original_numeric_verdict=original_grade,
                    oracle_decoding="portable nonfinite leaves decoded; comparator and policy unchanged",
                )
            observation = semantic_observation(grade, evidence)
        except Exception as exc:
            observation = {
                "status": "bundle_unavailable",
                "reason": f"{type(exc).__name__}: {exc}",
                "evidence": {"lane": "unassigned"},
            }
    # Existing runner writes verdict-bearing documents; stamp them before completion.
    for filename in ("core_aten_batch_map.json", "core_aten_batch_verdict.json", "spike-execution.json"):
        path = directory / filename
        if path.exists():
            doc = json.loads(path.read_text())
            doc["provenance"] = block
            atomic_json(path, doc)
    atomic_json(
        directory / "result.json",
        {
            "case_id": case["case_id"],
            "case_sha256": case_digest(case),
            "observation": observation,
            "seconds": time.time() - started,
            "provenance": block,
        },
    )


def launch(args, phase, case, block):
    key = case_capture_name(case)
    directory = args.output / "captures" / key if phase == "capture" else args.output / args.label / "shards" / key
    completion = directory / ("capture.json" if phase == "capture" else "result.json")
    if completion.exists() and json.loads(completion.read_text()).get("case_sha256") == case_digest(case):
        return "cached"
    directory.mkdir(parents=True, exist_ok=True)
    request = directory / "request.json"
    atomic_json(request, case)
    command = [sys.executable, str(Path(__file__).resolve()), *sys.argv[1:], "--worker", phase, "--case", str(request)]
    with (directory / "worker.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=args.repo, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            code = proc.wait(timeout=(args.capture_timeout + 120) if phase == "capture" else args.worker_timeout)
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait()
            code = "timeout"
    if code != 0 or not completion.exists():
        reason = f"{phase} worker {code}; " + (directory / "worker.log").read_text(errors="replace")[-3000:]
        record = {"case_id": case["case_id"], "case_sha256": case_digest(case), "provenance": block}
        if phase == "capture":
            record.update(status="capture_failed", reason=reason)
        else:
            record["observation"] = {"status": "execution_failed", "reason": reason, "evidence": {"lane": "unassigned"}}
        atomic_json(completion, record)
    return "finished"


def response(args, corpus, block):
    results = {}
    for case in corpus["selected_cases"]:
        path = args.output / args.label / "shards" / case_capture_name(case) / "result.json"
        results[case["case_id"]] = (
            json.loads(path.read_text())["observation"]
            if path.exists()
            else {"status": "not_run", "reason": "execution pending", "evidence": {"lane": "unassigned"}}
        )
    document = {"schema_version": 1, "results": results, "provenance": block}
    atomic_json(args.output / args.label / "response.json", document)
    return document


def refresh_portable_grades(args):
    """Upgrade cached byte grades using portable decoding, without re-execution."""
    block = provenance.record(
        sources=[Path(__import__("merlin.targetgen.core_aten_bounded_runner", fromlist=["x"]).__file__)],
        extra={"purpose": "portable oracle decoding; native comparator unchanged"},
    )
    changed = Counter()
    for path in (args.output / args.label / "shards").glob("*/result.json"):
        record = json.loads(path.read_text())
        if record["observation"]["status"] == "bundle_unavailable":
            captured = json.loads((args.output / "captures" / path.parent.name / "capture.json").read_text())
            reason = bundle_admission_reason(
                captured, (args.output / "captures" / path.parent.name / "capsule.linalg.mlir").read_text()
            )
            if reason and reason != record["observation"]["reason"]:
                record["pre_admission_diagnostic_observation"] = record["observation"]
                record["observation"] = {**record["observation"], "reason": reason}
                record["provenance"]["admission_diagnostic"] = block
                atomic_json(path, record)
        evidence = record["observation"].get("evidence", {})
        old = evidence.get("numeric_verdict", {})
        if old.get("status") != "ungradable" or evidence.get("execution", {}).get("status") != "ran":
            continue
        batch = json.loads(path.with_name("core_aten_batch_map.json").read_text())
        outputs = [bytes.fromhex(item) for item in json.loads(path.with_name("spike-output-bytes.json").read_text())]
        new = grade_portable_outputs(batch, outputs)["cases"][record["case_id"]]
        if new == old:
            continue
        record["pre_oracle_decoding_observation"] = record["observation"]
        evidence = {
            **evidence,
            "original_numeric_verdict": old,
            "numeric_verdict": new,
            "oracle_decoding": "portable nonfinite leaves decoded; comparator and policy unchanged",
        }
        record["observation"] = semantic_observation(new, evidence)
        record["provenance"]["oracle_decoding"] = block
        atomic_json(path, record)
        changed[new["status"]] += 1
    if changed:
        atomic_json(
            args.output / args.label / "oracle-decoding.json",
            {"case_count": sum(changed.values()), "numeric_status_counts": changed, "provenance": block},
        )
    return dict(changed)


def summarize(args, corpus, report, block, timings):
    by_overload = defaultdict(Counter)
    partitions = {axis: defaultdict(Counter) for axis in ("dtype", "layout", "values")}
    lanes = defaultdict(Counter)
    families = defaultdict(list)
    numeric = Counter()
    numeric_overloads = defaultdict(Counter)
    numeric_lanes = defaultdict(Counter)
    for case in corpus["selected_cases"]:
        item = report["cases"][case["case_id"]]
        status = item["status"]
        evidence = item.get("evidence", {})
        by_overload[case["overload"]][status] += 1
        lanes[evidence.get("lane", "unassigned")][status] += 1
        grade = evidence.get("numeric_verdict", {})
        numeric[grade.get("status", "unavailable")] += 1
        numeric_overloads[case["overload"]][grade.get("status", "unavailable")] += 1
        numeric_lanes[evidence.get("lane", "unassigned")][grade.get("status", "unavailable")] += 1
        for axis, buckets in partitions.items():
            buckets[case["partition_assignment"].get(axis, "unspecified")][status] += 1
        if status != "pass":
            reason = item.get("reason", "")
            # Group diagnostics structurally at their first line, keep complete examples.
            diagnostic = reason.splitlines()[-1] if reason.splitlines() else "no diagnostic"
            if status == "compile_lowering_failed":
                if "undefined reference" in reason:
                    diagnostic = "linker undefined reference"
                elif "returncode=-11" in reason:
                    diagnostic = "upstream lowering segmentation fault"
                else:
                    diagnostic = reason.splitlines()[0].split(":", 1)[0]
            family = status + ": " + diagnostic[:160]
            families[family].append({"case_id": case["case_id"], "reason": reason})
    summary = {
        "case_count": report["case_count"],
        "campaign_complete": report.get("campaign_complete", False),
        "status_counts": report["status_counts"],
        "by_lane": dict(lanes),
        "numeric_status_counts": numeric,
        "per_overload": dict(by_overload),
        "partitions": partitions,
        "numeric_per_overload": dict(numeric_overloads),
        "numeric_by_lane": dict(numeric_lanes),
        "failure_families": [
            {"family": k, "count": len(v), "examples": v[:2]}
            for k, v in sorted(families.items(), key=lambda pair: -len(pair[1]))
        ],
        "timings": timings,
        "provenance": block,
    }
    atomic_json(args.output / args.label / "summary.json", summary)
    lines = [
        f"# Bounded Core ATen: {args.label}",
        "",
        f"Cases: {report['case_count']}",
        "",
        f"Statuses: {report['status_counts']}",
        "",
        f"Numeric evidence: {dict(numeric)}",
        "",
        "Numeric successes are ungradable for the complete mutation/alias/metadata contract.",
        "",
        f"Timings (seconds): {timings}",
        "",
        f"Lanes: {dict(lanes)}",
        "",
        f"Numeric results by lane: {dict(numeric_lanes)}",
        "",
        "| Overload | Semantic pass | Numeric match | Total |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines.extend(
        f"| {name} | {counts.get('pass', 0)} | {numeric_overloads[name].get('pass', 0)} | {sum(counts.values())} |"
        for name, counts in sorted(by_overload.items())
    )
    for title, buckets in [("Lane", lanes), *[(axis, buckets) for axis, buckets in partitions.items()]]:
        statuses = sorted(report["status_counts"])
        lines += [
            "",
            f"{title} breakdown:",
            "",
            "| " + title + " | " + " | ".join(statuses) + " |",
            "| --- | " + " | ".join("---:" for _ in statuses) + " |",
        ]
        lines.extend(
            "| " + str(name) + " | " + " | ".join(str(counts.get(status, 0)) for status in statuses) + " |"
            for name, counts in sorted(buckets.items())
        )
    lines += ["", "Failure families:"]
    for item in summary["failure_families"]:
        lines += ["", f"- {item['count']}: {item['family']}"]
        for example in item["examples"]:
            lines += ["", f"  `{example['case_id']}`: {example['reason'].replace(chr(10), ' ')[:1500]}"]
    lines += ["", "Provenance:", "```json", json.dumps(block, indent=2), "```"]
    (args.output / args.label / "summary.md").write_text("\n".join(lines) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("suite", "output", "repo", "m2m-dir", "m2m-python"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--target")
    for name in ("device-package", "facts", "extension-library"):
        p.add_argument("--" + name, type=Path)
    p.add_argument("--pin", action="append", default=[])
    p.add_argument("--workers", type=int, default=48)
    p.add_argument("--capture-timeout", type=int, default=300)
    p.add_argument("--execution-timeout", type=int, default=120)
    p.add_argument("--worker-timeout", type=int, default=900)
    p.add_argument("--arena-mb", type=int, default=256)
    p.add_argument("--limit", type=int)
    p.add_argument("--worker", choices=["capture", "execute"])
    p.add_argument("--case", type=Path)
    p.add_argument("--evaluate-only", action="store_true")
    args = p.parse_args()
    if not 1 <= args.workers <= 64:
        p.error("workers must be between 1 and 64")
    block_path = args.output / args.label / "provenance.json"
    if args.worker:
        block = json.loads(block_path.read_text())
        block["worker_adapter_sources"] = provenance.record(
            sources=[
                Path(__file__),
                Path(__import__("merlin.targetgen.core_aten_bounded_runner", fromlist=["x"]).__file__),
            ]
        )
        case = json.loads(args.case.read_text())
        (capture if args.worker == "capture" else execute)(args, case, block)
        return
    args.output.mkdir(parents=True, exist_ok=True)
    block = stamp(args)
    atomic_json(block_path, block)
    corpus = json.loads(args.suite.read_text())
    cases = corpus["selected_cases"]
    if args.limit:
        # Spread smoke selection across overloads, layouts, and dtype partitions.
        cases = [cases[int(i * len(cases) / args.limit)] for i in range(args.limit)]
    timings = {}
    start = time.time()
    if not args.evaluate_only:
        check = subprocess.check_output(
            [str(args.m2m_python), "-c", "import m2m; print(m2m.__file__)"],
            env={**os.environ, "PYTHONPATH": str(args.m2m_dir)},
            text=True,
        )
        if not Path(check.strip()).is_relative_to(args.m2m_dir):
            raise RuntimeError(f"wrong m2m import: {check}")
        print("m2m verified", check.strip(), flush=True)
        for phase in ("capture", "execute"):
            began = time.time()
            with ThreadPoolExecutor(max_workers=args.workers) as pool:
                futures = [pool.submit(launch, args, phase, case, block) for case in cases]
                for count, future in enumerate(as_completed(futures), 1):
                    future.result()
                    if count % 50 == 0 or count == len(cases):
                        print(phase, count, "/", len(cases), "seconds", round(time.time() - began, 1), flush=True)
                        if phase == "execute":
                            response(args, corpus, block)
            timings[phase] = time.time() - began
    began = time.time()
    upgraded = refresh_portable_grades(args)
    if upgraded:
        print("portable oracle grades refreshed", upgraded, flush=True)
    if not args.evaluate_only:
        timings["oracle_decoding"] = time.time() - began
    document = response(args, corpus, block)
    if not args.evaluate_only:
        atomic_json(args.output / args.label / "timings.json", timings)
    if args.limit:
        counts = Counter(document["results"][c["case_id"]]["status"] for c in cases)
        atomic_json(
            args.output / args.label / "smoke.json",
            {
                "status_counts": counts,
                "case_ids": [c["case_id"] for c in cases],
                "timings": timings,
                "provenance": block,
            },
        )
        print("smoke", dict(counts), flush=True)
        return
    # The evaluator requires Torch for contract validation; keep capture import pinned.
    if "torch" not in sys.modules:
        command = [str(args.m2m_python), str(Path(__file__).resolve()), *sys.argv[1:], "--evaluate-only"]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(args.m2m_dir), str(args.repo / "src"), *[str(x) for x in (args.repo / "packages").glob("*/src")]]
        )
        subprocess.run(command, cwd=args.repo, env=env, check=True)
        return
    from merlin.targetgen.core_aten_eval import evaluate_bounded_core_aten_suite

    report = evaluate_bounded_core_aten_suite(
        SavedResponseAdapter(document),
        corpus,
        required_execution_kind="simulator",
        validate_live_denominator=False,
        validate_eager_oracles=False,
        validate_solver_certificate=False,
    )
    report["provenance"] = block
    report["campaign_complete"] = len(document["results"]) == len(corpus["selected_cases"]) and not any(
        item["status"] in {"not_run", "missing_result", "adapter_error"} for item in report["cases"].values()
    )
    report["validation_policy"] = (
        "sealed suite, case witnesses/schema/oracle digests checked; live denominator, eager reruns, solver reruns disabled"
    )
    atomic_json(args.output / args.label / "report.json", report)
    timings = json.loads((args.output / args.label / "timings.json").read_text())
    timings["evaluation"] = time.time() - start
    timings["total"] = sum(timings.values())
    summarize(args, corpus, report, block, timings)
    print(report["status_counts"], flush=True)


if __name__ == "__main__":
    # Torch is intentionally absent in the Merlin controller interpreter.
    if "--evaluate-only" in sys.argv:
        import torch
    main()
