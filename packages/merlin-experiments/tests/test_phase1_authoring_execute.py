"""Actual installed continuation with explicit synthetic already-admitted inputs.

Only admission verification and provider execution are substituted. This does not
qualify admission, source proof, brokers, OS isolation or hardware. Real background
cadence, no-submission grading, quota/restore, checkpoint, audit and completion
decisions run. The provider is a local stdout child, never a network client.
This file can be copied beside independently installed wheels outside a checkout.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import yaml


def _exercise(root: Path, resume: bool, coverage: bool = False, cost_cap: bool = False) -> int:
    import importlib.abc
    import socket
    from unittest.mock import patch

    class NoNative(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            if fullname in {"_common", "run_baseline_qa_loop", "run_agent_experiment", "chia", "ray"}:
                raise AssertionError(f"native or scheduler import: {fullname}")

    sys.meta_path.insert(0, NoNative())
    from merlin_experiments.phase1 import authoring, treatments
    from merlin_experiments.phase1.context import InvocationContext
    from merlin_experiments.phase1.options import parse_options
    from merlin_experiments.phase1.providers import execution
    from merlin_experiments.phase1.session import PreparedRun, RunRequest, WorkspaceTransport

    from merlin.common import arrival_stamp

    def no_network(*args, **kwargs):
        raise AssertionError("continuation fixture attempted network activity")

    def unused(*args, **kwargs):
        raise AssertionError("already-admitted continuation called an admission adapter")

    context = InvocationContext(
        root,
        root / "target_experiment.yaml",
        root,
        "fixture",
        root / "runs",
        root / "reports",
        root / "bundle",
        (),
    )
    options = parse_options(
        [
            "--run-id",
            "synthetic",
            "--arm",
            "raw_baseline",
            "--model",
            "fixture",
            "--driver",
            "codex",
            "--sandbox",
            "none",
            "--allow-unsandboxed",
            "--schedule",
            "rounds",
            "--max-rounds",
            "1",
            *(["--resume"] if resume else []),
            *(
                [
                    "--objective",
                    "device-coverage",
                    "--experiment",
                    "realistic",
                    "--schedule",
                    "continuous",
                    "--max-wall-s",
                    "1",
                    "--plateau-rounds",
                    "0",
                ]
                if coverage
                else []
            ),
        ]
    )
    request = RunRequest(
        context,
        options,
        treatments.Treatment(),
        root / "bundle/input_bundle_manifest.yaml",
        (),
        Path(__file__),
        False,
        {},
        lambda: (),
    )
    prepared = PreparedRun(
        request,
        root / "run",
        root / "workspace",
        root / "bundle",
        {"bundle_id": "synthetic"},
        {},
        None,
        None,
        None,
        None,
        resume,
        None,
        WorkspaceTransport(unused, unused),
        unused,
    )
    checks = []

    def verify(self):
        assert self is prepared
        checks.append("admitted-input seam observed")

    def launch(ws, run, model, effort, sandbox, bundle, rnd, timeout, **kwargs):
        assert ws == prepared.workspace and (coverage or rnd == 0)
        transcript = run / "rounds" / f"round_{rnd:02d}.transcript.jsonl"
        transcript.parent.mkdir(exist_ok=True)
        if not resume and not coverage:
            events = [
                {
                    "type": "rate_limit_event",
                    "rate_limit_info": {
                        "status": "rejected",
                        "rateLimitType": "seven_day",
                    },
                }
            ]
        else:
            events = [
                {
                    "type": "assistant",
                    "message": {
                        "model": "fixture",
                        "content": [
                            {
                                "type": "tool_use",
                                "id": "notes",
                                "name": "Read",
                                "input": {"file_path": "docs/notes.md"},
                            },
                        ],
                    },
                },
                {
                    "type": "user",
                    "message": {
                        "content": [
                            {"type": "tool_result", "tool_use_id": "notes", "content": "No compiler produced."},
                        ]
                    },
                },
                {
                    "type": "result",
                    "num_turns": 1,
                    "duration_ms": 1,
                    "duration_api_ms": 1,
                    "usage": {"input_tokens": 2, "output_tokens": 2},
                },
            ]
        program = "import json\n" + "\n".join(f"print({json.dumps(json.dumps(event))}, flush=True)" for event in events)
        rc = arrival_stamp.stream_stamped(
            [sys.executable, "-c", program],
            cwd=ws,
            transcript=transcript,
            stderr_path=transcript.with_suffix(".stderr.log"),
            timeout=10,
            raw_path=transcript.with_suffix(".raw.jsonl"),
        )
        if coverage:
            import time

            time.sleep(1.1 if cost_cap else 0.3)
            (ws / "submission").mkdir(exist_ok=True)
            (ws / "submission" / authoring.READY_MARKER).write_text("ready")
        return rc, transcript

    with (
        patch.object(socket, "socket", no_network),
        patch.object(PreparedRun, "verify_inputs", verify),
        patch.object(execution, "launch", launch),
        patch.object(authoring, "_spend_over_cap", lambda *a: (True, 2, 1)) if cost_cap else patch.dict({}, {}),
        patch.object(
            authoring.LG, "grade", lambda *a, **k: {"all_pass": True, "n_passed": 1, "n_capsules": 1, "per_capsule": []}
        )
        if coverage
        else patch.dict({}, {}),
        patch.object(
            authoring.LG,
            "fast_grade",
            lambda *a, **k: {"all_pass": True, "n_passed": 1, "n_capsules": 1, "per_capsule": []},
        )
        if coverage
        else patch.dict({}, {}),
    ):
        result = authoring.execute(prepared, authoring.AuthoringRuntime("synthetic", root / "timing.json"))
    assert checks, "the engine skipped admitted-input verification"
    (root / ("resumed_checks.json" if resume else "fresh_checks.json")).write_text(json.dumps(checks))
    return result


def test_quota_checkpoint_then_fresh_process_restore_and_honest_incompletion(tmp_path):
    root = tmp_path / "operator"
    for relative in ("run", "workspace", "bundle"):
        (root / relative).mkdir(parents=True)
    (root / "target_experiment.yaml").write_text(
        yaml.safe_dump(
            {
                "target": "fixture",
                "capsule_corpus": str(root / "absent-public"),
            }
        )
    )
    (root / "bundle/input_bundle_manifest.yaml").write_text("bundle_id: synthetic\n")
    env = {**os.environ, "MERLIN_CAPSULE_L3_CHECKPOINT": "0", "MERLIN_AET_SINK": "0"}
    first = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(root)],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert first.returncode == 42, first.stdout + first.stderr
    checkpoint = root / "run/qa_loop_state.yaml"
    before = yaml.safe_load(checkpoint.read_text())
    assert before["next_round"] == 0 and not before["converged"]
    second = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(root), "resume"],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert second.returncode == 1, second.stdout + second.stderr
    after = yaml.safe_load(checkpoint.read_text())
    assert after["next_round"] == 1, second.stdout + second.stderr
    assert after["cumulative"]["started_at"] == before["cumulative"]["started_at"]
    summary = yaml.safe_load((root / "run/qa_loop_summary.yaml").read_text())
    assert summary["n_rounds"] == 1 and summary["formal_complete"] is False
    assert summary["numeric_all_pass"] is False
    assert "restored" in second.stdout.lower()
    assert json.loads((root / "fresh_checks.json").read_text())
    assert json.loads((root / "resumed_checks.json").read_text())


def test_actual_coverage_loop_keeps_correctness_true_and_ignores_ready_until_budget(tmp_path):
    root = tmp_path / "coverage"
    for relative in ("run", "workspace", "bundle"):
        (root / relative).mkdir(parents=True)
    (root / "target_experiment.yaml").write_text(
        yaml.safe_dump({"target": "fixture", "capsule_corpus": str(root / "absent-public")})
    )
    (root / "bundle/input_bundle_manifest.yaml").write_text("bundle_id: synthetic\n")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(root), "coverage"],
        cwd=root,
        env={**os.environ, "MERLIN_CAPSULE_L3_CHECKPOINT": "0", "MERLIN_AET_SINK": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    summary = yaml.safe_load((root / "run/qa_loop_summary.yaml").read_text())
    assert summary["n_rounds"] >= 2 and summary["numeric_all_pass"] is True
    assert summary["converged"] is False and summary["formal_complete"] is False
    assert all(row["all_pass"] and not row["authoring_complete"] for row in summary["rounds"])
    assert summary["objective"]["termination"] == "max_wall_s" and summary["objective"]["budget_exhausted"]
    assert summary["objective"]["authoring_stop"]["active_wall_s"] >= 1
    assert "correctness/READY does not end" in result.stdout


def test_actual_cost_stop_is_not_mislabeled_when_wall_budget_also_expires(tmp_path):
    root = tmp_path / "cost-stop"
    for relative in ("run", "workspace", "bundle"):
        (root / relative).mkdir(parents=True)
    (root / "target_experiment.yaml").write_text(
        yaml.safe_dump({"target": "fixture", "capsule_corpus": str(root / "absent-public")})
    )
    (root / "bundle/input_bundle_manifest.yaml").write_text("bundle_id: synthetic\n")
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(root), "coverage-cost"],
        cwd=root,
        env={**os.environ, "MERLIN_CAPSULE_L3_CHECKPOINT": "0", "MERLIN_AET_SINK": "0"},
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 1, result.stdout + result.stderr
    summary = yaml.safe_load((root / "run/qa_loop_summary.yaml").read_text())
    assert summary["cost_capped"] is True and summary["n_rounds"] == 1
    assert summary["objective"]["termination"] == "cost_cap"
    assert summary["objective"]["budget_exhausted"] is True
    assert (
        summary["objective"]["final_active_wall_s"] >= summary["objective"]["authoring_stop"]["active_wall_s"] - 0.001
    )


if __name__ == "__main__":
    raise SystemExit(
        _exercise(
            Path(sys.argv[1]),
            len(sys.argv) > 2 and sys.argv[2] == "resume",
            len(sys.argv) > 2 and sys.argv[2] in {"coverage", "coverage-cost"},
            len(sys.argv) > 2 and sys.argv[2] == "coverage-cost",
        )
    )
