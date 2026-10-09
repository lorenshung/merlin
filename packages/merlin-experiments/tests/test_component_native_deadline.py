"""Actual ordinary private package processes obey a single selected deadline.

These deliberate evaluator-only wall-budget defects establish transport refusal
only. They do not seed a compiler or issue source/runtime/physical authority.
"""

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest
from merlin_experiments.phase2.component_runtime_control_execution import PrivateRuntimeControlExecutor
from merlin_experiments.phase2.contracts import exact_tree_record
from test_component_runtime_support import prepared  # noqa: F401 -- independent private control fixture

from merlin.common import invocation_record
from merlin.common.execution_deadline import ExecutionDeadline
from merlin.targetgen import native_component_execution as native
from merlin.targetgen import package_runtime as P
from merlin.targetgen.contract.readback_policy import FULL_VALUES_B64, ReadbackPolicy


def _arguments(context, fixture, output):
    return dict(
        package_dir=fixture.grade_arguments["package_dir"],
        capsule_dir=fixture.capsule_root,
        contract_root=context.contract_root,
        target=context.build_service.target,
        out_dir=output,
        build_service=context.build_service,
        execution_service=context.execution_service,
        source_verifier=context._source_verifier,
        readback_policy=ReadbackPolicy(FULL_VALUES_B64),
        timeout_s=1,
    )


def _records(work):
    return [json.loads(path.read_bytes()) for path in work.rglob("invocation.json")]


class BlockedBuild(PrivateRuntimeControlExecutor):
    def build_package(self, package, *, timeout=1800):
        super().build_package(package, timeout=timeout)
        invocation_record.run(
            [sys.executable, "-I", "-B", "-c", "import time; print('BUILD_START', flush=True); time.sleep(5)"],
            directory=self.evidence_root,
            stage="blocked_package_build_control",
            dependencies=(Path(__file__),),
            capture_output=True,
            text=True,
            timeout=timeout,
        )


def test_actual_package_build_uses_declared_budget_and_never_lowers_or_executes(prepared, tmp_path):  # noqa: F811
    fixture = prepared.prepare_control("source_correspondence.positive", tmp_path / "control")
    candidate = fixture.grade_arguments["package_dir"]
    output = tmp_path / "native"
    executor = BlockedBuild(candidate, tmp_path / "actual-build", exact_tree_record(candidate)["sha256"])
    start = time.monotonic()
    with P.scoped_package_executor(executor), pytest.raises(subprocess.TimeoutExpired):
        native.execute_component(**_arguments(prepared, fixture, output))
    assert time.monotonic() - start < 2.5
    rows = _records(executor.evidence_root)
    assert len(rows) == 1 and rows[0]["stage"] == "blocked_package_build_control"
    assert rows[0]["status"] == "interrupted"
    assert not (output / "generated").exists() and not (output / "build").exists()
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "unavailable" and report["failure"]["type"] == "TimeoutExpired"


def test_actual_normal_entrypoints_spend_one_budget_instead_of_resetting_each_stage(prepared, tmp_path):  # noqa: F811
    fixture = prepared.prepare_control("source_correspondence.positive", tmp_path / "control")
    candidate = fixture.grade_arguments["package_dir"]
    driver = candidate / "driver.py"
    driver.write_text(
        driver.read_text().replace(
            "from __future__ import annotations", "from __future__ import annotations\nimport time\ntime.sleep(0.4)", 1
        )
    )
    output = tmp_path / "native"
    executor = PrivateRuntimeControlExecutor(candidate, output, exact_tree_record(candidate)["sha256"])
    start = time.monotonic()
    with P.scoped_package_executor(executor), pytest.raises(subprocess.TimeoutExpired):
        native.execute_component(**_arguments(prepared, fixture, output))
    assert time.monotonic() - start < 2.5
    rows = _records(output)
    children = [row for row in rows if row["kind"] == "subprocess"]
    assert any(row["stage"] == "parse" and row["status"] == "completed" for row in children)
    assert any(row["status"] == "interrupted" for row in children)
    assert not {"component_native_execution", "elf", "execution"} & {row["stage"] for row in rows}
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "unavailable" and report["failure"]["type"] == "TimeoutExpired"


def test_source_check_boundary_refuses_expired_work_before_object_build(prepared, tmp_path):  # noqa: F811
    fixture = prepared.prepare_control("source_correspondence.positive", tmp_path / "control")
    candidate = fixture.grade_arguments["package_dir"]
    output = tmp_path / "native"
    executor = PrivateRuntimeControlExecutor(candidate, output, exact_tree_record(candidate)["sha256"])
    arguments = _arguments(prepared, fixture, output)

    def delayed_source_check(**kwargs):
        result = prepared._source_verifier(**kwargs)
        # A synchronous Python producer is checked at its completed boundary,
        # not preempted or mislabeled as a subprocess timer capability.
        time.sleep(3.1)
        return result

    arguments["timeout_s"] = 3
    arguments["source_verifier"] = delayed_source_check
    with P.scoped_package_executor(executor), pytest.raises(TimeoutError):
        native.execute_component(**arguments)
    assert not (output / "build").exists()
    rows = _records(output)
    assert not {"component_native_execution", "object", "elf", "execution"} & {row["stage"] for row in rows}
    report = json.loads((output / "result.json").read_text())
    assert report["status"] == "unavailable" and report["failure"]["type"] == "TimeoutError"


def test_late_diagnostic_publication_retracts_completed_status(tmp_path, monkeypatch):
    # This synthetic status is only a publication negative control. It issues
    # no numerical result, compiled artifact, compiler or runtime authority.
    record = {"status": "numeric_match_diagnostic"}
    original_write = native._write
    writes = []

    def delayed_write(path, value):
        writes.append(value["status"])
        if len(writes) == 1:
            subprocess.run([sys.executable, "-I", "-B", "-c", "import time; time.sleep(0.15)"], check=True)
        original_write(path, value)

    monkeypatch.setattr(native, "_write", delayed_write)
    with pytest.raises(TimeoutError):
        native._publish_result(tmp_path, record, ExecutionDeadline.start(0.05))
    actual = json.loads((tmp_path / "result.json").read_text())
    assert actual["status"] == "unavailable" and actual["failure"]["type"] == "TimeoutError"
    assert writes == ["numeric_match_diagnostic", "unavailable"]
