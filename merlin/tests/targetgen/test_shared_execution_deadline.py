"""Actual blocked ordinary tool processes consume one explicit wall budget.

These are evaluator-only process controls. The retained linked artifact and
timer refusals give no source, instruction, runtime or physical authority.
"""

import importlib.util
import json
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest

from merlin.common import invocation_record
from merlin.common.execution_deadline import ExecutionDeadline, selected_deadline
from merlin.llvmlower.codegen import CodegenError
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract.build_service import file_digest
from merlin.targetgen.contract.compile_only import prepare_linkage
from merlin.targetgen.contract.execution_service import FunctionalExecutionService

# Load exactly the archived sibling fixture under pytest's isolated importlib
# mode. It is test data, never a compiler/runtime import or qualification issuer.
_FIXTURE_SPEC = importlib.util.spec_from_file_location(
    "selected_stock_build_fixture", Path(__file__).with_name("test_compile_only_transport.py")
)
_FIXTURE = importlib.util.module_from_spec(_FIXTURE_SPEC)
_FIXTURE_SPEC.loader.exec_module(_FIXTURE)
_LLVM = _FIXTURE._LLVM
_buffer = _FIXTURE._buffer
_original_abi = _FIXTURE._original_abi
actual_transport = _FIXTURE.actual_transport


def _blocked_runner(elf, *, simulator, timeout, **kwargs):
    # The deliberate blocked producer consumes the actual linked ELF pin and
    # exits by timeout. It never produces simulated values or completion.
    invocation_record.run(
        [sys.executable, "-I", "-B", "-c", "import time; print('RUN_START', flush=True); time.sleep(5)"],
        directory=Path(elf).parent,
        stage="blocked_simulator_control",
        inputs=(Path(elf),),
        dependencies=(Path(__file__),),
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    raise AssertionError("blocked simulator completed")


def _unused_parser(_console):
    raise AssertionError("expired ordinary producer reached output decoding")


def _execution(build):
    owner = Path(__file__).resolve()
    return FunctionalExecutionService(
        build.target,
        "wall_control",
        _blocked_runner,
        _unused_parser,
        ((str(owner), file_digest(owner)),),
        '{"scope":"deliberately blocked process control; no simulated values or runtime authority"}',
    )


def _records(work):
    # Failed process records are useful diagnostics, never successful
    # invocation authority. Read them rather than calling success verification.
    return [json.loads(path.read_bytes()) for path in work.rglob("invocation.json")]


@pytest.mark.parametrize("budget", [None, True, 0, -1, float("nan"), float("inf"), 10**400])
def test_invalid_or_absent_wall_budget_cannot_create_a_deadline(budget):
    with pytest.raises((ValueError, TypeError)):
        ExecutionDeadline.start(budget)


def test_shared_parent_never_resets_and_second_or_unselected_build_budget_refuses():
    parent = ExecutionDeadline.start(0.05)
    with pytest.raises(ValueError):
        selected_deadline(seconds=1, parent=parent, build_service=object())
    with pytest.raises(ValueError):
        selected_deadline(seconds=None, parent=parent, build_service=None)
    time.sleep(0.08)
    with pytest.raises(TimeoutError):
        selected_deadline(seconds=None, parent=parent, build_service=object())
    assert selected_deadline(seconds=None, parent=None, build_service=None) is None


def test_actual_object_producer_is_killed_and_link_and_simulator_never_start(actual_transport, monkeypatch):  # noqa: F811
    _args, build, _gate = actual_transport
    actual_clang = Path(shutil.which("clang-18")).resolve(strict=True)
    wrapper = _args["package_dir"].parent / "blocked-clang"
    wrapper.write_text(
        "#!" + sys.executable + "\n"
        "import os, sys, time\n"
        "print('OBJECT_START', flush=True)\n"
        "time.sleep(5)\n"
        "os.execv(" + repr(str(actual_clang)) + ", [" + repr(str(actual_clang)) + ", *sys.argv[1:]])\n"
    )
    wrapper.chmod(0o700)
    monkeypatch.setenv("MERLIN_CLANG", str(wrapper))
    work = _args["package_dir"].parent / "blocked-object"
    deadline = ExecutionDeadline.start(1)
    start = time.monotonic()
    with pytest.raises(CodegenError, match="timed out after"):
        compiler.run_on_oracle(
            _buffer((1,)),
            _LLVM,
            target=build.target,
            simulator="wall_control",
            workdir=work,
            inputs={"input": [0]},
            _build_service=build,
            _execution_service=_execution(build),
            execution_deadline=deadline,
        )
    assert time.monotonic() - start < 2.5
    rows = _records(work)
    assert any(row["stage"] == "object" and row["status"] == "interrupted" for row in rows)
    assert not {"elf", "execution", "blocked_simulator_control"} & {row["stage"] for row in rows}
    assert not (work / "package_kernel.elf").exists()


def test_actual_linked_elf_then_blocked_simulator_shares_the_build_deadline(actual_transport, monkeypatch):  # noqa: F811
    _args, build, _gate = actual_transport
    cb = _buffer((1,))
    work = _args["package_dir"].parent / "blocked-simulator"
    linkage, _ = prepare_linkage(
        cb=cb, lowered_mlir=_LLVM, entry_symbol="fixture_entry", original_abi=_original_abi((1,))
    )
    # A retained reference links real stock translation/object/ELF products
    # without claiming the empty diagnostic entry computes tensor results.
    build = replace(build, renderer=lambda *_args, **_kwargs: linkage.render(cb))
    owner = Path(__file__).resolve()
    # BuildOnlyService requires a pinned source owner for this local renderer.
    build = replace(build, source_pins=(*build.source_pins, (str(owner), file_digest(owner))))
    original_object = compiler.llvm_mlir_to_object

    def measured_object(*args, **kwargs):
        assert kwargs["execution_deadline"] is deadline
        return original_object(*args, **kwargs)

    monkeypatch.setattr(compiler, "llvm_mlir_to_object", measured_object)
    deadline = ExecutionDeadline.start(2)
    start = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired):
        compiler.run_on_oracle(
            cb,
            _LLVM,
            target=build.target,
            simulator="wall_control",
            workdir=work,
            inputs={"input": [0]},
            _build_service=build,
            _execution_service=_execution(build),
            execution_deadline=deadline,
        )
    assert time.monotonic() - start < 3.5
    rows = _records(work)
    assert any(row["stage"] == "elf" and row["status"] == "completed" for row in rows)
    blocked = next(row for row in rows if row["stage"] == "blocked_simulator_control")
    assert blocked["status"] == "interrupted"
    assert (work / "package_kernel.elf").is_file()
    assert "RUN_START" in (work / "oracle_console.log").read_text()


def test_expired_deadline_refuses_before_tools_or_legacy_backend_discovery(actual_transport, monkeypatch):  # noqa: F811
    _args, build, _gate = actual_transport
    monkeypatch.setattr(compiler, "compile_lowered_to_elf", lambda *_args, **_kwargs: pytest.fail("late build"))
    deadline = ExecutionDeadline(time.monotonic() - 2, 1)
    with pytest.raises(TimeoutError):
        compiler.run_on_oracle(
            _buffer((1,)),
            _LLVM,
            target=build.target,
            simulator="wall_control",
            workdir=_args["package_dir"].parent / "late",
            inputs={"input": [0]},
            _build_service=build,
            _execution_service=_execution(build),
            execution_deadline=deadline,
        )
