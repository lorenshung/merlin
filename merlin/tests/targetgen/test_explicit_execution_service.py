"""Explicit functional transport executes bytes and cannot discover a backend.

The fixture substitutes only target building. The native process, complete
readback decoder, source drift guard and invocation recorder are real. These
dispatcher regressions do not qualify an accelerator runtime or compiler.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

from merlin.common import invocation_record
from merlin.runtime.backends.base import parse_console
from merlin.targetgen.contract import compile as compiler
from merlin.targetgen.contract import readback_policy as RB
from merlin.targetgen.contract.build_recipe import HarnessBuildRecipe
from merlin.targetgen.contract.build_service import BuildOnlyService, file_digest
from merlin.targetgen.contract.elf_admission import LinkedElfAdmissionService
from merlin.targetgen.contract.execution_service import FunctionalExecutionService


def _run_native(elf, *, simulator, timeout, **kwargs):
    assert simulator in ("native_control", "verilator") and not kwargs
    result = invocation_record.run(
        [sys.executable, "-I", "-B", str(elf)],
        directory=Path(elf).parent,
        stage="functional_engine",
        inputs=(Path(elf),),
        capture_output=True,
        text=True,
        timeout=timeout,
        check=True,
    )
    return result.stdout


def _parse(console):
    return parse_console(console)


def _render(_cb, *, inputs, readback_policy):
    raise AssertionError("this dispatcher test substitutes target building")


def _fixture(tmp_path, monkeypatch, console, *, simulator="native_control"):
    from merlin.runtime.backends import base as backends

    def forbidden(_target):
        raise AssertionError("explicit transport attempted target backend discovery")

    monkeypatch.setattr(backends, "get_backend", forbidden)
    owner = Path(__file__).resolve()
    pins = ((str(owner), file_digest(owner)),)
    execution = FunctionalExecutionService(
        "fixture", simulator, _run_native, _parse, pins, json.dumps({"binary": str(Path(sys.executable).resolve())})
    )
    recipe = HarnessBuildRecipe(tmp_path / "unused-compiler", (), (), tmp_path / "unused.ld", 0)
    build = BuildOnlyService("fixture", recipe, _render, pins)
    cb = {
        "kernel_abi": {"kind": "whole_program", "outputs": ["out"]},
        "tensors": {"out": {"shape": [1, 2], "dtype": "i8", "role": "output"}},
    }

    def substitute_build(_cb, _llvm, work, **kwargs):
        assert kwargs["_build_service"] is build
        work.mkdir()
        executable = work / "native.py"
        executable.write_text("import sys\nsys.stdout.write(" + repr(console) + ")\n")
        return executable

    def substitute_receipt(**kwargs):
        assert kwargs["build_service"] is build
        return {"elf_sha256": file_digest(Path(kwargs["elf_path"]))}

    monkeypatch.setattr(compiler, "compile_lowered_to_elf", substitute_build)
    monkeypatch.setattr(RB, "require_current_build_receipt", substitute_receipt)
    return cb, build, execution


def _full_console():
    return "OUT_B64_BEGIN v1 out 1 2 1 s\nOUT_B64_CHUNK 00000000 0002 AQI=\nOUT_B64_END\nDONE\n"


def _invoke(tmp_path, cb, build, execution, **kwargs):
    return compiler.run_on_oracle(
        cb,
        "substituted target build",
        simulator=execution.simulator,
        target="fixture",
        workdir=tmp_path / "run",
        timeout=10,
        inputs={"input": [1, 2]},
        readback_policy=RB.ReadbackPolicy(RB.FULL_VALUES_B64),
        _build_service=build,
        _execution_service=execution,
        **kwargs,
    )


def test_explicit_transport_skips_backend_and_retains_actual_process(tmp_path, monkeypatch):
    cb, build, execution = _fixture(tmp_path, monkeypatch, _full_console())
    actual = _invoke(tmp_path, cb, build, execution)
    assert actual["outputs"] == {"out": [[1, 2]]}
    assert actual["oracle"]["derived_from_rtl"] is False
    records = [invocation_record.verify(path) for path in (tmp_path / "run").rglob("invocation.json")]
    assert any(row["kind"] == "subprocess" and row["stage"] == "functional_engine" for row in records)
    assert (tmp_path / "run" / "oracle_console.log").read_text() == _full_console()


def test_explicit_transport_cannot_use_incomplete_output_roster(tmp_path, monkeypatch):
    cb, build, execution = _fixture(tmp_path, monkeypatch, "DONE\n")
    with pytest.raises(ValueError, match="omitted"):
        _invoke(tmp_path, cb, build, execution)
    assert (tmp_path / "run" / "oracle_console.log").read_text() == "DONE\n"


def test_known_engine_name_cannot_mint_timing_authority(tmp_path, monkeypatch):
    from merlin.perf import hw_counters

    console = f"{hw_counters.COUNTER_MARKER} diagnostic_counter 73\n" + _full_console()
    cb, build, execution = _fixture(tmp_path, monkeypatch, console, simulator="verilator")
    result = _invoke(tmp_path, cb, build, execution)
    assert result["counters"]["status"] == "unknown"
    assert result["counters"]["readings"] is None
    assert "functional transport" in result["counters"]["why"]
    assert result["oracle"]["derived_from_rtl"] is False


def test_explicit_transport_rechecks_selection_before_decoding(tmp_path, monkeypatch):
    cb, build, execution = _fixture(tmp_path, monkeypatch, _full_console())
    calls = []

    def changing():
        calls.append(1)
        return {"selected": len(calls) > 2}

    with pytest.raises(ValueError, match="selected execution engine changed"):
        _invoke(tmp_path, cb, build, execution, execution_revalidate=changing)
    assert len(calls) == 3


def test_transport_source_pin_changes_refuse_before_process(tmp_path):
    owner = Path(__file__).resolve()
    selected = tmp_path / "engine.source"
    selected.write_bytes(b"original")
    execution = FunctionalExecutionService(
        "fixture",
        "native_control",
        _run_native,
        _parse,
        ((str(owner), file_digest(owner)), (str(selected), file_digest(selected))),
        '{"kind":"diagnostic"}',
    )
    execution.verify("fixture", "native_control")
    selected.write_bytes(b"mutated")
    with pytest.raises(ValueError, match="source/tool pin changed"):
        execution.run_elf(tmp_path / "never", simulator="native_control", timeout=1)


def test_transport_requires_actual_inspected_callback_owner(tmp_path):
    selected = tmp_path / "not_callback.py"
    selected.write_text("# unrelated\n")
    execution = FunctionalExecutionService(
        "fixture",
        "native_control",
        _run_native,
        _parse,
        ((str(selected), file_digest(selected)),),
        '{"kind":"diagnostic"}',
    )
    with pytest.raises(ValueError, match="pinned inspected source owner"):
        execution.verify("fixture", "native_control")


def _artifact_gate(*, elf, evidence_root):
    # Dispatcher-only source policy; not an ISA audit or authority issuer.
    evidence_root.mkdir()
    report = evidence_root / "report.json"
    result = {
        "status": "refused" if b"private-forbidden-marker" in elf.read_bytes() else "accepted",
        "elf_sha256": file_digest(elf),
    }
    report.write_text(json.dumps(result))
    return {**result, "report_path": str(report), "report_sha256": file_digest(report)}


def _selected_gate():
    owner = Path(__file__).resolve()
    return LinkedElfAdmissionService("fixture", _artifact_gate, ((str(owner), file_digest(owner)),))


def test_linked_artifact_refusal_happens_before_any_process_dispatch(tmp_path, monkeypatch):
    cb, build, execution = _fixture(tmp_path, monkeypatch, "private-forbidden-marker")
    result = _invoke(tmp_path, cb, build, execution, _elf_admission=_selected_gate())
    assert result["status"] == "refused_before_execution"
    assert result["execution"] == "not_attempted"
    assert "outputs" not in result and "timing" not in result
    assert not list((tmp_path / "run").rglob("invocation.json"))
    assert not (tmp_path / "run" / "oracle_console.log").exists()


def test_accepted_linked_artifact_continues_to_actual_process_and_full_roster(tmp_path, monkeypatch):
    cb, build, execution = _fixture(tmp_path, monkeypatch, _full_console())
    actual = _invoke(tmp_path, cb, build, execution, _elf_admission=_selected_gate())
    assert actual["outputs"] == {"out": [[1, 2]]}
    records = [invocation_record.verify(path) for path in (tmp_path / "run").rglob("invocation.json")]
    assert any(row["stage"] == "functional_engine" and row["kind"] == "subprocess" for row in records)


def test_admission_report_and_linked_bytes_must_reopen_unchanged(tmp_path):
    elf = tmp_path / "elf"
    elf.write_bytes(b"original")
    service = _selected_gate()
    result = service.evaluate(elf=elf, target="fixture", evidence_root=tmp_path / "gate")
    elf.write_bytes(b"changed")
    with pytest.raises(ValueError, match="ELF changed after admission"):
        service.revalidate(elf=elf, result=result, target="fixture")
    elf.write_bytes(b"original")
    Path(result["report_path"]).write_text("{}")
    with pytest.raises(ValueError, match="report has no unchanged"):
        service.revalidate(elf=elf, result=result, target="fixture")
