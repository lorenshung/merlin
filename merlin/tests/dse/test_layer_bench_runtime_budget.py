"""The host-owned prepared-command path must share the GSim process budget."""

from contextlib import contextmanager
from importlib import import_module
from types import SimpleNamespace

import pytest

from merlin.runtime.backends import base as backends
from merlin.targetgen import gsim_emulator, rtl_engine_policy

RUN = import_module("merlin.perf.layer_bench.run")


def _prepared(monkeypatch, *, revalidate):
    command = SimpleNamespace(
        argv=("inert-emulator", "inert-elf"),
        env_overrides=(),
        revalidate=revalidate,
    )
    backend = SimpleNamespace(
        GSIM_EMU_ENV="INERT_GSIM",
        prepare_gsim_command=lambda *args, **kwargs: command,
    )
    monkeypatch.setattr(backends, "get_backend", lambda target: backend)
    monkeypatch.setattr(gsim_emulator, "citation", lambda *args, **kwargs: {"available": True, "refused": False})
    return command


def test_prepared_gsim_subprocess_and_revalidation_stay_inside_slot(monkeypatch, tmp_path):
    elf = tmp_path / "inert.elf"
    elf.write_bytes(b"inert test bytes")
    inside, events = False, []

    @contextmanager
    def slot(*, wait_timeout_s):
        nonlocal inside
        assert wait_timeout_s == 3
        inside = True
        try:
            yield
        finally:
            inside = False

    def revalidate():
        assert inside
        events.append("validated")
        return {"command_sha256": "c" * 64}

    _prepared(monkeypatch, revalidate=revalidate)

    def run(argv, **kwargs):
        assert inside
        assert argv == ["inert-emulator", "inert-elf"]
        assert kwargs["timeout"] == 3
        events.append("spawned")
        return SimpleNamespace(returncode=0, stdout="LB_RECORD x cycles=7\n", stderr="")

    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", slot)
    monkeypatch.setattr(RUN.subprocess, "run", run)
    result = RUN.run_on_gsim(elf, target="fixture", max_cycles=11, timeout_s=3)
    assert not inside
    assert events == ["validated", "spawned", "validated"]
    assert result.returncode == 0 and result.records[0].cycles == 7


def test_prepared_gsim_refused_slot_never_spawns(monkeypatch, tmp_path):
    elf = tmp_path / "inert.elf"
    elf.write_bytes(b"inert test bytes")
    events = []
    _prepared(monkeypatch, revalidate=lambda: events.append("validated"))

    @contextmanager
    def refused(*, wait_timeout_s):
        assert wait_timeout_s == 0
        raise TimeoutError("all five GSim slots busy")
        yield

    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", refused)
    monkeypatch.setattr(RUN.subprocess, "run", lambda *args, **kwargs: events.append("spawned"))
    with pytest.raises(TimeoutError, match="slots busy"):
        RUN.run_on_gsim(elf, target="fixture", max_cycles=11, timeout_s=0)
    assert events == []
