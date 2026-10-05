"""The FireSim machine refuses a submission before the queue sees it when either disk the queue needs
is too full: the filesystem holding the machine's chipyard, or the host root where the queue's child
processes stage into /tmp.  Either failure otherwise costs a whole board slot to discover."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from merlin_experiments.phase2.whole_model_measured import machines as M

GIB = 1024**3


def test_the_free_space_is_read_from_the_machines_own_chipyard_and_the_host_root(tmp_path, monkeypatch):
    seen = []
    monkeypatch.setattr(M.shutil, "disk_usage", lambda path: seen.append(str(path)) or SimpleNamespace(free=123))
    assert M.deploy_free_bytes(tmp_path / "chipyard") == 123 and M.host_root_free_bytes() == 123
    assert seen == [str(tmp_path / "chipyard"), "/"]


def _machine(tmp_path, calls, run=None):
    def runner(argv, **kwargs):
        calls.append(list(argv))
        if run is None:
            raise AssertionError(f"the queue must not be invoked: {argv}")
        return run(argv, **kwargs)

    machine = M.FiresimMachine(
        "toy",
        hw_config="hw",
        chipyard=tmp_path / "chipyard",
        workload="wl",
        bootbinary="boot",
        queue_command=("env", "-u", "HOME", "-u", "USER", "-u", "LOGNAME", "q"),
        jobs_root=tmp_path / "jobs",
        uart_relative="uart.log",
        lock_path=tmp_path / "lock",
        runner=runner,
    )
    machine.identity = lambda: SimpleNamespace(to_dict=lambda: {}, abi_header_sha256="x")
    machine.load_path.parent.mkdir(parents=True)
    return machine


@pytest.mark.parametrize(
    "deploy,root,named",
    [(5 * GIB, 50 * GIB, str(5 * GIB)), (50 * GIB, 2 * GIB, str(2 * GIB))],
    ids=["deploy_tree_low", "host_root_low"],
)
def test_a_low_disk_refuses_before_the_queue_is_invoked(tmp_path, monkeypatch, deploy, root, named):
    monkeypatch.setattr(M, "deploy_free_bytes", lambda chipyard: deploy)
    monkeypatch.setattr(M, "host_root_free_bytes", lambda: root)
    calls = []
    elf = tmp_path / "program.elf"
    elf.write_bytes(b"\x7fELF")
    result = _machine(tmp_path, calls).run(elf, tmp_path / "work", timeout_s=60)
    assert result["completed"] is False and result[M.INFRA_BOARD_UNAVAILABLE] is True
    assert named in result["incomplete_reason"] and calls == []


def test_with_room_on_both_the_submission_reaches_the_queue(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "deploy_free_bytes", lambda chipyard: 50 * GIB)
    monkeypatch.setattr(M, "host_root_free_bytes", lambda: 50 * GIB)
    calls = []
    elf = tmp_path / "program.elf"
    elf.write_bytes(b"\x7fELF")
    machine = _machine(tmp_path, calls, run=lambda argv, **kw: SimpleNamespace(returncode=0, stdout="", stderr=""))
    result = machine.run(elf, tmp_path / "work", timeout_s=1)
    assert calls and result.get(M.INFRA_BOARD_UNAVAILABLE) is not True
    assert "no job id" in result["incomplete_reason"]
