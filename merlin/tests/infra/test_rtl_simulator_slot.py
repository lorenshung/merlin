"""Cross-process Verilator admission is bounded and releases reservations."""

import multiprocessing

import pytest

from merlin.targetgen import rtl_simulator_slot as slots


def _try_slot(root, result):
    try:
        with slots.verilator_slot(root=root, timeout=0.05):
            result.put("acquired")
    except TimeoutError:
        result.put("timeout")


def test_other_process_waits_and_release_makes_capacity_available(tmp_path, monkeypatch):
    monkeypatch.setattr(slots, "capsule_worker_cap", lambda _: 1)
    root = tmp_path / "slots"
    context = multiprocessing.get_context("fork")
    result = context.Queue()
    with slots.verilator_slot(root=root, timeout=1):
        process = context.Process(target=_try_slot, args=(root, result))
        process.start()
        assert result.get(timeout=5) == "timeout"
        process.join(timeout=5)
        assert process.exitcode == 0
    with slots.verilator_slot(root=root, timeout=1) as remaining:
        assert 0 < remaining <= 1


def test_foreign_or_public_slot_directory_is_refused(tmp_path):
    root = tmp_path / "public"
    root.mkdir(mode=0o755)
    with pytest.raises(RuntimeError, match="not private"):
        with slots.verilator_slot(root=root, timeout=1):
            pytest.fail("public directory admitted")


def test_symlink_slot_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(slots, "capsule_worker_cap", lambda _: 1)
    root = tmp_path / "slots"
    root.mkdir(mode=0o700)
    (root / "slot_0.lock").symlink_to(tmp_path / "elsewhere")
    with pytest.raises(OSError):
        with slots.verilator_slot(root=root, timeout=1):
            pytest.fail("symlink admitted")
