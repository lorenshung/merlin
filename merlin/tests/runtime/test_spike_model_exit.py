"""A partial Spike log cannot qualify an ELF that exits unsuccessfully."""

from types import SimpleNamespace

import pytest

from merlin.runtime.backends import spike_model


def test_spike_run_rejects_nonzero_exit_after_out_and_done(monkeypatch):
    monkeypatch.setattr(spike_model._spike, "spike_path", lambda: "/unused/spike")
    monkeypatch.setattr(
        spike_model.subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            stdout=b"OUT 1 1065353216\nDONE\n", stderr=b"late failure\n", returncode=17
        ),
    )

    with pytest.raises(spike_model.SpikeModelError, match="exited 17"):
        spike_model.run("/unused/probe.elf")


def test_shared_model_console_preserves_bits_and_metrics():
    result = spike_model.parse_console(
        "METRIC cycles 7\nMETRIC build_hash abc\nOUT 2 1065353216 2147483648\nARGMAX 1 0\nSUM 1065353216\nDONE\n"
    )
    assert result["outputs"].view("uint32").tolist() == [1065353216, 2147483648]
    assert result["metrics"] == {"cycles": 7, "build_hash": "abc"}
    assert result["argmax"].tolist() == [0]
    assert result["sum"] == 1.0
    for console in (
        "OUT 2 1065353216\nDONE\n",
        "OUT 1 1065353216\nOUT 1 1065353216\nDONE\n",
        "OUT 1 1065353216\nNOT_DONE\n",
        "DONE\nOUT 1 1065353216\n",
        "OUT 1 4294967296\nDONE\n",
        "METRIC memref_rank_mismatch 1\nMETRIC memref_rank_mismatch 0\nOUT 0\nDONE\n",
        "OUT 0\nARGMAX 2 1\nDONE\n",
    ):
        with pytest.raises(spike_model.SpikeModelError):
            spike_model.parse_console(console)
