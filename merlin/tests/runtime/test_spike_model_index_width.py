"""Whole-model builds refuse an unbound cross-compiler index width before lowering."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "observation",
    [
        None,
        {},
        {
            "schema": "merlin.selected-index-lowering.v1",
            "compiler_requested": "fake-clang",
            "compiler_resolved": "/absent/fake-clang",
            "compiler_sha256": "0" * 64,
            "cross_flags": ["--target=riscv64-unknown-elf", "-march=rv64gc"],
            "data_layout": "e-p:64:64:64:32",
            "index_bits": 64,  # Refuses a width that disagrees with the reported GEP index width.
        },
        {
            "schema": "merlin.selected-index-lowering.v1",
            "compiler_requested": "fake-clang",
            "compiler_resolved": "/absent/fake-clang",
            "compiler_sha256": "0" * 64,
            "cross_flags": ["--target=riscv64-unknown-elf", "-march=rv64gc"],
            "data_layout": "e-p:32:32",
            "index_bits": 32,  # Valid generic lowering, but not the selected int64_t C descriptor ABI.
        },
    ],
)
def test_build_refuses_missing_or_mismatched_index_observation_before_lowering(tmp_path, monkeypatch, observation):
    from merlin.llvmlower import qinner, target_data_layout, toolchain, weight_prepack
    from merlin.runtime.backends import spike, spike_model, zephyr_model

    model = tmp_path / "capture"
    model.mkdir()
    (model / "model.mlir").write_text("module {}")
    monkeypatch.setattr(zephyr_model, "load_matrix_signatures", lambda *_: None)
    monkeypatch.setattr(weight_prepack, "prepare_build_bundle", lambda model_dir, *_: model_dir)
    monkeypatch.setattr(spike, "gcc_path", lambda: Path("/absent/gcc"))
    monkeypatch.setattr(toolchain, "clang", lambda: "fake-clang")
    monkeypatch.setattr(qinner, "plan_for_bundle", lambda *_: False)
    monkeypatch.setattr(target_data_layout, "observe_index_width", lambda *_: observation)
    monkeypatch.setattr(spike_model, "lower_model_file", lambda *_a, **_k: pytest.fail("lowering must not run"))

    with pytest.raises((spike_model.SpikeModelError, ValueError, TypeError)):
        spike_model.build(
            model,
            tmp_path / "build",
            backend="scalar",
            cflags_override=["-march=rv64gc"],
        )
    assert not (tmp_path / "build" / "model.o").exists()
    assert not list((tmp_path / "build").glob("*.elf"))
