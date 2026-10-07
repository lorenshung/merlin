"""The selected bare-metal math archive is a build input, not a driver guess."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower import toolchain
from merlin.runtime.backends import spike, spike_model


def _neutral_bundle(path: Path) -> Path:
    path.mkdir()
    (path / "model.mlir").write_text(
        """module {
          func.func @forward(%arg0: tensor<3xf32>) -> tensor<3xf32> {
            %init = tensor.empty() : tensor<3xf32>
            %out = linalg.generic {
              indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
              iterator_types = ["parallel"]
            } ins(%arg0 : tensor<3xf32>) outs(%init : tensor<3xf32>) {
              ^bb0(%value: f32, %unused: f32):
                %sin = math.sin %value : f32
                linalg.yield %sin : f32
            } -> tensor<3xf32>
            return %out : tensor<3xf32>
          }
        }"""
    )
    (path / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"values"}}')
    np.savez(path / "inputs.npz", in0=np.array([-1.0, 0.0, 1.0], dtype=np.float32))
    return path


@pytest.mark.parametrize("policy", ["native", "expf_via_double"])
def test_actual_link_records_selected_libm_bytes(tmp_path, policy):
    if not spike.available() or not toolchain.m2m_python().is_file() or not toolchain.clang().is_file():
        pytest.skip("bare-metal and upstream toolchain unavailable")
    built = spike_model.build(
        _neutral_bundle(tmp_path / "bundle"),
        tmp_path / "build",
        arena_mb=1,
        backend="scalar",
        host_math_policy=policy,
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
    )
    record = json.loads((tmp_path / "build/compilation_recipe.json").read_text())
    assert record["status"] == "completed"
    link = record["commands"][-1]
    assert Path(link["output"]["path"]) == built["elf"]
    archive_inputs = [item for item in link["inputs"] if Path(item["path"]).name == "libm.a"]
    assert len(archive_inputs) == 1
    assert archive_inputs[0]["path"] in link["argv"]
    assert "-lm" not in link["argv"]
    assert ("-Wl,--wrap=expf" in link["argv"]) == (policy == "expf_via_double")
    assert archive_inputs[0]["sha256"] == hashlib.sha256(Path(archive_inputs[0]["path"]).read_bytes()).hexdigest()
    assert (
        "sinf"
        in subprocess.run(
            [str(spike.gcc_path().with_name("riscv64-unknown-elf-nm")), str(built["elf"])],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    )


def test_archive_resolution_uses_selected_flags_and_refuses_missing_or_changed_driver(tmp_path, monkeypatch):
    driver, archive = tmp_path / "gcc", tmp_path / "libm.a"
    driver.write_bytes(b"selected driver")
    archive.write_bytes(b"!<arch>\n")
    commands = []

    def query(cmd):
        commands.append(cmd)
        return subprocess.CompletedProcess(cmd, 0, str(archive) + "\n", "")

    monkeypatch.setattr(spike_model, "_run", query)
    flags = ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany"]
    selected, archive_sha, driver_sha = spike_model._selected_libm_archive(driver, flags, ("-Wl,--wrap=expf",))
    assert selected == archive
    assert archive_sha == hashlib.sha256(archive.read_bytes()).hexdigest()
    assert driver_sha == hashlib.sha256(driver.read_bytes()).hexdigest()
    assert commands == [[driver, *flags, "-Wl,--wrap=expf", "-nostdlib", "-nostartfiles", "-print-file-name=libm.a"]]

    for search_flags, extra_link in (
        ([*flags, "-L/alternate"], ()),
        ([*flags, "--sysroot=/alternate"], ()),
        (flags, ("-Wl,-L,/alternate",)),
    ):
        with pytest.raises(spike_model.SpikeModelError, match="unsupported math-library search"):
            spike_model._selected_libm_archive(driver, search_flags, extra_link)
    assert len(commands) == 1

    monkeypatch.setattr(
        spike_model,
        "_run",
        lambda cmd: subprocess.CompletedProcess(cmd, 0, "libm.a\n", ""),
    )
    with pytest.raises(spike_model.SpikeModelError, match="resolve a regular libm.a"):
        spike_model._selected_libm_archive(driver, flags, ())

    def changed_driver(cmd):
        driver.write_bytes(b"changed driver")
        return subprocess.CompletedProcess(cmd, 0, str(archive) + "\n", "")

    monkeypatch.setattr(spike_model, "_run", changed_driver)
    with pytest.raises(spike_model.SpikeModelError, match="changed during resolution"):
        spike_model._selected_libm_archive(driver, flags, ())

    driver.write_bytes(b"selected driver")
    archive.write_bytes(b"!<thin>\n")
    monkeypatch.setattr(spike_model, "_run", query)
    with pytest.raises(spike_model.SpikeModelError, match="self-contained regular archive"):
        spike_model._selected_libm_archive(driver, flags, ())


def test_archive_bytes_change_marker_and_post_link_mutation_refuses(tmp_path, monkeypatch):
    if not spike.available() or not toolchain.m2m_python().is_file() or not toolchain.clang().is_file():
        pytest.skip("bare-metal and upstream toolchain unavailable")
    gcc = spike.gcc_path()
    original_archive = Path(
        subprocess.run(
            [gcc, "-march=rv64gc", "-mabi=lp64d", "-print-file-name=libm.a"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    )
    archive = tmp_path / "libm.a"
    shutil.copyfile(original_archive, archive)
    original_run = spike_model._run
    mutate_after_link = False

    def selected_run(cmd, **kwargs):
        if "-print-file-name=libm.a" in cmd:
            return subprocess.CompletedProcess(cmd, 0, str(archive) + "\n", "")
        result = original_run(cmd, **kwargs)
        if mutate_after_link and str(cmd[-1]).endswith("model.elf"):
            archive.write_bytes(archive.read_bytes() + b"changed after link")
        return result

    monkeypatch.setattr(spike_model, "_run", selected_run)
    bundle = _neutral_bundle(tmp_path / "bundle")
    flags = ["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"]
    first = spike_model.build(bundle, tmp_path / "first", arena_mb=1, backend="scalar", cflags_override=flags)
    unused = tmp_path / "unused.c"
    unused.write_text("int merlin_unused_archive_member(void) { return 1; }\n")
    subprocess.run([gcc, *flags, "-c", unused, "-o", tmp_path / "unused.o"], check=True, capture_output=True)
    subprocess.run(
        [gcc.with_name("riscv64-unknown-elf-ar"), "r", archive, tmp_path / "unused.o"],
        check=True,
        capture_output=True,
    )
    second = spike_model.build(bundle, tmp_path / "second", arena_mb=1, backend="scalar", cflags_override=flags)
    assert first["build_hash"] != second["build_hash"]
    assert first["elf"].read_bytes() != second["elf"].read_bytes()
    mutate_after_link = True
    with pytest.raises(spike_model.SpikeModelError, match="archive changed during link"):
        spike_model.build(bundle, tmp_path / "mutated", arena_mb=1, backend="scalar", cflags_override=flags)
    record = json.loads((tmp_path / "mutated/compilation_recipe.json").read_text())
    assert record["status"] != "completed"
    assert "executable" not in record
