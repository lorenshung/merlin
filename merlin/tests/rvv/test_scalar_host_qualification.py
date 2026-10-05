"""A scalar host receipt passes only a complete, exact saved-capture simulator result."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from contextlib import nullcontext

import numpy as np
import pytest
import yaml

from merlin.common.paths import repo_root
from merlin.compile.scalar_host_qualification import ScalarHostQualificationError, qualify
from merlin.runtime.backends import spike_model
from merlin.runtime.backends import base as backends
from merlin.targetgen import gsim_emulator, rtl_engine_policy


ROCKET_ISA = "rv64imafdcbzicsr_zifencei_zihpm_zfh_zba_zbb_zbs_xrocket"


def _inputs(tmp_path, monkeypatch):
    root = repo_root()
    generated = tmp_path / "out"
    monkeypatch.setenv("MERLIN_OUT_ROOT", str(generated))
    subprocess.run(
        [sys.executable, str(root / "build_tools/scripts/mint_scalar_host_package.py"),
         str(root / "examples/gemmini/target/scalar-host-recipe.yaml")],
        env={**os.environ, "MERLIN_OUT_ROOT": str(generated)},
        check=True, capture_output=True,
    )
    package = generated / "artifacts/targets/host/gemmini_rocket_scalar_int8_v0"
    capture = tmp_path / "capture"
    capture.mkdir()
    for name in ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
                 "inputs.npz", "input_order.json"):
        (capture / name).write_bytes(name.encode())
    np.save(capture / "golden.npy", np.array([0.5, -1.0, 2.0], dtype=np.float32))
    names = ("model.mlir", "weights.safetensors", "weights.safetensors.manifest.json",
             "inputs.npz", "input_order.json", "golden.npy")
    (capture / "capture_receipt.json").write_text(json.dumps({
        "schema": "m2m.capture-receipt.v1",
        "materialized_abi": {"complete": True, "inputs": 1},
        "artifacts": {name: {"sha256": hashlib.sha256((capture / name).read_bytes()).hexdigest()} for name in names},
    }), encoding="utf-8")
    dts = tmp_path / "rocket.dts"
    dts.write_text(f'device_type = "cpu";\nriscv,isa = "{ROCKET_ISA}";\n', encoding="utf-8")
    catalog = tmp_path / "boards.yaml"
    catalog.write_text(yaml.safe_dump({
        "schema_version": 1,
        "boards": {"rocket": {
            "target": "example", "rtl_sim_config": "ExampleConfig",
            "host_dts_sha256": hashlib.sha256(dts.read_bytes()).hexdigest(),
            "dram_bytes": 256 << 20, "dram_base": 0x80000000, "harts": 1,
            "vector_harts": 0, "console": "htif", "flow": "baremetal",
            "code_reserve": 64 << 20,
        }},
    }), encoding="utf-8")
    return capture, package, catalog, dts, generated


def _stub_spike(monkeypatch, *, outputs, console):
    seen = {}

    def build(_capture, work, **kwargs):
        seen["build"] = kwargs
        work.mkdir(parents=True)
        elf = work / "model.elf"
        elf.write_bytes(b"synthetic scalar ELF")
        return {"elf": elf, "mem_bytes": 256 << 20, "build_hash": "test"}

    def run(_elf, **kwargs):
        seen["run"] = kwargs
        return {"outputs": np.asarray(outputs, dtype=np.float32), "console": console,
                "metrics": {"memref_rank_mismatch": 0, "cycles": 10}}

    monkeypatch.setattr(spike_model, "build", build)
    monkeypatch.setattr(spike_model, "run", run)
    monkeypatch.setattr(spike_model, "arch_extensions", lambda _elf: ["rv64i", "m", "a", "f", "d", "c"])
    return seen


def _console(values):
    bits = np.asarray(values, dtype=np.float32).view(np.uint32)
    return "OUT " + str(len(bits)) + " " + " ".join(str(int(value)) for value in bits) + "\nDONE\n"


def _stub_native_gsim(tmp_path, monkeypatch, *, console):
    firrtl = tmp_path / "selected.fir"
    firrtl.write_bytes(b"selected elaborated FIRRTL")
    digest = hashlib.sha256(firrtl.read_bytes()).hexdigest()
    facts = tmp_path / "rtl-facts.json"
    facts.write_text(json.dumps({
        "inputs": {"target": "example", "fir_sha256": digest,
                   "firrtl_inputs": [{"path": str(firrtl), "sha256": digest}]},
        "facts": {"source": {"config": "ExampleConfig"}},
    }), encoding="utf-8")
    emulator = tmp_path / "emulator"
    emulator.write_bytes(b"selected engine")
    citation = {
        "target": "example", "engine": "gsim", "path": str(emulator),
        "available": True, "refused": False, "flavour": "binary", "receipt_status": "bound",
        "binary_sha256": hashlib.sha256(emulator.read_bytes()).hexdigest(),
        "receipt": {"schema_version": gsim_emulator.STRICT_RECEIPT_SCHEMA, "firrtl_sha256": digest},
    }
    seen = {}

    class Prepared:
        def revalidate(self):
            seen["revalidated"] = seen.get("revalidated", 0) + 1

    class Backend:
        GSIM_EMU_ENV = "TEST_GSIM_EMULATOR"

        def available(self, simulator):
            return simulator == "gsim"

        def gsim_path(self):
            return emulator

        def prepare_gsim_command(self, elf, **kwargs):
            seen["prepared"] = (elf, kwargs)
            return Prepared()

        def run_elf(self, elf, **kwargs):
            seen["run"] = (elf, kwargs)
            return console

    monkeypatch.setattr(backends, "get_backend", lambda _target: Backend())
    monkeypatch.setattr(gsim_emulator, "citation", lambda _target, **_kw: citation)
    monkeypatch.setattr(rtl_engine_policy, "gsim_runtime_slot", lambda **_kw: nullcontext())
    return facts, citation, seen


def test_exact_single_output_receipt_binds_every_input(tmp_path, monkeypatch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    seen = _stub_spike(monkeypatch, outputs=[0.5, -1.0, 2.0], console=_console([0.5, -1.0, 2.0]))
    output = generated / "qualifications" / "one"
    receipt = qualify(capture=capture, package=package, board_catalog=catalog,
                      board="rocket", dts=dts, output=output, arena_mb=32)
    assert receipt["status"] == "passed_saved_capture_spike"
    assert receipt["inputs"]["host_isa"] == ROCKET_ISA
    assert receipt["inputs"]["simulator_isa"] == "rv64gc_zba_zbb_zbs_zfh"
    assert receipt["inputs"]["package_tree"]["sha256"]
    assert receipt["inputs"]["capture_tree"]["sha256"]
    assert receipt["inputs"]["dts_sha256"] == hashlib.sha256(dts.read_bytes()).hexdigest()
    assert receipt["output"]["elf_sha256"] == hashlib.sha256(b"synthetic scalar ELF").hexdigest()
    assert receipt["output"]["mismatched_elements"] == 0
    assert seen["run"]["isa"] == receipt["inputs"]["simulator_isa"]
    assert seen["build"]["backend"] == "scalar"
    assert json.loads((output / "qualification.json").read_text())["status"] == receipt["status"]


@pytest.mark.parametrize("outputs,multiout,reason", [
    ([0.5, -1.0], False, "partial or nonfinite output"),
    ([0.5, -1.0, 3.0], False, "exact golden mismatch"),
    ([0.5, -1.0, 2.0], True, "exactly one OUT"),
])
def test_partial_mismatch_and_multiout_never_pass(tmp_path, monkeypatch, outputs, multiout, reason):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    console = _console(outputs)
    if multiout:
        console += _console([1.0])
    _stub_spike(monkeypatch, outputs=outputs, console=console)
    output = generated / "qualifications" / "failed"
    with pytest.raises(ScalarHostQualificationError, match=reason):
        qualify(capture=capture, package=package, board_catalog=catalog,
                board="rocket", dts=dts, output=output, arena_mb=32)
    receipt = json.loads((output / "qualification.json").read_text())
    assert receipt["status"] == "failed"


def test_unpinned_dts_refuses_before_build(tmp_path, monkeypatch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    dts.write_text('riscv,isa = "rv64gcv";\n', encoding="utf-8")
    monkeypatch.setattr(spike_model, "build", lambda *_a, **_kw: pytest.fail("build before DTS pin"))
    with pytest.raises(ValueError, match="pinned SHA256"):
        qualify(capture=capture, package=package, board_catalog=catalog,
                board="rocket", dts=dts, output=generated / "qualifications" / "bad-dts", arena_mb=32)


def test_output_inside_selected_package_refuses_before_writing(tmp_path, monkeypatch):
    capture, package, catalog, dts, _generated = _inputs(tmp_path, monkeypatch)
    output = package / "qualification"
    monkeypatch.setattr(spike_model, "build", lambda *_a, **_kw: pytest.fail("build before output preflight"))
    with pytest.raises(ScalarHostQualificationError, match="nested inside an input"):
        qualify(capture=capture, package=package, board_catalog=catalog,
                board="rocket", dts=dts, output=output, arena_mb=32)
    assert not output.exists()


def test_native_gsim_uses_shared_model_parser_and_records_exact_engine(tmp_path, monkeypatch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    _stub_spike(monkeypatch, outputs=[0.5, -1.0, 2.0], console="")
    console = _console([0.5, -1.0, 2.0]).replace("DONE", "METRIC memref_rank_mismatch 0\nDONE")
    facts, citation, seen = _stub_native_gsim(tmp_path, monkeypatch, console=console)
    output = generated / "qualifications" / "native"
    receipt = qualify(capture=capture, package=package, board_catalog=catalog,
                      board="rocket", dts=dts, output=output, arena_mb=32,
                      simulator="gsim", rtl_facts=facts)
    assert receipt["schema"] == "merlin.scalar_host_native_qualification.v1"
    assert receipt["status"] == "passed_saved_capture_native_rtl"
    assert receipt["engine"]["citation"] == citation
    assert receipt["output"]["mismatched_elements"] == 0
    assert receipt["output"]["metrics"]["memref_rank_mismatch"] == 0
    assert receipt["output"]["rtl_console_sha256"] == hashlib.sha256(console.encode()).hexdigest()
    assert seen["run"][1]["simulator"] == "gsim"
    assert seen["revalidated"] == 2


@pytest.mark.parametrize("console,reason", [
    ("OUT 3 1056964608 3212836864 1073741824\nMETRIC memref_rank_mismatch 0\n", "DONE"),
    ("OUT 2 1056964608 3212836864\nMETRIC memref_rank_mismatch 0\nDONE\n", "partial"),
])
def test_native_incomplete_result_never_passes(tmp_path, monkeypatch, console, reason):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    _stub_spike(monkeypatch, outputs=[0.5, -1.0, 2.0], console="")
    facts, _citation, seen = _stub_native_gsim(tmp_path, monkeypatch, console=console)
    output = generated / "qualifications" / "native-incomplete"
    with pytest.raises((ScalarHostQualificationError, spike_model.SpikeModelError), match=reason):
        qualify(capture=capture, package=package, board_catalog=catalog,
                board="rocket", dts=dts, output=output, arena_mb=32,
                simulator="gsim", rtl_facts=facts)
    assert "run" in seen
    assert json.loads((output / "qualification.json").read_text())["status"] == "failed"


def test_native_facts_target_mismatch_refuses_before_build(tmp_path, monkeypatch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    facts, _citation, _seen = _stub_native_gsim(tmp_path, monkeypatch, console="")
    document = json.loads(facts.read_text())
    document["inputs"]["target"] = "different-target"
    facts.write_text(json.dumps(document))
    monkeypatch.setattr(spike_model, "build", lambda *_a, **_kw: pytest.fail("build before RTL facts pin"))
    with pytest.raises(ScalarHostQualificationError, match="board target/config"):
        qualify(capture=capture, package=package, board_catalog=catalog,
                board="rocket", dts=dts, output=generated / "qualifications" / "wrong-target",
                arena_mb=32, simulator="gsim", rtl_facts=facts)


@pytest.mark.parametrize("arch", [
    ["rv64i", "m", "a", "f", "d", "c", "v"],
    ["rv64i", "m", "a", "f", "d", "c", "zicond"],
])
def test_elf_with_vector_or_undeclared_extension_refuses(tmp_path, monkeypatch, arch):
    capture, package, catalog, dts, generated = _inputs(tmp_path, monkeypatch)
    _stub_spike(monkeypatch, outputs=[0.5, -1.0, 2.0], console=_console([0.5, -1.0, 2.0]))
    monkeypatch.setattr(spike_model, "arch_extensions", lambda _elf: arch)
    output = generated / "qualifications" / "bad-elf"
    with pytest.raises(ScalarHostQualificationError, match="ELF ISA is not scalar-compatible"):
        qualify(capture=capture, package=package, board_catalog=catalog,
                board="rocket", dts=dts, output=output, arena_mb=32)
    assert json.loads((output / "qualification.json").read_text())["status"] == "failed"
