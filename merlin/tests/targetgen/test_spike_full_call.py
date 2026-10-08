"""Plain-ISA Spike oracle: routed from the contract, functional tier only, ISA never defaulted."""

from __future__ import annotations

import pytest

from merlin.common.paths import repo_root

capsule_runner = pytest.importorskip("merlin.targetgen.capsule_runner")
spike_full_call = pytest.importorskip("merlin.targetgen.spike_full_call")

TARGET = "riscv_scalar"


def test_scalar_contract_declares_isa_and_spike_engine():
    from merlin.targetgen.target_experiment import load_capability_manifest

    contract = load_capability_manifest(TARGET, contract_path=_contract()).contract
    assert contract["isa"]["march"]
    assert contract["runner"]["sim_via"] == "spike"


def _contract():
    return repo_root() / "examples" / "riscv_scalar" / "target" / "contracts" / "target_contract.yaml"


def test_isa_is_derived_not_defaulted(monkeypatch):
    class Manifest:
        contract = {"name": "x"}

    monkeypatch.setattr("merlin.targetgen.target_experiment.load_capability_manifest", lambda target, **_: Manifest())
    with pytest.raises(ValueError, match="isa.march"):
        spike_full_call.sim_adapters("anything")


def test_adapter_exposes_full_call_boundary_only():
    adapter = spike_full_call.full_call_adapter("t", isa="rv64gc")
    assert callable(adapter.run_full_call)
    with pytest.raises(capsule_runner.OracleUnavailable):
        adapter(None, "", None, 1)


def test_build_refusal_returns_durable_execution_error(tmp_path, monkeypatch):
    from merlin.runtime.backends import spike_model
    from merlin.targetgen import core_aten_provenance

    def refuse(*args, **kwargs):
        raise ValueError("candidate build refused")

    monkeypatch.setattr(spike_model, "build", refuse)
    monkeypatch.setattr(core_aten_provenance, "batch_provenance", lambda *a, **k: {"scope": "test"})
    adapter = spike_full_call.full_call_adapter("t", isa="declared-isa")
    result = adapter.run_full_call(bundle=tmp_path, llvm_mlir="submitted LLVM")
    assert result["output_bytes"] is None
    assert result["execution_error"] == "ValueError: candidate build refused"
    assert result["lane"] == "host" and result["executed_instructions"] == 0


def _provider_adapter(tmp_path, monkeypatch, routing):
    from merlin.runtime.backends import spike_model
    from merlin.targetgen import core_aten_device, core_aten_provenance

    provider = tmp_path / "provider.py"
    provider.write_text("TARGET = 't'\n")
    seen = {}

    def build(*args, **kwargs):
        seen["device"] = kwargs.get("device")
        raise ValueError("stop after routing")

    from contextlib import nullcontext

    monkeypatch.setattr(spike_model, "build", build)
    monkeypatch.setattr(core_aten_provenance, "batch_provenance", lambda *a, **k: {"scope": "test"})
    monkeypatch.setattr(core_aten_device, "load_execution_provider", lambda *a, **k: object())
    monkeypatch.setattr(core_aten_device, "selected_facts", lambda *a, **k: nullcontext({}))
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a, **k: (routing, "declined"))
    adapter = spike_full_call.full_call_adapter("t", isa="declared-isa", execution_provider=provider)
    return adapter, seen


def test_any_lane_capsule_falls_back_to_the_host_lane(tmp_path, monkeypatch):
    adapter, seen = _provider_adapter(tmp_path, monkeypatch, None)
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert seen["device"] is None and result["execution_error"] == "ValueError: stop after routing"
    assert result["lane"] == "host"


def test_device_lane_capsule_still_refuses_without_routing(tmp_path, monkeypatch):
    adapter, seen = _provider_adapter(tmp_path, monkeypatch, None)
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "device"}
    )
    assert "did not route" in result["execution_error"] and "device" not in seen


def test_scored_host_capsule_skips_device_routing_even_if_backend_covers_it(tmp_path, monkeypatch):
    adapter, seen = _provider_adapter(tmp_path, monkeypatch, object())
    result = adapter.run_full_call(bundle=tmp_path, llvm_mlir="", capsule={"scored": True, "lane_expectation": "host"})
    assert seen["device"] is None
    assert result["lane"] == "host" and result["execution_error"] == "ValueError: stop after routing"
