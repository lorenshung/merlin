"""Plain-ISA Spike oracle: routed from the contract, functional tier only, ISA never defaulted."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest

capsule_runner = pytest.importorskip("merlin.targetgen.capsule_runner")
spike_full_call = pytest.importorskip("merlin.targetgen.spike_full_call")


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


def test_explicit_transport_cannot_be_an_arbitrary_candidate_callback():
    with pytest.raises(TypeError, match="selected candidate product transport"):
        spike_full_call.full_call_adapter(
            "t", isa="declared-isa", candidate_transport=SimpleNamespace(compile=lambda: None)
        )


def test_native_binding_requires_every_source_oracle_to_accept_transport():
    def legacy(*args):
        pass

    legacy.compiles_source_bundle = True
    with pytest.raises(NotImplementedError, match="cannot consume explicit"):
        capsule_runner._with_candidate_transport({"L2": legacy}, object())
    with pytest.raises(NotImplementedError, match="no oracle"):
        capsule_runner._with_candidate_transport({}, object())


@pytest.mark.parametrize("build", [{"command": "run-candidate"}, {}, None, []])
def test_source_product_scope_refuses_build_commands_and_other_capsules(build):
    def oracle(*args):
        pass

    oracle._merlin_candidate_transport = object()
    adapters = {"L2": oracle}
    package = SimpleNamespace(manifest={"build": build})
    with pytest.raises(capsule_runner.CertFailure, match="submitted package build commands"):
        capsule_runner.require_candidate_product_scope(package, adapters)
    package.manifest = {}
    with pytest.raises(capsule_runner.CertFailure, match="exported full-call capsules"):
        capsule_runner.require_candidate_product_scope(package, adapters, capsule={"kind": "model"})
    capsule_runner.require_candidate_product_scope(package, adapters)
    package.manifest = {"build": {"command": "legacy-execution"}}
    capsule_runner.require_candidate_product_scope(package, {})


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


@pytest.fixture
def source_adapter(tmp_path, monkeypatch):
    from merlin.llvmlower import device_offload
    from merlin.runtime.backends import spike_model
    from merlin.targetgen import core_aten_capsules, core_aten_device, core_aten_provenance

    provider_path = tmp_path / "provider.py"
    provider_path.write_text("# selected support\n")
    provider = SimpleNamespace(prepare_bundle=lambda *a, **k: None, validate_package=lambda *a, **k: None)
    monkeypatch.setattr(core_aten_device, "load_execution_provider", lambda *a: provider)
    monkeypatch.setattr(core_aten_device, "selected_facts", lambda *a: nullcontext({}))
    monkeypatch.setattr(core_aten_provenance, "batch_provenance", lambda *a, **k: {"scope": "test"})
    builds = []

    def build(*a, **kwargs):
        builds.append(kwargs.get("device"))
        return {"elf": tmp_path / "model.elf", "mem_bytes": 4096, "build_hash": "bound-build"}

    monkeypatch.setattr(spike_model, "build", build)
    monkeypatch.setattr(
        spike_model,
        "run",
        lambda *a, **k: {"console": "done", "output_bytes": [b"data"], "metrics": {"build_hash": "bound-build"}},
    )
    monkeypatch.setattr(device_offload, "load_sidecar", lambda p: {"routed": ["source-bound-call"]})
    monkeypatch.setattr(core_aten_device, "executed_device_instructions", lambda *a: {"executed_instructions": 7})
    monkeypatch.setattr(core_aten_capsules, "load_private", lambda c: {"batch_map": {"cases": [{"case_id": "source"}]}})
    return spike_full_call.full_call_adapter("t", isa="declared-isa", execution_provider=provider_path), builds


def test_selected_transport_validates_inventory_before_host_fallback(tmp_path, monkeypatch, source_adapter):
    from merlin_experiments.phase1.candidate_products import CandidateProductTransport

    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter
    transport = object.__new__(CandidateProductTransport)
    events = []
    prepared = tmp_path / "prepared.mlir"
    prepared.write_text("module {}")
    names = ("catalog_source.mlir", "device_catalog.json", "kernel.gemmini.mlir")
    tools = {"python": "/selected/bin/python", "llvm_bin": "/selected/llvm/bin"}
    monkeypatch.setattr(CandidateProductTransport, "tool_substitutions", lambda self: tools)

    def compile(self, package, source, **kwargs):
        assert self is transport and source == prepared and package == tmp_path
        assert tuple(p.relative_path for p in kwargs["products"]) == names
        assert sum(p.max_bytes for p in kwargs["products"]) <= 256 * 1024 * 1024
        assert kwargs["private_paths"] == (tmp_path,)
        assert kwargs["substitutions"] == tools
        events.append("isolated-inventory")
        return {"synthetic": "API-control only"}

    monkeypatch.setattr(CandidateProductTransport, "compile", compile)

    def validate(directory, *, compile_products, **kwargs):
        assert directory == tmp_path
        compile_products(prepared, tmp_path / "retained", inventory_only=True)
        events.append("validated")

    provider = SimpleNamespace(
        validate_package_candidate_products=validate,
        validate_package=lambda *a, **k: pytest.fail("legacy candidate Python validation was selected"),
        candidate_catalog_products=lambda *, inventory_only: names if inventory_only else (),
        candidate_catalog_command=lambda **kwargs: ("{python}", "-m", "selected.compiler"),
    )
    monkeypatch.setattr(core_aten_device, "load_execution_provider", lambda *a: provider)

    def decline(*args, **kwargs):
        assert callable(kwargs["compile_products"])
        assert events == ["isolated-inventory", "validated"]
        return None, "no eligible contraction"

    monkeypatch.setattr(core_aten_device, "routing_for_bundle", decline)
    bound = adapter._merlin_with_candidate_transport(transport)
    result = bound.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert result["execution_error"] is None
    assert result["lane"] == "host" and builds == [None]
    assert events == ["isolated-inventory", "validated"]


@pytest.mark.parametrize(
    "reason", ["no eligible contraction in captured source", "provider declines captured contraction semantics"]
)
def test_optional_device_decline_runs_real_host_path(tmp_path, monkeypatch, source_adapter, reason):
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: (None, reason))
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert result["execution_error"] is None and result["output_bytes"] == [b"data"]
    assert builds == [None] and result["lane"] == "host" and result["executed_instructions"] == 0
    assert result["routing"]["reason"] == reason


def test_strict_device_decline_still_refuses(tmp_path, monkeypatch, source_adapter):
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: (None, "declined"))
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "device"}
    )
    assert "did not route" in result["execution_error"] and not builds


def test_host_guard_never_invokes_candidate_routing(tmp_path, monkeypatch, source_adapter):
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: pytest.fail("guard must stay host"))
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": False, "lane_expectation": "host-guard"}
    )
    assert result["execution_error"] is None and builds == [None] and result["lane"] == "host"


@pytest.mark.parametrize("failure", ["catalog source hash mismatch", "invalid catalog", "compiler failure"])
def test_optional_device_errors_do_not_fall_back(tmp_path, monkeypatch, source_adapter, failure):
    from merlin.runtime.backends import spike_model
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter

    def refuse(*a, **k):
        raise ValueError(failure)

    if failure == "compiler failure":
        monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: (object(), "bound"))
        monkeypatch.setattr(spike_model, "build", refuse)
    else:
        monkeypatch.setattr(core_aten_device, "routing_for_bundle", refuse)
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert failure in result["execution_error"] and result["output_bytes"] is None and not builds


@pytest.mark.parametrize("instructions", [0, 7])
def test_optional_device_requires_bound_calls_and_executed_instructions(
    tmp_path, monkeypatch, source_adapter, instructions
):
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter
    route = object()
    monkeypatch.setattr(core_aten_device, "routing_for_bundle", lambda *a: (route, "bound"))
    monkeypatch.setattr(
        core_aten_device, "executed_device_instructions", lambda *a: {"executed_instructions": instructions}
    )
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert builds == [route]
    if instructions:
        assert result["execution_error"] is None and result["routing"]["routed"] is True
        assert result["executed_instructions"] == instructions
    else:
        assert "no target instruction" in result["execution_error"] and result["output_bytes"] is None


@pytest.mark.parametrize("validation", [None, "raises", "invalid-return"])
def test_optional_fallback_requires_independent_validation(tmp_path, monkeypatch, source_adapter, validation):
    from merlin.targetgen import core_aten_device

    adapter, builds = source_adapter

    def validate(*a, **k):
        if validation == "raises":
            raise ValueError("malformed unused catalog")
        return True

    provider = SimpleNamespace(validate_package=validate if validation else None)
    monkeypatch.setattr(core_aten_device, "load_execution_provider", lambda *a: provider)
    monkeypatch.setattr(
        core_aten_device, "routing_for_bundle", lambda *a: pytest.fail("validation must precede decline")
    )
    result = adapter.run_full_call(
        bundle=tmp_path, llvm_mlir="", package_dir=tmp_path, capsule={"scored": True, "lane_expectation": "any"}
    )
    assert result["execution_error"] and not builds and result["output_bytes"] is None
