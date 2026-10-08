"""Selection remains core-only; evaluation keeps one mutable plugin registry."""

import os
import subprocess
import sys

import pytest

from merlin.common.paths import python_import_roots


def test_core_selection_and_capsule_io_do_not_import_optional_evaluation():
    code = """
import importlib.abc, sys
class BlockEvaluation(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "aet" or fullname.startswith("aet.") or fullname in {
            "merlin.targetgen.capsule_runner", "merlin.targetgen.capsule_grade",
            "merlin.targetgen.capsule_golden", "merlin.targetgen.package_certification",
            "merlin.targetgen.sandbox",
        }:
            raise ModuleNotFoundError("evaluation deliberately unavailable", name=fullname)
sys.meta_path.insert(0, BlockEvaluation())
from merlin.targetgen import oracle_policy as policy, capsule_common, runtime_build
from merlin.compile import mesh_backend, mesh
policy.chipyard_l3_selection = lambda target: {"engine": "synthetic"}
assert mesh_backend._resolve_oot_mesh_simulator("synthetic") == "synthetic"
policy.register_sim_oracle("fixture", adapters=lambda target: {}, available=lambda target: (True, ""),
                           exclusive=True, has_memmap=True)
assert policy.sim_oracle_caps("fixture").has_memmap
assert policy.selected_screen_tiers(None) == ()
assert not any(name == "aet" or name.startswith("aet.") for name in sys.modules)
assert "merlin.targetgen.capsule_runner" not in sys.modules
"""
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(str(path) for path in python_import_roots()))
    environment.pop("MERLIN_REQUIRED_RTL_ENGINE", None)
    environment.pop("MERLIN_MESH_SIM", None)
    subprocess.run([sys.executable, "-c", code], env=environment, check=True, capture_output=True, text=True)


def test_legacy_registry_class_and_registration_have_one_identity(monkeypatch):
    from merlin.targetgen import capsule_runner as legacy
    from merlin.targetgen import oracle_policy as policy

    assert legacy._SIM_ORACLES is policy._SIM_ORACLES
    assert legacy._SimOracle is policy._SimOracle
    assert legacy.register_sim_oracle is policy.register_sim_oracle
    assert legacy._ensure_sim_oracles_discovered is policy._ensure_sim_oracles_discovered
    sentinel = policy._SimOracle(lambda target: {}, lambda target: (False, "unavailable"), True)
    monkeypatch.setitem(legacy._SIM_ORACLES, "identity-test", sentinel)
    assert policy.sim_oracle_caps("identity-test") is sentinel
    monkeypatch.setattr(legacy, "_sim_engine_adapters", lambda engine, target: {"test": (engine, target)})
    assert policy._SIM_ORACLES["chipyard"].adapters("target") == {"test": ("chipyard", "target")}


def test_legacy_selector_overrides_still_reach_compiler(monkeypatch):
    from merlin.compile import mesh_backend
    from merlin.targetgen import capsule_runner as legacy
    from merlin.targetgen import oracle_policy as policy

    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    monkeypatch.delenv("MERLIN_MESH_SIM", raising=False)
    monkeypatch.setattr(legacy, "chipyard_l3_selection", lambda target: {"engine": "selected-by-test"})
    monkeypatch.setattr(legacy, "_bespoke_sim_via", lambda target: "registered-by-test")
    assert mesh_backend._resolve_oot_mesh_simulator("synthetic") == "selected-by-test"
    assert policy.selected_sim_via("synthetic") == "registered-by-test"


def test_evaluators_and_worker_live_only_in_experiments_distribution():
    from merlin.common.paths import module_source_path, python_source_dir

    names = (
        "capsule_runner",
        "capsule_grade",
        "capsule_golden",
        "trace_check",
        "coverage_report",
        "_capsule_bundle_worker",
        "package_certification",
        "sandbox",
    )
    for name in names:
        source = module_source_path("merlin.targetgen." + name)
        assert "merlin-experiments" in source.parts
        assert not (python_source_dir() / "merlin/targetgen" / (name + ".py")).exists()
    runner = module_source_path("merlin.targetgen.capsule_runner")
    assert runner.with_name("_capsule_bundle_worker.py").is_file()


def test_chipyard_preflight_uses_declared_provider_options(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin.runtime.backends import base
    from merlin.targetgen import oracle_policy as policy

    binary = tmp_path / "spike"
    binary.write_text("fixture")
    binary.chmod(0o755)
    library = tmp_path / "isolated.so"
    library.write_bytes(b"model")
    provider = SimpleNamespace(
        runner_options=lambda: {
            "spike_binary": binary,
            "extension": "fixture",
            "extlib": library,
        }
    )
    monkeypatch.setattr(policy, "selected_full_call_provider", lambda target: provider)
    monkeypatch.setattr(base, "get_backend", lambda target: pytest.fail("provider must own selection"))
    ok, reason = policy._chipyard_available("synthetic")
    assert ok
    assert str(library) in reason
    library.unlink()
    ok, reason = policy._chipyard_available("synthetic")
    assert not ok
    assert "extension is absent" in reason
    library.write_bytes(b"model")
    binary.chmod(0o644)
    assert not policy._chipyard_available("synthetic")[0]


def test_chipyard_broken_provider_refuses_arc_fallback(monkeypatch):
    from merlin.targetgen import oracle_policy as policy
    from merlin.targetgen.rtl import mlc_bridge

    def broken(target):
        raise ValueError("selected model unavailable")

    monkeypatch.setattr(policy, "selected_full_call_provider", broken)
    monkeypatch.setattr(mlc_bridge, "arc_available", lambda target: True)
    ok, reason = policy._chipyard_available("synthetic")
    assert not ok
    assert "selected model unavailable" in reason


@pytest.mark.parametrize("spike,arc,expected", [(True, False, True), (False, True, True), (False, False, False)])
def test_chipyard_without_provider_preserves_backend_fallback(monkeypatch, spike, arc, expected):
    from types import SimpleNamespace

    from merlin.runtime.backends import base
    from merlin.targetgen import oracle_policy as policy
    from merlin.targetgen.rtl import mlc_bridge

    monkeypatch.setattr(policy, "selected_full_call_provider", lambda target: None)
    monkeypatch.setattr(base, "get_backend", lambda target: SimpleNamespace(available=lambda engine: spike))
    monkeypatch.setattr(mlc_bridge, "arc_available", lambda target: arc)
    assert policy._chipyard_available("synthetic")[0] is expected


@pytest.mark.parametrize("absolute", [False, True])
def test_full_call_provider_resolution_is_shared_with_grading(tmp_path, monkeypatch, absolute):
    from types import SimpleNamespace

    from merlin.targetgen import oracle_policy as policy
    from merlin.targetgen import plugins, target_experiment

    source = tmp_path / "support/model.py"
    reference = str(source if absolute else source.relative_to(tmp_path))
    monkeypatch.setattr("merlin.common.paths.repo_root", lambda: tmp_path)
    monkeypatch.setattr(
        target_experiment,
        "load_capability_manifest",
        lambda target: SimpleNamespace(contract={"runner": {"full_call_provider": reference}}),
    )
    seen = []
    provider = object()

    def load(root, name, *, package_name):
        seen.append((root, name, package_name))
        return provider

    monkeypatch.setattr(plugins, "load_module", load)
    assert policy.selected_full_call_provider("synthetic") is provider
    import hashlib

    identity = hashlib.sha256(str(source.parent.resolve()).encode()).hexdigest()
    assert seen == [(source.parent, source.name, f"full_call_execution_{identity}")]
