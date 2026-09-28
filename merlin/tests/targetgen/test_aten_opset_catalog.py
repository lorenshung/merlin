"""Framework catalogs use registered schemas, preserve uncertainty and never cache observations."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

from merlin.targetgen import _aten_opset_worker as worker
from merlin.targetgen import aten_coverage


class _Schema:
    def __init__(self, name, overload_name=""):
        self.name = name
        self.overload_name = overload_name

    def __str__(self):
        return f"{self.name}.{self.overload_name or 'default'}(Tensor input) -> Tensor"


class _LazyAten:
    def __init__(self, **packets):
        self.packets = packets

    def __dir__(self):
        return ["first"]

    def __getattr__(self, name):
        return self.packets[name]


def _torch(monkeypatch, *, table_available=True, tags_available=True):
    core_tag = "core_tag"
    core = SimpleNamespace(tags=[core_tag])
    non_core = SimpleNamespace(tags=[])
    torch = ModuleType("torch")
    torch.__version__ = "test-version"
    torch.Tag = SimpleNamespace(core=core_tag)
    torch.ops = SimpleNamespace(
        aten=_LazyAten(
            first=SimpleNamespace(default=core),
            unlisted=SimpleNamespace(Tensor=non_core if tags_available else object()),
        )
    )
    # ``unlisted`` need not be present in dir(aten) for schema enumeration to find it.
    schemas = [_Schema("aten::first"), _Schema("aten::unlisted", "Tensor"), _Schema("custom::op")]
    torch._C = SimpleNamespace(_jit_get_all_schemas=lambda: schemas + [schemas[0]])
    decompositions = ModuleType("torch._decomp")
    if table_available:
        decompositions.core_aten_decompositions = lambda: {"aten.composite.default": object()}
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "torch._decomp", decompositions)


def test_catalog_keeps_full_registry_core_and_decompositions_separate(monkeypatch):
    _torch(monkeypatch)
    catalog = worker.core_opset()
    assert catalog["schema"] == "merlin.pytorch_opset.v1"
    assert catalog["status"] == "available"
    assert catalog["all_ops"] == ["aten.first.default", "aten.unlisted.Tensor"]
    assert catalog["n_all_aten"] == 2
    assert catalog["ops"] == ["aten.first.default"]
    assert catalog["n_core"] == 1
    assert catalog["decomposed"] == ["aten.composite.default"]
    assert catalog["registered_namespace_counts"] == {"aten": 2, "custom": 1}
    assert catalog["registered_schema_count"] == 3
    assert catalog["support_proven"] is False
    assert catalog["aten_schemas"][1]["overload_name"] == "Tensor"


def test_component_failures_do_not_erase_registered_operators(monkeypatch):
    _torch(monkeypatch, table_available=False, tags_available=False)
    catalog = worker.core_opset()
    assert catalog["status"] == "available"
    assert catalog["n_all_aten"] == 2
    assert catalog["components"]["core"]["status"] == "partial"
    assert catalog["aten_schemas"][1]["core"] is None
    assert catalog["components"]["decompositions"]["status"] == "not_available"
    assert catalog["n_decomposed"] is None


def test_missing_observer_interpreter_is_unknown_and_never_launches(monkeypatch, tmp_path):
    def forbidden_launch(*args, **kwargs):
        raise AssertionError("missing capture environment must not launch a fallback")

    monkeypatch.setattr(aten_coverage.subprocess, "run", forbidden_launch)
    catalog = aten_coverage.observe_opset(python=tmp_path / "missing-python")
    assert catalog["status"] == "not_available"
    assert catalog["n_all_aten"] is None
    assert catalog["n_core"] is None
    assert catalog["diagnostics"]
    assert list(tmp_path.iterdir()) == []
