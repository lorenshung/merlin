"""The target edge supplies source-bound backend hooks and simulator options."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.core_aten_device import load_execution_provider, selected_facts


def test_runner_settings_follow_the_selected_spike_installation(monkeypatch):
    from merlin.runtime.backends import spike

    provider = load_execution_provider("gemmini")
    monkeypatch.setattr(spike, "spike_path", lambda: "/selected/env/riscv-tools/bin/spike")
    options = provider.runner_options()
    assert options["extension"] == "gemmini"
    assert options["extlib"] == Path("/selected/env/riscv-tools/lib/libgemmini.so")
    assert options["path_prepend"] == [Path("/selected/env/bin")]


def test_provider_binds_catalog_and_derives_precision_from_selected_facts(tmp_path, monkeypatch):
    provider = load_execution_provider("gemmini")
    (tmp_path / "model.mlir").write_text("caller source")

    def builder(*args):
        return None

    backend = SimpleNamespace(
        build_catalog=lambda source: (None, {"covered_contractions": 1}), merlin_builder=lambda llvm_bin: builder
    )
    monkeypatch.setattr(provider, "load_module", lambda *a, **kw: backend)
    fixture = repo_root() / "merlin/tests/fixtures/core_aten_device/facts.json"
    eligible = [(None, SimpleNamespace(dtypes=("i8", "i8", "i32")))]
    with selected_facts("synthetic_core_aten", fixture) as document:
        route = provider.routing(
            tmp_path, target="synthetic_core_aten", package=tmp_path, facts=document, eligible=eligible
        )
        assert (route.operand_dtype, route.accum_dtype) == ("i8", "i32")
        assert route.select is None and route.catalog_builder is builder
        with pytest.raises(ValueError, match="underivable"):
            provider.routing(
                tmp_path,
                target="synthetic_core_aten",
                package=tmp_path,
                facts=document,
                eligible=[(None, SimpleNamespace(dtypes=("f32", "f32", "f32")))],
            )
        backend.build_catalog = lambda source: (None, {"covered_contractions": 0})
        assert (
            provider.routing(
                tmp_path, target="synthetic_core_aten", package=tmp_path, facts=document, eligible=eligible
            )
            is None
        )


def test_other_targets_cannot_load_this_provider():
    path = repo_root() / "examples/gemmini/phase0/core_aten/execution_provider.py"
    with pytest.raises(ValueError, match="selected target"):
        load_execution_provider("synthetic_core_aten", path)
