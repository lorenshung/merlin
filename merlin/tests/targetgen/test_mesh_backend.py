"""Whole-model simulator selection reuses a stable policy decision per process."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from merlin.compile import mesh_backend as backend
from merlin.targetgen import oracle_policy


@pytest.fixture(autouse=True)
def clear_simulator_cache():
    backend._MESH_SIM_CACHE.clear()
    yield
    backend._MESH_SIM_CACHE.clear()


def test_repeated_layers_probe_once_even_with_concurrent_requests(monkeypatch):
    monkeypatch.delenv("MERLIN_MESH_SIM", raising=False)
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    calls = []
    monkeypatch.setattr(
        oracle_policy, "select_chipyard_engine", lambda target: calls.append(target) or {"engine": "fixture"}
    )

    # Coarse 128-layer proxy for a model with repeated contraction groups.
    with ThreadPoolExecutor(max_workers=8) as pool:
        selected = list(pool.map(backend._resolve_oot_mesh_simulator, ["model"] * 128))
    assert selected == ["fixture"] * 128
    assert calls == ["model"]  # 128 policy probes before caching, one after


def test_selection_key_includes_target_and_requirement(monkeypatch):
    monkeypatch.delenv("MERLIN_MESH_SIM", raising=False)
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    calls = []
    monkeypatch.setattr(
        oracle_policy, "select_chipyard_engine", lambda target: calls.append(target) or {"engine": "fixture"}
    )
    assert backend._resolve_oot_mesh_simulator("a") == "fixture"
    assert backend._resolve_oot_mesh_simulator("b") == "fixture"
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "fixture")
    assert backend._resolve_oot_mesh_simulator("a") == "fixture"
    assert calls == ["a", "b", "a"]


def test_failed_selection_is_not_cached(monkeypatch):
    monkeypatch.delenv("MERLIN_MESH_SIM", raising=False)
    monkeypatch.setenv("MERLIN_REQUIRED_RTL_ENGINE", "expected")
    calls = []
    monkeypatch.setattr(
        oracle_policy, "select_chipyard_engine", lambda target: calls.append(target) or {"engine": "other"}
    )
    for _ in range(2):
        with pytest.raises(RuntimeError, match="differs from required"):
            backend._resolve_oot_mesh_simulator("a")
    assert calls == ["a", "a"]
    assert backend._MESH_SIM_CACHE == {}


def test_explicit_unpinned_simulator_skips_policy(monkeypatch):
    monkeypatch.delenv("MERLIN_REQUIRED_RTL_ENGINE", raising=False)
    monkeypatch.setattr(
        oracle_policy, "select_chipyard_engine", lambda target: pytest.fail("explicit request must not probe")
    )
    assert backend._resolve_oot_mesh_simulator("model", "explicit") == "explicit"
    assert backend._MESH_SIM_CACHE == {}
