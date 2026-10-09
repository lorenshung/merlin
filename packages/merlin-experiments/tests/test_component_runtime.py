"""Metadata and diagnostic observations cannot issue independent runtime tools."""
from dataclasses import fields
from pathlib import Path
from types import SimpleNamespace

import pytest
from component_baseline_fixture import unissued_runtime
from merlin_experiments.phase2 import component_providers as P
from merlin_experiments.phase2 import component_runtime as R
from merlin_experiments.phase2.contracts import StageGateError


def test_runtime_data_constructor_and_diagnostic_fixture_are_unissued(tmp_path):
    values = {field.name: None for field in fields(R.IndependentComponentRuntime)}
    with pytest.raises(StageGateError, match="evaluated live"):
        R.IndependentComponentRuntime(**values).verify()
    descriptor = tmp_path / "descriptor.yaml"
    descriptor.write_text("target: fixture\n")
    fixture = unissued_runtime(target_descriptor=descriptor)
    with pytest.raises(StageGateError, match="evaluated live"):
        fixture.verify(required_roles=("grade", "stage_verifier"))


def test_runtime_metadata_never_substitutes_for_actual_controls():
    with pytest.raises(StageGateError, match="independent runtime"):
        R.admit_independent_component_runtime(SimpleNamespace(status="qualified", sources_sha256="a" * 64))
    with pytest.raises(StageGateError, match="independent runtime"):
        R.require_independent_runtime({"status": "qualified"}, required_roles=("grade",),
                                      target_descriptor=Path("unused"))


def test_independent_provider_factory_has_no_reference_backend_resolution(monkeypatch):
    from merlin.runtime.backends import base

    monkeypatch.setattr(base, "get_backend", lambda *_: pytest.fail("compiler-provider backend was resolved"))
    with pytest.raises(StageGateError, match="fresh Phase 1 baseline"):
        P.build_independent_component_cca_provider(independent_runtime=object(), baseline_admission=None)
    with pytest.raises(StageGateError, match="fresh Phase 1 baseline"):
        P.build_independent_component_analytical_provider(
            independent_runtime=object(), baseline_admission=None, dependencies={},
        )
