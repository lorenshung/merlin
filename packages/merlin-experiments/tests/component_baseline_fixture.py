"""Synthetic controller boundary for broker tests, never experimental authority.

These tests exercise reporting and pin checks without paying for a fresh author
or running a target. Their baseline admissions are deliberately unissued: the
production verifier rejects them. Tests that need them explicitly install this
fixture, while the admission tests exercise the real verifier separately.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_baseline as CB
from merlin_experiments.phase2.component_runtime import IndependentComponentRuntime, IndependentRuntimeServices
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, sha256_file

from merlin.benchharness import hash_tree


def unissued_baseline(*, baseline, corpus, target_descriptor):
    return CB.ComponentBaselineAdmission(
        Path(baseline), str(hash_tree(baseline)["sha256"]), SimpleNamespace(),
        document_sha256("synthetic qualification"), document_sha256("synthetic fresh origin"),
        corpus, Path(target_descriptor), sha256_file(target_descriptor), object(),
    )


def _diagnostic_only(*_args, **_kwargs):
    raise AssertionError("synthetic runtime never executes a functional compiler")


def unissued_runtime(*, target_descriptor, feature_provider=None, cca_provider=None, rtl_executor=None, source_pins=()):
    services = IndependentRuntimeServices(_diagnostic_only, _diagnostic_only,
                                          feature_provider, cca_provider, rtl_executor)
    roles = tuple(role for role in ("grade", "stage_verifier", "feature_provider", "cca_provider", "rtl_executor")
                  if getattr(services, role) is not None)
    return IndependentComponentRuntime(
        SimpleNamespace(verify_feedback_binding=lambda **_inputs: None,
                        verify_component_binding=lambda _admission: None), services,
        SimpleNamespace(sha256=document_sha256("synthetic hardware")),
        Path(target_descriptor), tuple(source_pins), roles, document_sha256("synthetic runtime qualification"), (),
    )


@pytest.fixture
def synthetic_component_controller(monkeypatch):
    def verify(self, *, baseline=None, corpus=None, target_descriptor=None):
        selected = self.baseline if baseline is None else Path(baseline)
        if selected != self.baseline:
            raise StageGateError("synthetic fixture selected a different baseline")
        if str(hash_tree(selected)["sha256"]) != self.baseline_sha256:
            raise StageGateError("component analytical baseline freeze changed")
        if corpus is not None and corpus != self.corpus:
            raise StageGateError("component corpus bytes changed: synthetic fixture selected another cohort")
        descriptor = self.target_descriptor if target_descriptor is None else Path(target_descriptor)
        if descriptor != self.target_descriptor or sha256_file(descriptor) != self.target_sha256:
            raise StageGateError("component target configuration changed")
        return self.sha256

    monkeypatch.setattr(CB.ComponentBaselineAdmission, "verify", verify)
    def runtime_verify(self, *, required_roles=()):
        if not set(required_roles) <= set(self.qualified_roles):
            raise StageGateError("synthetic runtime has unqualified requested roles")
        return self.sha256

    # Unissued fixtures are only accepted inside these reporting tests.
    monkeypatch.setattr(IndependentComponentRuntime, "verify", runtime_verify)
    return unissued_baseline
