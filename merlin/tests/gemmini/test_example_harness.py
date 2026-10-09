"""Fresh experiment sources contain no copied reference runtime or kernel helpers."""

import json

import pytest
from merlin_experiments.phase1 import component_witness

from merlin.common.paths import repo_root
from merlin.targetgen.target_experiment import load_target_experiment

pytestmark = pytest.mark.target("gemmini")


def test_reference_provider_and_headers_are_absent():
    root = repo_root()
    for relative in (
        "examples/gemmini/support",
        "examples/gemmini/SOURCE.yaml",
        "examples/gemmini/phase1/contracts/harness_curated",
        "merlin/experiments/capsule_bench/targets/gemmini/contracts/harness_curated",
        "merlin/experiments/capsule_bench/targets/gemmini_universal/contracts/harness_curated",
        "merlin/experiments/capsule_bench/targets/gemmini_universal/contracts/isa_include",
    ):
        path = root / relative
        assert not path.exists() and not path.is_symlink(), relative


def test_reference_runtime_is_not_registered_or_selected_as_evidence():
    root = repo_root()
    document = json.loads((root / "build_tools/upstreams/target_support.json").read_bytes())
    assert all(row["target"] != "gemmini" for row in document["companions"])
    descriptor = load_target_experiment(root / "examples/gemmini/target/descriptor.yaml")
    assert descriptor.isa_headers == ()
    assert descriptor.hwbringup_set is None
    assert descriptor.curated_harness is None
    assert not hasattr(component_witness, "selected_component_witness_verifier")
