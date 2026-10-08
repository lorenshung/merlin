"""One capture-input tracer bullet; no simulator or headline model is involved."""

from __future__ import annotations

import hashlib
import importlib.util
import json

import pytest
from merlin_experiments.phase0.workload_variants import derive, materialize

from merlin.common.paths import repo_root

ROOT = repo_root()


def test_residual_cnn_capacity_variants_are_bound_and_executable(tmp_path):
    loader = ROOT / "examples/workloads/residual_cnn/loader.py"
    template = ROOT / "examples/gemmini/phase2/iteration-profiles.yaml"
    facts = tmp_path / "facts.json"
    facts.write_text(
        json.dumps(
            {
                "source_consistency": {"status": "verified"},
                "facts": {
                    "target": "gemmini",
                    "arrays": [{"name": "mesh", "cols": 4}],
                    "memories": [{"name": "scratchpad.mem", "bytes": 1024, "elem_bits": 8}],
                },
            }
        )
    )
    output = tmp_path / "variants"
    result = materialize(facts_path=facts, template_path=template, loader_path=loader, output_root=output)
    capacity = result["boundary"]["address_space_bytes"]
    below = result["cases"]["below"]
    above = result["cases"]["above"]
    assert result["boundary"]["live_tensor_roles"] == ["skip_input", "branch_result"]
    assert below["combined_live_tensor_bytes"] < capacity < above["combined_live_tensor_bytes"]
    assert all(case["feature_tensor_bytes"] < capacity for case in (below, above))
    assert all(case["combined_live_tensor_bytes"] == 2 * case["feature_tensor_bytes"] for case in (below, above))
    assert result["loader_sha256"] == hashlib.sha256(loader.read_bytes()).hexdigest()
    for label, case in (("below", below), ("above", above)):
        member = output / label
        profile_bytes = (member / "profile.json").read_bytes()
        manifest = json.loads((member / "variant-manifest.json").read_bytes())
        assert (member / "loader.py").read_bytes() == loader.read_bytes()
        assert manifest["profile_sha256"] == hashlib.sha256(profile_bytes).hexdigest()
        assert manifest["facts_sha256"] == hashlib.sha256(facts.read_bytes()).hexdigest()
        assert manifest["feature_tensor_bytes"] == case["feature_tensor_bytes"]
        assert manifest["combined_live_tensor_bytes"] == case["combined_live_tensor_bytes"]

    torch = pytest.importorskip("torch")
    for label, case in (("below", below), ("above", above)):
        source = output / label / "loader.py"
        spec = importlib.util.spec_from_file_location(f"variant_loader_{label}", source)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        model, (image,) = module.get_model_and_inputs()
        side = case["profile"]["spatial_side"]
        assert tuple(image.shape) == (1, 3, side, side)
        assert model.stem.out_channels == case["profile"]["channels"]
        assert hasattr(model, "conv1") and hasattr(model, "conv2")
        assert len(model.residuals) == 1
        assert model.session_provenance["profile_sha256"] == case["profile_sha256"]
        assert tuple(model(image).shape) == (1, 4)
        assert torch.isfinite(model(image)).all()
        with (output / label / "profile.json").open("rb") as stream:
            malformed = json.load(stream)
        malformed["unrecognized"] = 1
        (output / label / "profile.json").write_text(json.dumps(malformed))
        with pytest.raises(ValueError, match="unsupported shape"):
            module.get_model_and_inputs()


def test_concurrent_variant_requires_named_distinct_roles():
    template = (ROOT / "examples/gemmini/phase2/iteration-profiles.yaml").read_text()
    facts = json.dumps(
        {
            "source_consistency": {"status": "verified"},
            "facts": {
                "target": "gemmini",
                "arrays": [{"name": "mesh", "cols": 16}],
                "memories": [{"name": "scratchpad.mem", "bytes": 262144, "elem_bits": 8}],
            },
        }
    ).encode()
    result = derive(facts, template.encode())
    assert result["cases"]["below"]["profile"]["spatial_side"] == 90
    assert result["cases"]["above"]["profile"]["spatial_side"] == 91
    with pytest.raises(ValueError, match="distinct named live tensors"):
        derive(facts, template.replace("branch_result", "skip_input").encode())
