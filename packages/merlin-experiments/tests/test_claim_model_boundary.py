"""Held-out evaluation models cannot enter a public Phase 0/1 candidate payload."""

import pytest
import yaml
from merlin_experiments.phase0.claim_boundary import assert_no_claim_capsules


@pytest.mark.parametrize(
    "entry",
    [
        {"cat": "model", "name": "SY_model_tiny_llama", "model": "tiny_llama"},
        {"cat": "model", "name": "other_name", "model": "resnet50_v1_5"},
        {"cat": "model", "name": "other_name", "loader": "workloads/resnet50_v1_5/loader.py"},
    ],
)
def test_claim_model_source_is_rejected_even_when_renamed(entry):
    with pytest.raises(ValueError, match="held-out claim model"):
        assert_no_claim_capsules([entry], ["resnet50", "tiny_llama"])


def test_derivation_model_with_overlapping_word_is_allowed():
    assert_no_claim_capsules(
        [{"cat": "model", "name": "M0_small_llama", "model": "small_llama"}],
        ["resnet50", "tiny_llama"],
    )


def test_manifest_records_count_only_owner_obligation(tmp_path):
    from merlin_experiments.phase0.provenance import update_provenance_manifest

    obligation = {
        "schema": "claim_model_evaluation_v1",
        "source": "workload_spec.models",
        "model_count": 2,
        "visibility": "owner_only_after_phase1_freeze",
        "public_capsules_emitted": 0,
        "status": "awaiting_phase1_freeze",
    }
    path = update_provenance_manifest(
        [],
        cap_root=tmp_path,
        target="sample_target",
        performance_record={"phase": {"category": "_perf"}},
        unbuilt_roster=[],
        claim_model_evaluation=obligation,
    )
    manifest = yaml.safe_load(path.read_text())
    assert manifest["claim_model_evaluation"]["sample_target"] == obligation
    assert manifest["phase_corpora"]["sample_target"]["phase1"]["generated_members"] == []


# --- claim-model objective variants ---------------------------------------------------------------


def _gemmini_descriptor():
    from merlin.common.paths import repo_root
    from merlin.targetgen.target_experiment import load_target_experiment

    return load_target_experiment(repo_root() / "examples/gemmini/target/descriptor.yaml")


def test_the_declared_objective_variants_are_capture_capabilities_of_a_claim_model():
    from merlin_experiments.phase0.claim_boundary import claim_objective_variants

    te = _gemmini_descriptor()
    claims = set(te.workload_spec["models"])
    variants = claim_objective_variants(te)
    assert variants, "the example declares its claim-model objective variants"
    for entry in variants:
        assert entry["model"] in claims
        assert entry["quant_activation_contractions"] is True
        assert entry["quant_scheme"] == "int8_dyn_act_int8_weight"
    by_variant = {entry["objective_variant"]: entry for entry in variants}
    assert set(by_variant) == {"int8attn", "int8full"}
    assert by_variant["int8full"]["quant_integer_nonlinear"] is True
    assert "quant_integer_nonlinear" not in by_variant["int8attn"]


def test_an_objective_variant_is_never_a_derivation_source():
    """Every variant names its claim model, so the boundary refuses it -- renamed or not."""
    from merlin_experiments.phase0.claim_boundary import claim_objective_variants, held_out_models

    te = _gemmini_descriptor()
    held = held_out_models(te)
    for entry in claim_objective_variants(te):
        with pytest.raises(ValueError, match="held-out claim model"):
            assert_no_claim_capsules([entry], held)
        with pytest.raises(ValueError, match="held-out claim model"):
            assert_no_claim_capsules([{**entry, "name": "MF_" + entry["name"]}], held)
        with pytest.raises(ValueError, match="held-out claim model"):
            assert_no_claim_capsules([{"cat": "model", "name": "x", "model_form": {"model": entry["model"]}}], held)


def test_no_derived_claim_model_form_is_tracked():
    """Equivalent forms come from an application, never from a claim model's capture."""
    from merlin_experiments.phase0.claim_boundary import held_out_models

    from merlin.common.paths import repo_root

    held = held_out_models(_gemmini_descriptor())
    root = repo_root() / "merlin" / "contract" / "capsules"
    derived = [path.name for path in root.rglob("MF_*") if path.is_dir()]
    for name in derived:
        assert_no_claim_capsules([{"name": name}], held)


@pytest.mark.parametrize(
    ("declared", "why"),
    [
        ({"not_a_claim": {"v": {"quant_activation_contractions": True}}}, "not a claim model"),
        ({"m": {"v": {"loader": "x.py"}}}, "only"),
        ({"m": {"v": {"quant_integer_nonlinear": True}}}, "without activation contractions"),
        ({"m": {}}, "declares no objective variants"),
    ],
)
def test_a_malformed_variant_declaration_is_refused(declared, why):
    from types import SimpleNamespace

    from merlin_experiments.phase0.claim_boundary import claim_objective_variants

    te = SimpleNamespace(workload_spec={"models": ["m"], "claim_objective_variants": declared})
    with pytest.raises(ValueError, match=why):
        claim_objective_variants(te)
