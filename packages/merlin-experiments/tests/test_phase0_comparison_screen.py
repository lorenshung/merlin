"""A comparison family whose unfused part the declarations refuse is withdrawn at derivation."""

from __future__ import annotations

from types import SimpleNamespace

from merlin_experiments.phase0 import comparison_screen
from merlin_experiments.phase0.sweeps import expand_sweeps

from merlin.targetgen import corpus_spec as CS
from merlin.targetgen.software_spec import validate_software_spec

_SPEC = {
    "schema": "merlin.software_spec.v1",
    "target": "fixture",
    "status": "reviewed",
    "numerical_semantics": {
        "model": {"engine": "integer_reference"},
        "operand_dtype": "int8",
        "accumulator_dtype": "i32",
        "readout_dtype": "i32",
        "subnormal_operand_flush": False,
        "overflow": "wrap_internal_mac",
    },
    "operations": {
        "contraction": {"placement": "accelerator", "operand_dtypes": ["int8"], "accumulator_dtype": "i32"},
        "elementwise_map": {
            "placement": "fused_accelerator",
            "dtypes": ["int8", "i32"],
            "composed_with": ["contraction"],
            "epilogues": ["relu", "acc_scale", "bias"],
            "scale_granularity": ["tensor"],
        },
    },
}


def _spec(**override):
    return validate_software_spec(dict(_SPEC, **override), "fixture")


def _comparison_sweep():
    """One neutral tiled comparison, with its complete performance claim contract."""
    return {
        "id": "PF",
        "fit_axes": ["K"],
        "comparison_roles": ["fused", "part"],
        "base": {
            "cat": "_perf",
            "kind": "model_slice",
            "out": "Y0",
            "performance": {
                "level": "L5_fusion",
                "family": "PF",
                "lever": "epilogue_fusion",
                "member_class": "OBJECTIVE",
                "claim": "DIFFERENTIAL",
                "comparand": {
                    "kind": "group_arithmetic",
                    "against": "sum_of_parts",
                    "cancels": ["shape"],
                    "demand_equal": ["M", "N"],
                },
                "falsifier": {
                    "observation": "differential_cycles",
                    "fires_when": "fused_not_faster",
                    "negative_control": "separate_parts",
                },
                "gate": {
                    "traits": ["structural_pipeline_depth"],
                    "instrument": "cycle_count",
                    "capacity": "complete_comparison_group",
                    "on_missing": "skip_with_evidence",
                },
                "regime": {"separation": "same_shape", "layout": "shared_inputs"},
                "emitter": {
                    "status": "existing",
                    "entry": "merlin.targetgen.corpus_spec.build",
                    "knobs": {"members": ["fused_matmul_bias", "matmul", "bias_add"]},
                },
                "cost": {
                    "tier": 1,
                    "runs": "three_per_shape",
                    "projected_cycles": "preflight",
                    "basis": "fused_plus_parts",
                },
            },
        },
        "variants": [
            {"op": "fused_matmul_bias", "comparison_group": {"name": "g", "role": "fused"}},
            {"op": "matmul", "comparison_group": {"name": "g", "role": "part"}},
            {"op": "bias_add", "comparison_group": {"name": "g", "role": "part"}},
        ],
        "axes": {"M": ["tile"], "N": ["tile"], "K": ["tile", "2*tile"]},
        "name": "{id}{i:02d}_{op}_m{M}k{K}n{N}",
    }


def test_a_fused_only_part_withdraws_its_group():
    binding = SimpleNamespace(operand_dtype="int8", accum_dtype="i32")
    variants = [
        {"op": "fused_matmul_bias", "comparison_group": {"name": "g", "role": "fused"}},
        {"op": "matmul", "comparison_group": {"name": "g", "role": "part"}},
        {"op": "bias_add", "comparison_group": {"name": "g", "role": "part"}},
    ]
    refusal = comparison_screen.refused_part(variants, {}, software_spec=_spec(), binding=binding)
    assert refusal["member"] == "bias_add" and "composition" in refusal["reason"]
    standalone = dict(_SPEC["operations"]["elementwise_map"])
    standalone.pop("composed_with")
    permissive = _spec(operations={**_SPEC["operations"], "elementwise_map": standalone})
    assert comparison_screen.refused_part(variants, {}, software_spec=permissive, binding=binding) is None
    # Without selected declarations nothing is withdrawn.
    assert comparison_screen.refused_part(variants, {}, software_spec=None, binding=binding) is None


def test_unrelated_typed_host_rules_do_not_rescue_an_unbuildable_comparison_part(monkeypatch):
    """A partial entry screen cannot give a different operator an unknown host lane."""
    unrelated = {
        "placement": "host",
        "ops": ["aten.add.Tensor"],
        "families": ["elementwise_map"],
        "ordered_operand_dtypes": ["i64", "i64", "i64"],
        "ordered_result_dtypes": ["i64"],
        "ranks": [1],
    }
    spec = _spec(operations={**_SPEC["operations"], "host_integer_add": unrelated})
    sweep = _comparison_sweep()
    binding = CS.CorpusBinding(
        target="fixture",
        tile_dim=4,
        operand_dtype="int8",
        accum_dtype="i32",
        integer=True,
        tiers=["L0", "L1", "L2", "L3"],
        compare="exact_int",
        classes_for=lambda **_: ["CONTRACTION"],
    )
    fact = {"satisfied": True, "tier": "fixture", "evidence": "fixture fact", "missing": []}
    evidence = SimpleNamespace(
        target="fixture",
        software_spec=spec,
        contract={
            "name": "fixture",
            "compute_units": [
                {
                    "name": "array",
                    "kind": "systolic",
                    "dtypes": ["int8"],
                    "ops": ["matmul"],
                    "semantic_capabilities": [
                        {"family": "contraction", "dtypes": ["int8"], "ranks": [2, 3, 4]},
                        {
                            "family": "elementwise_map",
                            "dtypes": ["int8"],
                            "result_dtypes": ["i32"],
                            "composed_with": ["contraction"],
                        },
                    ],
                }
            ],
        },
        host_capabilities={
            "profile": {
                "capability_spec": {
                    "schema": "merlin.host_capabilities.v1",
                    "status": "reviewed",
                    "compiler": {"package_sha256": "a" * 64, "dtype_strategy": "lane"},
                    "operations": [
                        {
                            "id": "integer_add",
                            "ops": ["aten.add.Tensor"],
                            "placement": "host",
                            "signature": {
                                "ordered_operand_dtypes": ["i64", "i64", "i64"],
                                "ordered_result_dtypes": ["i64"],
                                "ranks": [1],
                            },
                        }
                    ],
                    "evidence": {},
                },
                "package_sha256": "a" * 64,
                "capability_spec_sha256": "b" * 64,
                "dtype_strategy": "lane",
            }
        },
        performance_facts={"traits": {"structural_pipeline_depth": fact}, "execution_capabilities": {}},
        readout_facets=[],
    )
    from merlin.targetgen import target_experiment

    monkeypatch.setattr(
        target_experiment,
        "load_capability_manifest",
        lambda _target: SimpleNamespace(contract={"runner": {"tier_sim": {"L2": "spike", "L3": "verilator"}}}),
    )
    skipped = []
    entries = expand_sweeps(
        {"capsules": [], "sweeps": [sweep], "_performance_oracles": {"L2": "spike", "L3": "verilator"}},
        binding,
        evidence=evidence,
        trait_facts=evidence.performance_facts,
        skipped=skipped,
    )
    assert entries == []
    assert len(skipped) == 1 and skipped[0]["family"] == "PF"
    assert "bias_add" in skipped[0]["reason"]
    assert skipped[0]["refused_part"]["basis"] == "builder_emitted_typed_interface"

    # Lack of reviewed declarations is UNKNOWN, not proof that a whole family is inapplicable.
    original_host = evidence.host_capabilities
    evidence.software_spec = None
    evidence.host_capabilities = None
    skipped = []
    entries = expand_sweeps(
        {"capsules": [], "sweeps": [sweep], "_performance_oracles": {"L2": "spike", "L3": "verilator"}},
        binding,
        evidence=evidence,
        trait_facts=evidence.performance_facts,
        skipped=skipped,
    )
    assert len(entries) == 6 and skipped == [], skipped

    # A separately reviewed exact standalone lane also retains the whole comparison.
    standalone = {
        "placement": "host",
        "ops": ["bias_add"],
        "families": ["elementwise_map"],
        "ordered_operand_dtypes": ["i32", "i32"],
        "ordered_result_dtypes": ["i32"],
        "ranks": [2],
    }
    evidence.software_spec = _spec(operations={**_SPEC["operations"], "host_bias_add": standalone})
    evidence.host_capabilities = original_host
    evidence.host_capabilities["profile"]["capability_spec"]["operations"].append(
        {
            "id": "bias_add",
            "ops": ["merlin_iface.bias_add"],
            "placement": "host",
            "signature": {"ordered_operand_dtypes": ["i32", "i32"], "ordered_result_dtypes": ["i32"], "ranks": [2]},
        }
    )
    skipped = []
    entries = expand_sweeps(
        {"capsules": [], "sweeps": [sweep], "_performance_oracles": {"L2": "spike", "L3": "verilator"}},
        binding,
        evidence=evidence,
        trait_facts=evidence.performance_facts,
        skipped=skipped,
    )
    assert len(entries) == 6 and skipped == [], skipped
