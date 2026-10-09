"""Selected exact DAGs prove every intermediate before ordinary allocation.

The target declarations here are synthetic. Source interval proofs establish no
RTL correspondence, physical arithmetic or target runtime correctness.
"""

import copy

import pytest
import test_component_coverage as coverage_fixtures
import test_component_execution_budget as budget_fixtures
import test_component_generation as generation_fixtures
import yaml
from merlin_experiments.phase0 import component_coverage as coverage
from merlin_experiments.phase0 import component_integer_bounds as bounds
from merlin_experiments.phase0 import component_numerics, generation

from merlin.targetgen import golden_store, input_palette

independent = generation_fixtures.independent


def semantics(options, *, overflow, operand="int8", internal=None, elementwise=False):
    contract = yaml.safe_load(options["capability_contract"].read_bytes())
    unit = contract["compute_units"][0]
    unit["dtypes"] = [operand]
    unit["accumulate"] = [{"in": operand, "weight": operand, "acc": "i32"}]
    for declaration in unit["semantic_capabilities"]:
        declaration["dtypes"] = [operand]
    if elementwise:
        contract["compute_units"].append(
            {
                "name": "scalar",
                "kind": "simt",
                "dtypes": ["i32"],
                "ops": ["add"],
                "semantic_capabilities": [{"family": "elementwise_map", "dtypes": ["i32"], "ranks": [2]}],
            }
        )
    generation_fixtures.write(options["capability_contract"], contract)
    coverage_fixtures.update_hardware(options)
    software = yaml.safe_load(options["software_spec"].read_bytes())
    software["numerical_semantics"].update(overflow=overflow, operand_dtype=operand)
    if internal is not None:
        software["numerical_semantics"]["internal_arithmetic"] = internal
    for declaration in software["operations"].values():
        declaration["dtypes"] = [operand]
    if elementwise:
        software["operations"]["elementwise_map"] = {"placement": "accelerator", "dtypes": ["i32"], "ranks": [2]}
    generation_fixtures.write(options["software_spec"], software)


def palette(**values):
    return {
        "schema": input_palette.SCHEMA,
        "inputs": [{"name": name, "axis": "linear", "values": items, "offset": 0} for name, items in values.items()],
    }


def unavailable(options, monkeypatch):
    original_writer = generation._write_capsule

    def check_writer(entry, *args, **kwargs):
        assert entry["op"] != "component_program", "refused numerical source reached its builder"
        return original_writer(entry, *args, **kwargs)

    monkeypatch.setattr(generation, "_write_capsule", check_writer)
    monkeypatch.setattr(component_numerics, "evaluate", lambda *a, **k: pytest.fail("refused reference evaluated"))
    monkeypatch.setattr(input_palette, "realize", lambda *a, **k: pytest.fail("refused palette allocated"))
    with pytest.raises(RuntimeError, match="component coverage"):
        generation.generate_target("fixture", **options)
    report = budget_fixtures.receipt(options)
    row = report["obligations"][0]
    assert row["mandatory"] and row["state"] == "unavailable" and len(row["members"]) == 1
    member = row["members"][0]
    assert not (options["output_root"] / member["requested_member"]).exists()
    return member["reason"]


def test_actual_bounded_dag_publishes_original_complete_oracle(independent):
    obligation = budget_fixtures.contraction(independent)
    semantics(independent, overflow="bounded_exact")
    budget_fixtures.select(independent, [obligation])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    member = report["obligations"][0]["members"][0]
    directory = independent["output_root"] / member["member"]
    capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
    proof = capsule["integer_partial_sum_bound"]
    assert proof["schema"] == bounds.SCHEMA and proof["status"] == "proven_safe"
    assert proof["nodes"][0]["product_interval"] == [0, 9]
    assert proof["nodes"][0]["partial_sum_interval"] == [0, 27]
    from merlin.targetgen.capsule_inputs import materialize_capsule_leaves

    env = materialize_capsule_leaves(capsule)
    expected = [
        [sum(env["A"].data[i * 3 + p] * env["W"].data[p * 2 + j] for p in range(3)) for j in range(2)] for i in range(2)
    ]
    assert golden_store.load_golden(directory)["outputs"] == {"Y": expected}
    assert golden_store.load_golden(directory)["integer_partial_sum_bound"] == proof


def test_reduction_prefix_overflow_refuses_even_when_complete_result_is_zero(independent, monkeypatch):
    obligation = budget_fixtures.contraction(independent, m=1, k=4, n=1)
    semantics(independent, overflow="bounded_exact", operand="int32")
    maximum = (1 << 31) - 1
    weights = [maximum, maximum, -maximum, -maximum]
    assert sum(weights) == 0 and sum(weights[:2]) > maximum
    obligation["base"]["input_palette"] = palette(A=[1], W=weights)
    budget_fixtures.select(independent, [obligation], max_scalar_bits=128)
    reason = unavailable(independent, monkeypatch)
    assert "P: partial sum may overflow signed i32" in reason


def test_declared_narrow_mac_prefix_is_checked_separately_from_wide_result(independent, monkeypatch):
    obligation = budget_fixtures.contraction(independent, m=1, k=6, n=1)
    semantics(
        independent,
        overflow="bounded_exact",
        internal={
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
            "signed_operand_bits": 8,
            "mac_result_bits": 16,
        },
    )
    weights = [127, 127, 127, -127, -127, -127]
    assert sum(127 * value for value in weights) == 0
    assert sum(127 * value for value in weights[:3]) > (1 << 15) - 1
    obligation["base"]["input_palette"] = palette(A=[127], W=weights)
    budget_fixtures.select(independent, [obligation])
    assert "P: partial sum may overflow signed i16" in unavailable(independent, monkeypatch)


def cancellation_graph():
    return {
        "inputs": [
            {"name": name, "role": "input", "shape": [1, 1], "dtype": "accumulator"} for name in ("A", "B", "C")
        ],
        "nodes": [
            {"name": "Overflow", "op": "add", "inputs": ["A", "B"]},
            {"name": "Final", "op": "add", "inputs": ["Overflow", "C"]},
        ],
        "outputs": [{"name": "Y", "value": "Final"}],
    }


def cancellation_obligation(options, *, overflow):
    obligation = budget_fixtures.contraction(options)
    semantics(options, overflow=overflow, elementwise=True)
    obligation["operations"] = ["elementwise_map"]
    obligation["base"]["program"] = cancellation_graph()
    obligation["base"]["input_palette"] = palette(A=[(1 << 31) - 1], B=[1], C=[-1])
    budget_fixtures.select(options, [obligation])
    return obligation


def test_overflowing_node_cannot_be_hidden_by_a_later_cancellation(independent, monkeypatch):
    cancellation_obligation(independent, overflow="bounded_exact")
    assert "Overflow: node result may overflow signed i32" in unavailable(independent, monkeypatch)


def test_same_source_wraps_only_under_explicit_modular_contract(independent):
    cancellation_obligation(independent, overflow="modular_wrap")
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    directory = independent["output_root"] / report["obligations"][0]["members"][0]["member"]
    capsule = yaml.safe_load((directory / "capsule.yaml").read_bytes())
    assert capsule["integer_partial_sum_bound"]["status"] == "explicit_modular_wrap"
    assert golden_store.load_golden(directory)["outputs"] == {"Y": [[(1 << 31) - 1]]}


@pytest.mark.parametrize("overflow", ["unknown", "saturate", "wrap_internal_mac"])
def test_unknown_or_unimplemented_arithmetic_never_defaults_to_modular(independent, monkeypatch, overflow):
    obligation = budget_fixtures.contraction(independent)
    semantics(independent, overflow=overflow)
    budget_fixtures.select(independent, [obligation])
    assert "requires explicit bounded_exact or modular_wrap" in unavailable(independent, monkeypatch)


def test_unknown_declared_partial_width_does_not_grant_exactness(independent, monkeypatch):
    obligation = budget_fixtures.contraction(independent)
    semantics(
        independent,
        overflow="bounded_exact",
        internal={
            "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
            "signed_operand_bits": 8,
        },
    )
    budget_fixtures.select(independent, [obligation])
    assert "unknown signed widths" in unavailable(independent, monkeypatch)


def test_replayed_capsule_cannot_change_exact_gate_or_use_a_stale_proof(independent, monkeypatch):
    obligation = budget_fixtures.contraction(independent)
    semantics(independent, overflow="bounded_exact")
    budget_fixtures.select(independent, [obligation])
    generation.generate_target("fixture", **independent)
    report = coverage.verify_report(independent["output_root"])
    directory = independent["output_root"] / report["obligations"][0]["members"][0]["member"]
    original = yaml.safe_load((directory / "capsule.yaml").read_bytes())
    altered = copy.deepcopy(original)
    altered["numerical_semantics"]["overflow"] = "modular_wrap"
    from merlin_experiments.phase0.component_execution_budget import source_for_capsule

    altered["integer_partial_sum_bound"] = bounds.derive(source_for_capsule(altered), altered["numerical_semantics"])
    with pytest.raises(ValueError, match="protected selected software semantics"):
        bounds.verify_selected_capsule(
            altered, semantics_sha256=report["numerical_semantics_sha256"], require_bound=True
        )
    original["integer_partial_sum_bound"]["nodes"][0]["interval"] = [0, 0]
    monkeypatch.setattr(
        component_numerics, "materialize_capsule_leaves", lambda *a, **k: pytest.fail("stale proof allocated inputs")
    )
    with pytest.raises(ValueError, match="source bound differs"):
        component_numerics.evaluate(original)
