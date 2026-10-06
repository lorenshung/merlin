"""An integer model capsule is admitted only once every device group it forms is verified in isolation."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import model_qualification as MQ
from merlin_experiments.phase0 import writer

from merlin.targetgen import group_capsule_entries as G

_TE = SimpleNamespace(target="synthetic", sim_via="sim")
_CONTRACT = {
    "name": "synthetic",
    "compute_units": [{"name": "array", "kind": "systolic", "dtypes": ["int8"]}],
    "capabilities": {"mesh": {"rows": 4, "cols": 4}},
    "encoding": {
        "semantic_class": {"a": "MVIN", "b": "COMPUTE", "c": "MVOUT"},
        "corpus_issue_order": ["MVIN", "COMPUTE", "MVOUT"],
    },
}
_FACTS = {"facts": {"datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}]}}
_SEMANTICS = {
    "internal_arithmetic": {
        "signed_operand_bits": 8,
        "mac_result_bits": 20,
        "full_operation_overflow_policy": "bounded_exact_requires_each_partial_sum",
    }
}


def _binding():
    return G.group_binding(
        _TE, {"required_oracle_tiers": ["L0"], "compare": "exact_int"}, contract=_CONTRACT, facts=_FACTS, taxonomy={}
    )


def _row(name, m, k, n, groups):
    entry = {
        "op": "matmul",
        "M": m,
        "K": k,
        "N": n,
        "epilogue": [],
        "operand_dtype": "int8",
        "name": name,
        "cat": "layers",
        "kind": "layer",
        "label": "dev",
        "source_role": G.SOURCE_ROLE,
    }
    return {"name": name, "count": len(groups), "groups": groups, "entry": entry}


def _residual_row(name, groups):
    entry = {
        "op": "residual_add",
        "M": 4,
        "N": 4,
        "epilogue": ["relu"],
        "lhs_scale": 1.0,
        "rhs_scale": 0.5,
        "bound_lsb": 1,
        "operand_dtype": "int8",
        "stimulus_range": [-128, 127],
        "name": name,
        "cat": "layers",
        "kind": "layer",
        "label": "dev",
        "source_role": G.SOURCE_ROLE,
    }
    return {"name": name, "count": len(groups), "groups": groups, "entry": entry}


def _model(tmp_path, stated, monkeypatch, *, semantics=_SEMANTICS, gate=None):
    directory = tmp_path / "capsules" / "model" / "SY_source_fixture"
    directory.mkdir(parents=True)
    (directory / "capsule.interface.mlir").write_text("// fixture program\n")
    monkeypatch.setattr(MQ, "_weight_args", lambda *_: None)
    import merlin.common.mlir_query as mq

    monkeypatch.setattr(mq, "parse", lambda *_: object())
    monkeypatch.setattr(G, "entries", lambda *args, **kwargs: stated)
    entry = {"name": "SY_source_fixture", "kind": "model", "op": "model", "numerical_semantics": semantics}
    if gate is not None:
        entry["reference_gate"] = str(gate)
    cap = {"name": "SY_source_fixture", "kind": "model", "interface_mlir": "capsule.interface.mlir"}
    return entry, cap, directory


def _stated(*rows, unstated=None, groups=None):
    total = sum(r["count"] for r in rows)
    return {
        "entries": list(rows),
        "accelerator_groups": groups if groups is not None else total + sum((unstated or {}).values()),
        "stated": total,
        "unstated": dict(unstated or {}),
    }


def test_without_evidence_the_refusal_is_unchanged(tmp_path, monkeypatch):
    entry, cap, directory = _model(tmp_path, _stated(_row("G_a", 4, 8, 4, [1])), monkeypatch, semantics={})
    with pytest.raises(ValueError) as refused:
        MQ.qualify(entry, cap, directory, _binding(), tmp_path / "capsules")
    assert (
        str(refused.value)
        == MQ.REFUSAL
        == "source-backed integer contraction lacks a verified isolated i8 matmul and exact inputs"
    )
    assert isinstance(refused.value.__cause__, MQ.QualificationMissing)


def test_a_model_whose_groups_are_all_covered_is_admitted_with_its_evidence(tmp_path, monkeypatch):
    stated = _stated(_row("G_a", 4, 8, 4, [1, 3]), _row("G_b", 2, 16, 4, [5]))
    entry, cap, directory = _model(tmp_path, stated, monkeypatch)
    record = MQ.qualify(entry, cap, directory, _binding(), tmp_path / "capsules")
    assert record["status"] == "qualified" and record["device_groups"] == record["covered_groups"] == 3
    assert [row["capsule"] for row in record["group_capsules"]] == ["layers/G_a", "layers/G_b"]
    assert record["reference_gate"] == {"status": "not_declared"}
    for row in record["group_capsules"]:
        capsule = yaml.safe_load((tmp_path / "capsules" / row["capsule"] / "capsule.yaml").read_text())
        assert capsule["label"] == "public"
        golden = yaml.safe_load((tmp_path / "capsules" / row["capsule"] / "golden.yaml").read_text())
        assert golden["integer_partial_sum_bound"]["status"] == "proven_safe"
    # A second model sharing a group reuses the same capsule rather than writing another.
    again = MQ.qualify(entry, cap, directory, _binding(), tmp_path / "capsules")
    assert again["group_capsules"] == record["group_capsules"]


def test_a_residual_add_group_needs_a_falsifiable_full_range_golden_not_a_mac_bound(tmp_path, monkeypatch):
    stated = _stated(_row("G_matmul", 4, 8, 4, [1]), _residual_row("G_residual", [2]))
    entry, cap, directory = _model(tmp_path, stated, monkeypatch)
    record = MQ.qualify(entry, cap, directory, _binding(), tmp_path / "capsules")
    residual = next(row for row in record["group_capsules"] if row["capsule"] == "layers/G_residual")
    assert residual["arithmetic_qualification"] == {
        "kind": "bounded_integer_residual_add",
        "output_step_bound": 1,
    }
    golden = yaml.safe_load((tmp_path / "capsules/layers/G_residual/golden.yaml").read_text())
    assert golden["integer_partial_sum_bound"]["status"] == "not_applicable"
    capsule_path = tmp_path / "capsules/layers/G_residual/capsule.yaml"
    capsule = yaml.safe_load(capsule_path.read_text())
    capsule["stimulus_range"] = [-4, 3]
    capsule_path.write_text(yaml.safe_dump(capsule))
    with pytest.raises(ValueError, match="verified isolated i8 matmul") as refused:
        MQ.qualify(entry, cap, directory, _binding(), tmp_path / "capsules")
    assert "full-domain sampled" in str(refused.value.__cause__)


def test_same_shape_residual_groups_with_distinct_scales_keep_distinct_goldens(tmp_path, monkeypatch):
    first = _residual_row("G_residual", [1])
    second = _residual_row("G_residual", [2])
    second["entry"]["rhs_scale"] = 0.25
    entry, cap, directory = _model(tmp_path, _stated(first, second), monkeypatch)
    record = MQ.qualify(entry, cap, directory, _binding(), tmp_path / "capsules")
    paths = [row["capsule"] for row in record["group_capsules"]]
    assert len(set(paths)) == 2
    assert sorted(
        yaml.safe_load((tmp_path / "capsules" / path / "capsule.yaml").read_text())["operation"]["attributes"][
            "rhs_scale"
        ]
        for path in paths
    ) == [0.25, 0.5]
    assert (
        MQ.qualify(entry, cap, directory, _binding(), tmp_path / "capsules")["group_capsules"]
        == record["group_capsules"]
    )
    altered = tmp_path / "capsules" / paths[0] / "capsule.yaml"
    capsule = yaml.safe_load(altered.read_text())
    capsule["operation"]["attributes"]["rhs_scale"] = 0.75
    altered.write_text(yaml.safe_dump(capsule))
    with pytest.raises(ValueError, match="verified isolated i8 matmul") as refused:
        MQ.qualify(entry, cap, directory, _binding(), tmp_path / "capsules")
    assert "differs from the requested numerical program" in str(refused.value.__cause__)


def test_real_group_enumeration_preserves_numerical_variants_for_qualification(monkeypatch):
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import group_command as GC

    groups = [SimpleNamespace(placement="accelerator", root=object(), index=index) for index in (1, 2)]
    monkeypatch.setattr(CG, "form_groups", lambda *_args, **_kwargs: groups)

    def program(group, *, weight_args=None):
        entry = _residual_row("G_residual", [group.index])["entry"]
        entry["rhs_scale"] = 0.5 if group.index == 1 else 0.25
        return SimpleNamespace(entry=entry, to_dict=lambda: {})

    monkeypatch.setattr(GC, "program", program)
    assert G.entries("synthetic", object(), with_raw=False)["distinct"] == 1
    selected = G.entries("synthetic", object(), with_raw=False, numerical_variants=True)
    assert selected["accelerator_groups"] == selected["stated"] == selected["distinct"] == 2
    assert {row["entry"]["rhs_scale"] for row in selected["entries"]} == {0.25, 0.5}


def test_one_uncovered_group_refuses(tmp_path, monkeypatch):
    stated = _stated(_row("G_a", 4, 8, 4, [1]), unstated={"a stage with no epilogue name": 1})
    entry, cap, directory = _model(tmp_path, stated, monkeypatch)
    with pytest.raises(ValueError, match="verified isolated i8 matmul") as refused:
        MQ.qualify(entry, cap, directory, _binding(), tmp_path / "capsules")
    assert "cannot state" in str(refused.value.__cause__)
    overflowing = _stated(_row("G_deep", 4, 1 << 16, 4, [1]))
    entry, cap, directory = _model(tmp_path / "deep", overflowing, monkeypatch)
    entry["numerical_semantics"] = {"internal_arithmetic": {**_SEMANTICS["internal_arithmetic"], "mac_result_bits": 4}}
    with pytest.raises(ValueError, match="verified isolated i8 matmul"):
        MQ.qualify(entry, cap, directory, _binding(), tmp_path / "deep" / "capsules")


@pytest.mark.parametrize(
    "gate,admitted",
    [
        ({"words": {"dispatches": 1551, "differ": 0}}, True),
        ({"words": {"dispatches": 1551, "differ": 3}}, False),
        ({"mismatches": 0, "of": 0}, False),
        (None, False),
    ],
)
def test_a_declared_reference_gate_must_record_zero_mismatches(tmp_path, monkeypatch, gate, admitted):
    path = tmp_path / "gate.json"
    if gate is not None:
        path.write_text(json.dumps(gate))
    entry, cap, directory = _model(tmp_path, _stated(_row("G_a", 4, 8, 4, [1])), monkeypatch, gate=path)
    if admitted:
        record = MQ.qualify(entry, cap, directory, _binding(), tmp_path / "capsules")
        assert record["reference_gate"]["status"] == "exact" and record["reference_gate"]["dispatches"] == 1551
    else:
        with pytest.raises(ValueError, match="verified isolated i8 matmul"):
            MQ.qualify(entry, cap, directory, _binding(), tmp_path / "capsules")


def test_a_non_model_source_capsule_keeps_the_isolated_matmul_rule():
    with pytest.raises(ValueError, match="verified isolated i8 matmul"):
        writer._source_integer_reference_bound(
            {"kind": "model", "numerical_semantics": _SEMANTICS}, {"operation": {"op": "model"}}, None
        )
