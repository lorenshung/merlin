import pytest

from merlin.targetgen.core_aten_prune import apply_ledger, load_ledger


def _case(case_id, overload, dtype, layout, obligations):
    return {
        "case_id": case_id,
        "overload": overload,
        "partition_assignment": {"dtype": dtype, "layout": layout},
        "covered_obligations": obligations,
    }


def _suite():
    cases = [
        _case(
            "a",
            "aten.add.Tensor",
            "float32",
            "contiguous",
            ["overload::aten.add.Tensor", "single::aten.add.Tensor::dtype=float32"],
        ),
        _case(
            "b",
            "aten.add.Tensor",
            "int64",
            "zero_stride",
            [
                "single::aten.add.Tensor::dtype=int64",
                "single::aten.add.Tensor::layout=zero_stride",
                "pair::aten.add.Tensor::dtype=int64::layout=zero_stride",
            ],
        ),
        _case(
            "c",
            "aten.any.default",
            "bool",
            "contiguous",
            ["overload::aten.any.default", "single::aten.any.default::dtype=bool"],
        ),
    ]
    obligations = sorted({o for c in cases for o in c["covered_obligations"]})
    return {
        "claim": "complete",
        "complete": True,
        "overloads": ["aten.add.Tensor", "aten.any.default"],
        "witnessed_obligations": obligations,
        "selected_cases": cases,
        "selected_count": len(cases),
    }


def _ledger(drop):
    return {"schema_version": 1, "parent": {"sha256": "p"}, "cuts": [{"id": "cut", "drop": drop}]}


def test_cut_separates_by_design_from_collateral_losses():
    pruned, steps = apply_ledger(_suite(), _ledger({"dtype": ["int64", "bool"]}), "p")
    assert [c["case_id"] for c in pruned["selected_cases"]] == ["a"]
    assert pruned["complete"] is False and pruned["pruning"]["case_ids"] == ["a"]
    cut = steps[-1]
    assert cut["cases_removed"] == 2
    assert cut["overloads_emptied"] == ["aten.any.default"]
    # dtype=int64 / dtype=bool obligations are intended; the layout single and the any overload are not.
    assert cut["obligations_lost_by_design"] == 3
    assert cut["obligations_lost_collateral"] == [
        "overload::aten.any.default",
        "single::aten.add.Tensor::layout=zero_stride",
    ]
    assert steps[0]["case_count"] == 3 and cut["axes"]["dtype"] == {"float32": 1}


def test_parent_digest_mismatch_fails_closed():
    with pytest.raises(ValueError, match="does not match"):
        apply_ledger(_suite(), _ledger({"dtype": ["bool"]}), "other")


def test_committed_ledger_is_well_formed():
    from merlin.common.paths import repo_root

    ledger = load_ledger(repo_root() / "experiments/reference-data/core_aten_prune/cuts.yaml")
    assert len(ledger["parent"]["sha256"]) == 64
    assert all(cut["drop"] for cut in ledger["cuts"])
