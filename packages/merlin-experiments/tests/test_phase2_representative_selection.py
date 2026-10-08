"""The development selector preserves complete claims and reports its limits."""

from __future__ import annotations

from merlin_experiments.phase2 import representative_selection as RS


def _member(family, capsule, demand, trait, *, complete=True):
    return {
        "family": family,
        "capsule": capsule,
        "capsule_tree_sha256": capsule.ljust(64, "0"),
        "purpose": {"lever": family},
        "matched_comparator": {"matching_status": "planned_unverified" if complete else "incomplete_group"},
        "hardware_basis": {"traits": {trait: {"satisfied": True}}},
        "workload_need": {
            "matched_demands": [
                {
                    "application": "development",
                    "signature_index": demand,
                    "form_match": {"status": "exact_arithmetic_form"},
                }
            ]
        },
    }


def test_family_closed_subset_reports_refused_and_omitted_levers():
    receipt = {
        "schema": "merlin.phase2.test_justification.v1",
        "target": "fixture",
        "workload_accounting": {"uncovered_demands": [{"operation": "unsupported_map"}]},
        "members": [
            _member("PAIR", "left", 0, "mesh"),
            _member("PAIR", "right", 0, "control"),
            {**_member("REDUNDANT", "solo", 0, "mesh"), "purpose": {"lever": "PAIR"}},
            _member("SECOND", "s1", 1, "store"),
            _member("SECOND", "s2", 1, "store"),
            _member("REFUSED", "broken", 2, "port", complete=False),
        ],
    }
    scope = {
        "performance": {
            "status": "no_eligible_chain",
            "required": [],
            "excluded": [
                {
                    "instance_id": "chain-1",
                    "signature": "map -> matmul",
                    "status": "software_refused",
                    "reason": "dtype",
                }
            ],
            "unresolved": [],
        }
    }
    selected = RS.derive(receipt, scope)
    assert selected == RS.derive(receipt, scope)
    assert selected["selected_families"] == ["PAIR", "SECOND"]
    assert {row["capsule"] for row in selected["selected_members"]} == {"left", "right", "s1", "s2"}
    assert {row["family"] for row in selected["unselected_families"]} == {"REDUNDANT", "REFUSED"}
    assert selected["uncovered_static_axes"] == [
        {"kind": "exact_form", "value": "development:2"},
        {"kind": "lever", "value": "REFUSED"},
        {"kind": "rtl_trait", "value": "port"},
    ]
    assert selected["connected_slice"]["excluded"][0]["status"] == "software_refused"
    assert selected["uncovered_source_demands"] == [{"operation": "unsupported_map"}]
    assert selected["status"] == "diagnostic_subset_unmeasured"


def test_distinct_capsule_shapes_are_not_collapsed_by_a_shared_lever():
    small = _member("SMALL", "small", 0, "mesh")
    large = _member("LARGE", "large", 0, "mesh")
    for row, extent in ((small, 8), (large, 16)):
        row["purpose"] = {"lever": "same_optimization"}
        row["workload_need"]["capsule_form"] = {
            "operation": "matmul",
            "inputs": [{"role": "input", "shape": [extent, 16], "dtype": "i8"}],
        }
    receipt = {
        "schema": "merlin.phase2.test_justification.v1",
        "target": "fixture",
        "members": [small, large],
    }
    selected = RS.derive(receipt, None)
    assert selected["selected_families"] == ["LARGE", "SMALL"]
    assert selected["unselected_families"] == []
