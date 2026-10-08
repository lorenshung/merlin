"""The Phase 1 coverage inventory: one fail-closed row per required item, development workloads only.

Two kinds of evidence. The synthetic cases drive :func:`build_inventory` (the pure join) with a
made-up device so each fail-closed rule is pinned by an exact input. The tracked cases run the whole
reader path (:func:`inventory_for_target`) over two registered targets' tracked spec, contract and
capsule corpus, with no simulator, toolchain or ``.env``.
"""

from __future__ import annotations

import copy
import functools
import json

import pytest

from merlin.targetgen import claim_models as CM
from merlin.targetgen import coverage_inventory as CI
from merlin.targetgen.contract.materialize import capsule_cell_rows, cert_capsule_cover

TRACKED_TARGETS = ("gemmini", "atlas")

# --- a made-up device: one array unit, a fused-only epilogue family, one undecidable family -------

CONTRACT = {
    "name": "device",
    "compute_units": [
        {
            "name": "u0",
            "kind": "systolic",
            "dtypes": ["int8"],
            "ops": ["matmul"],
            "semantic_capabilities": [
                {"family": "contraction", "dtypes": ["int8"]},
                {"family": "elementwise_map", "dtypes": ["int8"], "composed_with": ["contraction"]},
            ],
        }
    ],
    "semantic_capabilities_unknown": [{"family": "normalization"}],
}


def _spec(target: str = "device") -> dict:
    return {
        "target": target,
        "boundaries": {"tile_edge": 4},
        "cells": [
            {
                "cell": "contraction/i8/aligned",
                "family": "contraction",
                "dtype": "i8",
                "alignment": "aligned",
                "observed_in": ["dev_model_a"],
            },
            {
                "cell": "elementwise_map/i8/aligned",
                "family": "elementwise_map",
                "dtype": "i8",
                "alignment": "aligned",
                "observed_in": ["dev_model_a"],
            },
            {
                "cell": "normalization/i8/aligned",
                "family": "normalization",
                "dtype": "i8",
                "alignment": "aligned",
                "observed_in": ["dev_model_a"],
            },
            # A cell the spec demands on the accelerator but the manifest refuses: spec/manifest drift.
            {
                "cell": "contraction/f32/aligned",
                "family": "contraction",
                "dtype": "f32",
                "alignment": "aligned",
                "observed_in": ["dev_model_a"],
            },
        ],
        "host_lane": {"required": [{"family": "movement", "dtype": "f32"}]},
        "composition": {"required": {"A": ["dev_model_a"], "H->A->H": ["dev_model_b"]}},
    }


def _census(on_accelerator: int, on_host: int, silent=()) -> dict:
    return {
        "placement_census": {
            "silent_fallbacks_status": "offloaded",
            "silent_fallbacks": list(silent),
            "coverage": {"on_accelerator": on_accelerator, "on_host": on_host},
        }
    }


WITNESSES = {
    "cell": {
        "contraction/i8/aligned": ["W_mm"],
        "elementwise_map/i8/aligned": ["W_fused"],
        "normalization/i8/aligned": ["W_norm"],
        "contraction/f32/aligned": ["W_f32"],
    },
    "host_lane": {"movement/f32": ["W_host"]},
    "composition": {"A": ["W_mm"], "H->A->H": ["W_seam"]},
}


def _by_key(inv) -> dict:
    return {(r.kind, r.key): r for r in inv.rows}


def test_no_results_means_nothing_is_covered():
    inv = CI.build_inventory("device", _spec(), CONTRACT, WITNESSES)
    rows = _by_key(inv)
    assert all(r.status != CI.COVERED for r in inv.rows)
    mm = rows[("cell", "contraction/i8/aligned")]
    assert mm.required_placement == CI.MUST_ACCELERATE
    assert mm.status == CI.STATUS_UNKNOWN and mm.verdict == CI.NOT_RUN
    assert [o.verdict for o in mm.observations] == [CI.NOT_RUN]


def test_pass_with_matching_placement_is_the_only_covered():
    results = {
        "W_mm": {"status": "pass", **_census(3, 0)},
        "W_fused": {"status": "pass", **_census(2, 0)},
        "W_host": {"status": "pass", **_census(0, 2)},
        "W_seam": {"status": "pass", **_census(1, 2)},
    }
    rows = _by_key(CI.build_inventory("device", _spec(), CONTRACT, WITNESSES, results=results))
    for key in (
        ("cell", "contraction/i8/aligned"),
        ("cell", "elementwise_map/i8/aligned"),
        ("host_lane", "movement/f32"),
        ("composition", "A"),
        ("composition", "H->A->H"),
    ):
        assert rows[key].status == CI.COVERED, (key, rows[key].reasons)
    assert rows[("cell", "elementwise_map/i8/aligned")].boundary == "fused_with:contraction"
    assert rows[("host_lane", "movement/f32")].required_placement == CI.HOST
    assert rows[("composition", "H->A->H")].required_placement == CI.MIXED


@pytest.mark.parametrize(
    ("result", "status"),
    [
        (None, CI.STATUS_UNKNOWN),  # never graded
        ({"status": "pass"}, CI.STATUS_UNKNOWN),  # passed, placement never measured
        (
            {"status": "pass", "placement_census": {"silent_fallbacks_status": "incomplete", "silent_fallbacks": []}},
            CI.STATUS_UNKNOWN,
        ),
        ({"status": "pass", **_census(2, 1, silent=[7])}, CI.MISSING),  # silent host fallback
        ({"status": "pass", **_census(0, 3)}, CI.MISSING),  # ran on the host, required on the unit
        ({"status": "fail", **_census(3, 0)}, CI.MISSING),
        ({"status": "incomplete"}, CI.MISSING),
    ],
)
def test_unknown_or_not_run_never_counts_as_covered(result, status):
    results = {} if result is None else {"W_mm": result}
    row = _by_key(CI.build_inventory("device", _spec(), CONTRACT, WITNESSES, results=results))[
        ("cell", "contraction/i8/aligned")
    ]
    assert row.status == status, row.reasons
    assert row.status != CI.COVERED


def test_undecided_placement_is_unknown_even_with_a_perfect_witness():
    results = {"W_norm": {"status": "pass", **_census(3, 0)}, "W_f32": {"status": "pass", **_census(3, 0)}}
    rows = _by_key(CI.build_inventory("device", _spec(), CONTRACT, WITNESSES, results=results))
    norm = rows[("cell", "normalization/i8/aligned")]
    assert norm.required_placement == CI.UNKNOWN and norm.status == CI.STATUS_UNKNOWN
    drift = rows[("cell", "contraction/f32/aligned")]
    assert drift.required_placement == CI.HOST and drift.status == CI.STATUS_UNKNOWN
    assert any("disagree" in reason for reason in drift.reasons)


def test_unwitnessed_item_is_missing():
    witnesses = copy.deepcopy(WITNESSES)
    witnesses["host_lane"] = {}
    row = _by_key(CI.build_inventory("device", _spec(), CONTRACT, witnesses))[("host_lane", "movement/f32")]
    assert row.status == CI.MISSING and row.witnesses == ()


# --- holdout discipline ----------------------------------------------------------------------------


def test_holdout_workload_is_refused():
    held = CM.claim_models()[0]
    with pytest.raises(CI.HoldoutInputError):
        CI.build_inventory("device", _spec(), CONTRACT, WITNESSES, workloads=["dev_model_a", held])
    with pytest.raises(CI.HoldoutInputError):
        CI.build_inventory("device", _spec(), CONTRACT, WITNESSES, workloads=[f"{held}_fp32_full"])


def test_holdout_results_are_refused():
    capsule = f"X0_{CM.claim_models()[0]}_device"
    with pytest.raises(CI.HoldoutInputError):
        CI.build_inventory("device", _spec(), CONTRACT, WITNESSES, results={capsule: {"status": "pass"}})


def test_holdout_evidence_and_witnesses_are_separated_not_trusted():
    held = CM.claim_models()[0]
    spec = _spec()
    spec["cells"][0]["observed_in"] = [held, f"{held}_int8"]
    witnesses = copy.deepcopy(WITNESSES)
    witnesses["cell"]["contraction/i8/aligned"] = ["W_mm", f"M9_{held}_device"]
    results = {"W_mm": {"status": "pass", **_census(3, 0)}}
    inv = CI.build_inventory("device", spec, CONTRACT, witnesses, results=results)
    row = _by_key(inv)[("cell", "contraction/i8/aligned")]
    assert row.development_evidence == ()
    assert set(row.holdout_evidence_excluded) == {held, f"{held}_int8"}
    assert row.status == CI.STATUS_UNKNOWN  # rests only on held-out evidence
    assert row.witnesses == ("W_mm",)
    assert inv.holdout_witnesses_excluded == (f"M9_{held}_device",)


def test_selected_workloads_restrict_evidence():
    rows = _by_key(CI.build_inventory("device", _spec(), CONTRACT, WITNESSES, workloads=["dev_model_b"]))
    assert rows[("composition", "H->A->H")].development_evidence == ("dev_model_b",)
    assert rows[("composition", "A")].status == CI.STATUS_UNKNOWN
    assert any("held-out or unselected" in r for r in rows[("composition", "A")].reasons)


def test_claim_model_mention_is_whole_token():
    held = CM.claim_models()[0]
    assert CM.mentioned_claim_model(f"M4_{held}_denoise") == held
    assert CM.mentioned_claim_model(f"M4_{held}x_denoise") is None


# --- determinism and target-agnosticism ------------------------------------------------------------


def test_deterministic_and_input_bound():
    results = {"W_mm": {"status": "pass", **_census(3, 0)}}
    a = CI.build_inventory("device", _spec(), CONTRACT, WITNESSES, results=results)
    shuffled = {k: {kk: list(reversed(vv)) for kk, vv in reversed(list(v.items()))} for k, v in WITNESSES.items()}
    spec = _spec()
    spec["cells"] = list(reversed(spec["cells"]))
    b = CI.build_inventory("device", spec, CONTRACT, shuffled, results=dict(results))
    assert json.dumps(a.to_json(), sort_keys=True) == json.dumps(
        {**b.to_json(), "input_digests": a.input_digests, "inputs_digest": a.inputs_digest}, sort_keys=True
    )
    again = CI.build_inventory("device", _spec(), CONTRACT, WITNESSES, results=results)
    assert again.to_json() == a.to_json()
    other = CI.build_inventory("device", _spec(), CONTRACT, WITNESSES, results={"W_mm": {"status": "fail"}})
    assert other.inputs_digest != a.inputs_digest
    assert [(r.kind, r.key) for r in a.rows] == sorted(
        ((r.kind, r.key) for r in a.rows), key=lambda kk: (CI.KINDS.index(kk[0]), kk[1])
    )


def test_target_is_only_a_parameter():
    one = CI.build_inventory("device_one", _spec("device_one"), CONTRACT, WITNESSES)
    two = CI.build_inventory("device_two", _spec("device_two"), CONTRACT, WITNESSES)
    assert [r.to_json() for r in one.rows] == [r.to_json() for r in two.rows]
    with pytest.raises(ValueError):
        CI.build_inventory("device_one", _spec("device_two"), CONTRACT, WITNESSES)


def test_cell_rows_are_the_cover_definition():
    from merlin.common.paths import merlin_dir

    root = merlin_dir() / "contract" / "capsules" / "layers"
    rows = capsule_cell_rows([root], labels={"public", "dev"}, tile_dim=16)
    cells = {"/".join(x for x in cell if x) for row in rows for cell in row["cells"]}
    assert cells == set(cert_capsule_cover([root], labels={"public", "dev"}, tile_dim=16)["cells"])


# --- two tracked targets, whole reader path --------------------------------------------------------


@functools.cache
def _tracked(target: str):
    return CI.inventory_for_target(target)


@pytest.fixture(params=TRACKED_TARGETS)
def tracked(request):
    return request.param, _tracked(request.param)


def test_tracked_target_inventory_is_complete_and_fail_closed(tracked):
    target, inv = tracked
    assert inv.target == target and inv.rows
    counts = inv.counts()
    assert sum(counts["total"].values()) == len(inv.rows)
    kinds = {r.kind for r in inv.rows}
    assert {"cell", "composition"} <= kinds
    for row in inv.rows:
        assert row.status in CI.STATUSES
        assert row.required_placement in CI.REQUIRED_PLACEMENTS
        # nothing was graded, so nothing may be covered
        assert row.status != CI.COVERED
        if row.required_placement == CI.UNKNOWN:
            assert row.status == CI.STATUS_UNKNOWN
        if row.witnesses:
            assert all(o.verdict == CI.NOT_RUN for o in row.observations)
        for name in (*row.development_evidence, *row.witnesses):
            assert CM.model_of(name) is None and CM.mentioned_claim_model(name) is None
    assert inv.to_json() == CI.inventory_for_target(target).to_json()


def test_tracked_targets_share_one_code_path():
    """Both tracked inventories are the same schema and vocabulary; nothing branches on the target."""
    first, second = (_tracked(t) for t in TRACKED_TARGETS)
    assert set(first.to_json()) == set(second.to_json())
    assert first.inputs_digest != second.inputs_digest
    assert {r.signature_identity for r in first.rows if r.kind == "composition"} == {
        r.signature_identity for r in second.rows if r.kind == "composition"
    }
