"""A target recipe may withdraw a shared performance family only by name, with reason and decision."""

from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.phase0 import profiles
from merlin_experiments.phase0.sweeps import expand_sweeps

from merlin.common.paths import repo_root

_TEMPLATE = repo_root() / "experiments/templates/phase0/performance.yaml"
_DECISION = "operator decision recorded for the test"


def _recipe(tmp_path, withdrawals):
    path = tmp_path / "recipe.yaml"
    document = {"capsules": []}
    if withdrawals is not None:
        document["performance_withdrawals"] = withdrawals
    path.write_text(yaml.safe_dump(document))
    return path


def _load(tmp_path, withdrawals):
    return profiles.load_profile(
        "fixture", recipe=_recipe(tmp_path, withdrawals), performance_template=_TEMPLATE, include_holdouts=False
    )


def test_withdrawal_is_recorded_with_its_reason_and_decision(tmp_path):
    row = {"family": "PB", "reason": "the island member's host map has no declaration", "decided_by": _DECISION}
    profile = _load(tmp_path, [row])
    assert "performance_withdrawals" not in profile
    assert profile["_performance_withdrawals"] == {"PB": row}
    assert profile["_performance_template"]["withdrawn"] == [row]
    # The shared family stays declared: a withdrawal declines it, it does not redefine the template.
    assert "PB" in {family["family"] for family in profile["_performance_template"]["families"]}


def test_no_withdrawals_leaves_every_family_selected(tmp_path):
    profile = _load(tmp_path, None)
    assert "_performance_withdrawals" not in profile
    assert "withdrawn" not in profile["_performance_template"]


@pytest.mark.parametrize(
    "rows, message",
    [
        ([{"family": "PZ_not_declared", "reason": "r", "decided_by": _DECISION}], "undeclared family"),
        ([{"family": "PB", "reason": "  ", "decided_by": _DECISION}], "nonempty strings"),
        ([{"family": "PB", "reason": "r"}], "exactly family, reason and decided_by"),
        ([{"family": "PB", "reason": "r", "decided_by": _DECISION}] * 2, "withdrawn twice"),
        ([], "nonempty list"),
    ],
)
def test_malformed_or_stale_withdrawals_are_refused(tmp_path, rows, message):
    with pytest.raises(ValueError, match=message):
        _load(tmp_path, rows)


def test_withdrawn_family_is_skipped_as_withdrawn_not_as_a_gate_refusal():
    row = {"family": "PX", "reason": "declined for the test", "decided_by": _DECISION}
    profile = {"sweeps": [{"id": "PX", "base": {"cat": "_perf"}}], "_performance_withdrawals": {"PX": row}}
    skipped = []
    entries = expand_sweeps(profile, SimpleNamespace(tile_dim=16, target="fixture"), skipped=skipped)
    assert entries == []
    assert skipped == [
        {
            "family": "PX",
            "sweep": "PX",
            "status": "withdrawn",
            "reason": row["reason"],
            "decided_by": _DECISION,
            "basis": "target recipe performance_withdrawals",
            "fit_axes": [],
            "comparison_roles": [],
        }
    ]
