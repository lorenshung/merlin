"""Launch profiles: data, checked before anything launches -- the model must run as named and be priced."""

from __future__ import annotations

import pytest
from merlin_experiments.phase2.whole_model_measured import profiles as P

from merlin.common.paths import repo_root

PRICES = repo_root() / "merlin" / "experiments" / "capsule_bench" / "bedrock_prices.yaml"


def test_the_codex_profile_runs_the_model_it_names_and_that_model_is_priced():
    profile = P.load("codex-gpt-6-sol")
    checked = P.check(profile, price_table=PRICES)
    assert checked["resolved_model"] == "gpt-6-sol" and checked["rate"][1] > 0


def test_a_stale_alias_that_the_driver_would_replace_is_refused_at_launch():
    """A leftover model name once ran 40 rounds under the driver's default instead."""
    profile = {**P.load("codex-gpt-6-sol"), "model": "claude-opus-5"}
    with pytest.raises(P.ProfileError, match="resolves"):
        P.check(profile, price_table=PRICES)


def test_an_unpriced_model_is_refused_at_launch(tmp_path):
    table = tmp_path / "prices.yaml"
    table.write_text("gpt-5.6-sol: [5.0, 30.0, 0.5, 5.0]\n")
    with pytest.raises(P.ProfileError, match="prices no model"):
        P.check(P.load("codex-gpt-6-sol"), price_table=table)
    with pytest.raises(P.ProfileError, match="no launch profile"):
        P.load("absent")
