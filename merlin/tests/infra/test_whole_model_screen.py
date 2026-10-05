"""The structure screen: a whole-model program on the functional simulator reads correctness per group
and never feeds an objective; a partial console is refused, not read."""

from __future__ import annotations

from merlin.perf import whole_model_screen as S

CONSOLE = "\n".join(
    [
        "GM_GROUP 1 conv 100 sum=1 fnv1a=2",
        "GM_LOCAL 1 mismatches=0 of=64 first=-1",
        "GM_GROUP 2 add 50 sum=UNKNOWN fnv1a=UNKNOWN",
        "GM_BOUND 2 max_abs=3 over=2 bound=1",
        "FM full model cycles: 150",
        "GM_ARGMAX got=4 want=4 agrees=1",
        "MERLIN_WINDOW end label=m",
    ]
)


def test_each_group_is_read_by_its_own_local_check_and_never_feeds_an_objective():
    screen = S.screen_console(CONSOLE, {"1": "exact", "2": "bounded_int"})
    assert screen["status"] == "screened" and screen["feeds_objective"] is False
    rows = {r["group"]: r for r in screen["groups"]}
    assert rows[1]["local"] == "correct" and rows[2]["local"] == "wrong" and rows[2]["failure"]["over"] == 2
    assert screen["groups_not_correct"] == [2] and screen["argmax"] == [4, 4, 1]


def test_a_console_missing_an_expected_group_is_refused():
    screen = S.screen_console(CONSOLE, {"1": "exact", "2": "bounded_int", "3": "exact"})
    assert screen["status"] == "refused" and "missing ['3']" in screen["refusal"]
