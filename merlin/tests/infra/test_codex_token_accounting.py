"""Counter reconciliation must distinguish usage from duplicated subsets."""

import json

import pytest

from merlin.agentreport.tokens import read_codex_tokens


def event(timestamp, total):
    return dict(
        timestamp=timestamp, type="event_msg", payload=dict(type="token_count", info=dict(total_token_usage=total))
    )


def counters(n=1):
    return dict(
        input_tokens=100 * n,
        cached_input_tokens=70 * n,
        cache_write_input_tokens=5 * n,
        output_tokens=20 * n,
        reasoning_output_tokens=10 * n,
        total_tokens=120 * n,
    )


def write(tmp_path, records, tail=""):
    p = tmp_path / "rollout.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in records) + tail)
    return p


def test_exact_buckets_ignore_duplicate_reports_and_live_tail(tmp_path):
    a = event("2026-10-05T12:00:00Z", counters())
    p = write(tmp_path, [a, a, event("2026-10-05T12:01:00Z", counters(2))], "{")
    facts = read_codex_tokens(p)
    assert (facts.input_tokens, facts.cache_read_tokens, facts.cache_creation_tokens) == (50, 140, 10)
    assert (facts.output_tokens, facts.reasoning_tokens, facts.total_tokens) == (40, 20, 240)
    assert facts.cost_usd is None and facts.notional_usd is None


def test_completed_request_window_subtracts_prior_counter(tmp_path):
    p = write(tmp_path, [event(f"2026-10-05T12:0{i}:00Z", counters(i + 1)) for i in range(3)])
    facts = read_codex_tokens(p, since="2026-10-05T12:01:00Z", until="2026-10-05T12:01:30Z")
    assert facts.total_tokens == 120 and facts.input_tokens == 25


@pytest.mark.parametrize(
    "problem", ["missing", "negative", "subsets", "total", "reset", "order", "no_event", "malformed"]
)
def test_unprovable_usage_is_unavailable(tmp_path, problem):
    values = counters()
    if problem == "missing":
        del values["cached_input_tokens"]
    if problem == "negative":
        values["output_tokens"] = -1
    if problem == "subsets":
        values["cached_input_tokens"] = 101
    if problem == "total":
        values["total_tokens"] = 999
    records = [event("2026-10-05T12:00:00Z", values)]
    if problem == "reset":
        records.append(event("2026-10-05T12:01:00Z", counters(0)))
    if problem == "order":
        records.append(event("2026-10-05T11:59:00Z", counters(2)))
    if problem == "no_event":
        records = []
    p = write(tmp_path, records, "invalid\n" if problem == "malformed" else "")
    facts = read_codex_tokens(p)
    assert facts.availability.get("tokens").kind == "unavailable"


def test_empty_window_and_missing_file_are_unavailable(tmp_path):
    p = write(tmp_path, [event("2026-10-05T12:00:00Z", counters())])
    assert read_codex_tokens(p, since="2026-10-06T00:00:00Z").total_tokens == 0
    assert read_codex_tokens(tmp_path / "absent").availability.get("tokens").kind == "unavailable"


@pytest.mark.parametrize(
    "record",
    [[], {"type": "event_msg", "payload": []}, {"type": "event_msg", "payload": {"type": "token_count", "info": [1]}}],
)
def test_malformed_record_shapes_refuse(tmp_path, record):
    facts = read_codex_tokens(write(tmp_path, [record]))
    assert facts.availability.get("tokens").kind == "unavailable"


@pytest.mark.parametrize("subset", ["cached_input_tokens", "reasoning_output_tokens"])
def test_impossible_incremental_subsets_refuse(tmp_path, subset):
    later = counters()
    later[subset] += 1  # Cumulative split looks legal; increment has no new parent tokens.
    p = write(tmp_path, [event("2026-10-05T12:00:00Z", counters()), event("2026-10-05T12:01:00Z", later)])
    assert read_codex_tokens(p).availability.get("tokens").kind == "unavailable"


@pytest.mark.parametrize(
    "window", [dict(since="2026-10-05T12:00:00"), dict(since="2026-10-06T00:00:00Z", until="2026-10-05T00:00:00Z")]
)
def test_ambiguous_or_inverted_window_refuses(tmp_path, window):
    p = write(tmp_path, [event("2026-10-05T12:00:00Z", counters())])
    assert read_codex_tokens(p, **window).availability.get("tokens").kind == "unavailable"
