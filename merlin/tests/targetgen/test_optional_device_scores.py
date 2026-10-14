"""Correct host fallback never manufactures device coverage or guard success."""

from merlin.targetgen.capsule_common import host_guards_pass, score_lanes


def test_optional_host_correctness_and_device_evidence_remain_separate():
    rows = [
        {"status": "pass", "lane": "host", "scored": True},
        {"status": "pass", "lane": "device", "executed_instructions": 7, "scored": True},
        {"status": "fail", "lane": "device", "executed_instructions": 9, "scored": True},
        {"status": "pass", "lane": "device", "executed_instructions": 0, "scored": True},
        {"status": "pass", "lane": "host", "scored": False},
    ]
    score = {}
    scored = score_lanes(rows, score)
    assert len(scored) == 4
    assert score == {
        "host_guard_total": 1,
        "host_guard_pass": 1,
        "device_lane_scored_pass": 1,
        "device_evidence_count": 2,
        "host_lane_scored_pass": 1,
    }
    assert host_guards_pass(score)
    rows[-1]["lane"] = "device"
    score_lanes(rows, score)
    assert not host_guards_pass(score)
