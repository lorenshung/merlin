"""The external functional rules are commitments, never observed placement."""

import copy
import json
from pathlib import Path

from merlin.targetgen.functional_truth import expectation


def test_lane_commitments_are_value_and_placement_independent():
    data = json.loads((Path(__file__).parents[1] / "data/exactness/bounded_final_gate.json").read_text())
    for case in data["selected_cases"]:
        baseline = expectation(case)
        changed = copy.deepcopy(case)

        def mutate(v):
            if isinstance(v, dict):
                if v.get("kind") == "tensor":
                    v["values"] = ["intentionally unavailable"]
                else:
                    for value in v.values():
                        mutate(value)
            elif isinstance(v, list):
                for value in v:
                    mutate(value)

        mutate(changed)
        changed["current_run13_lane"] = "arbitrary"
        assert expectation(changed) == baseline
        assert baseline["lane"] in {"device", "no-work", "host-guard"}


def test_strict_placement_check_refuses_both_directions_and_device_no_work():
    import pytest

    from merlin.targetgen.functional_truth import require_placement

    for lane in ("device", "host-guard", "no-work"):
        commitment = dict(lane=lane, reason="semantic test commitment")
        require_placement(commitment, device=(lane == "device"))
        with pytest.raises(ValueError, match="truth mismatch"):
            require_placement(commitment, device=(lane != "device"))
