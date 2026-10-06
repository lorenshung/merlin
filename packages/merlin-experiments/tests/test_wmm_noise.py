"""Each machine's noise comes from its own solo repeats: the improvement margin and the batch drift
tolerance follow it, and a machine without two same-day repeats is flagged rather than guessed."""

from __future__ import annotations

from pathlib import Path

from merlin_experiments.phase2.whole_model_measured import noise as N
from merlin_experiments.phase2.whole_model_measured import objective as O
from merlin_experiments.phase2.whole_model_measured.identity import write_json_atomic

from merlin.perf import whole_model_verdict as V

STOCK, LEAN = "s" * 64, "l" * 64


def _reading(path: Path, *, device: str, cycles: int, day: str, elf: str = "e" * 64, **fields) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    write_json_atomic(
        path / "result.json",
        {
            "timing_status": V.TIMING_MEASURED,
            "verdict": {"whole_window_cycles": cycles, "objective_cycles": cycles},
            "objective_cycles": cycles,
            "build": {"elf_sha256": elf},
            "device": {"binary_sha256": device},
            "finished_at": f"{day}T120000Z",
            **fields,
        },
    )
    return path / "result.json"


def test_same_day_and_cross_day_spreads_are_per_device(tmp_path):
    store = tmp_path / "store"
    _reading(store / "a", device=STOCK, cycles=1000, day="20261001")
    _reading(store / "b", device=STOCK, cycles=1010, day="20261001")
    _reading(store / "c", device=STOCK, cycles=1024, day="20261006")
    _reading(store / "d", device=LEAN, cycles=2000, day="20261001")
    readings = N.solo_readings([store])
    stock = N.machine_noise(readings, device=STOCK)
    assert stock["established"] and stock["flag"] is None
    assert stock["same_day"] == {"pairs": 1, "max": 0.01, "median": 0.01}
    assert stock["cross_day"]["max"] == 0.024
    lean = N.machine_noise(readings, device=LEAN)
    assert not lean["established"] and lean["flag"] == N.NOT_ESTABLISHED


def test_batched_carried_and_unmeasured_results_are_never_solo_readings(tmp_path):
    store = tmp_path / "store"
    _reading(store / "a", device=STOCK, cycles=1000, day="20261001")
    _reading(store / "b", device=STOCK, cycles=1500, day="20261001", batch={"size": 8})
    _reading(store / "c", device=STOCK, cycles=1500, day="20261001", same_program_as="a")
    _reading(store / "d", device=STOCK, cycles=1500, day="20261001", timing_status=V.TIMING_MEASURED_INVALID)
    _reading(store / "e", device=STOCK, cycles=1500, day="20261001", elf="f" * 64)  # another program
    assert not N.machine_noise(N.solo_readings([store]), device=STOCK)["established"]
    # MUTATION: one real same-day solo repeat of the same program establishes it.
    _reading(store / "f", device=STOCK, cycles=1001, day="20261001")
    assert N.machine_noise(N.solo_readings([store]), device=STOCK)["established"]


def test_every_attempt_of_a_job_is_a_reading(tmp_path):
    store = tmp_path / "store"
    _reading(store / "job", device=STOCK, cycles=1000, day="20261001")
    _reading(store / "job" / "attempts" / "0", device=STOCK, cycles=1020, day="20261001")
    assert N.machine_noise(N.solo_readings([store]), device=STOCK)["same_day"]["max"] == 0.02


def test_the_margin_is_the_largest_measured_noise_and_says_which():
    noisy = {"established": True, "same_day": {"max": 0.024}, "flag": None}
    assert N.margin(noisy, floor=0.001, batched_vs_solo=0.004) == {
        "margin": 0.024,
        "basis": "same_day_solo_spread",
        "candidates": {"floor": 0.001, "same_day_solo_spread": 0.024, "batched_vs_solo_median": 0.004},
        "established": True,
        "flag": None,
    }
    unknown = N.margin(None, floor=0.001)
    assert unknown["margin"] == 0.001 and unknown["basis"] == "floor" and unknown["flag"] == N.NOT_ESTABLISHED


def test_the_drift_tolerance_is_never_narrower_than_declared_and_widens_to_the_machines_drift():
    noise = {"established": True, "same_day": {"max": 0.0036}, "cross_day": {"max": 0.024}}
    stale = N.drift_tolerance(noise, declared=0.02, same_day=False)
    assert stale["tolerance"] == 0.024 and stale["basis"] == "cross_day_solo_spread"
    today = N.drift_tolerance(noise, declared=0.02, same_day=True)
    assert today["tolerance"] == 0.02 and today["basis"] == "declared"
    unknown = N.drift_tolerance({"established": False, "flag": N.NOT_ESTABLISHED}, declared=0.02, same_day=True)
    assert unknown["tolerance"] == 0.02 and unknown["flag"] == N.NOT_ESTABLISHED


class _Screen:
    """The objective's view of a screen store: its root and its (empty) job list."""

    def __init__(self, root: Path):
        self.root = root
        self.machine = {"kind": "spike"}
        self.build_options = {}

    def jobs(self):
        return []

    def result(self, digest):
        return None

    def result_by_key(self, key):
        return None

    def history(self):
        return []

    def attributable(self, key):
        return True


def test_the_objective_crowns_only_beyond_its_own_machines_noise(tmp_path):
    """The same objective over two machines' stores: 2.4% same-day noise on one, 0.36% on the other."""
    margins = {}
    for device, spread in ((STOCK, 0.024), (LEAN, 0.0036)):
        root = tmp_path / device[:1]
        _reading(root / "a", device=device, cycles=100_000, day="20261001")
        _reading(root / "b", device=device, cycles=int(100_000 * (1 + spread)), day="20261001")
        objective = O.WholeModelObjective(screen=_Screen(root), screen_reference=None)
        margins[device] = objective.noise_margin()
        assert objective.summary()["noise"]["established"] is True
    assert margins == {STOCK: 0.024, LEAN: 0.0036}


def test_a_machine_without_two_same_day_repeats_is_flagged_in_the_summary(tmp_path):
    root = tmp_path / "store"
    _reading(root / "a", device=STOCK, cycles=1000, day="20261001")
    _reading(root / "b", device=STOCK, cycles=1100, day="20261005")
    objective = O.WholeModelObjective(screen=_Screen(root), screen_reference=None)
    noise = objective.summary()["noise"]
    assert noise["established"] is False and noise["flag"] == N.NOT_ESTABLISHED
    assert noise["margin"] == O.NOISE_FLOOR and noise["cross_day"]["max"] == 0.1
