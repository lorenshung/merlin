"""The cheap cost proxy, held to the hardware it was validated against.

A cost model that is not confronted with measurement is a random number generator: measured on this
tree, a block-DMA change scored a 23% WIN under the functional model and was 2.8%-12.8% WORSE on
hardware. So the correlation this proxy reaches against 71 FireSim-measured compute groups is not
recorded in a report that nobody re-runs -- it is asserted here, and a change that breaks the ranking
fails the suite.

The validation points are target data and live with the target that produced them
(``examples/<target>/phase2/schedule-proxy-validation.json``). They carry the contraction each group
presents, the cycles the hardware spent on it in both arms, and the machine the proxy was scored with,
so this test needs neither an RTL toolchain nor the measurement artifact on disk. Nothing under test
names a target: the proxy reads one from the fixture like any other caller.
"""

from __future__ import annotations

import json
import math

import pytest
import yaml

from merlin.common.paths import merlin_dir, repo_root
from merlin.perf.decompose import UNKNOWN, is_unknown
from merlin.perf.derived_bound import Gemm, Machine, select_tiling
from merlin.perf.mesh_occupancy import mesh_occupancy
from merlin.perf.schedule_proxy import (
    LOAD_CRITICAL,
    MESH_CRITICAL,
    UNDECIDED_REGIME,
    Contraction,
    Pricing,
    schedule_cost,
)

#: The target whose FireSim measurements the proxy was validated against.
MEASURED_TARGET = "gemmini"
FIXTURE = repo_root() / "examples" / MEASURED_TARGET / "phase2" / "schedule-proxy-validation.json"

pytestmark = pytest.mark.target(MEASURED_TARGET)


def _spearman(xs, ys):
    def rank(values):
        order = sorted(range(len(values)), key=lambda i: values[i])
        out = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for t in range(i, j + 1):
                out[order[t]] = (i + j) / 2.0 + 1
            i = j + 1
        return out

    a, b = rank(xs), rank(ys)
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    num = sum((a[i] - ma) * (b[i] - mb) for i in range(n))
    da = math.sqrt(sum((x - ma) ** 2 for x in a))
    db = math.sqrt(sum((x - mb) ** 2 for x in b))
    return num / (da * db)


@pytest.fixture(scope="module")
def validation():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def machine(validation):
    """The machine the proxy was scored with, rebuilt from the values the fixture recorded.

    The PRODUCTION path is ``derived_bound.machine_from_facts(target)``; this rebuild exists only so
    the correlation is checkable without an RTL toolchain. Every value here carries the provenance
    string the facts read produced, and ``test_machine_comes_from_the_target_s_own_facts`` holds the
    two together.
    """
    block = validation["machine"]

    def value(name):
        raw = block[name]
        return UNKNOWN if raw == "UNKNOWN" else raw

    return Machine(
        array_rows=value("array_rows"),
        array_cols=value("array_cols"),
        muls_per_element=value("muls_per_element"),
        operand_bytes=value("operand_bytes"),
        accumulate_bytes=value("accumulate_bytes"),
        readout_bytes=value("readout_bytes"),
        ping_pong_ways=value("ping_pong_ways"),
        operand_store_bytes=value("operand_store_bytes_per_loop"),
        accumulate_store_bytes=value("accumulate_store_bytes_per_loop"),
        fill_drain_cycles=value("fill_drain_cycles"),
        dram_bytes_per_cycle=value("dram_bytes_per_cycle"),
        provenance=block["provenance"],
        refusals=block["refusals"],
    )


def _scored(validation, machine, pricing):
    target = validation["target"]
    scores, measured, groups = [], [], []
    for row in validation["groups"]:
        work = Contraction(**row["contraction"])
        scores.append(schedule_cost(work, target=target, machine=machine, pricing=pricing).transactions)
        measured.append(row["explicit_cycles"])
        groups.append(row["group"])
    return groups, scores, measured


def test_the_validation_set_is_the_one_the_numbers_came_from(validation):
    assert len(validation["groups"]) == 71
    assert validation["measurement_source"]["explicit_total_cycles"] == 43900637
    assert validation["measurement_source"]["fsm_total_cycles"] == 24359749
    ratios = sorted(row["explicit_cycles"] / row["fsm_cycles"] for row in validation["groups"])
    # genuine spread to be wrong about: a set whose points all agree cannot discriminate a model
    assert ratios[0] < 1.2 < 5.0 < ratios[-1]
    by_kind = validation["measurement_source"]["by_kind"]
    assert sum(entry["n"] for entry in by_kind.values()) == 71
    assert sum(entry["explicit"] for entry in by_kind.values()) == 43900637


def test_ranks_the_71_measured_groups(validation, machine):
    """The headline number. Uncalibrated, so this is the floor a caller gets with no measurement."""
    _, scores, measured = _scored(validation, machine, Pricing.uncalibrated())
    rho = _spearman(scores, measured)
    assert rho > 0.90, f"rank correlation against measured hardware fell to {rho:.4f}"


def test_the_calibrated_exchange_rate_ranks_better(validation, machine):
    calibrated = Pricing(movement_per_compute=3.5, provenance="calibrated on this fixture's 71 groups")
    _, scores, measured = _scored(validation, machine, calibrated)
    assert _spearman(scores, measured) > 0.96


def test_picks_the_more_expensive_of_two_candidates(validation, machine):
    """The decision a search actually makes, scored directly rather than through a correlation."""
    _, scores, measured = _scored(validation, machine, Pricing(movement_per_compute=3.5))
    correct = total = 0
    for i in range(len(scores)):
        for j in range(i + 1, len(scores)):
            if measured[i] == measured[j]:
                continue
            total += 1
            correct += (scores[i] > scores[j]) == (measured[i] > measured[j])
    assert correct / total > 0.90, f"pairwise decision accuracy fell to {correct / total:.1%}"


def test_ranks_the_four_worst_groups_worst(validation, machine):
    """The groups the campaign is actually blocked on: deep-stage convolutions whose output rows are
    7 wide against a 16-row array. A proxy that cannot see why THOSE are bad is not useful here."""
    groups, scores, measured = _scored(validation, machine, Pricing.uncalibrated())
    order = sorted(range(len(groups)), key=lambda i: -scores[i])
    measured_order = sorted(range(len(groups)), key=lambda i: -measured[i])
    assert {groups[i] for i in order[:4]} == {60, 58, 67, 63}
    assert {groups[i] for i in measured_order[:4]} == {60, 58, 67, 63}


def test_predicts_the_load_limited_versus_mesh_limited_split(validation, machine):
    """The decision the loop makes PER GROUP, and the reason a single scalar is not enough.

    Two of the scheduling levers are anti-correlated: a hoisted move-in burst costs runtime only
    where the load side is critical, a stationary-operand reload only where the mesh is. Measured on
    pinned elaborated RTL over four shapes, no shape was limited by both. The proxy has to get the
    regime right, because the two take opposite optimisations.
    """
    block = validation["lever_sensitivity"]
    pricing = Pricing(
        movement_per_compute=3.5,
        mesh_critical_reload_multiplicity=block["boundary"]["mesh_critical_at_or_above"],
        provenance="measured on pinned elaborated RTL",
    )
    for row in block["shapes"]:
        spec = row["contraction"]
        work = Contraction(
            lhs_bytes=spec["m"] * spec["k"],
            rhs_bytes=spec["k"] * spec["n"],
            result_bytes=spec["m"] * spec["n"],
            label=row["shape"],
            **spec,
        )
        cost = schedule_cost(work, target=validation["target"], machine=machine, pricing=pricing)
        assert cost.regime == row["regime"], f"{row['shape']}: {cost.regime_basis}"
        # and the lever that pays is the one with pressure behind it
        if row["regime"] == MESH_CRITICAL:
            assert row["remove_stationary_reload"] < -0.10
            assert cost.reload_pressure > 0
        else:
            assert row["burst_to_just_in_time"] < -0.10
            assert cost.burst_pressure > 0


def test_the_regime_is_undecided_without_a_measured_boundary(machine):
    """The two regimes take opposite optimisations, so a guessed verdict is worse than none."""
    work = Contraction(m=3136, n=64, k=64, lhs_bytes=200704, rhs_bytes=4096, result_bytes=200704, stream_run=3136)
    cost = schedule_cost(work, target="any", machine=machine)
    assert cost.regime == UNDECIDED_REGIME
    assert "regime" in cost.unresolved
    assert "OPPOSITE" in cost.reasons["regime"]
    assert cost.resolved  # the RANK signal is still usable

    decided = schedule_cost(
        work,
        target="any",
        machine=machine,
        pricing=Pricing(mesh_critical_reload_multiplicity=5.5, provenance="measured"),
    )
    assert decided.regime == MESH_CRITICAL
    assert "regime" not in decided.unresolved


def test_the_regime_is_not_a_reading_of_the_scalar_cost(validation, machine):
    """Two shapes in DIFFERENT regimes can sit either side of each other on the scalar cost.

    This is why the verdict is separate: on the four measured shapes the compute/movement ratio is
    3.96 and 8.80 for the mesh-critical pair and 6.74 and 6.10 for the load-critical pair -- it
    interleaves, so no threshold on the cost recovers the split.
    """
    ratios = {}
    for row in validation["lever_sensitivity"]["shapes"]:
        spec = row["contraction"]
        work = Contraction(
            lhs_bytes=spec["m"] * spec["k"],
            rhs_bytes=spec["k"] * spec["n"],
            result_bytes=spec["m"] * spec["n"],
            **spec,
        )
        cost = schedule_cost(work, target=validation["target"], machine=machine, pricing=Pricing(1.0))
        ratios[row["regime"]] = ratios.get(row["regime"], []) + [cost.terms["compute"] / cost.terms["movement"]]
    mesh, load = ratios[MESH_CRITICAL], ratios[LOAD_CRITICAL]
    assert min(mesh) < min(load) < max(load) < max(mesh), (mesh, load)


def test_the_refuted_lever_is_not_rewarded(validation):
    """Hoisting the load-configuration state measured -0.4% to +0.5% everywhere and is REFUTED.

    Nothing in the model gives it credit; this holds the recorded evidence so a future change that
    starts rewarding it has to argue with the measurement.
    """
    deltas = [row["hoist_load_states"] for row in validation["lever_sensitivity"]["shapes"]]
    assert max(abs(d) for d in deltas) <= 0.07  # inside the binary-composition noise floor
    assert "Not rewarded here" in validation["lever_sensitivity"]["refuted_lever"]


def test_traffic_alone_does_not_rank(validation, machine):
    """The negative control. If bytes alone ranked these points, the pairing would be unmotivated."""
    bytes_only, measured = [], []
    for row in validation["groups"]:
        work = Contraction(**row["contraction"])
        bytes_only.append(work.moved_bytes)
        measured.append(row["explicit_cycles"])
    assert _spearman(bytes_only, measured) < 0.55


def test_is_not_vacuous_on_two_schedules_with_identical_bytes(machine):
    """The built-in check that this is not a traffic model wearing a hat.

    ``mesh_occupancy`` measured a whole model's array issue count moving 21.38M -> 16.05M cycles with
    the DRAM bytes per operand role BIT-IDENTICAL. A proxy that scores those two the same cannot see
    a 25% swing, so this asserts the two orientations of one shape are separated -- and separated in
    the right direction, by the one with the idle array columns.
    """
    common = {"k": 1024, "lhs_bytes": 200704, "rhs_bytes": 2097152, "result_bytes": 100352}
    packed = Contraction(m=49, n=2048, stream_run=49, **common)
    idle = Contraction(m=2048, n=49, stream_run=2048, **common)
    a = schedule_cost(packed, target="any", machine=machine)
    b = schedule_cost(idle, target="any", machine=machine)

    assert a.moved_bytes == b.moved_bytes
    assert a.transactions < b.transactions
    assert a.array_issue_cycles < b.array_issue_cycles
    assert a.items[0].idle_slot_share == 0.0
    assert b.items[0].idle_slot_share > 0.2


def test_a_short_stream_run_costs_more_than_a_long_one(machine):
    """The other half of the pair: identical shape, identical bytes, different tile granularity."""
    common = {
        "m": 49,
        "n": 2048,
        "k": 1024,
        "lhs_bytes": 200704,
        "rhs_bytes": 2097152,
        "result_bytes": 100352,
    }
    narrow = schedule_cost(Contraction(stream_run=7, **common), target="any", machine=machine)
    wide = schedule_cost(Contraction(stream_run=49, **common), target="any", machine=machine)
    assert narrow.moved_bytes == wide.moved_bytes
    assert narrow.transactions > wide.transactions


def test_array_floor_matches_mesh_occupancy(machine):
    """The per-shape aggregate is the instrument's own answer, not a second implementation.

    ``schedule_cost`` prices each DISTINCT tile shape once with a multiplicity because an inner-loop
    cost model cannot materialise a million tile descriptors. This holds that shortcut to
    ``mesh_occupancy`` over the fully expanded tile list on a shape small enough to expand.
    """
    rows, cols = int(machine.array_rows), int(machine.array_cols)
    work = Contraction(m=37, n=53, k=71, lhs_bytes=2627, rhs_bytes=3763, result_bytes=1961, stream_run=11)
    cost = schedule_cost(work, target="any", machine=machine)

    run = min(work.stream_run, work.m)
    tiles = []
    m = 0
    while m < work.m:
        k = 0
        while k < work.k:
            n = 0
            while n < work.n:
                tiles.append(
                    {
                        "rows": min(run, work.m - m),
                        "depth": min(rows, work.k - k),
                        "cols": min(cols, work.n - n),
                    }
                )
                n += cols
            k += rows
        m += run
    expanded = mesh_occupancy(tiles, array_rows=rows, array_cols=cols)
    assert cost.items[0].compute_transactions == expanded["tiles_read"]
    assert cost.items[0].array_issue_cycles == expanded["issue_cycles"]
    assert cost.items[0].idle_slot_share == expanded["idle_slot_share"]


def test_cycles_are_unknown_without_a_measured_slot_cost(machine):
    work = Contraction(m=64, n=64, k=64, lhs_bytes=4096, rhs_bytes=4096, result_bytes=4096, stream_run=64)
    uncalibrated = schedule_cost(work, target="any", machine=machine)
    assert is_unknown(uncalibrated.cycles)
    assert "cycles" in uncalibrated.unresolved
    assert not uncalibrated.pricing.calibrated
    assert uncalibrated.transactions > 0  # the RANK signal survives the refusal

    priced = schedule_cost(
        work,
        target="any",
        machine=machine,
        pricing=Pricing(movement_per_compute=3.5, cycles_per_transaction=16.0, provenance="measured"),
    )
    assert not is_unknown(priced.cycles)
    assert priced.cycles >= priced.array_issue_cycles  # the array floor is never hidden


def test_fails_closed_when_the_geometry_is_not_derivable():
    """An input the facts do not ground yields no cost at all, with the reason attached."""
    blind = Machine(
        *([UNKNOWN] * 11),
        provenance={},
        refusals={"array_rows": "these facts evidence no compute array", "array_cols": "same", "operand_bytes": "same"},
    )
    work = Contraction(m=16, n=16, k=16, lhs_bytes=256, rhs_bytes=256, result_bytes=256)
    cost = schedule_cost(work, target="any", machine=blind)
    assert not cost.resolved
    assert "array_rows" in cost.unresolved
    assert "no compute array" in cost.reasons["array_rows"]
    assert math.isnan(cost.transactions)


def test_every_result_carries_what_it_cannot_see(machine):
    """The blind axis travels with the number. Naming it is what stops the next silent regression."""
    work = Contraction(m=64, n=64, k=64, lhs_bytes=4096, rhs_bytes=4096, result_bytes=4096)
    cost = schedule_cost(work, target="any", machine=machine)
    assert cost.blind_to
    joined = " ".join(cost.blind_to)
    assert "spill" in joined
    assert "any other device" in joined
    assert cost.to_dict()["blind_to"] == list(cost.blind_to)


def test_names_the_operand_at_risk_of_a_spill(machine):
    """A schedule the compulsory-traffic assumption does not hold for is REPORTED, never silently
    priced at zero -- the model under-prices exactly that case and says so per operand."""
    capacity = int(machine.operand_store_bytes)
    work = Contraction(
        m=4096,
        n=4096,
        k=4096,
        lhs_bytes=capacity * 4,
        rhs_bytes=capacity * 4,
        result_bytes=1024,
        stream_run=4096,
    )
    cost = schedule_cost(work, target="any", machine=machine)
    assert cost.items[0].tiling is not None
    assert cost.spill  # both operands are larger than one loop context's store


def test_a_program_is_the_sum_of_its_contractions(machine):
    parts = [
        Contraction(m=64, n=64, k=64, lhs_bytes=4096, rhs_bytes=4096, result_bytes=4096, label="a"),
        Contraction(m=32, n=128, k=64, lhs_bytes=2048, rhs_bytes=8192, result_bytes=4096, label="b"),
    ]
    whole = schedule_cost(parts, target="any", machine=machine, workload="program")
    singles = [schedule_cost(p, target="any", machine=machine) for p in parts]
    assert whole.transactions == pytest.approx(sum(s.transactions for s in singles))
    assert whole.moved_bytes == sum(s.moved_bytes for s in singles)
    assert [item.label for item in whole.items] == ["a", "b"]


def test_intensity_comes_through_the_ledger(machine):
    """MAC/byte is consumed through ``optimization_ledger.arithmetic_intensity``, which refuses the
    bound-ness verdict without a MEASURED machine balance rather than drawing a guessed ridge."""
    work = Contraction(m=64, n=64, k=64, lhs_bytes=4096, rhs_bytes=4096, result_bytes=4096)
    cost = schedule_cost(work, target="any", machine=machine)
    assert cost.intensity["status"] == "derived"
    assert cost.intensity["macs_per_byte"] == pytest.approx(work.macs / work.moved_bytes)
    assert cost.intensity["bound_by"] == "UNKNOWN"

    with_balance = schedule_cost(work, target="any", machine=machine, machine_macs_per_byte=16.0)
    assert with_balance.intensity["machine_macs_per_byte"] == 16.0


def test_the_tiling_search_runs_without_a_readout_width(machine):
    """``select_tiling`` used to refuse on a fact that cannot change its answer.

    ``|C|`` is the same for every tiling of a shape, so the readout width cannot move the argmin --
    yet it was required, and on every target in this tree it is UNKNOWN, which is why the search had
    no reachable caller. A caller supplying its own footprints now gets a tiling.
    """
    assert is_unknown(machine.readout_bytes)
    gemm = Gemm(196, 256, 2304)
    assert isinstance(select_tiling(gemm, machine), str)  # the dense path still needs the width
    chosen = select_tiling(gemm, machine, lhs_bytes=50176, rhs_bytes=589824, result_bytes=50176)
    assert not isinstance(chosen, str)
    assert chosen.tile_m * chosen.tile_n * int(machine.accumulate_bytes) <= int(machine.accumulate_store_bytes)


def test_machine_comes_from_the_targets_own_facts(validation):
    """The fixture's machine is a RECORD of a facts read, not a set of constants someone chose."""
    block = validation["machine"]
    for name in ("array_rows", "array_cols", "operand_bytes", "accumulate_bytes"):
        assert block["provenance"][name].startswith("facts.")
    for name in ("readout_bytes", "dram_bytes_per_cycle"):
        assert block[name] == "UNKNOWN"
        assert block["refusals"][name]


def test_the_measurement_ladder_names_this_module_and_its_points(validation):
    """The ``cost_proxy`` rung is implemented by THIS module and cites THESE points.

    The rung's evidence block states the figures the tests above re-derive; naming the module and the
    points file there keeps the contract and the implementation from drifting apart unnoticed.
    """
    ladder = yaml.safe_load((merlin_dir() / "contract" / "measurement_ladder.yaml").read_text(encoding="utf-8"))
    rung = ladder["rungs"]["cost_proxy"]
    assert rung["implemented_by"] == "merlin.perf.schedule_proxy"
    assert rung["establishes"] == ["ranking"] and "cycle_count" in rung["refuses"]
    assert (repo_root() / rung["evidence"]["points"]).resolve() == FIXTURE.resolve()
    assert rung["evidence"]["measured_points"] == len(validation["groups"])
    assert validation["target"] == MEASURED_TARGET
