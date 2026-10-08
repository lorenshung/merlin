"""The memory-order tier on the race it exists for: the cand_07 residual add, and its fenced control.

Phase-1 candidate cand_07 lowered a residual add as an overwriting accumulator load followed by an
accumulating load into the same rows, with nothing ordering their COMPLETION. It passed L2 and L3 and
failed on FireSim with the first operand alone in the failing elements. Its fenced rebuild passed on the
same board. ``fixtures/memory_order_cand07.json`` holds what the runner decoded from both builds and
what the memory-perturbing engine answered for the unfenced one; its ``provenance`` names the probe, the
engine pin and the RTL revision.

Asserted here, on those recorded traces rather than on a synthetic one:

* the unfenced trace triggers the sweep (its loads sit in a rolled loop, so their addresses are
  unresolved, and "could not tell" never reads as clear) and the recorded sweep fails the certificate
  with the seeds the engine named;
* the fenced control is clear and is never swept;
* the fences are what clear it: delete them and the same resolved loads are a hazard.
"""

from __future__ import annotations

import json

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen import load_order as LO

FIXTURE = merlin_dir() / "tests" / "gemmini" / "fixtures" / "memory_order_cand07.json"


@pytest.fixture(scope="module")
def recorded():
    return json.loads(FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def facts(recorded):
    layout = recorded["accumulator_layout"]
    acc, accum, full = layout["ACC_I8"], layout["ACC_ACCUM"], layout["FULL_C_BIT"]
    return LO.OrderFacts(acc, accum, acc | accum | full, layout["block_columns"])


def _replay(recorded_runs):
    """A sweep that answers each seed exactly as the engine did, and records what it was asked."""
    outcome = {r["seed"]: r["outcome"] for r in recorded_runs}
    calls = []

    def sweep(engine, elf, seeds, judge, **kw):
        calls.append({"seeds": list(seeds), **kw})
        order = ([None] if kw.get("include_in_order") else []) + list(seeds)
        runs = [{"seed": s, "rc": 0, "wall_s": 0.0, "outcome": outcome[s]} for s in order]
        return {"runs": runs, "elapsed_s": 0.0, "emulator_sha256": "e", "elf_sha256": "f"}

    return sweep, calls


def test_the_unfenced_residual_add_triggers_the_sweep(recorded, facts):
    got = LO.static_check(recorded["unfenced"]["trace"], facts)
    assert got["verdict"] == "unresolved"
    pairs = {(p["first"], p["second"]) for p in got["unresolved_pairs"]}
    then = {(p["first"], p["second"]) for p in recorded["unfenced"]["recorded_sweep"]["static_pairs_then"]}
    assert then <= pairs, "the overwrite/accumulate pair the engine was swept for must still trigger"


def test_the_recorded_sweep_fails_the_certificate_with_its_seeds(recorded, facts, tmp_path, monkeypatch):
    sweep_record = recorded["unfenced"]["recorded_sweep"]
    monkeypatch.setenv(LO.SEEDS_ENV, str(sweep_record["knobs"]["seeds"]))
    elf = tmp_path / "package_kernel.elf"
    elf.write_bytes(b"replayed")
    sweep, calls = _replay(sweep_record["runs"])
    out = LO.run_order_check(
        target="gemmini", trace=recorded["unfenced"]["trace"], elf=elf, judge=None,
        facts=facts, engine=(tmp_path / "emulator", "replayed"), sweep=sweep,
    )  # fmt: skip
    assert out.failed and out.record["status"] == "order_sensitive"
    assert out.seeds == sweep_record["failing_seeds"] == [1, 2, 3, 4, 6, 8]
    # today's dealing of seeds over profiles is the dealing the recorded sweep ran under
    dealt = {r["seed"]: r["profile"] for r in out.record["runs"]}
    assert dealt == {r["seed"]: r["profile"] for r in sweep_record["runs"]}
    assert [c["max_latency"] for c in calls] == [p["max_latency"] for p in sweep_record["knobs"]["profiles"]]


def test_the_fenced_control_is_clear_and_never_swept(recorded, facts, tmp_path, monkeypatch):
    monkeypatch.setenv(LO.SEEDS_ENV, "8")
    assert LO.static_check(recorded["fenced"]["trace"], facts)["verdict"] == "clear"
    sweep, calls = _replay([])
    out = LO.run_order_check(
        target="gemmini", trace=recorded["fenced"]["trace"], elf=tmp_path / "absent.elf", judge=None,
        facts=facts, engine=(tmp_path / "emulator", "replayed"), sweep=sweep,
    )  # fmt: skip
    assert out.record["status"] == "static_clear" and calls == []


def test_the_fences_are_what_clear_the_control(recorded, facts):
    """MUTATION: the fenced build with its fences removed is the unfenced program with resolved
    addresses, and every overwrite/accumulate pair in it is a proven hazard."""
    rows = recorded["fenced"]["trace"]["instructions"]
    stripped = {"instructions": [r for r in rows if r["class"] != "FENCE"]}
    got = LO.static_check(stripped, facts)
    assert got["verdict"] == "hazard"
    loads = [r["index"] for r in rows if (r.get("decoded") or {}).get("spad_addr") is not None]
    overwrite_then_accumulate = set(zip(loads[0::2], loads[1::2]))
    assert overwrite_then_accumulate <= {(p["first"], p["second"]) for p in got["hazard_pairs"]}
