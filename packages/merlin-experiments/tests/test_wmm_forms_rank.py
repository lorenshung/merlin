"""Each stated group carries the validated schedule cost proxy's rank beside its derived bound."""

from __future__ import annotations

from merlin_experiments.phase2.whole_model_measured import forms as FORMS

from merlin.perf import derived_bound as DB
from merlin.perf import group_headroom as GH
from merlin.perf import whole_model_build as W


def _machine(operand_bytes) -> DB.Machine:
    return DB.Machine(
        array_rows=16,
        array_cols=16,
        muls_per_element=1,
        operand_bytes=operand_bytes,
        accumulate_bytes=4,
        readout_bytes=DB.UNKNOWN,
        ping_pong_ways=2,
        operand_store_bytes=131072,
        accumulate_store_bytes=32768,
        fill_drain_cycles=DB.UNKNOWN,
        dram_bytes_per_cycle=DB.UNKNOWN,
        refusals={} if operand_bytes is not DB.UNKNOWN else {"operand_bytes": "no datapath width"},
    )


def _entry(m, k, n) -> dict:
    return {"op": "matmul", "M": m, "K": k, "N": n, "operand_dtype": "i8", "output_dtype": "i8", "epilogue": []}


def _stated(monkeypatch, machine) -> list[dict]:
    program = {
        "whole_program": {
            "per_group": [
                {"group": 1, "entry": _entry(64, 64, 64), "operands": {}},
                {"group": 2, "entry": _entry(512, 1024, 512), "operands": {}},
                {"group": 3, "entry": {"op": "residual_add"}, "operands": {}},
            ]
        }
    }
    monkeypatch.setattr(W, "load_model_capsule", lambda path: path)
    monkeypatch.setattr(W, "state", lambda capsule, target: program)
    monkeypatch.setattr(GH, "machine_for", lambda target: machine)
    return {row["group"]: row for row in FORMS.statement_forms("/capsule", target="any")}


def test_a_stated_group_is_ranked_by_the_proxy_beside_its_bound(monkeypatch):
    rows = _stated(monkeypatch, _machine(1))
    small, large = rows["1"], rows["2"]
    assert small["predicted_cycles"] is not None
    assert 0 < small["rank_transactions"] < large["rank_transactions"]
    assert small["rank_regime"] == "UNDECIDED"  # no measured boundary was supplied
    # a group with no contraction is unranked, never ranked zero
    assert rows["3"]["rank_transactions"] is None and rows["3"]["predicted_cycles"] is None


def test_an_unsized_transaction_leaves_the_rank_unknown_not_zero(monkeypatch):
    rows = _stated(monkeypatch, _machine(DB.UNKNOWN))
    assert rows["1"]["rank_transactions"] is None
    assert rows["1"]["predicted_cycles"] is not None  # the bound does not need the operand width
