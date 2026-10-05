"""Efficiency diagnostics: a group's instructions decoded by derived role, counted by EXECUTION (the
functional model's PC histogram, per invocation the program states), and reported against its roofline
as numbers -- never a fix."""

from __future__ import annotations

from pathlib import Path

from merlin.perf import group_efficiency as E
from merlin.perf import isa_prohibition as ISA


def test_the_histogram_is_read_only_after_its_banner():
    text = (
        "Gemmini extension configured with:\n    dim = 16\nPC Histogram size:3\n80000000 4\n80000004 2\nbad line here\n"
    )
    assert E.pc_histogram(text) == {0x80000000: 4, 0x80000004: 2}
    assert E.pc_histogram("80000000 4\n") == {}


def test_the_program_states_how_often_it_ran_the_group():
    assert E.invocations("x\nMERLIN_INVOCATIONS warmup=1 measured=1\n") == 2
    assert E.invocations("MERLIN_INVOCATIONS warmup=0 measured=3") == 3
    assert E.invocations("nothing here") is None


_OPCODE = 0x0B
_ROLES = {0: {"roles": ["config"]}, 2: {"roles": ["operand_load"]}, 4: {"roles": ["accumulate"]},
          3: {"roles": ["commit", "readout"]}}  # fmt: skip


def _word(selector: int) -> int:
    return (selector << ISA.SELECTOR_SHIFT) | _OPCODE


def _rig(monkeypatch, tmp_path, *, histogram: str, console: str, returncode: int = 0):
    listing = [
        {"address": 0x100, "word": _word(0), "word_text": "", "mnemonic": ".insn", "function": "kernel_g7"},
        {"address": 0x104, "word": _word(2), "word_text": "", "mnemonic": ".insn", "function": "kernel_g7"},
        {"address": 0x108, "word": _word(4), "word_text": "", "mnemonic": ".insn", "function": "kernel_g7"},
        {"address": 0x10C, "word": _word(3), "word_text": "", "mnemonic": ".insn", "function": "kernel_g7"},
        {"address": 0x110, "word": 0x0FF0000F, "word_text": "", "mnemonic": "fence", "function": "kernel_g7"},
        # The harness's own accelerator code is not the group's and is not counted.
        {"address": 0x200, "word": _word(4), "word_text": "", "mnemonic": ".insn", "function": "main"},
    ]

    def run(spec, elf, out, *, timeout_s):
        out.mkdir(parents=True, exist_ok=True)
        (out / "c.txt").write_text(console)
        (out / "h.txt").write_text(histogram)
        return {"returncode": returncode, "console": out / "c.txt", "histogram": out / "h.txt"}

    monkeypatch.setattr(E, "run_functional_model", run)
    monkeypatch.setattr(ISA, "disassembler_for", lambda compiler: Path("/tc/riscv-objdump"))
    monkeypatch.setattr(ISA, "_owner_of", lambda objects, nm: {"kernel_g7": "7"})
    monkeypatch.setattr(ISA, "_declared_by_selector", lambda target: _ROLES)
    monkeypatch.setattr(ISA, "custom_opcode", lambda target: _OPCODE)
    monkeypatch.setattr(ISA, "listing", lambda elf, objdump: listing)
    return E.dynamic_census(
        tmp_path / "p.elf", target="t", compiler="/tc/riscv-gcc", group_objects={"7": tmp_path / "g7.o"},
        functional_model={"command": ["fm"]}, out=tmp_path / "fm",
    )  # fmt: skip


def test_each_role_is_counted_by_execution_per_invocation(monkeypatch, tmp_path):
    histogram = "PC Histogram size:6\n100 2\n104 8\n108 64\n10c 4\n110 2\n200 1000\n"
    census = _rig(monkeypatch, tmp_path, histogram=histogram, console="MERLIN_INVOCATIONS warmup=1 measured=1\n")
    row = census["per_group"]["7"]
    assert census["invocations"] == 2
    assert row["roles"] == {"config": 1, "operand_load": 4, "accumulate": 32, "commit": 2, "readout": 2}
    assert row["host_ordering"] == 1 and row["uneven"] == []


def test_a_count_that_does_not_divide_says_the_runs_differed(monkeypatch, tmp_path):
    histogram = "PC Histogram size:1\n108 3\n"
    census = _rig(monkeypatch, tmp_path, histogram=histogram, console="MERLIN_INVOCATIONS warmup=1 measured=1\n")
    assert census["per_group"]["7"]["uneven"] == ["accumulate"]


def test_no_stated_invocations_or_a_failed_run_is_a_refusal_not_a_count(monkeypatch, tmp_path):
    histogram = "PC Histogram size:1\n108 3\n"
    assert "refusal" in _rig(monkeypatch, tmp_path, histogram=histogram, console="no line\n")
    failed = _rig(monkeypatch, tmp_path, histogram=histogram, console="MERLIN_INVOCATIONS warmup=1 measured=1\n",
                  returncode=3)  # fmt: skip
    assert failed["refusal"] == "the functional model exited 3"


def test_efficiency_is_issued_work_against_the_roofline_per_output_tile():
    roofline = {"status": "derived", "roofline_cycles": 1000, "min_computes": 50, "output_tiles": 10,
                "cycles_per_compute_floor": 16.0, "limiter": "compute"}  # fmt: skip
    census = {"roles": {"accumulate": 100, "config": 20, "operand_load": 30, "readout": 10}, "host_ordering": 5}
    report = E.efficiency(census, roofline=roofline, cycles=3200)
    assert report["cycles_over_roofline"] == 3.2 and report["computes_over_min"] == 2.0
    assert report["cycles_per_compute"] == 32.0
    assert report["per_output_tile"] == {"config": 2.0, "operand_load": 3.0, "weight_load": 0.0, "accumulate": 10.0,
                                         "readout": 1.0, "sync": 0.0, "fence": 0.5}  # fmt: skip
    line = E.describe(report)
    assert "3,200 cyc vs roofline 1,000 (3.2x, compute-bound)" in line and "fence=0.5" in line


def test_a_refuted_roofline_is_not_a_bar():
    report = E.efficiency({"roles": {}}, roofline={"status": "refuted", "roofline_cycles": 10}, cycles=5)
    assert report["roofline_cycles"] is None and report["cycles_over_roofline"] is None
