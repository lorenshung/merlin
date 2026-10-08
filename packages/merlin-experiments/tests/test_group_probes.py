"""Single-group probes: one group's counters and timing, from the target's own facts and the run's own record.

Everything target-shaped here is MADE UP -- a counter header with its own prefix and engine tokens, an
instruction set with its own opcode and roles, an engine the trust table is told about -- so the probes
are held to deriving what they report, never to knowing one target's spellings.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from merlin_experiments import group_probes as GP

from merlin.perf import counter_trust as CT
from merlin.perf import hw_counters as HWC
from merlin.perf import isa_prohibition as ISA
from merlin.perf import task_instruction_evidence as TIE

#: Two engines (A, B), their overlap, and a duration counter outside the busy block.
HEADER = """
#define UNIT_A_CYCLES 10
#define UNIT_B_CYCLES 11
#define UNIT_A_B_CYCLES 12
#define UNIT_A_WAIT_CYCLES 20
#define UNIT_A_BYTES 30
"""
CONSOLE = """
noise MERLIN_HWCOUNTER not-a-reading
MERLIN_HWCOUNTER UNIT_A_CYCLES 700
MERLIN_HWCOUNTER UNIT_B_CYCLES 300
MERLIN_HWCOUNTER UNIT_A_B_CYCLES 90
MERLIN_HWCOUNTER UNIT_A_WAIT_CYCLES 41
"""
TRUSTED, FABRICATING = "fixture-rtl", "fixture-iss"


@pytest.fixture
def header(tmp_path, monkeypatch):
    path = tmp_path / "counters.h"
    path.write_text(HEADER, encoding="utf-8")
    digest = hashlib.sha256(HEADER.encode()).hexdigest()
    monkeypatch.setattr(
        HWC,
        "counters_for_target",
        lambda _target: {"status": "derived", "header": str(path), "header_sha256": digest},
    )
    monkeypatch.setattr(
        CT,
        "_declared",
        lambda: {TRUSTED: CT.Verdict(TRUSTED, CT.REAL), FABRICATING: CT.Verdict(FABRICATING, CT.FABRICATED)},
    )
    return digest


def test_counter_names_are_factored_out_of_the_targets_own_header(header):
    facts = GP.counter_facts("any")
    assert facts["status"] == "derived" and facts["engines"] == ["A", "B"] and facts["complete"]
    assert facts["busy_counters"] == {"A": "UNIT_A_CYCLES", "B": "UNIT_B_CYCLES", "A+B": "UNIT_A_B_CYCLES"}
    # A duration counter outside the busy block is still reported, under its own name; a byte counter is not.
    assert facts["other_duration_counters"] == ["UNIT_A_WAIT_CYCLES"]


def test_values_are_read_only_from_a_trusted_engines_console(header):
    facts = GP.counter_facts("any")
    measured = GP.counter_values(facts, CONSOLE, engine=TRUSTED)
    assert measured["status"] == "measured" and measured["schema"] == GP.UNKNOWN
    assert measured["busy_cycles"] == {"A": 700, "B": 300, "A+B": 90}
    assert measured["other_duration_counters"] == {"UNIT_A_WAIT_CYCLES": 41}
    for engine in (FABRICATING, "never-declared", None):
        refused = GP.counter_values(facts, CONSOLE, engine=engine)
        assert refused["status"] == "refused" and "busy_cycles" not in refused, engine


def test_a_console_without_readings_is_unknown_never_zero(header):
    facts = GP.counter_facts("any")
    assert GP.counter_values(facts, "", engine=TRUSTED)["status"] == GP.UNKNOWN
    silent = GP.counter_values(facts, "the program printed no counters\n", engine=TRUSTED)
    assert silent["status"] == GP.UNKNOWN and "did not print" in silent["why"]


def test_a_console_built_against_another_counter_header_is_refused(header):
    facts = GP.counter_facts("any")
    same = GP.counter_values(facts, f"MERLIN_COUNTER_SCHEMA {header}\n" + CONSOLE, engine=TRUSTED)
    assert same["status"] == "measured" and same["schema"] == header
    other = GP.counter_values(facts, f"MERLIN_COUNTER_SCHEMA {'e' * 64}\n" + CONSOLE, engine=TRUSTED)
    assert other["status"] == "refused" and "schema" in other["why"]


def test_a_target_with_no_readable_header_is_unavailable_not_absent(monkeypatch):
    monkeypatch.setattr(HWC, "counters_for_target", lambda _t: {"status": "unavailable", "why": "no header"})
    facts = GP.counter_facts("any")
    assert facts["status"] == "unavailable"
    assert GP.counter_values(facts, CONSOLE, engine=TRUSTED)["status"] == GP.UNKNOWN


# ------------------------------------------------------------------------------------------ census

OPCODE = 0x2B
FACTS = {
    "isa": {"CUSTOM_OPCODE": OPCODE},
    "instruction_names": {"1": "MOVE", "4": "STEP", "9": "WAIT"},
    "roles_by_selector": {"1": ["data_movement"], "4": ["compute"], "9": ["sync"]},
}


def _listing(functions: dict[str, list[int]]) -> str:
    lines, address = [], 0x80000000
    for function, selectors in functions.items():
        lines.append(f"{address:016x} <{function}>:")
        for selector in selectors:
            word = f"{(selector << ISA.SELECTOR_SHIFT) | OPCODE:08x}"
            lines.append(f"    {address:x}:\t{word}          \t.insn\t4, 0x{word}")
            address += 4
    return "\n".join(lines) + "\n"


@pytest.fixture
def program(tmp_path, monkeypatch):
    """A one-group program record whose ELF holds the group's kernel and the program's own code."""
    monkeypatch.setattr(TIE, "target_instruction_facts", lambda _target: FACTS)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("cross-cc", "cross-objdump", "cross-nm"):
        (tools / name).write_text("")
    listing = _listing({"group_kernel": [1, 1, 4, 4, 4, 1], "main": [9, 1]})

    def run(argv, **_kw):
        if Path(argv[0]).name.endswith("objdump"):
            return subprocess.CompletedProcess(argv, 0, stdout=listing, stderr="")
        names = ["group_kernel"] if Path(argv[-1]).name == "g7.o" else []
        return subprocess.CompletedProcess(argv, 0, stdout="".join(f"0 T {n}\n" for n in names), stderr="")

    monkeypatch.setattr(ISA.subprocess, "run", run)
    for name in ("program.elf", "g7.o"):
        (tmp_path / name).write_bytes(b"\x00")
    return {
        "group": 7,
        "linked": "submission",
        "elf": str(tmp_path / "program.elf"),
        "variant": {"objects": [str(tmp_path / "g7.o")], "program": {"compiler": str(tools / "cross-cc")}},
    }


def test_the_census_splits_the_groups_kernel_from_the_programs_own_code(program):
    census = GP.instruction_census(program, target="any")
    assert census["status"] == "measured"
    assert census["kernel"]["by_kind"] == {"data_movement": 3, "compute": 3}
    assert census["program_code"]["by_kind"] == {"sync": 1, "data_movement": 1}


def test_a_program_that_did_not_build_has_no_census(program):
    census = GP.instruction_census({**program, "elf": None, "refusal": "link failed"}, target="any")
    assert census["status"] == GP.UNKNOWN and "no program" in census["why"]


def test_an_underived_instruction_set_leaves_the_census_unknown(program, monkeypatch):
    monkeypatch.setattr(TIE, "target_instruction_facts", lambda _target: {"isa": {}})
    assert GP.instruction_census(program, target="any")["status"] == GP.UNKNOWN


# ------------------------------------------------------------------------------------------ timing


@pytest.fixture
def emulator(monkeypatch):
    """time_group_programs replaced by one that writes the run directory a real run writes."""
    from merlin.perf import whole_model_group_timing as GT

    calls = []

    def time_group_programs(programs, *, out, cache, max_parallel, **kwargs):
        calls.append({"groups": sorted(programs), "cache": cache, "max_parallel": max_parallel, **kwargs})
        rows = {}
        for group in programs:
            run = Path(out) / f"g{group}"
            run.mkdir(parents=True)
            (run / "console.txt").write_text(CONSOLE, encoding="utf-8")
            adjudication = {"instrument": TRUSTED, "state": "UNADJUDICATED", "claim_kind": "cycle_count"}
            (run / "verdict.json").write_text(json.dumps({"cycles_adjudication": adjudication}), encoding="utf-8")
            rows[group] = {"group": group, "status": "graded", "cycles": 1234, "correct": True, "run_dir": str(run)}
        return rows

    monkeypatch.setattr(GT, "time_group_programs", time_group_programs)
    return calls


def test_timing_is_one_fresh_run_tagged_a_ranking_signal(emulator, tmp_path):
    timing = GP.time_group({"group": 7}, target="any", model_capsule="capsule", out=tmp_path / "t")
    assert emulator == [
        {
            "groups": [7],
            "cache": None,
            "max_parallel": 1,
            "target": "any",
            "model_capsule": "capsule",
            "max_cycles": 60_000_000,
            "timeout_s": 1800.0,
        }
    ]
    assert timing["status"] == "graded" and timing["cycles"] == 1234 and timing["correct"] is True
    assert timing["signal"] == "ranking" and "never quoted" in timing["not_a_claim"]
    assert timing["cycles_adjudication"]["state"] == "UNADJUDICATED"
    assert GP.engine_of(timing) == TRUSTED and Path(timing["console"]).is_file()


def test_a_run_without_its_own_record_says_so(monkeypatch, tmp_path):
    from merlin.perf import whole_model_group_timing as GT

    monkeypatch.setattr(GT, "time_group_programs", lambda programs, **_kw: {7: {"group": 7, "status": "refused"}})
    timing = GP.time_group({"group": 7}, target="any", model_capsule="c", out=tmp_path)
    assert timing["cycles_adjudication"]["status"] == GP.UNKNOWN and timing["console"] is None
    assert GP.engine_of(timing) is None
