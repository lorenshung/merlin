"""Semantic hardware facts are READ out of a design's own sources, and a changed source changes the fact.

The fixture below is a small invented design, spelled nothing like any real target, so every test here
exercises the derivation through the probe DATA and not through a name the library could know. Each
behavioural fact has a test that flips it by mutating exactly the source construct it rests on, and an
UNKNOWN test that removes what it needs -- a derivation that cannot be falsified that way is not one.
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from merlin.targetgen.rtl import firrtl_struct as F
from merlin.targetgen.rtl import semantic_facts as SF

TARGET = "toy_semantic_target"

FIR = """FIRRTL version 3.3.0
circuit Top :
  module Tracker : @[src/Tracker.scala 1:1]
    input clock : Clock
    output io : @TRACKER_IO@

    reg entries_ld : { valid : UInt<1>, bits : { issued : UInt<1>, opa : UInt<8>}}[2], clock @[src/Tracker.scala 5:1]
    reg entries_st : { valid : UInt<1>, bits : { issued : UInt<1>, opa : UInt<8>}}[2], clock @[src/Tracker.scala 6:1]
    wire new_entry : { deps_ld : UInt<1>[2], deps_st : UInt<1>[2]} @[src/Tracker.scala 7:1]
    node is_load = eq(io.alloc.bits.funct, UInt<3>(0h5)) @[src/Tracker.scala 10:1]
    node is_store = eq(io.alloc.bits.funct, UInt<3>(0h6)) @[src/Tracker.scala 11:1]
    when is_load : @[src/Tracker.scala 12:1]
      node _T_1 = eq(entries_ld[0].bits.issued, UInt<1>(0h0)) @[src/Tracker.scala 13:5]
      node _T_2 = and(entries_ld[0].valid, _T_1) @[src/Tracker.scala 13:4]
      node _T_3 = eq(entries_ld[1].bits.issued, UInt<1>(0h0)) @[src/Tracker.scala 13:5]
      node _T_4 = and(entries_ld[1].valid, _T_3) @[src/Tracker.scala 13:4]
      connect new_entry.deps_ld[0], _T_2 @[src/Tracker.scala 13:3]
      connect new_entry.deps_ld[1], _T_4 @[src/Tracker.scala 13:3]
      node _T_5 = eq(entries_st[0].bits.opa, UInt<8>(0h1)) @[src/Tracker.scala 14:3]
      node _T_6 = and(entries_st[0].valid, _T_5) @[src/Tracker.scala 14:2]
      connect new_entry.deps_st[0], _T_6 @[src/Tracker.scala 14:1]
    else :
      connect new_entry.deps_ld[0], entries_ld[0].valid @[src/Tracker.scala 16:1]
      connect new_entry.deps_st[0], entries_st[0].valid @[src/Tracker.scala 17:1]
    when io.completed.valid : @[src/Tracker.scala 20:1]
      connect entries_ld[0].valid, UInt<1>(0h0) @[src/Tracker.scala 21:1]
      connect entries_st[0].valid, UInt<1>(0h0) @[src/Tracker.scala 22:1]
    when io.issue.ready : @[src/Tracker.scala 24:1]
      connect entries_ld[0].bits.issued, UInt<1>(0h1) @[src/Tracker.scala 25:1]
      connect entries_st[0].bits.issued, UInt<1>(0h1) @[src/Tracker.scala 26:1]
    node _busy = or(entries_ld[0].valid, entries_st[0].valid) @[src/Tracker.scala 30:1]
    connect io.busy, _busy @[src/Tracker.scala 30:2]

  module Host : @[src/Host.scala 1:1]
    output io : { flip coproc : { busy : UInt<1>}}
    wire dec : { barrier : UInt<1>} @[src/Host.scala 2:1]
    connect dec.barrier, UInt<1>(0h1) @[src/Host.scala 2:2]
    node _stall = and(io.coproc.busy, dec.barrier) @[src/Host.scala 3:1]
    wire hold : UInt<1> @[src/Host.scala 4:1]
    connect hold, _stall @[src/Host.scala 4:2]

  module Accel : @[src/Accel.scala 1:1]
    output io : { busy : UInt<1>, flip cmd : { funct : UInt<7>}}
    inst tracker of Tracker @[src/Accel.scala 2:1]
    inst mmu of Mmu @[src/Accel.scala 2:2]
    connect io.busy, tracker.io.busy @[src/Accel.scala 3:1]
    node is_drain = eq(io.cmd.funct, UInt<3>(0h7)) @[src/Accel.scala 4:1]
    when is_drain : @[src/Accel.scala 5:1]
      connect mmu.io.drop, UInt<1>(0h1) @[src/Accel.scala 5:2]

  module Buffers : @[src/Buffers.scala 1:1]
    inst scaler of Scaler @[src/Buffers.scala 2:1]
    wire bank : { write : { data : SInt<32>[2]}}[1] @[src/Buffers.scala 3:1]
    node sign = bits(scaler.io.out[0], 7, 7) @[src/Buffers.scala 4:1]
    node _hi = cat(sign, sign) @[src/Buffers.scala 4:2]
    node _cat = cat(_hi, asUInt(scaler.io.out[0])) @[src/Buffers.scala 4:3]
    wire _w : SInt<32> @[src/Buffers.scala 4:4]
    connect _w, asSInt(_cat) @[src/Buffers.scala 4:5]
    connect bank[0].write.data[0], _w @[src/Buffers.scala 5:1]

  module Writer : @[src/Writer.scala 1:1]
    output io : { flip req : { valid : UInt<1>, data : UInt<64>}}
""".replace(
    "@TRACKER_IO@",
    "{ flip alloc : { valid : UInt<1>, bits : { funct : UInt<7>}}, flip completed : { valid : UInt<1>}, "
    "flip issue : { ready : UInt<1>}, busy : UInt<1>}",
)

HEADER = """#ifndef TOY_H
#define TOY_H
#include <stdint.h>
#define ROWS 2
typedef int8_t op_t;
typedef int32_t wide_t;
#define LOAD_SCALE(x, scale) \\
    ({float y = RNE((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (op_t)y);})
#define LOAD_SCALE_WIDE(x, scale) (x)
#define READ_NARROW
#define READ_WIDE
#endif
"""

PROBES = {
    "schema": "semantic_probes_v1",
    "target": TARGET,
    "sources": {},  # filled per test
    "load_completion_ordering": {
        "source": "elaboration",
        "tracker": {
            "module": "Tracker",
            "command_field": "io.alloc.bits.funct",
            "subject": "load",
            "queues": {
                "load": {
                    "entries": "entries_ld",
                    "dependency": "new_entry.deps_ld",
                    "guard": ["is_load"],
                    "membership": "is_load",
                },
                "store": {
                    "entries": "entries_st",
                    "dependency": "new_entry.deps_st",
                    "guard": ["not is_load"],
                    "membership": "is_store",
                },
            },
            "entry_fields": {"live": "valid", "issued": "bits.issued"},
            "address_fields": ["bits.opa"],
            "retire_port": "io.completed",
            "issue_port": "io.issue",
            "busy_output": "io.busy",
        },
        "barrier": {
            "host_module": "Host",
            "stall": "hold",
            "decode_field": "dec.barrier",
            "accelerator_busy_input": "io.coproc.busy",
            "instruction": "drainall",
            "accelerator_module": "Accel",
            "accelerator_busy_output": "io.busy",
            "tracker_instance": "tracker",
        },
        "sync_decode": {
            "module": "Accel",
            "command_field": "io.cmd.funct",
            "selectors": [{"guard": ["is_drain"], "membership": "is_drain"}],
        },
    },
    "load_scale_saturation": {
        "header": "header",
        "macros": {"operand": "LOAD_SCALE", "accumulator": "LOAD_SCALE_WIDE"},
        "types": {"operand": "op_t", "accumulator": "wide_t"},
        "accumulator_entry": {
            "source": "elaboration",
            "module": "Buffers",
            "accumulator_write": "bank[*].write.data",
            "scaled_operand": "scaler.io.out",
        },
    },
    "accumulator_readout_width": {
        "macros": {"full": "READ_WIDE", "narrow": "READ_NARROW"},
        "types": {"full": "wide_t", "narrow": "op_t"},
        "row_elements_macro": "ROWS",
        "writer": {"module": "Writer", "port": "io", "path": ["req", "data"]},
        "machines": {"m_one": {"header": "header", "firrtl": "elaboration"}},
    },
}


def _setup(tmp_path: Path, fir: str = FIR, header: str = HEADER, probes: dict | None = None) -> Path:
    (tmp_path / "design.fir").write_text(fir)
    (tmp_path / "params.h").write_text(header)
    doc = json.loads(json.dumps(probes or PROBES))
    doc["sources"] = {
        "elaboration": {"file": str(tmp_path / "design.fir")},
        "header": {"file": str(tmp_path / "params.h")},
    }
    p = tmp_path / "semantic_probes.yaml"
    p.write_text(yaml.safe_dump(doc))
    return p


def _derive(tmp_path: Path, **kw) -> dict:
    return SF.derive(TARGET, probes_file=_setup(tmp_path, **kw))


# ------------------------------------------------------------------------------------ the FIRRTL reader


def test_reader_keeps_when_else_guards_and_walks_through_nodes_and_wires(tmp_path):
    (tmp_path / "d.fir").write_text(FIR)
    mod = F.load_modules(tmp_path / "d.fir", ["Tracker"])["Tracker"]
    guarded = {(c.lhs.text(), tuple(g.text() for g in c.guards)) for c in mod.connects}
    assert ("new_entry.deps_ld[0]", ("is_load",)) in guarded
    assert ("new_entry.deps_ld[0]", ("not is_load",)) in guarded
    lhs = next(c for c in mod.connects if c.lhs.text() == "new_entry.deps_ld[0]" and c.guards[0].positive)
    assert F.Cone(mod).leaves(lhs.rhs) == {"entries_ld[0].valid", "entries_ld[0].bits.issued"}
    # an undriven flipped field of an OUTPUT bundle is an input the module reads: a leaf, never dropped
    assert F.Cone(mod).leaves(F.parse_expr("io.completed.valid")) == {"io.completed.valid"}
    assert F.locator_citation(lhs.locator) == {"file": "src/Tracker.scala", "line": 13}


def test_type_parser_reads_a_nested_port_width():
    typ = F.parse_type("{ flip req : { valid : UInt<1>, data : UInt<64>}, busy : UInt<1>[3]}")
    assert F.type_width(F.field_type(typ, ["req", "data"])) == 64
    assert F.type_width(F.field_type(typ, ["busy"])) == 3


# ------------------------------------------------------------------------------ load_completion_ordering


def test_ordering_is_derived_from_the_dependency_expression(tmp_path):
    fact = _derive(tmp_path)["facts"]["load_completion_ordering"]
    assert fact["status"] == SF.DERIVED
    v = fact["value"]
    assert v["same_class"] == {"load": SF.ISSUE_ORDERED_ONLY, "store": SF.COMPLETION_ORDERED}
    assert v["classes"]["load"]["selectors"] == [5]
    load_st = next(p for p in v["pairs"] if p["later"] == "load" and p["earlier"] == "store")
    assert load_st["released_at"] == SF.RELEASED_AT_COMPLETION and load_st["address_conditional"] is True
    assert v["field_checks"] == {"live_cleared_under_retire_port": True, "issued_set_under_issue_port": True}
    assert fact["provenance"]["kind"] == "firrtl_structural" and fact["provenance"]["derived"] is True
    assert fact["provenance"]["sources"][0]["sha256"]


def test_the_barrier_is_the_instruction_whose_stall_waits_on_the_trackers_busy(tmp_path):
    v = _derive(tmp_path)["facts"]["load_completion_ordering"]["value"]
    b = v["completion_barrier"]
    assert b["status"] == SF.DERIVED and b["instruction"] == "drainall"
    assert "drainall" in v["summary"]
    # the declared sync instruction's decode never touches the tracker: not a barrier
    (sync,) = v["sync_role_instructions"]
    assert sync["selector"] == 7 and sync["touches_dependency_tracker"] is False
    assert sync["is_completion_barrier"] is False


def test_mutating_the_dependency_to_wait_for_retirement_flips_the_fact(tmp_path):
    mutated = FIR.replace(
        "node _T_2 = and(entries_ld[0].valid, _T_1)", "node _T_2 = and(entries_ld[0].valid, UInt<1>(0h1))"
    ).replace("node _T_4 = and(entries_ld[1].valid, _T_3)", "node _T_4 = and(entries_ld[1].valid, UInt<1>(0h1))")
    v = _derive(tmp_path, fir=mutated)["facts"]["load_completion_ordering"]["value"]
    assert v["same_class"]["load"] == SF.COMPLETION_ORDERED
    assert "COMPLETION-ORDERED" in v["summary"]


def test_a_stall_that_ignores_the_accelerator_is_not_a_barrier(tmp_path):
    mutated = FIR.replace(
        "node _stall = and(io.coproc.busy, dec.barrier)", "node _stall = and(UInt<1>(0h0), dec.barrier)"
    )
    b = _derive(tmp_path, fir=mutated)["facts"]["load_completion_ordering"]["value"]["completion_barrier"]
    assert b["status"] == SF.UNKNOWN and b["stall_waits_on_accelerator_busy"] is False


def test_ordering_is_unknown_without_the_elaboration_or_the_module(tmp_path):
    probes = _setup(tmp_path)
    (tmp_path / "design.fir").unlink()
    fact = SF.derive(TARGET, probes_file=probes)["facts"]["load_completion_ordering"]
    assert fact["status"] == SF.UNKNOWN and "not present" in fact["unknown_reason"]

    renamed = _derive(tmp_path, fir=FIR.replace("module Tracker :", "module Tracker2 :"))
    fact = renamed["facts"]["load_completion_ordering"]
    assert fact["status"] == SF.UNKNOWN and "Tracker" in fact["unknown_reason"]


# --------------------------------------------------------------------------------- load_scale_saturation


def test_saturation_is_derived_from_the_header_and_the_accumulator_entry(tmp_path):
    fact = _derive(tmp_path)["facts"]["load_scale_saturation"]
    assert fact["status"] == SF.DERIVED
    op = fact["value"]["operand_load"]
    assert op["saturates"] is True and op["range"] == [-128, 127] and op["saturates_to_bits"] == 8
    assert op["stage"] == "before_accumulate" and op["stage_provenance"] == SF.DERIVED
    assert fact["value"]["accumulator_load"]["scaled"] is False
    assert fact["value"]["accumulator_entry"]["enters_at_bits"] == [8]


def test_a_load_scale_without_a_clamp_does_not_saturate(tmp_path):
    header = HEADER.replace(
        "({float y = RNE((x) * (scale)); y > INT8_MAX ? INT8_MAX : (y < INT8_MIN ? INT8_MIN : (op_t)y);})",
        "((op_t)RNE((x) * (scale)))",
    )
    fact = _derive(tmp_path, header=header)["facts"]["load_scale_saturation"]
    assert fact["value"]["operand_load"]["saturates"] is False
    assert fact["status"] == SF.UNKNOWN


def test_a_full_width_accumulator_entry_is_not_before_accumulate(tmp_path):
    mutated = FIR.replace("node sign = bits(scaler.io.out[0], 7, 7)", "node sign = bits(scaler.io.out[0], 31, 31)")
    fact = _derive(tmp_path, fir=mutated)["facts"]["load_scale_saturation"]
    assert fact["value"]["operand_load"]["stage"] == SF.UNKNOWN
    assert fact["status"] == SF.UNKNOWN


def test_saturation_is_unknown_without_a_header(tmp_path):
    probes = _setup(tmp_path)
    (tmp_path / "params.h").unlink()
    fact = SF.derive(TARGET, probes_file=probes)["facts"]["load_scale_saturation"]
    assert fact["status"] == SF.UNKNOWN and "header unavailable" in fact["unknown_reason"]


# ---------------------------------------------------------------------------- accumulator_readout_width


def test_readout_width_agrees_between_header_and_elaboration(tmp_path):
    m = _derive(tmp_path)["facts"]["accumulator_readout_width"]["value"]["machines"]["m_one"]
    assert m["status"] == SF.DERIVED and m["full_width_readout"] is True
    assert m["firrtl"] == {**m["firrtl"], "writer_data_bits": 64, "bits_per_element": 32, "agrees_with_header": True}


def test_removing_the_full_width_macro_flips_the_header_and_conflicts_with_the_design(tmp_path):
    fact = _derive(tmp_path, header=HEADER.replace("#define READ_WIDE\n", ""))["facts"]["accumulator_readout_width"]
    m = fact["value"]["machines"]["m_one"]
    assert m["full_width_readout"] is False and m["status"] == SF.CONFLICT
    assert fact["status"] == SF.CONFLICT


def test_a_narrow_design_with_a_narrow_header_is_a_narrow_machine(tmp_path):
    fir = FIR.replace("data : UInt<64>}}", "data : UInt<16>}}")
    fact = _derive(tmp_path, fir=fir, header=HEADER.replace("#define READ_WIDE\n", ""))["facts"]
    v = fact["accumulator_readout_width"]["value"]
    assert v["machines"]["m_one"]["status"] == SF.DERIVED and v["narrow_only_machines"] == ["m_one"]


# ------------------------------------------------------------------------------------ document surfaces


def test_a_target_without_probes_is_unknown_and_renders_nothing(tmp_path):
    doc = SF.derive("no_such_target_for_semantic_facts", probes_file=tmp_path / "absent.yaml")
    assert doc["status"] == SF.UNKNOWN and doc["facts"] == {}
    assert SF.brief(doc) == ""


def test_compact_and_brief_carry_status_provenance_and_the_fix_instruction(tmp_path):
    doc = _derive(tmp_path)
    c = SF.compact(doc)
    row = c["facts"]["load_completion_ordering"]
    assert row["completion_barrier"]["instruction"] == "drainall"
    assert row["provenance"] == "firrtl_structural" and row["needs_review"] is False
    text = SF.brief(doc)
    assert "load_completion_ordering" in text and "drainall" in text
    assert "load_completion_ordering" in SF.task_paragraph(c)


# ------------------------------------------------------------------- a machine's header by its registry digest


def test_a_machine_header_is_found_by_the_registry_abi_digest_among_supplied_headers(tmp_path, monkeypatch):
    """No host path in the probes: the header a machine's programs compile against is whichever supplied
    header has the ABI digest the hardware registry declares for that machine."""
    import hashlib

    probes = json.loads(json.dumps(PROBES))
    probes["accumulator_readout_width"]["machines"] = {"m_one": {"header": {"abi_header_of": "m_one"}}}
    path = _setup(tmp_path, probes=probes)
    other = tmp_path / "other.h"
    other.write_text(HEADER.replace("#define READ_WIDE\n", ""))
    digest = hashlib.sha256((tmp_path / "params.h").read_bytes()).hexdigest()
    monkeypatch.setattr(SF, "registry_abi_header", lambda machine: ("m_one", digest))
    fact = SF.derive(
        TARGET, probes_file=path, headers=[other, tmp_path / "params.h"], only=["accumulator_readout_width"]
    )
    m = fact["facts"]["accumulator_readout_width"]["value"]["machines"]["m_one"]
    assert m["status"] == SF.DERIVED and m["full_width_readout"] is True
    assert set(fact["facts"]) == {"accumulator_readout_width"}  # only what was asked for
    missing = SF.derive(TARGET, probes_file=path, headers=[other], only=["accumulator_readout_width"])
    row = missing["facts"]["accumulator_readout_width"]
    assert row["status"] == SF.UNKNOWN
    assert "declared ABI digest" in row["value"]["machines"]["m_one"]["why"]


def test_probes_are_read_from_the_selected_providers_contracts(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin.targetgen import target_registry

    contracts = tmp_path / "provider" / "contracts"
    contracts.mkdir(parents=True)
    monkeypatch.setattr(
        target_registry, "resolve", lambda target: SimpleNamespace(contract_path=contracts / "target_contract.yaml")
    )
    assert SF.probes_path(TARGET) == contracts / SF.PROBES_FILE
