"""The accelerator's memory path is read structurally from an elaborated FIRRTL: the module that RECEIVES a
host command and MASTERS a memory edge, with each direction's data width -- no module or config names."""

from __future__ import annotations

from merlin.targetgen.rtl import introspect as I

_CMD = (
    "{ flip ready : UInt<1>, valid : UInt<1>, bits : { inst : { funct : UInt<7>, rs2 : UInt<5>, "
    "opcode : UInt<7>}, rs1 : UInt<64>}}"
)


def _edge(name: str, *, a_bits: int, d_bits: int, manager: bool = False) -> str:
    a = (
        f"a : {{ flip ready : UInt<1>, valid : UInt<1>, bits : {{ opcode : UInt<3>, address : UInt<32>, "
        f"mask : UInt<16>, data : UInt<{a_bits}>}}}}"
    )
    d = f"d : {{ flip ready : UInt<1>, valid : UInt<1>, bits : {{ opcode : UInt<3>, data : UInt<{d_bits}>}}}}"
    return f"{name} : {{ flip {a}, {d}}}" if manager else f"{name} : {{ {a}, flip {d}}}"


def _module(name: str, *ports: str) -> str:
    body = "".join(f"    {p} @[x.scala 1:1]\n" for p in ports)
    return f"  module {name} : @[gen/x.scala 1:1]\n{body}    wire w : UInt<1>\n"


def _fir(tmp_path, *modules: str):
    path = tmp_path / "model.fir"
    path.write_text("FIRRTL version 4.0.0\ncircuit Top :\n" + "".join(modules))
    return path


def test_the_module_that_receives_the_command_and_masters_an_edge_sets_the_widths(tmp_path):
    fir = _fir(
        tmp_path,
        # A command queue receives the command but masters nothing: not the accelerator.
        _module("CmdQueue", "input clock : Clock", f"output io : {{ flip enq : {_CMD}, deq : {_CMD}}}"),
        # The host core masters an edge but SENDS the command: not the accelerator either.
        _module("Core", f"output auto : {{ {_edge('mem', a_bits=64, d_bits=64)}}}", f"output io : {{ cmd : {_CMD}}}"),
        _module(
            "Accel",
            "output auto : { "
            + _edge("mem_out", a_bits=128, d_bits=256)
            + ", "
            + _edge("cfg_in", a_bits=32, d_bits=32, manager=True)
            + "}",
            f"output io : {{ flip cmd : {_CMD}, busy : UInt<1>}}",
        ),
    )
    found = I.memory_path(fir)
    assert found["status"] == "derived" and found["module"] == "Accel"
    # The manager-side edge (a request coming IN) is not a path the accelerator moves data on.
    assert (found["read_bytes_per_cycle"], found["write_bytes_per_cycle"]) == (32, 16)
    assert [m["module"] for m in found["modules"]] == ["Accel"]


def test_two_master_edges_add_up_per_direction(tmp_path):
    fir = _fir(
        tmp_path,
        _module(
            "Accel",
            f"output auto : {{ {_edge('rd', a_bits=128, d_bits=128)}, {_edge('wr', a_bits=128, d_bits=128)}}}",
            f"output io : {{ flip cmd : {_CMD}}}",
        ),
    )
    found = I.memory_path(fir)
    assert (found["read_bytes_per_cycle"], found["write_bytes_per_cycle"]) == (32, 32)


def test_no_accelerator_means_unknown_never_a_default(tmp_path):
    fir = _fir(tmp_path, _module("Core", f"output auto : {{ {_edge('mem', a_bits=64, d_bits=64)}}}"))
    found = I.memory_path(fir)
    assert found["status"] == "unknown" and "read_bytes_per_cycle" not in found
