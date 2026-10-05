"""The whole-program instruction prohibition, held to what it must refuse and what it must pass.

* the prohibited set is whatever the target's facts give a prohibited ROLE -- change the roles and
  the set follows, with no instruction name in the check;
* a prohibited instruction in the program's OWN code (a group the library answers, or host code) is
  refused, as is one in a group's kernel, each attributed to where it sits;
* a clean program passes, and a check that cannot run refuses rather than passes;
* the service refuses before any run, leading with ``isa_prohibited: <instr> in <where>``, and the
  reference -- the program the rule is measured against -- is exempt;
* the library header's copy loses every macro that would issue a prohibited instruction, and the
  library's groups are called with the loop-free path they were routed to.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.perf import isa_prohibition as ISA
from merlin.perf import task_instruction_evidence as TIE

#: A made-up target: its own opcode, its own names, its own roles. Nothing here is a real encoding.
OPCODE = 0x2B
FACTS = {
    "isa": {"CUSTOM_OPCODE": OPCODE},
    "instruction_names": {"1": "MOVE", "5": "STEP_A", "6": "STEP_B", "9": "FENCE_LIKE"},
    "roles_by_selector": {"1": ["data_movement"], "5": ["loop_descriptor"], "6": ["loop_descriptor"], "9": ["sync"]},
}


def _word(selector: int) -> str:
    return f"{(selector << ISA.SELECTOR_SHIFT) | OPCODE:08x}"


def _listing(functions: dict[str, list[int]]) -> str:
    lines, address = [], 0x80000000
    for function, selectors in functions.items():
        lines.append(f"{address:016x} <{function}>:")
        for selector in selectors:
            lines.append(f"    {address:x}:\t{_word(selector)}          \t.insn\t4, 0x{_word(selector)}")
            address += 4
        lines.append(f"    {address:x}:\t00000013          \tnop")
        address += 4
    return "\n".join(lines) + "\n"


@pytest.fixture
def target(monkeypatch, tmp_path):
    monkeypatch.setattr(TIE, "target_instruction_facts", lambda _target: FACTS)
    tools = tmp_path / "tools"
    tools.mkdir()
    for name in ("cross-gcc", "cross-objdump", "cross-nm"):
        (tools / name).write_text("")
    return SimpleNamespace(compiler=tools / "cross-gcc", tmp=tmp_path)


def _fake_tools(monkeypatch, listing: str, symbols: dict[str, list[str]]):
    def run(argv, **_kw):
        tool = Path(argv[0]).name
        if tool.endswith("objdump"):
            return subprocess.CompletedProcess(argv, 0, stdout=listing, stderr="")
        names = symbols.get(Path(argv[-1]).name, [])
        return subprocess.CompletedProcess(argv, 0, stdout="".join(f"0 T {n}\n" for n in names), stderr="")

    monkeypatch.setattr(ISA.subprocess, "run", run)


def test_the_prohibited_set_is_derived_from_role_data(target, monkeypatch):
    assert ISA.prohibited_instructions("any", ["loop_descriptor"]) == {5: "STEP_A", 6: "STEP_B"}
    assert ISA.prohibited_instructions("any", ["sync"]) == {9: "FENCE_LIKE"}
    moved = {**FACTS, "roles_by_selector": {**FACTS["roles_by_selector"], "1": ["loop_descriptor"]}}
    monkeypatch.setattr(TIE, "target_instruction_facts", lambda _target: moved)
    assert ISA.prohibited_instructions("any", ["loop_descriptor"]) == {1: "MOVE", 5: "STEP_A", 6: "STEP_B"}


def test_an_underived_instruction_set_fails_closed(monkeypatch):
    monkeypatch.setattr(TIE, "target_instruction_facts", lambda _target: {"isa": {}})
    with pytest.raises(ValueError, match="not derived"):
        ISA.prohibited_instructions("any", ["loop_descriptor"])


def test_a_prohibited_instruction_in_a_library_group_is_refused(target, monkeypatch):
    kernel = target.tmp / "g2.o"
    kernel.write_text("")
    _fake_tools(
        monkeypatch,
        _listing({"main": [1], "library_tiled_conv": [5, 5, 6, 1], "kernel_g2": [1, 9]}),
        {"g2.o": ["kernel_g2"]},
    )
    report = ISA.check_program(
        target.tmp / "p.elf",
        target="any",
        roles=["loop_descriptor"],
        compiler=target.compiler,
        group_objects={"2": kernel},
        library_groups=["1"],
    )
    assert not report["clean"]
    assert report["summary"] == {"STEP_A in the program code": 2, "STEP_B in the program code": 1}
    assert {h["function"] for h in report["hits"]} == {"library_tiled_conv"}
    assert ISA.refusal_line(report).startswith("isa_prohibited: STEP_A in the program code")


def test_a_prohibited_instruction_in_a_kernel_names_its_group(target, monkeypatch):
    kernel = target.tmp / "g7.o"
    kernel.write_text("")
    _fake_tools(monkeypatch, _listing({"main": [1], "kernel_g7": [6]}), {"g7.o": ["kernel_g7"]})
    report = ISA.check_program(
        target.tmp / "p.elf",
        target="any",
        roles=["loop_descriptor"],
        compiler=target.compiler,
        group_objects={"7": kernel},
    )
    assert report["summary"] == {"STEP_B in g7": 1}


def test_check_program_carries_a_per_group_instruction_census(target, monkeypatch):
    """The census is diagnostic, not a verdict: it counts EVERY custom instruction by its derived
    role and the group (or the program's own code) that issues it, whether or not that role is
    prohibited -- and a selector the target's facts do not name is still counted, under its own
    selector, rather than dropped."""
    kernel = target.tmp / "g2.o"
    kernel.write_text("")
    _fake_tools(
        monkeypatch,
        _listing({"main": [1], "kernel_g2": [1, 9, 3]}),
        {"g2.o": ["kernel_g2"]},
    )
    report = ISA.check_program(
        target.tmp / "p.elf",
        target="any",
        roles=["loop_descriptor"],
        compiler=target.compiler,
        group_objects={"2": kernel},
    )
    census = report["census"]
    assert census["schema"] == ISA.CENSUS_SCHEMA
    per_group = census["per_group"]
    assert per_group["2"] == {
        "total": 3,
        "by_kind": {"data_movement": 1, "sync": 1, f"{ISA.UNNAMED_SELECTOR}_3": 1},
        "by_symbol": {"kernel_g2": 3},
    }
    assert per_group[ISA.PROGRAM_CODE] == {
        "total": 1,
        "by_kind": {"data_movement": 1},
        "by_symbol": {"main": 1},
    }
    # Nothing prohibited was asked for in either group here, so the verdict stays clean; the census
    # still saw every instruction -- the two are independent readings of the same disassembly.
    assert report["clean"]


def test_a_clean_program_passes(target, monkeypatch):
    kernel = target.tmp / "g2.o"
    kernel.write_text("")
    _fake_tools(monkeypatch, _listing({"main": [1, 9], "kernel_g2": [1]}), {"g2.o": ["kernel_g2"]})
    report = ISA.check_program(
        target.tmp / "p.elf",
        target="any",
        roles=["loop_descriptor"],
        compiler=target.compiler,
        group_objects={"2": kernel},
    )
    assert report["clean"] and report["summary"] == {}


def _driver():
    from selected_driver import load

    return load("gemmini", "group_model_program.py")


HEADER = """#define k_STEP_A 5
#define k_STEP_B 6
#define k_MOVE 1
#define issue_step(a, b) \\
  ROCC(OPC, a, b, k_STEP_A)
#define issue_both(a) { ROCC(OPC, a, 0, k_STEP_B); ROCC(OPC, a, 0, k_MOVE); }
#define issue_move(a) ROCC(OPC, a, 0, k_MOVE)
static void keep(void) {}
"""


def test_the_library_header_loses_every_prohibited_macro():
    G = _driver()
    text, trapped = G.loop_free_header(HEADER, [5, 6])
    assert trapped == ["issue_step", "issue_both"]
    assert "k_STEP_A)" not in text and "k_STEP_B)" not in text
    assert "#define issue_move(a) ROCC(OPC, a, 0, k_MOVE)" in text
    assert "#define issue_step(a, b) do {" in text
    assert "static void keep(void) {}" in text
    assert G.loop_free_header(HEADER, [])[1] == []


def test_the_library_groups_are_called_on_their_routed_path():
    G = _driver()
    routes = G.library_path_without_loops("#define DIM 16\n")
    assert routes["conv2d"]["path"] == G.LIBRARY_HOST and routes["matmul"]["path"] == G.LIBRARY_HOST
    stated = G.library_path_without_loops("#define GEMMINI_OS_DATAFLOW 1\n")
    assert stated["matmul"]["path"] == G.LIBRARY_OUTPUT_STATIONARY
    conv = {
        "kind": "conv2d",
        "in_dim": 8,
        "ci": 3,
        "n": 4,
        "out_dim": 4,
        "stride": 2,
        "padding": 1,
        "kernel": 3,
        "in": "IN",
        "weight": "W",
        "bias": None,
        "out": "OUT",
        "relu": True,
        "scale": 0.5,
        "pool": {"size": 0, "stride": 0, "padding": 0},
    }
    assert G._call(conv).rstrip(";").endswith(f"{G.LIBRARY_DEFAULT})")
    # A host-routed convolution is the program's own loop-free routine, never the vendor's CPU path:
    # that path still reaches the loop-descriptor macro for some shapes.
    assert G._call(conv, routes).startswith("hr_conv2d(")


def _fake_toolchain(tmp_path, functions: dict[str, list[int]]):
    """An objdump that honours --start/--stop-address over a fixed listing, and an nm that lists the
    functions' start addresses -- enough to run the range-split scan for real, in worker processes."""
    import json
    import sys

    layout, address = [], 0x80000000
    for function, selectors in functions.items():
        layout.append({"function": function, "start": address, "selectors": selectors})
        address += 4 * (len(selectors) + 1)
    (tmp_path / "layout.json").write_text(json.dumps({"layout": layout, "opcode": OPCODE, "shift": ISA.SELECTOR_SHIFT}))
    objdump = tmp_path / "tools" / "cross-objdump"
    objdump.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"spec = json.load(open({str(tmp_path / 'layout.json')!r}))\n"
        "lo = hi = None\n"
        "for a in sys.argv[1:]:\n"
        "    if a.startswith('--start-address='): lo = int(a.split('=', 1)[1], 16)\n"
        "    if a.startswith('--stop-address='): hi = int(a.split('=', 1)[1], 16)\n"
        "out = []\n"
        "for f in spec['layout']:\n"
        "    addr = f['start']\n"
        "    words = [(s << spec['shift']) | spec['opcode'] for s in f['selectors']] + [0x13]\n"
        "    labelled = False\n"
        "    for w in words:\n"
        "        if (lo is None or addr >= lo) and (hi is None or addr < hi):\n"
        "            if not labelled:\n"
        "                out.append(f\"{f['start']:016x} <{f['function']}>:\")\n"
        "                labelled = True\n"
        "            out.append(f'    {addr:x}:\\t{w:08x}          \\t.insn\\t4')\n"
        "        addr += 4\n"
        "print('\\n'.join(out))\n"
    )
    objdump.chmod(0o755)
    nm = tmp_path / "tools" / "cross-nm"
    nm.write_text(
        f"#!{sys.executable}\n"
        "import json, sys\n"
        f"spec = json.load(open({str(tmp_path / 'layout.json')!r}))\n"
        "if '-n' in sys.argv:\n"
        "    print('\\n'.join(f\"{f['start']:016x} T {f['function']}\" for f in spec['layout']))\n"
    )
    nm.chmod(0o755)


def test_the_range_split_scan_reports_exactly_what_one_pass_reports(target, monkeypatch):
    """A large program is scanned in address ranges by worker processes; folding them back in order
    gives the one-pass report byte for byte: counts, first-seen orders and the first hits."""
    import json

    functions = {f"fn{i}": [1, 5, 9, 6, 1][: 1 + i % 5] * (1 + i % 3) for i in range(40)}
    _fake_toolchain(target.tmp, functions)
    elf = target.tmp / "p.elf"
    elf.write_bytes(b"\0" * 64)
    groups = {"3": target.tmp / "g3.o"}

    def report():
        return ISA.check_program(
            elf, target="t", roles=["loop_descriptor"], compiler=target.compiler, group_objects=groups
        )

    monkeypatch.setattr(ISA, "PARALLEL_SCAN_MIN_BYTES", 1 << 60)
    one_pass = report()
    from merlin.perf import isa_scan

    parts = []
    real_scan = isa_scan.scan
    monkeypatch.setattr(isa_scan, "scan", lambda *a, **k: parts.append(real_scan(*a, **k)) or parts[-1])
    monkeypatch.setattr(ISA, "PARALLEL_SCAN_MIN_BYTES", 0)
    monkeypatch.setattr(ISA, "SCAN_PARTS", 6)
    split = report()
    assert len(parts[0]) >= 4, "the program was not split into ranges"
    assert not one_pass["clean"] and one_pass["hits"]
    assert json.dumps(split) == json.dumps(one_pass)
