"""Neutral writable-ELF and coherent-dump checks for selected caller output storage."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import shutil
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.feedback.native_memory_readback import (
    NativeMemoryReadback,
    execute_memory_elf,
    select_memory_engine,
)
from merlin_experiments.phase1.feedback.native_output_readback import (
    admit_coherent_output_layout,
    admit_fixed_address_output_layout,
    admit_htif_signature_bounds,
    decode_coherent_output_dump,
    decode_htif_signature,
)

from merlin.runtime.backends import base as backends
from merlin.targetgen import plugins


@pytest.fixture
def selected_layout(tmp_path, monkeypatch):
    root = tmp_path / "provider"
    root.mkdir()
    source = root / "renderer.py"
    source.write_text("LAYOUT = 1\n")
    contract = root / "contract.yaml"
    contract.write_text("role: renderer\n")
    (root / "provider.yaml").write_text("role: selected\n")
    info = SimpleNamespace(base=root, contract_path=contract)

    def describe(cb, *, target, facts):
        assert cb["target"] == target == facts["inputs"]["target"]

        def row(name, spec):
            shape = spec["shape"]
            rows, cols = math.prod(shape[:-1]), shape[-1]
            prows, pcols = ((rows + 3) // 4) * 4, ((cols + 3) // 4) * 4
            return {
                "tensor": name,
                "dtype": spec["dtype"],
                "logical_shape": shape,
                "physical_extents": [prows, pcols],
                "logical_strides_elements": [math.prod(shape[axis + 1 : -1]) * pcols for axis in range(len(shape) - 1)]
                + [1],
                "storage_elements": prows * pcols,
                "offset_elements": 0,
            }

        return {
            "schema": "caller_storage_layout_v1",
            "policy": {"mode": "legacy_aligned_row_major_v1", "row_alignment_elements": 4},
            "tensors": [row(name, spec) for name, spec in cb["tensors"].items()],
        }

    module = SimpleNamespace(
        __file__=str(source),
        caller_layout_source_paths=lambda: (source,),
        describe_caller_layout=describe,
    )
    monkeypatch.setattr(plugins, "resolve_support", lambda target: info)
    monkeypatch.setattr(backends, "get_backend", lambda target: module)
    submission = tmp_path / "submission"
    submission.mkdir()
    cb = {
        "target": "neutral",
        "params": {},
        "commands": [],
        "kernel_abi": {
            "kind": "whole_program",
            "outputs": ["Y"],
            "args": [{"tensor": "A", "access": "read"}, {"tensor": "Y", "access": "write"}],
        },
        "tensors": {
            "A": {"shape": [2, 3], "dtype": "i8", "role": "input"},
            "Y": {"shape": [2, 3], "dtype": "i8", "role": "output"},
        },
    }
    (submission / "command_buffer.json").write_text(json.dumps(cb))
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps({"inputs": {"target": "neutral"}}))
    return submission, facts, cb


def _elf(
    tmp_path: Path,
    *,
    ctype: str = "int8_t",
    count: int = 16,
    const: bool = False,
    duplicate: bool = False,
    second_output: bool = False,
    signature_aliases: str = "none",
    fixed_exec: bool = True,
) -> Path:
    cc = shutil.which("cc")
    if cc is None:
        pytest.skip("a neutral host C compiler is unavailable")
    source = tmp_path / "output.c"
    alias_source = ""
    if signature_aliases != "none":
        byte_count = count * {"int8_t": 1, "int32_t": 4, "uint32_t": 4}[ctype]
        end = byte_count - 1 if signature_aliases == "short_end" else byte_count
        if signature_aliases != "missing_begin":
            alias_source += '__asm__(".globl begin_signature\\n.set begin_signature,T_Y\\n");\n'
        alias_source += f'__asm__(".globl end_signature\\n.set end_signature,T_Y+{end}\\n");\n'
    source.write_text(
        "#include <stdint.h>\n"
        f"static {'const ' if const else ''}{ctype} T_Y[{count}] __attribute__((used))"
        f"{' = {1}' if const else ''};\n"
        + (f"static {ctype} T_Z[{count}] __attribute__((used));\n" if second_output else "")
        + alias_source
        + "int main(void) { return 0; }\n"
    )
    sources = [str(source)]
    if duplicate:
        extra = tmp_path / "extra.c"
        extra.write_text(f"#include <stdint.h>\nstatic {ctype} T_Y[{count}] __attribute__((used));\n")
        sources.append(str(extra))
    elf = tmp_path / "program.elf"
    subprocess.run(
        [cc, "-O0", "-no-pie" if fixed_exec else "-pie", *sources, "-o", str(elf)], check=True, capture_output=True
    )
    return elf


def _admit(selected_layout, elf: Path) -> dict:
    submission, facts, _ = selected_layout
    return admit_coherent_output_layout(
        submission=submission,
        command_buffer_member="command_buffer.json",
        target="neutral",
        facts_path=facts,
        elf_path=elf,
        expected_elf_sha256=hashlib.sha256(elf.read_bytes()).hexdigest(),
    )


def _dump(path: Path, *, address: int, payload: bytes, source: int = 0) -> Path:
    return _dump_regions(path, [(address, payload)], source=source)


def _dump_regions(path: Path, regions: list[tuple[int, bytes]], *, source: int = 0) -> Path:
    count = len(regions)
    total = sum(len(payload) for _, payload in regions)
    path.write_bytes(
        b"GSIMDMP1"
        + struct.pack("<QQ", source, count)
        + b"".join(struct.pack("<QQ", address, len(payload)) + payload for address, payload in regions)
        + b"GSIMEND1"
        + struct.pack("<QQ", count, total)
    )
    return path


def _decode(selected_layout, elf: Path, admission: dict, dump: Path) -> dict:
    submission, facts, _ = selected_layout
    fixed = admit_fixed_address_output_layout(
        admission=admission,
        submission=submission,
        command_buffer_member="command_buffer.json",
        target="neutral",
        facts_path=facts,
        elf_path=elf,
    )
    return decode_coherent_output_dump(
        admission=admission,
        fixed_address_admission=fixed,
        submission=submission,
        command_buffer_member="command_buffer.json",
        target="neutral",
        facts_path=facts,
        elf_path=elf,
        dump_path=dump,
    )


def _signature(selected_layout, elf: Path, admission: dict, path: Path) -> dict:
    submission, facts, _ = selected_layout
    bounds = admit_htif_signature_bounds(
        admission=admission,
        submission=submission,
        command_buffer_member="command_buffer.json",
        target="neutral",
        facts_path=facts,
        elf_path=elf,
    )
    return decode_htif_signature(
        admission=admission,
        bounds_admission=bounds,
        submission=submission,
        command_buffer_member="command_buffer.json",
        target="neutral",
        facts_path=facts,
        elf_path=elf,
        signature_path=path,
    )


def _write_signature(path: Path, physical: bytes) -> Path:
    path.write_bytes(b"".join(f"{byte:02x}\n".encode("ascii") for byte in physical))
    return path


def test_strided_dump_decodes_only_every_logical_signed_value(selected_layout, tmp_path):
    elf = _elf(tmp_path)
    admission = _admit(selected_layout, elf)
    assert admission["status"] == "layout_only"
    assert admission["byte_order"] == "little"
    row = admission["outputs"][0]
    assert row["physical_word_bytes"] == 1
    assert row["bytes"] == 16
    physical = bytearray([99] * 16)
    for offset, value in zip((0, 1, 2, 4, 5, 6), (1, -2, 3, 4, -5, 6), strict=True):
        physical[offset] = value & 255
    dump = _dump(tmp_path / "coherent.dump", address=row["address"], payload=bytes(physical))
    decoded = _decode(selected_layout, elf, admission, dump)
    assert decoded["outputs"] == {"Y": [[1, -2, 3], [4, -5, 6]]}
    assert decoded["elf_sha256"] == admission["elf_sha256"]
    assert decoded["dump_sha256"] == hashlib.sha256(dump.read_bytes()).hexdigest()
    assert 99 not in [value for line in decoded["outputs"]["Y"] for value in line]


def test_float_codes_use_existing_declared_dtype_decoder(selected_layout, tmp_path):
    submission, _, cb = selected_layout
    cb["tensors"]["Y"]["dtype"] = "f32"
    (submission / "command_buffer.json").write_text(json.dumps(cb))
    elf = _elf(tmp_path, ctype="uint32_t")
    admission = _admit(selected_layout, elf)
    row = admission["outputs"][0]
    physical = bytearray(row["bytes"])
    for offset, bits in zip((0, 1, 2, 4, 5, 6), (0x3F800000, 0x80000000, 0, 0xBF000000, 0, 0x40000000), strict=True):
        struct.pack_into("<I", physical, offset * 4, bits)
    result = _decode(
        selected_layout, elf, admission, _dump(tmp_path / "float.dump", address=row["address"], payload=physical)
    )
    assert result["outputs"]["Y"] == [[1.0, -0.0, 0.0], [-0.5, 0.0, 2.0]]
    assert math.copysign(1, result["outputs"]["Y"][0][1]) == -1


def test_i32_width_is_declared_and_every_logical_value_is_read(selected_layout, tmp_path):
    submission, _, cb = selected_layout
    cb["tensors"]["Y"]["dtype"] = "i32"
    (submission / "command_buffer.json").write_text(json.dumps(cb))
    elf = _elf(tmp_path, ctype="int32_t")
    admission = _admit(selected_layout, elf)
    row = admission["outputs"][0]
    assert row["physical_word_bytes"] == 4
    assert row["bytes"] == 64
    physical = bytearray(row["bytes"])
    logical = (-2147483648, -100, 7, 0, 201, 2147483647)
    for offset, value in zip((0, 1, 2, 4, 5, 6), logical, strict=True):
        struct.pack_into("<i", physical, offset * 4, value)
    decoded = _decode(
        selected_layout, elf, admission, _dump(tmp_path / "i32.dump", address=row["address"], payload=physical)
    )
    assert decoded["outputs"] == {"Y": [list(logical[:3]), list(logical[3:])]}


def test_rank_three_logical_coordinates_use_full_selected_strides(selected_layout, tmp_path):
    submission, _, cb = selected_layout
    cb["tensors"]["Y"]["shape"] = [2, 2, 3]
    (submission / "command_buffer.json").write_text(json.dumps(cb))
    elf = _elf(tmp_path)
    admission = _admit(selected_layout, elf)
    row = admission["outputs"][0]
    assert row["logical_strides_elements"] == [8, 4, 1]
    physical = bytearray([99] * row["bytes"])
    for value, index in enumerate((0, 1, 2, 4, 5, 6, 8, 9, 10, 12, 13, 14), start=1):
        physical[index] = value
    decoded = _decode(
        selected_layout, elf, admission, _dump(tmp_path / "rank3.dump", address=row["address"], payload=physical)
    )
    assert decoded["outputs"] == {"Y": [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]}


def test_two_outputs_require_the_exact_complete_ordered_dump_roster(selected_layout, tmp_path):
    submission, _, cb = selected_layout
    cb["kernel_abi"]["args"].append({"tensor": "Z", "access": "write"})
    cb["kernel_abi"]["outputs"].append("Z")
    cb["tensors"]["Z"] = {"shape": [2, 3], "dtype": "i8", "role": "output"}
    (submission / "command_buffer.json").write_text(json.dumps(cb))
    elf = _elf(tmp_path, second_output=True)
    admission = _admit(selected_layout, elf)
    y, z = admission["outputs"]
    assert [row["tensor"] for row in admission["outputs"]] == ["Y", "Z"]
    regions = [(y["address"], bytes([1] * y["bytes"])), (z["address"], bytes([2] * z["bytes"]))]
    complete = _dump_regions(tmp_path / "both.dump", regions)
    decoded = _decode(selected_layout, elf, admission, complete)
    assert decoded["outputs"] == {"Y": [[1] * 3, [1] * 3], "Z": [[2] * 3, [2] * 3]}
    with pytest.raises(ValueError, match="size differs"):
        _decode(selected_layout, elf, admission, _dump_regions(tmp_path / "one.dump", regions[:1]))
    with pytest.raises(ValueError, match="regions asked for"):
        _decode(selected_layout, elf, admission, _dump_regions(tmp_path / "reordered.dump", regions[::-1]))
    with pytest.raises(ValueError, match="exactly one output"):
        _signature(selected_layout, elf, admission, _write_signature(tmp_path / "both.sig", bytes(y["bytes"])))


def test_htif_signature_decodes_the_same_selected_signed_physical_region(selected_layout, tmp_path):
    elf = _elf(tmp_path, signature_aliases="exact")
    admission = _admit(selected_layout, elf)
    submission, facts, _ = selected_layout
    bounds = admit_htif_signature_bounds(
        admission=admission,
        submission=submission,
        command_buffer_member="command_buffer.json",
        target="neutral",
        facts_path=facts,
        elf_path=elf,
    )
    row = admission["outputs"][0]
    assert bounds["begin"] == row["address"]
    assert bounds["end"] == row["address"] + row["bytes"]
    assert bounds["signature_granularity_bytes"] == 1
    assert bounds["signature_file_bytes"] == 3 * row["bytes"]
    physical = bytearray([99] * row["bytes"])
    for offset, value in zip((0, 1, 2, 4, 5, 6), (1, -2, 3, 4, -5, 6), strict=True):
        physical[offset] = value & 255
    signature = _write_signature(tmp_path / "output.sig", physical)
    decoded = _signature(selected_layout, elf, admission, signature)
    assert decoded["source"] == "htif_signature"
    assert decoded["outputs"] == {"Y": [[1, -2, 3], [4, -5, 6]]}
    assert decoded["elf_sha256"] == admission["elf_sha256"]
    assert decoded["signature_sha256"] == hashlib.sha256(signature.read_bytes()).hexdigest()


def test_htif_signature_preserves_declared_f32_bits_including_negative_zero(selected_layout, tmp_path):
    submission, _, cb = selected_layout
    cb["tensors"]["Y"]["dtype"] = "f32"
    (submission / "command_buffer.json").write_text(json.dumps(cb))
    elf = _elf(tmp_path, ctype="uint32_t", signature_aliases="exact")
    admission = _admit(selected_layout, elf)
    row = admission["outputs"][0]
    physical = bytearray(row["bytes"])
    for offset, bits in zip((0, 1, 2, 4, 5, 6), (0x3F800000, 0x80000000, 0, 0xBF000000, 0, 0x40000000), strict=True):
        struct.pack_into("<I", physical, offset * 4, bits)
    decoded = _signature(selected_layout, elf, admission, _write_signature(tmp_path / "float.sig", physical))
    assert decoded["outputs"] == {"Y": [[1.0, -0.0, 0.0], [-0.5, 0.0, 2.0]]}
    assert math.copysign(1, decoded["outputs"]["Y"][0][1]) == -1


@pytest.mark.parametrize("alias_case", ["none", "missing_begin", "short_end"])
def test_htif_signature_refuses_missing_or_wrong_elf_aliases(selected_layout, tmp_path, alias_case):
    elf = _elf(tmp_path, signature_aliases=alias_case)
    admission = _admit(selected_layout, elf)
    with pytest.raises(ValueError, match="signature alias|signature aliases"):
        _signature(selected_layout, elf, admission, _write_signature(tmp_path / "output.sig", bytes(16)))


def test_layout_only_may_inspect_pie_but_no_execution_reader_accepts_unknown_load_bias(selected_layout, tmp_path):
    submission, facts, _ = selected_layout
    elf = _elf(tmp_path, signature_aliases="exact", fixed_exec=False)
    admission = _admit(selected_layout, elf)
    assert admission["status"] == "layout_only"
    kwargs = {
        "admission": admission,
        "submission": submission,
        "command_buffer_member": "command_buffer.json",
        "target": "neutral",
        "facts_path": facts,
        "elf_path": elf,
    }
    with pytest.raises(ValueError, match="ET_EXEC"):
        admit_fixed_address_output_layout(**kwargs)
    with pytest.raises(ValueError, match="ET_EXEC"):
        admit_htif_signature_bounds(**kwargs)
    with pytest.raises(ValueError, match="ET_EXEC"):
        _decode(
            selected_layout,
            elf,
            admission,
            _dump(tmp_path / "pie.dump", address=admission["outputs"][0]["address"], payload=bytes(16)),
        )


@pytest.mark.parametrize("simulator,transport", [("spike", "htif_signature_v1"), ("gsim", "gsim_coherent_dump_v1")])
def test_trusted_hook_preflights_before_native_launch_then_rechecks_complete_output(
    selected_layout, tmp_path, monkeypatch, simulator, transport
):
    _, facts, cb = selected_layout
    backend = backends.get_backend("neutral")
    monkeypatch.setattr(backend, "memory_readback_transport", lambda engine: transport, raising=False)
    elf = _elf(tmp_path, signature_aliases="exact" if simulator == "spike" else "none")
    workdir = tmp_path / "run"
    workdir.mkdir()
    hook = NativeMemoryReadback(facts_path=facts)
    selected = hook.prepare(
        cb=cb, target="neutral", elf_path=elf, workdir=workdir, simulator=simulator, backend=backend
    )
    assert set(selected) == {"memory_readback"}
    request = selected["memory_readback"]
    assert request["schema"] == "oracle_memory_readback_v1"
    assert request["transport"] == transport
    assert request["output_path"] == str(
        workdir / "memory_readback" / ("output.signature" if simulator == "spike" else "output.dump")
    )
    assert not Path(request["output_path"]).exists()
    assert len(request["regions"]) == 1
    region = request["regions"][0]
    physical = bytearray([99] * region["bytes"])
    for offset, value in zip((0, 1, 2, 4, 5, 6), (1, -2, 3, 4, -5, 6), strict=True):
        physical[offset] = value & 255
    if simulator == "spike":
        assert request["regions_path"] is None
        _write_signature(Path(request["output_path"]), physical)
    else:
        regions_path = Path(request["regions_path"])
        assert regions_path.read_bytes() == f"{region['base']:#x} {region['bytes']}\n".encode("ascii")
        _dump(Path(request["output_path"]), address=region["base"], payload=physical)
    outputs, evidence = hook.decode("METRIC cycles 7\nDONE\n")
    assert outputs == {"Y": [[1, -2, 3], [4, -5, 6]]}
    assert evidence["status"] == "complete"
    assert evidence["transport"] == transport
    assert evidence["elf_sha256"] == request["elf_sha256"]
    with pytest.raises(ValueError, match="cannot prepare twice"):
        hook.prepare(cb=cb, target="neutral", elf_path=elf, workdir=workdir, simulator=simulator, backend=backend)


def test_trusted_hook_refuses_wrong_capability_and_mutated_staged_inputs(selected_layout, tmp_path, monkeypatch):
    _, facts, cb = selected_layout
    backend = backends.get_backend("neutral")
    monkeypatch.setattr(backend, "memory_readback_transport", lambda engine: "unknown", raising=False)
    elf = _elf(tmp_path)
    workdir = tmp_path / "run"
    workdir.mkdir()
    hook = NativeMemoryReadback(facts_path=facts)
    kwargs = {
        "cb": cb,
        "target": "neutral",
        "elf_path": elf,
        "workdir": workdir,
        "simulator": "gsim",
        "backend": backend,
    }
    with pytest.raises(ValueError, match="cannot run"):
        hook.prepare(**kwargs)
    assert not (workdir / "memory_readback").exists()
    monkeypatch.setattr(backend, "memory_readback_transport", lambda engine: "gsim_coherent_dump_v1")
    request = hook.prepare(**kwargs)["memory_readback"]
    region = request["regions"][0]
    _dump(Path(request["output_path"]), address=region["base"], payload=bytes(region["bytes"]))
    Path(request["regions_path"]).write_text("not the selected regions\n")
    with pytest.raises(ValueError, match="region request changed"):
        hook.decode("DONE\n")
    Path(request["regions_path"]).write_text(f"{region['base']:#x} {region['bytes']}\n")
    cb["params"]["changed"] = 1
    with pytest.raises(ValueError, match="request or source changed"):
        hook.decode("DONE\n")


@pytest.mark.parametrize(
    "outcome",
    ["complete", "failed_exit", "post_run_refusal", "missing_done", "missing_dump", "partial_roster", "swapped_elf"],
)
def test_existing_backend_execution_seam_never_substitutes_incomplete_memory(
    selected_layout, tmp_path, monkeypatch, outcome
):
    _, facts, cb = selected_layout
    backend = backends.get_backend("neutral")
    monkeypatch.setattr(backend, "memory_readback_transport", lambda engine: "gsim_coherent_dump_v1", raising=False)
    parsed = []

    def parse(console):
        parsed.append(True)
        return backends.parse_console(console)

    monkeypatch.setattr(backend, "parse_output", parse, raising=False)
    if outcome == "partial_roster":
        cb["kernel_abi"]["args"].append({"tensor": "Z", "access": "write"})
        cb["kernel_abi"]["outputs"].append("Z")
        cb["tensors"]["Z"] = {"shape": [2, 3], "dtype": "i8", "role": "output"}
    elf = _elf(tmp_path, second_output=outcome == "partial_roster")
    selected_sha = hashlib.sha256(elf.read_bytes()).hexdigest()
    workdir = tmp_path / "native-exec"
    launched = []
    checked = []

    def post_run_revalidate():
        checked.append(True)
        if outcome == "post_run_refusal":
            raise ValueError("selected build changed after native run")

    def run_elf(path, *, simulator, timeout, memory_readback):
        launched.append(True)
        assert simulator == "gsim" and timeout == 13
        assert Path(path) == elf
        assert memory_readback["elf_sha256"] == selected_sha
        assert not Path(memory_readback["output_path"]).exists()
        if outcome == "failed_exit":
            raise subprocess.CalledProcessError(
                1, ["neutral-native"], output=b"partial-stdout", stderr=b"partial-stderr"
            )
        if outcome != "missing_dump":
            region = memory_readback["regions"][0]
            _dump(Path(memory_readback["output_path"]), address=region["base"], payload=bytes(region["bytes"]))
        return "METRIC cycles 7\n" + ("" if outcome == "missing_done" else "DONE\n")

    monkeypatch.setattr(backend, "run_elf", run_elf, raising=False)
    kwargs = {
        "cb": cb,
        "target": "neutral",
        "elf_path": elf,
        "workdir": workdir,
        "simulator": "gsim",
        "backend": backend,
        "timeout": 13,
        "expected_elf_sha256": selected_sha,
        "facts_path": facts,
        "post_run_revalidate": post_run_revalidate,
    }
    if outcome == "swapped_elf":
        original_prepare = NativeMemoryReadback.prepare

        def swap_after_preflight(self, **passed):
            request = original_prepare(self, **passed)
            elf.write_bytes(elf.read_bytes() + b"changed-before-launch")
            return request

        monkeypatch.setattr(NativeMemoryReadback, "prepare", swap_after_preflight)
        with pytest.raises(ValueError, match="changed after preflight"):
            execute_memory_elf(**kwargs)
        assert not launched
    elif outcome == "failed_exit":
        with pytest.raises(subprocess.CalledProcessError):
            execute_memory_elf(**kwargs)
        assert (workdir / "console.partial").read_bytes() == b"partial-stdout"
        assert (workdir / "stderr.partial").read_bytes() == b"partial-stderr"
        assert not parsed and not checked
    elif outcome == "post_run_refusal":
        with pytest.raises(ValueError, match="selected build changed"):
            execute_memory_elf(**kwargs)
        assert (workdir / "console.txt").read_text().endswith("DONE\n")
        assert checked == [True] and not parsed
    elif outcome == "missing_done":
        with pytest.raises(RuntimeError, match="did not reach DONE"):
            execute_memory_elf(**kwargs)
        assert (workdir / "console.txt").read_text() == "METRIC cycles 7\n"
    elif outcome == "missing_dump":
        with pytest.raises(ValueError, match="ordinary complete file"):
            execute_memory_elf(**kwargs)
        assert (workdir / "console.txt").read_text().endswith("DONE\n")
    elif outcome == "partial_roster":
        with pytest.raises(ValueError, match="size differs"):
            execute_memory_elf(**kwargs)
        assert (workdir / "console.txt").read_text().endswith("DONE\n")
    else:
        console, outputs, metrics, evidence = execute_memory_elf(**kwargs)
        assert console.endswith("DONE\n")
        assert outputs == {"Y": [[0] * 3, [0] * 3]}
        assert metrics == {"cycles": 7}
        assert evidence["status"] == "complete"
        assert evidence["elf_sha256"] == selected_sha
        assert evidence["console_path"] == str(workdir / "console.txt")
        assert checked == [True] and parsed == [True]
    assert workdir.stat().st_mode & 0o777 == 0o700


@pytest.mark.parametrize("simulator", ["spike", "gsim"])
@pytest.mark.parametrize("mutation", ["none", "facts", "firrtl", "engine", "backend", "selection"])
def test_memory_engine_citation_rechecks_explicit_facts_and_selected_runner(
    selected_layout, tmp_path, monkeypatch, simulator, mutation
):
    from merlin.compile import model_execution_inputs
    from merlin.targetgen import native_model_execution, oracle_policy

    _, _, _ = selected_layout
    backend = backends.get_backend("neutral")
    firrtl = tmp_path / "selected.fir"
    firrtl.write_bytes(b"selected FIRRTL bytes")
    fir_sha = hashlib.sha256(firrtl.read_bytes()).hexdigest()
    facts = tmp_path / "selected-facts.json"
    facts.write_text(
        json.dumps(
            {
                "facts": {"source": {"config": "public-board"}},
                "inputs": {
                    "target": "neutral",
                    "fir_sha256": fir_sha,
                    "firrtl_inputs": [{"path": str(firrtl), "sha256": fir_sha}],
                },
            }
        )
    )
    state = {"engine": "engine-v1", "selected": "gsim"}

    def engine_recheck():
        if state["engine"] != "engine-v1":
            raise ValueError("selected engine bytes drifted")

    monkeypatch.setattr(
        native_model_execution,
        "_functional_engine",
        lambda target: (backend, {"selected_binary": state["engine"]}, engine_recheck),
    )
    monkeypatch.setattr(
        model_execution_inputs,
        "native_engine",
        lambda target, selected_sim, selected_facts: (
            backend,
            {"selected_binary": state["engine"], "firrtl": selected_facts["firrtl_sha256"]},
            engine_recheck,
            None,
        ),
    )
    monkeypatch.setattr(
        oracle_policy,
        "selected_l3_engine_report",
        lambda target: {"available": True, "engine": state["selected"]},
    )
    citation, revalidate = select_memory_engine(
        target="neutral", simulator=simulator, backend=backend, facts_path=facts
    )
    assert citation["rtl_facts"]["firrtl_sha256"] == fir_sha
    assert citation["simulator"] == simulator
    assert revalidate() == citation and revalidate() is not citation
    if mutation == "facts":
        facts.write_bytes(facts.read_bytes() + b"\n")
    elif mutation == "firrtl":
        firrtl.write_bytes(b"changed FIRRTL")
    elif mutation == "engine":
        state["engine"] = "engine-v2"
    elif mutation == "backend":
        monkeypatch.setattr(backends, "get_backend", lambda target: object())
    elif mutation == "selection":
        state["selected"] = "verilator"
    if mutation == "selection" and simulator == "spike":
        # The L2 functional selector is checked by _functional_engine itself;
        # an L3-only selector change does not alter the selected Spike path.
        assert revalidate() == citation
    elif mutation != "none":
        with pytest.raises(ValueError):
            revalidate()


@pytest.mark.parametrize(
    "mutation", ["truncated", "oversized", "uppercase", "no_newline", "bad_hex", "changed_elf", "changed_context"]
)
def test_htif_signature_refuses_incomplete_noncanonical_or_stale_inputs(selected_layout, tmp_path, mutation):
    submission, _, cb = selected_layout
    elf = _elf(tmp_path, signature_aliases="exact")
    admission = _admit(selected_layout, elf)
    signature = _write_signature(tmp_path / "output.sig", bytes([0xAB] * 16))
    if mutation == "truncated":
        signature.write_bytes(signature.read_bytes()[:-1])
    elif mutation == "oversized":
        signature.write_bytes(signature.read_bytes() + b"00\n")
    elif mutation == "uppercase":
        signature.write_bytes(signature.read_bytes().replace(b"ab", b"AB", 1))
    elif mutation == "no_newline":
        signature.write_bytes(signature.read_bytes().replace(b"\n", b" ", 1))
    elif mutation == "bad_hex":
        signature.write_bytes(signature.read_bytes().replace(b"ab", b"ag", 1))
    elif mutation == "changed_elf":
        elf.write_bytes(elf.read_bytes() + b"stale")
    elif mutation == "changed_context":
        cb["params"]["changed"] = True
        (submission / "command_buffer.json").write_text(json.dumps(cb))
    with pytest.raises(ValueError):
        _signature(selected_layout, elf, admission, signature)


def test_output_roster_and_selected_legacy_strides_must_agree(selected_layout, tmp_path, monkeypatch):
    submission, _, cb = selected_layout
    elf = _elf(tmp_path)
    cb["kernel_abi"]["outputs"] = ["A", "Y"]
    (submission / "command_buffer.json").write_text(json.dumps(cb))
    with pytest.raises(ValueError, match="writable pointer roster"):
        _admit(selected_layout, elf)
    cb["kernel_abi"]["outputs"] = ["Y"]
    (submission / "command_buffer.json").write_text(json.dumps(cb))
    module = backends.get_backend("neutral")
    describe = module.describe_caller_layout

    def altered(*args, **kwargs):
        layout = describe(*args, **kwargs)
        layout["tensors"][1]["logical_strides_elements"] = [3, 1]
        return layout

    monkeypatch.setattr(module, "describe_caller_layout", altered)
    with pytest.raises(ValueError, match="legacy address map"):
        _admit(selected_layout, elf)


@pytest.mark.parametrize("dtype", ["i1", "f16", "bf16", "f64"])
def test_unselected_physical_dtype_refuses_even_if_elf_size_fits(selected_layout, tmp_path, dtype):
    submission, _, cb = selected_layout
    cb["tensors"]["Y"]["dtype"] = dtype
    (submission / "command_buffer.json").write_text(json.dumps(cb))
    elf = _elf(tmp_path, ctype="uint64_t" if dtype == "f64" else "int8_t")
    with pytest.raises(ValueError, match="logical dtype or shape"):
        _admit(selected_layout, elf)


@pytest.mark.parametrize("change", ["short", "readonly", "duplicate", "big_endian", "relocatable"])
def test_symbol_admission_refuses_incomplete_elf_output(selected_layout, tmp_path, change):
    elf = _elf(
        tmp_path, count=15 if change == "short" else 16, const=change == "readonly", duplicate=change == "duplicate"
    )
    if change == "big_endian":
        raw = bytearray(elf.read_bytes())
        raw[5] = 2
        elf.write_bytes(raw)
    if change == "relocatable":
        raw = bytearray(elf.read_bytes())
        raw[16:18] = (1).to_bytes(2, "little")
        elf.write_bytes(raw)
    with pytest.raises(ValueError, match="coherent output"):
        _admit(selected_layout, elf)


def test_dump_refuses_stale_or_incomplete_sources_and_changed_admission(selected_layout, tmp_path):
    elf = _elf(tmp_path)
    admission = _admit(selected_layout, elf)
    row = admission["outputs"][0]
    good = _dump(tmp_path / "good.dump", address=row["address"], payload=bytes(row["bytes"]))
    for source in (1, 2):
        other = _dump(
            tmp_path / f"source-{source}.dump", address=row["address"], payload=bytes(row["bytes"]), source=source
        )
        with pytest.raises(ValueError, match="coherent port"):
            _decode(selected_layout, elf, admission, other)
    short = tmp_path / "short.dump"
    short.write_bytes(good.read_bytes()[:-1])
    with pytest.raises(ValueError, match="size differs"):
        _decode(selected_layout, elf, admission, short)
    malformed = tmp_path / "malformed.dump"
    raw = bytearray(good.read_bytes())
    raw[-24:-16] = b"WRONGEND"
    malformed.write_bytes(raw)
    with pytest.raises(ValueError, match="complete trailer"):
        _decode(selected_layout, elf, admission, malformed)
    wrong_region = _dump(tmp_path / "wrong-region.dump", address=row["address"] + 1, payload=bytes(row["bytes"]))
    with pytest.raises(ValueError, match="regions asked for"):
        _decode(selected_layout, elf, admission, wrong_region)
    changed = copy.deepcopy(admission)
    changed["outputs"][0]["logical_strides_elements"] = [3, 1]
    with pytest.raises(ValueError, match="layout admission changed"):
        _decode(selected_layout, elf, changed, good)
    elf.write_bytes(elf.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="linked bytes"):
        _decode(selected_layout, elf, admission, good)
