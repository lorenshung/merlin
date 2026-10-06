"""Actual compiler twins and negative binary/source attribution gates."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.perf.debug_companion import attribute_symbolized_pcs, verify_debug_companion


@pytest.fixture(scope="module")
def binaries(tmp_path_factory):
    clang = shutil.which("clang")
    if not clang:
        pytest.skip("clang unavailable")
    root = tmp_path_factory.mktemp("debug-companion")
    source = root / "input.c"
    source.write_text('extern int puts(const char*);\nint main(void){return puts("original source");}\n')
    paths = {}
    for mode, flags in [("control", []), ("debug", ["-gline-tables-only"])]:
        obj, elf = root / (mode + ".o"), root / mode
        subprocess.run([clang, "-O2", "-fno-builtin", *flags, "-c", str(source), "-o", str(obj)], check=True)
        subprocess.run([clang, "-Wl,--build-id=none", str(obj), "-o", str(elf)], check=True)
        paths[mode] = (obj.read_bytes(), elf.read_bytes(), elf)
    return paths


def test_actual_compiler_object_and_final_image(binaries):
    for index, relocatable in [(0, True), (1, False)]:
        result = verify_debug_companion(binaries["control"][index], binaries["debug"][index], relocatable=relocatable)
        assert result["allocated_sections"] > 0
        if relocatable:
            assert result["relocations"] > 0


def test_actual_symbolizer_conserves_pcs(binaries):
    tool = shutil.which("llvm-symbolizer")
    nm = shutil.which("nm")
    if not tool or not nm:
        pytest.skip("symbolizer or nm unavailable")
    elf = binaries["debug"][2]
    symbols = subprocess.check_output([nm, str(elf)], text=True)
    address = next(int(line.split()[0], 16) for line in symbols.splitlines() if line.split()[-1:] == ["main"])
    raw = subprocess.check_output(
        [tool, "--obj=" + str(elf), "--output-style=JSON", "--inlining"], input=hex(address) + "\n", text=True
    )
    result = attribute_symbolized_pcs({address: 19}, [json.loads(raw)])
    assert result["total"] == 19
    assert result["functions"] == {"main": 19}
    assert sum(result["lines"].values()) == 19


@pytest.mark.parametrize("offset", [18, 24, 48])
def test_changed_identity_refused(binaries, offset):
    changed = bytearray(binaries["debug"][1])
    changed[offset] ^= 1
    with pytest.raises(ValueError):
        verify_debug_companion(binaries["control"][1], bytes(changed))


def test_changed_allocated_literal_refused(binaries):
    changed = bytearray(binaries["debug"][1])
    offset = changed.index(b"original source\0")
    changed[offset] = ord("x")
    with pytest.raises(ValueError, match="allocated section"):
        verify_debug_companion(binaries["control"][1], bytes(changed))


def test_relocation_symbol_change_refused(tmp_path):
    clang = shutil.which("clang")
    if not clang:
        pytest.skip("clang unavailable")
    objects = []
    for name in ["alpha", "bravo"]:
        source = tmp_path / (name + ".c")
        obj = source.with_suffix(".o")
        source.write_text(f"extern int {name}(int); int call(int x){{return {name}(x);}}")
        subprocess.run([clang, "-O2", "-c", str(source), "-o", str(obj)], check=True)
        objects.append(obj.read_bytes())
    with pytest.raises(ValueError, match="relocations differ"):
        verify_debug_companion(*objects, relocatable=True)


@pytest.mark.parametrize("blob", [b"", b"not elf", b"\x7fELF" + bytes(60)])
def test_unknown_or_truncated_refused(blob):
    with pytest.raises(ValueError):
        verify_debug_companion(blob, blob)


def test_explicit_file_kind_required(binaries):
    with pytest.raises(ValueError):
        verify_debug_companion(binaries["control"][0], binaries["debug"][0])
    with pytest.raises(ValueError):
        verify_debug_companion(binaries["control"][1], binaries["debug"][1], relocatable=1)


def record(address, frames=None):
    return {"Address": hex(address), "Symbol": frames or []}


def test_missing_debug_kept_and_inline_stack_not_double_counted():
    frames = [
        {"FunctionName": "inner", "FileName": "a.h", "Line": 2, "Column": 3},
        {"FunctionName": "outer", "FileName": "b.c", "Line": 7, "Column": 1},
    ]
    result = attribute_symbolized_pcs({16: 5, 20: 7}, [record(16, frames), record(20)])
    assert result["total"] == 12
    assert result["functions"] == {"inner": 5, "": 7}
    assert sum(result["inline_stacks"].values()) == 12


@pytest.mark.parametrize("records", [[], [record(16), record(16)], [record(20)]])
def test_missing_duplicate_unexpected_addresses_refused(records):
    with pytest.raises(ValueError):
        attribute_symbolized_pcs({16: 5}, records)


@pytest.mark.parametrize("counts", [{16: -1}, {True: 1}, {16: True}])
def test_invalid_counts_refused(counts):
    with pytest.raises(ValueError):
        attribute_symbolized_pcs(counts, [])
