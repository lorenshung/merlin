"""Selected-header usage changes follow the source, and contradictions refuse."""

import hashlib
import shutil

import pytest

from merlin.targetgen.rtl.rocc_header import decode_expansion, extract


def _header(tmp_path, *, code=41, flags=3, destination="x0", extra=""):
    header = tmp_path / "abi.h"
    header.write_text(
        f"""#define SLOT 2
#define COMMAND {code}
#define STR_IMPL(x) #x
#define STR(x) STR_IMPL(x)
#define ROCC_TEST(x, a, b, f) \\
 asm volatile(".insn r CUSTOM_2, " STR({flags}) ", " STR(f) ", {destination}, %0, %1" : : "r"(a), "r"(b))
#define SUBMIT(a,b) ROCC_TEST(SLOT, a, b, COMMAND)
{extra}
"""
    )
    return header


@pytest.mark.skipif(shutil.which("cc") is None, reason="requires the C preprocessor")
def test_usage_tracks_selected_header_and_source_digests(tmp_path):
    header = _header(tmp_path)
    first = extract(header, include_root=tmp_path, prefix="ROCC_")
    assert first["by_funct"]["41"]["registers"] == {"rd": False, "rs1": True, "rs2": True}
    assert first["by_funct"]["41"]["funct3"] == 3
    source = next(row for row in first["sources"] if row["path"] == str(header))
    assert source["sha256"] == hashlib.sha256(header.read_bytes()).hexdigest()
    _header(tmp_path, code=57, flags=7, destination="%2")
    second = extract(header, include_root=tmp_path, prefix="ROCC_")
    assert set(second["by_funct"]) == {"57"}
    assert second["by_funct"]["57"]["registers"]["rd"] is True
    assert second["by_funct"]["57"]["funct3"] == 7
    assert first["preprocessed_sha256"] != second["preprocessed_sha256"]


def test_usage_refuses_flag_operand_disagreement():
    with pytest.raises(ValueError, match="disagree"):
        decode_expansion('asm volatile(".insn r CUSTOM_2, 7, 41, x0, %0, %1" : : "r"(a))')


@pytest.mark.skipif(shutil.which("cc") is None, reason="requires the C preprocessor")
def test_usage_refuses_conflicting_forms(tmp_path):
    header = _header(
        tmp_path,
        extra=(
            '#define ROCC_OTHER(x,a,b,f) asm volatile(".insn r CUSTOM_2, 7, " '
            'STR(f) ", %0, %1, %2" : "=r"(a) : "r"(b))\n'
            "#define OTHER(a,b) ROCC_OTHER(SLOT,a,b,COMMAND)"
        ),
    )
    with pytest.raises(ValueError, match="conflicting"):
        extract(header, include_root=tmp_path, prefix="ROCC_")


def test_absent_calls_refuse(tmp_path):
    header = tmp_path / "empty.h"
    header.write_text("#define COMMAND 41\n")
    with pytest.raises(ValueError, match="no RoCC call"):
        extract(header, include_root=tmp_path, prefix="ROCC_")
