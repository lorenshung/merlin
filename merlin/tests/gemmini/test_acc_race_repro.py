"""The accumulator load-order reproducers parse, classify and patch exactly what they claim to.

The reproducers themselves need a RISC-V cross compiler, a Gemmini harness tree and a simulator or
board. What runs everywhere is the part that turns a console into a verdict and the part that builds
the fenced control: a classifier that miscounted, or a patch that changed more than the ordering,
would make a measured race unreadable. The build itself runs when the toolchain is configured.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess

import pytest

from merlin.common.paths import repo_root

HERE = repo_root() / "examples/gemmini/verification/acc_race"

#: Counts from a recorded FireSim console of the full reproducer (2026-09-24), with the column
#: histograms shortened. The `VEND cold` line is synthetic: it carries two elements that are
#: neither operand alone, which must not read as the race.
FIRESIM_LINES = """\
ACC_RACE begin race=256x512 reps=4
RACE cand07 rep=0 bad=32 lhs_only=32 rhs_only=0 other=0 first=52016 bins512=0,0,0,16,0,16
RACE cand07 rep=1 bad=48 lhs_only=48 rhs_only=0 other=0 first=30304 bins512=0,16,0,16,0,16
RACE cand07 rep=2 bad=96 lhs_only=96 rhs_only=0 other=0 first=43184 bins512=16,16,16,16,0,16
RACE cand07 rep=3 bad=128 lhs_only=128 rhs_only=0 other=0 first=14320 bins512=16,32,16,16,32,16
RACE fenced rep=0 bad=0 lhs_only=0 rhs_only=0 other=0 first=-1 bins512=0,0,0,0,0,0
RACE fenced rep=1 bad=0 lhs_only=0 rhs_only=0 other=0 first=-1 bins512=0,0,0,0,0,0
RACE sameunit rep=0 bad=320 lhs_only=320 rhs_only=0 other=0 first=2672 bins512=0,16,0,0,16,0
VEND view512 rep=0 bad=0 lhs_only=0 rhs_only=0 other=0 first=-1 bins512=0,0,0,0,0,0
VEND view512 rep=1 bad=32 lhs_only=32 rhs_only=0 other=0 first=245216 bins512=0,0,0,0,32,0
VEND cold rows=98 bad=32 lhs_only=30 rhs_only=0 other=2 first=2560 bins=0,0,0,32
ACC_RACE end
"""


def _load(name: str):
    spec = importlib.util.spec_from_file_location(f"acc_race_{name}", HERE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_console_rollup_separates_the_race_from_other_wrong_answers():
    summary = _load("classify").summarize(FIRESIM_LINES.splitlines())
    cand07 = summary["RACE cand07"]
    assert (cand07["reps"], cand07["failing_reps"], cand07["bad"]) == (4, 4, 304)
    assert cand07["race_signature"] is True
    fenced = summary["RACE fenced"]
    assert (fenced["reps"], fenced["failing_reps"], fenced["race_signature"]) == (2, 0, False)
    assert summary["VEND view512"]["failing_reps"] == 1 and summary["VEND view512"]["race_signature"]
    # One element that is neither operand alone means "some other wrong answer", never the race.
    assert summary["VEND cold"]["race_signature"] is False
    assert "ACC_RACE begin" not in str(summary)


@pytest.mark.parametrize(
    "line",
    [
        "RACE cand07 rep=0 bad=32 lhs_only=32 rhs_only=0",  # truncated: no `other` count
        "RACE cand07 rep=0 bad=33 lhs_only=32 rhs_only=0 other=0",  # counts do not add up
        "RACE cand07 rep=0 bad=32 lhs_only=32 rhs_only=0 other=0 garbage",
    ],
)
def test_a_damaged_reproducer_line_is_refused_not_skipped(line):
    with pytest.raises(ValueError):
        _load("classify").parse_line(line)


def test_host_side_classification_of_an_elementwise_sum():
    classify = _load("classify").classify_elements
    lhs, rhs = [1, 2, 3, 4], [40, 50, 60, 70]
    assert classify(lhs, rhs, [41, 2, 60, 9]) == {"ok": 1, "lhs_only": 1, "rhs_only": 1, "other": 1}
    with pytest.raises(ValueError, match="lengths differ"):
        classify(lhs, rhs, [41])


_HEADER = """\
static void sp_tiled_resadd(const size_t I, const size_t J,
        const elem_t * A, const elem_t * B, elem_t * C,
        size_t A_row_stride, size_t B_row_stride, size_t C_row_stride, bool relu) {
    int tile_I = I / DIM;
    gemmini_loop_ws(tile_I, 1, 1, 0, 0, 0, A, B, NULL, C,
        A_row_stride, B_row_stride, 0, C_row_stride, false, false, false, false, false, 0, 0, true);
}

static void tiled_resadd(void) {
    gemmini_loop_ws(1, 1, 1, 0, 0, 0, NULL, NULL, NULL, NULL, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
}
"""


def test_the_fenced_control_changes_only_the_residual_adds_unroller_call():
    fence = _load("fence_vendor_resadd")
    patched = fence.fence_resadd(_HEADER)
    assert "sp_tiled_resadd_fenced(I, J, A, B, C, A_row_stride, B_row_stride, C_row_stride);" in patched
    # The overwriting phase, a fence, then the accumulating phase on the other load unit.
    body = patched[patched.index("static void sp_tiled_resadd_fenced") :]
    assert body.index("gemmini_extended_mvin(") < body.index("gemmini_fence();") < body.index("gemmini_extended_mvin2(")
    # The other unroller call in the header is untouched, and so is everything before the residual add.
    assert patched.count("gemmini_loop_ws(") == _HEADER.count("gemmini_loop_ws(") - 1
    assert "int tile_I = I / DIM;" in patched


@pytest.mark.parametrize(
    "header",
    ["static void unrelated(void) {}\n", _HEADER + _HEADER, _HEADER.replace("gemmini_loop_ws(tile_I", "other(tile_I")],
)
def test_the_fenced_control_refuses_a_header_it_cannot_patch_exactly(header):
    with pytest.raises(ValueError):
        _load("fence_vendor_resadd").fence_resadd(header)


def test_cold_operand_data_is_fixed_and_distinguishable(tmp_path):
    out = tmp_path / "vend_data.h"
    _load("gen_vend_data").main(str(out), 2, 32)
    text = out.read_text()
    assert "#define VEND_ROWS 2\n#define VEND_COLS 32\n" in text
    first = text.split("VA[64] row_align(1) = {", 1)[1].split("}", 1)[0]
    second = text.split("VB[64] row_align(1) = {", 1)[1].split("}", 1)[0]
    lhs, rhs = [int(v) for v in first.split(",")], [int(v) for v in second.split(",")]
    # a, b and a+b never collide, so every wrong element names the operand that survived.
    assert max(lhs) < min(rhs) and max(lhs) + max(rhs) <= 127
    again = tmp_path / "again.h"
    _load("gen_vend_data").main(str(again), 2, 32)
    assert again.read_bytes() == out.read_bytes()


def test_build_script_parses():
    subprocess.run(["bash", "-n", str(HERE / "build.sh")], check=True)


@pytest.mark.skipif(
    not (os.environ.get("RISCV_CC") and os.environ.get("GEMMINI_HARNESS")),
    reason="needs RISCV_CC and GEMMINI_HARNESS (a bare-metal Gemmini harness tree)",
)
def test_small_reproducer_builds_against_the_selected_harness(tmp_path):
    if shutil.which(os.environ["RISCV_CC"]) is None and not os.path.isfile(os.environ["RISCV_CC"]):
        pytest.skip("RISCV_CC does not name an executable")
    done = subprocess.run(
        [str(HERE / "build.sh"), str(tmp_path), "-DRACE_ROWS=32", "-DRACE_COLS=32", "-DREPS=1", "-DSKIP_VENDOR"],
        capture_output=True,
        text=True,
        check=False,
    )
    assert done.returncode == 0, done.stderr[-2000:]
    assert (tmp_path / "acc_race_repro.elf").is_file() and (tmp_path / "build.json").is_file()
