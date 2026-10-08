"""The open-coded nest placement probe derives its geometry and varies only what each arm names.

The probe's verdicts are ratios between arms in one binary, so an arm that differed from its
neighbour in anything but its stated policy would turn a measured ratio into a comparison of two
programs. These tests hold the generated arms to their policy, the blocking to the harness's own
capacities, and the geometry to the headers it is read from. The build itself runs when a harness
and compiler are configured.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from merlin.common.paths import repo_root

PROBE = repo_root() / "examples/gemmini/verification/probe_load_placement.py"


def _probe():
    spec = importlib.util.spec_from_file_location("gemmini_load_placement_probe", PROBE)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _harness(tmp_path: Path, *, dim=16, acc_rows=1024, banks=4, bank_rows=4096, mvins=("k_MVIN", "k_MVIN2", "k_MVIN3")):
    include = tmp_path / "include"
    include.mkdir()
    (include / "gemmini_params.h").write_text(
        f"#define DIM {dim}\n#define ADDR_LEN 32\n#define BANK_NUM {banks}\n#define BANK_ROWS {bank_rows}\n"
        f"#define ACC_ROWS {acc_rows}\n#define MAX_BYTES 64\n#define ROUND_NEAR_EVEN(x) (x)\n"
    )
    (include / "gemmini.h").write_text("".join(f"#define {name} {i}\n" for i, name in enumerate(mvins)))
    (include / "gemmini_counter.h").write_text(
        "#define INCREMENTAL_COUNTERS 44\n#define MAIN_LD_CYCLES 1\n"
        "#define RESERVATION_STATION_LD_COUNT (INCREMENTAL_COUNTERS + 1)\n"
    )
    return tmp_path


def test_geometry_and_load_states_are_read_from_the_harness(tmp_path):
    probe = _probe()
    facts = probe.Facts(_harness(tmp_path))
    assert (facts.dim, facts.acc_rows, facts.spad_rows, facts.load_states) == (16, 1024, 16384, 3)
    assert facts.acc_bit == 1 << 31 and facts.accumulate_bit == 1 << 30
    # A build with fewer move-in opcodes has fewer load states; the probe reports it, never assumes three.
    fewer = tmp_path / "fewer"
    fewer.mkdir()
    assert probe.Facts(_harness(fewer, mvins=("k_MVIN",))).load_states == 1
    codes = probe.counter_codes(tmp_path)
    assert codes == {"MAIN_LD_CYCLES": 1, "RESERVATION_STATION_LD_COUNT": 45}


def test_blocking_fits_the_capacities_and_reproduces_the_recorded_shape(tmp_path):
    probe = _probe()
    facts = probe.Facts(_harness(tmp_path))
    # The 64x256x1024 study shape was blocked bm=4 bn=12 bk=64 over tiles (4, 16, 64).
    assert probe.block_shape(facts, 4, 16, 64) == (4, 12, 64)
    for shape in ((196, 4, 4), (7, 32, 64), (4, 32, 64), (1, 1, 1)):
        bm, bn, bk = probe.block_shape(facts, *shape)
        assert bm * bn * facts.dim <= facts.acc_rows
        assert (bm + bn) * bk * facts.dim <= facts.spad_rows
    small = tmp_path / "small"
    small.mkdir()
    tight = probe.Facts(_harness(small, acc_rows=64, bank_rows=64))
    bm, bn, bk = probe.block_shape(tight, 4, 16, 64)
    assert bm * bn * tight.dim <= tight.acc_rows and (bm + bn) * bk * tight.dim <= tight.spad_rows


def _arm(probe, facts, name):
    states, placement, store, stationary = probe.ARMS[name]
    return probe.arm_source(name, min(states, facts.load_states), placement, store, stationary, facts, 0, 0)


def test_each_arm_differs_only_in_the_policy_it_names(tmp_path):
    probe = _probe()
    facts = probe.Facts(_harness(tmp_path))
    reuse, reload = _arm(probe, facts, "ld3_hoist"), _arm(probe, facts, "ld3_hoist_reload")
    # The stationary operand: named once per run of rows, or on every compute.
    assert "compute_accumulated" in reuse and "GARBAGE_ADDR," in reuse
    assert "compute_accumulated" not in reload
    # Move-in placement: a burst before the contraction, or each tile just before its first use.
    jit = _arm(probe, facts, "ld3_jit")
    contraction = jit[jit.index("for (int b = 0; b < bkc; b++)") :]
    assert "gemmini_extended_mvin(" in contraction and "if (d == 0)" in contraction
    assert "if (d == 0)" not in reuse
    # One load state pays the reconfigurations a single-slot backend would issue; three configure once.
    assert "live_ld" in _arm(probe, facts, "ld1_hoist") and "live_ld" not in reuse
    # Readout placement: fused into the last k-contribution, or one block readout afterwards.
    fused = _arm(probe, facts, "ld3_jit_store")
    assert "k0 + BK >= KT" in fused and "the whole block read out" not in fused
    assert "the whole block read out" in jit


def test_a_build_with_one_load_state_shares_it_across_operands(tmp_path):
    probe = _probe()
    facts = probe.Facts(_harness(tmp_path, mvins=("k_MVIN",)))
    source = _arm(probe, facts, "ld3_hoist")
    assert "gemmini_extended_mvin2(" not in source and "gemmini_extended_mvin3(" not in source
    assert "live_ld" in source


@pytest.mark.skipif(
    not (os.environ.get("RISCV_CC") and os.environ.get("GEMMINI_HARNESS")),
    reason="needs RISCV_CC and GEMMINI_HARNESS (a bare-metal Gemmini harness tree)",
)
def test_the_probe_builds_every_arm_into_one_binary(tmp_path):
    import json

    probe = _probe()
    code = probe.main(
        [
            "--target",
            "gemmini",
            "--shape",
            "64x64x64",
            "--out",
            str(tmp_path),
            "--build-only",
            "--harness",
            os.environ["GEMMINI_HARNESS"],
            "--cc",
            os.environ["RISCV_CC"],
        ]
    )
    assert code == 0
    record = json.loads((tmp_path / "arms.json").read_text())
    assert record["arms"][0] == "loop_ws" and len(record["arms"]) == 1 + len(probe.ARMS)
    assert (tmp_path / "ldprobe.elf").is_file()
