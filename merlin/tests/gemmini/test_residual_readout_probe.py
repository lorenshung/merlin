"""The target-local direct-command residual witness has a closed numerical contract."""

from __future__ import annotations

import importlib.util

import pytest

from merlin.common.paths import repo_root


def _probe():
    path = repo_root() / "examples/gemmini/verification/probe_residual_readout.py"
    spec = importlib.util.spec_from_file_location("gemmini_residual_readout_probe", path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_selected_gain_above_one_uses_readout_scale_and_full_domain_stays_bounded():
    probe = _probe()
    report = {"entries": [{"name": "sum", "entry": {"op": "residual_add", "epilogue": ["relu"],
                 "operand_dtype": "int8", "lhs_scale": 1.5, "rhs_scale": 0.75, "bound_lsb": 2}}]}
    group = probe.select_group(report)
    expected = probe.expected_outputs(group, "full")
    assert len(expected["reference"]) == len(expected["unit_model"]) == 256
    assert expected["gain"] == {"lhs_load": 1.0, "rhs_load": 0.5, "readout": 1.5}
    assert probe.compare(expected["unit_model"], expected, group["bound"])["n_over_bound"] == 0
    source = probe.render_source(group, "full")
    assert "gemmini_extended_mvin(&A[i][j], acc" in source
    assert "gemmini_extended_mvin(&B[i][j], acc | accumulate" in source
    assert "gemmini_extended_config_st(COLS*sizeof(elem_t), RELU, 1.5f)" in source
    assert "static const int8_t expected[ROWS][COLS]" in source
    assert 'printf("CHECK %d\\n", mismatches)' in source
    assert "gemmini_loop_ws(" not in source and "tiled_resadd" not in source
    # A direct >1 load would clip before the cancelling operand arrived; this pair exposes it.
    a, b = 127, -128
    direct = min(127, max(0, min(127, round(a * group["lhs"])) + round(b * group["rhs"])))
    assert abs(direct - expected["reference"][a + 128][b + 128]) > group["bound"]


def test_transcript_and_derived_opcode_screen_fail_closed():
    probe = _probe()
    assert probe._c_float(1.0) == "1.0f"
    assert probe.parse_rows("ROW 0 1 2\nROW 1 3 4\nDONE\n", (2, 2)) == [[1, 2], [3, 4]]
    with pytest.raises(ValueError, match="incomplete"):
        probe.parse_rows("ROW 0 1 2\nDONE\n", (2, 2))
    assert probe.parse_full_check("CHECK 0\nDONE\n") == 0
    with pytest.raises(ValueError, match="incomplete"):
        probe.parse_full_check("CHECK 0\n")
    assert tuple(map(len, probe.campaign_values("boundary_rows"))) == (16, 256)
    assert tuple(map(len, probe.campaign_values("boundary_cols"))) == (256, 16)
    with pytest.raises(ValueError, match="invalid"):
        probe.parse_full_check("CHECK 2\nDONE\n", pairs=1)
    names = {"0": "CONFIG_CMD", "2": "LOAD_CMD", "3": "STORE_CMD", "7": "FLUSH_CMD",
             "8": "LOOP_WS"}
    facts = {"facts": {"interfaces": [{"name": "funct_decode_table", "custom_opcode": 123,
                                          "names": names}]}}
    def lines(*functs):
        return "\n".join(f"  {i * 4:x}: {(funct << 25) | 123:08x}  .insn" for i, funct in enumerate(functs))
    assert len(probe.decode_custom(lines(0, 2, 3, 7), facts)) == 4
    with pytest.raises(ValueError, match="direct command proof"):
        probe.decode_custom(lines(0, 2, 3, 7, 8), facts)
