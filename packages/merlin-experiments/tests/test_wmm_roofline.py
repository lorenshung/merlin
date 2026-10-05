"""The per-group roofline: a tile-granular compute floor minimised over orientation, a per-direction
movement floor over the group's own tensors, the larger of the two, refuted by any measurement below it."""

from __future__ import annotations

import pytest
from merlin_experiments.phase2.whole_model_measured import roofline as R

_MACHINE = {
    "array_rows": 16,
    "array_cols": 16,
    "memory": {"status": "derived", "read_bytes_per_cycle": 16, "write_bytes_per_cycle": 16},
    "unresolved": {},
}


def test_full_blocks_cost_their_streamed_rows():
    floor = R.compute_floor(3136, 64, 64, array_rows=16, array_cols=16)
    # 196 x 4 x 4 computes, each streaming 16 rows.
    assert floor["min_computes"] == 3136 and floor["cycles"] == 3136 * 16


def test_a_single_row_pays_for_each_block_entering_the_array():
    """M=1 streams one row per block, but every 16-deep block still takes 16 cycles to enter."""
    floor = R.compute_floor(1, 2048, 1000, array_rows=16, array_cols=16)
    assert floor["options"]["stream_m"] == {"cycles": 63 * 2048, "computes": 1 * 128 * 63}
    # Holding the activation instead streams 1000 rows past each of 128 one-wide blocks.
    assert floor["options"]["stream_n"]["cycles"] == 128 * 1000
    assert floor["cycles"] == 128_000 and floor["orientation"] == "stream_n"


def test_a_short_reduction_edge_costs_only_its_rows():
    # K = 147 -> nine 16-deep blocks and one 3-deep block; M >= 16 so the stream dominates each.
    floor = R.compute_floor(12544, 147, 64, array_rows=16, array_cols=16)
    assert floor["min_computes"] == 784 * 10 * 4 and floor["cycles"] == 4 * 10 * 12544


def _conv_shape():
    entry = {"op": "conv2d", "N": 64, "ci": 3, "Himg": 224, "Wimg": 224, "kh": 7, "kw": 7}
    entry.update(stride=[2, 2], padding=[3, 3, 3, 3])
    return R._extents("conv2d", entry)


def test_a_convolution_is_its_own_im2col_contraction():
    assert _conv_shape() == {"M": 112 * 112, "K": 147, "N": 64}


def test_an_elementwise_group_is_bounded_by_movement_alone():
    shape = {"op": "residual_add", "extents": {"M": 3136, "N": 256}, "reads": {"a": 802816, "b": 802816},
             "writes": {"c": 802816}}  # fmt: skip
    doc = R.group_roofline(shape, _MACHINE)
    assert doc["compute_floor_cycles"] == 0 and doc["limiter"] == "movement"
    assert doc["roofline_cycles"] == 1605632 // 16 and doc["output_tiles"] == 196 * 16


def test_read_and_write_are_separate_channels():
    shape = {"op": "matmul", "extents": {"M": 1, "K": 2048, "N": 1000}, "reads": {"x": 2048, "w": 2048000},
             "writes": {"y": 4000}}  # fmt: skip
    doc = R.group_roofline(shape, _MACHINE)
    assert doc["movement_floor_cycles"] == pytest.approx((2048 + 2048000) / 16)
    assert doc["limiter"] == "movement" and doc["roofline_cycles"] == 128128


def test_an_unknown_memory_path_drops_its_term_and_says_so():
    machine = {**_MACHINE, "memory": {"status": "unknown", "reason": "no edge"}, "unresolved": {"movement": "no edge"}}
    shape = {"op": "matmul", "extents": {"M": 16, "K": 16, "N": 16}, "reads": {"a": 256}, "writes": {"c": 256}}
    doc = R.group_roofline(shape, machine)
    assert doc["roofline_cycles"] == 16 and doc["unresolved"] == {"movement": "no edge"}
    elementwise = R.group_roofline({**shape, "op": "residual_add", "extents": {"M": 16, "N": 16}}, machine)
    # No contraction and no movement term: no roofline -- never a zero one passed off as a bound.
    assert elementwise["roofline_cycles"] is None and elementwise["status"] == "unknown"


def test_a_tensor_of_undeclared_size_leaves_movement_unresolved():
    shape = {"op": "residual_add", "extents": {"M": 16, "N": 16}, "reads": {"a": None}, "writes": {"c": 256}}
    assert "a" in R.group_roofline(shape, _MACHINE)["unresolved"]["movement"]


def test_a_measurement_below_the_roofline_refutes_it():
    doc = {"roofline_cycles": 1000, "status": "derived"}
    assert R.confront(doc, [("ours", 1200), ("vendor", None)])["status"] == "derived"
    refuted = R.confront(doc, [("ours", 1200), ("other", 999)])
    assert refuted["status"] == "refuted" and refuted["refuted_by"] == [{"label": "other", "cycles": 999}]


def test_the_form_table_sums_whole_forms_or_says_what_is_missing():
    shapes = {"1": {"op": "matmul", "form_text": "f"}, "2": {"op": "matmul", "form_text": "f"},
              "3": {"op": "residual_add", "form_text": "r"}}  # fmt: skip
    roofs = {"1": {"roofline_cycles": 100, "status": "derived", "limiter": "compute", "min_computes": 4},
             "2": {"roofline_cycles": 50, "status": "derived", "limiter": "compute", "min_computes": 2},
             "3": {"roofline_cycles": 10, "status": "refuted", "limiter": "movement"}}  # fmt: skip
    rows = {r["form_text"]: r for r in R.form_table(shapes, roofs, {"ours": {"1": 300, "2": 150, "3": 40}})}
    assert rows["f"]["roofline_cycles"] == 150 and rows["f"]["ours_over_roofline"] == 3.0
    assert rows["f"]["min_computes"] == 6
    # A refuted group's bound is not a bar: its form has no roofline, and says which group.
    assert rows["r"]["roofline_cycles"] is None and rows["r"]["roofline_missing"] == ["3"]


def test_group_shapes_reads_the_buffers_tensors_and_omits_a_constant_operand(monkeypatch):
    from merlin_experiments.phase2.whole_model_measured import forms as PC

    from merlin.perf import whole_model_build as W

    buffer = {
        "tensors": {
            "ONES_g2": {"shape": [1, 49], "dtype": "i8"},
            "B_g1": {"shape": [49, 2048], "dtype": "i8"},
            "B_g2": {"shape": [2048, 1], "dtype": "i8"},
            "W": {"shape": [8, 4], "dtype": "i8"},
            "X": {"shape": [2, 8], "dtype": "i8"},
            "Y": {"shape": [2, 4], "dtype": "i32"},
        },  # fmt: skip
        "whole_program": {
            "per_group": [
                {
                    "group": 2,
                    "operands": {"lhs": "ONES_g2", "rhs": "B_g1", "dst": "B_g2"},
                    "entry": {"op": "matmul", "M": 2048, "K": 49, "N": 1},
                },
                {
                    "group": 3,
                    "operands": {"lhs": "X", "rhs": "W", "bias": "b", "dst": "Y"},
                    "entry": {"op": "matmul", "M": 2, "K": 8, "N": 4},
                },
            ]  # fmt: skip
        },
    }
    monkeypatch.setattr(W, "load_model_capsule", lambda path: {})
    monkeypatch.setattr(W, "state", lambda capsule, target: buffer)
    monkeypatch.setattr(PC, "form_of_entry", lambda entry, window_mean: ({}, f"{entry['op']}:{window_mean}"))
    shapes = R.group_shapes("/m", target="t")
    assert shapes["2"]["reads"] == {"B_g1": 49 * 2048} and shapes["2"]["form_text"] == "matmul:True"
    assert shapes["3"]["reads"] == {"X": 16, "W": 32} and shapes["3"]["writes"] == {"Y": 32}
    assert shapes["3"]["omitted"] == ["bias:b"]


def test_a_census_that_cannot_be_taken_is_recorded_and_never_raised(tmp_path):
    from merlin_experiments.phase2.whole_model_measured import group_capsules as G

    roofline = {"status": "derived", "roofline_cycles": 100, "limiter": "movement"}
    record = {"group": 3, "arm": G.ARM_PACKAGE, "elf": str(tmp_path / "p.elf"), "variant": {}}
    report = G.efficiency_row(record, {"cycles": 250}, {"rooflines": {"3": roofline}}, target="t", out=tmp_path)
    assert "refusal" in report and report["cycles_over_roofline"] == 2.5
