"""Per-group HEADROOM: measured cycles against two bounds derived from the target's own facts, never
a hardcoded rate -- a diagnostic the whole-model feedback reads, never a suggested fix.
"""

from __future__ import annotations

from merlin.perf import derived_bound as DB
from merlin.perf import group_headroom as GH


def _machine(*, rows=16, cols=16, muls=1, dram=None) -> DB.Machine:
    """A synthetic machine: only the two facts this module reads are given real values, the rest are
    UNKNOWN, exactly as a real, partially-derived facts document would leave them."""
    return DB.Machine(
        array_rows=DB.UNKNOWN if rows is None else rows,
        array_cols=DB.UNKNOWN if cols is None else cols,
        muls_per_element=DB.UNKNOWN if muls is None else muls,
        operand_bytes=DB.UNKNOWN,
        accumulate_bytes=DB.UNKNOWN,
        readout_bytes=DB.UNKNOWN,
        ping_pong_ways=DB.UNKNOWN,
        operand_store_bytes=DB.UNKNOWN,
        accumulate_store_bytes=DB.UNKNOWN,
        fill_drain_cycles=DB.UNKNOWN,
        dram_bytes_per_cycle=DB.UNKNOWN if dram is None else dram,
    )


# --------------------------------------------------------------------------------------- macs_and_bytes


def test_matmul_macs_and_bytes_are_the_plain_gemm_count():
    counted = GH.macs_and_bytes("matmul", {"M": 4, "K": 8, "N": 16, "operand_dtype": "i8", "output_dtype": "i8"})
    assert counted == (4 * 8 * 16, 4 * 8 + 8 * 16 + 4 * 16)


def test_conv_bytes_are_the_units_own_tensors_not_the_im2col_matrix():
    """MUTATION THIS CATCHES: charge conv traffic as ``positions * reduced`` (the im2col matrix a
    native unit never materializes) and a 3x3 conv's moved bytes come out ~9x too high."""
    facts = {
        "N": 8,
        "ci": 4,
        "Himg": 10,
        "Wimg": 10,
        "kh": 3,
        "kw": 3,
        "stride": [1, 1],
        "padding": [0, 0, 0, 0],
        "operand_dtype": "i8",
        "output_dtype": "i8",
    }
    counted = GH.macs_and_bytes("conv2d", facts)
    hout = wout = 10 - 3 + 1
    assert counted[0] == hout * wout * 8 * 4 * 3 * 3
    image_bytes = 10 * 10 * 4
    weight_bytes = 8 * 4 * 3 * 3
    result_bytes = hout * wout * 8
    assert counted[1] == image_bytes + weight_bytes + result_bytes
    # NOT the im2col-inflated figure: that double-counts by roughly kh*kw for a stride-1 conv.
    im2col_bytes = hout * wout * (4 * 3 * 3) + weight_bytes + result_bytes
    assert counted[1] < im2col_bytes


def test_an_op_this_module_has_no_formula_for_is_none():
    assert GH.macs_and_bytes("residual_add", {"anything": 1}) is None
    assert GH.macs_and_bytes("window_mean", {}) is None


def test_a_missing_extent_or_dtype_is_none_not_a_default():
    assert GH.macs_and_bytes("matmul", {"M": 4, "K": 8, "operand_dtype": "i8"}) is None  # no N
    assert GH.macs_and_bytes("matmul", {"M": 4, "K": 8, "N": 16}) is None  # no dtype
    assert GH.macs_and_bytes("conv2d", {"N": 8, "ci": 4, "Himg": 10, "Wimg": 10, "kh": 3, "kw": 3}) is None


# --------------------------------------------------------------------------------------- group_bound


def test_the_compute_bound_is_macs_over_the_derived_mesh_rate():
    bound = GH.group_bound(macs=256_000, moved_bytes=1_000, machine=_machine(rows=16, cols=16, muls=1))
    assert bound["mesh_macs_per_cycle"] == 256
    assert bound["compute_bound_cycles"] == 1000.0
    assert bound["traffic_bound_cycles"] is None  # dram rate UNKNOWN in this machine
    assert bound["bound_cycles"] == 1000.0 and bound["limiter"] == GH.COMPUTE_TERM


def test_the_traffic_bound_is_bytes_over_the_derived_dma_rate():
    bound = GH.group_bound(macs=256_000, moved_bytes=8_000, machine=_machine(rows=16, cols=16, muls=1, dram=4.0))
    assert bound["traffic_bound_cycles"] == 2000.0
    assert bound["compute_bound_cycles"] == 1000.0
    # The tighter (larger) bound is the binding one.
    assert bound["bound_cycles"] == 2000.0 and bound["limiter"] == GH.TRAFFIC_TERM


def test_an_unresolved_mesh_geometry_drops_the_compute_term_not_the_whole_bound():
    """Dropping a term LOOSENS the bound and says so, matching derived_bound's own discipline: an
    unresolved fact must never silently zero a term or crash the diagnostic."""
    bound = GH.group_bound(macs=256_000, moved_bytes=8_000, machine=_machine(rows=None, dram=4.0))
    assert bound["compute_bound_cycles"] is None
    assert bound["mesh_macs_per_cycle"] is None
    assert bound["bound_cycles"] == bound["traffic_bound_cycles"] == 2000.0


def test_both_terms_unresolved_bounds_nothing():
    bound = GH.group_bound(macs=256_000, moved_bytes=8_000, machine=_machine(rows=None, dram=None))
    assert bound["compute_bound_cycles"] is None and bound["traffic_bound_cycles"] is None
    assert bound["bound_cycles"] is None and bound["limiter"] is None


# --------------------------------------------------------------------------------------- group_headroom


def test_group_headroom_states_ours_minus_reference_and_the_multiple_of_the_bound():
    machine = _machine(rows=16, cols=16, muls=1)
    facts = {"M": 100, "K": 100, "N": 100, "operand_dtype": "i8", "output_dtype": "i8"}
    document = GH.group_headroom(
        op="matmul", shape_facts=facts, ours_cycles=10_000, reference_cycles=6_000, machine=machine
    )
    macs = 100 * 100 * 100
    expected_bound = macs / 256
    assert document["headroom_cycles"] == 4_000
    assert document["bound_cycles"] == expected_bound
    assert document["over_bound"] == round(10_000 / expected_bound, 4)


def test_group_headroom_is_none_for_an_op_with_no_macs_formula():
    assert (
        GH.group_headroom(
            op="residual_add", shape_facts={"whatever": 1}, ours_cycles=100, reference_cycles=90, machine=_machine()
        )
        is None
    )


def test_group_headroom_is_none_without_shape_facts_or_an_int_cycle_count():
    machine = _machine()
    assert (
        GH.group_headroom(op="matmul", shape_facts=None, ours_cycles=100, reference_cycles=90, machine=machine) is None
    )
    facts = {"M": 4, "K": 4, "N": 4, "operand_dtype": "i8"}
    assert (
        GH.group_headroom(op="matmul", shape_facts=facts, ours_cycles=None, reference_cycles=90, machine=machine)
        is None
    )


# --------------------------------------------------------------------------------------- the rank signal


def _priced_machine() -> DB.Machine:
    """The facts the schedule proxy needs to size a transaction: array geometry and operand width."""
    return DB.Machine(
        array_rows=16,
        array_cols=16,
        muls_per_element=1,
        operand_bytes=1,
        accumulate_bytes=4,
        readout_bytes=DB.UNKNOWN,
        ping_pong_ways=2,
        operand_store_bytes=131072,
        accumulate_store_bytes=32768,
        fill_drain_cycles=DB.UNKNOWN,
        dram_bytes_per_cycle=DB.UNKNOWN,
    )


_CONV = {
    "N": 8,
    "ci": 4,
    "Himg": 10,
    "Wimg": 10,
    "kh": 3,
    "kw": 3,
    "stride": [1, 1],
    "padding": [0, 0, 0, 0],
    "operand_dtype": "i8",
    "output_dtype": "i8",
}


def test_the_bound_and_the_rank_price_the_same_contraction():
    """``macs_and_bytes`` is the contraction's own count, so the two signals cannot drift apart."""
    for op, facts in (
        ("matmul", {"M": 4, "K": 8, "N": 16, "operand_dtype": "i8", "output_dtype": "i8"}),
        ("conv2d", _CONV),
    ):
        work = GH.contraction_for(op, facts)
        assert GH.macs_and_bytes(op, facts) == (work.macs, work.moved_bytes)


def test_a_convolution_streams_one_output_row_and_reads_its_image_not_its_im2col():
    work = GH.contraction_for("conv2d", _CONV)
    assert (work.m, work.n, work.k) == (8 * 8, 8, 4 * 3 * 3)
    assert work.stream_run == 8  # one output row of an 8x8 output
    assert work.lhs_bytes == 10 * 10 * 4


def test_group_rank_is_the_proxys_order_and_never_a_cycle_count():
    machine = _priced_machine()
    narrow = GH.group_rank(GH.contraction_for("conv2d", _CONV), machine, target="any")
    wide = GH.group_rank(GH.contraction_for("conv2d", {**_CONV, "Himg": 34, "Wimg": 34}), machine, target="any")
    assert narrow["transactions"] is not None and wide["transactions"] > narrow["transactions"]
    assert set(narrow["terms"]) == {"compute", "movement"}
    assert "cycles" in narrow["unresolved"]  # uncalibrated: an order, not a magnitude
    assert narrow["regime"] == "UNDECIDED"


def test_group_rank_refuses_when_the_facts_do_not_size_a_transaction():
    rank = GH.group_rank(GH.contraction_for("conv2d", _CONV), _machine(), target="any")
    assert rank["transactions"] is None
    assert "operand_bytes" in rank["unresolved"]


def test_group_headroom_carries_the_rank_when_the_target_is_named():
    machine = _priced_machine()
    facts = {"M": 64, "K": 64, "N": 64, "operand_dtype": "i8", "output_dtype": "i8"}
    without = GH.group_headroom(op="matmul", shape_facts=facts, ours_cycles=900, reference_cycles=800, machine=machine)
    assert "rank" not in without
    with_rank = GH.group_headroom(
        op="matmul", shape_facts=facts, ours_cycles=900, reference_cycles=800, machine=machine, target="any"
    )
    assert with_rank["rank"]["transactions"] > 0
    assert with_rank["headroom_cycles"] == 100
