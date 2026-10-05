"""A model is emitted as a program of its groups only when its groups are the whole program."""

from __future__ import annotations

import json
import struct
from pathlib import Path

import numpy as np
import pytest
from fake_quant_layer import Oracle, module

from merlin.common.paths import merlin_dir


def _script():
    from selected_driver import load

    return load("gemmini", "group_model_program.py")


def _capture(tmp_path: Path, text: str) -> Path:
    payload = np.zeros(16, dtype="<f4").tobytes()
    header = json.dumps({"layer.bias": {"dtype": "F32", "shape": [16], "data_offsets": [0, len(payload)]}}).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(header)) + header + payload)
    (tmp_path / "weights.safetensors.manifest.json").write_text(
        json.dumps({"0": {"kind": "input", "name": "x"}, "4": {"kind": "param", "weight": "layer.bias"}})
    )
    (tmp_path / "linalg.mlir").write_text(text, encoding="utf-8")
    return tmp_path


def _capture_with_stored_weight(tmp_path: Path, text: str) -> Path:
    """The same capture with the contraction's WEIGHT named too, so its group has a device form."""
    weight, bias = np.zeros((8, 16), dtype=np.int8), np.zeros(16, dtype="<f4")
    blob, header, offset = b"", {}, 0
    for name, array, dtype in (("layer.weight", weight, "I8"), ("layer.bias", bias, "F32")):
        data = array.tobytes()
        header[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + len(data)]}
        blob, offset = blob + data, offset + len(data)
    raw = json.dumps(header).encode()
    (tmp_path / "weights.safetensors").write_bytes(struct.pack("<Q", len(raw)) + raw + blob)
    (tmp_path / "weights.safetensors.manifest.json").write_text(
        json.dumps(
            {
                "0": {"kind": "input", "name": "x"},
                "1": {"kind": "param", "weight": "layer.weight"},
                "4": {"kind": "param", "weight": "layer.bias"},
            }
        )
    )
    (tmp_path / "linalg.mlir").write_text(text, encoding="utf-8")
    return tmp_path


def test_the_float_input_quantize_is_kept_on_the_host_and_does_not_stop_the_program(tmp_path: Path) -> None:
    # A captured model's first region is its input ARGUMENT put on the integer grid, and its operand
    # is a float32 tensor no integer unit reads. It is the one host region a closed model keeps.
    #
    # A target that correctly admits `elementwise_map` STANDALONE -- as one running the residual add
    # with no contraction in front of it does -- must not thereby acquire this region: the format
    # question is about what the region reads, and there is no device form for it either way.
    # Routed onto a unit, it refused the whole model with "group 0 cannot be stated as a device
    # program: a host region has no contraction root to demand" -- naming a contraction root, about
    # a group that was not on the host at all.
    script = _script()
    text = module(weight_dequantize="per_tensor", quantized_input=True)
    capture = _capture_with_stored_weight(tmp_path, text)
    with pytest.raises(script.NotClosed) as refused:
        script.extract(capture, "synthetic", oracle=Oracle(standalone=("elementwise_map",)))
    # The refusal is this fixture's TAIL -- one layer is not a classifier, so its last group does not
    # leave as an accumulator to compare against a golden. Reaching that check at all means the input
    # quantize stayed on the host and every group after it was stated as a device program.
    assert "does not leave as an accumulator" in str(refused.value)
    assert "cannot be stated as a device program" not in str(refused.value)
    assert "host region(s) compute between the groups" not in str(refused.value)


def test_a_capture_with_host_work_between_its_groups_is_refused_by_name(tmp_path: Path) -> None:
    # A per-channel weight the readout cannot hold leaves the requantization on the host: a number
    # for "the groups" of this model would silently run the rest somewhere nobody measured.
    script = _script()
    with pytest.raises(script.NotClosed, match="host region"):
        script.extract(_capture(tmp_path, module(weight_dequantize="per_channel")), "synthetic", oracle=Oracle())


def test_a_group_that_cannot_be_stated_as_a_device_program_is_refused(tmp_path: Path) -> None:
    # The fixture's weight is a bare model argument with no manifest entry: nothing says which
    # operand is stored, so the group has no device form and the program is not built around it.
    script = _script()
    with pytest.raises(script.NotClosed, match="cannot be stated as a device program"):
        script.extract(_capture(tmp_path, module(weight_dequantize="per_tensor")), "synthetic", oracle=Oracle())


def _model() -> dict:
    rng = np.random.default_rng(3)
    arrays = {
        "IMAGE_DATA": rng.integers(-128, 128, size=(1, 6, 6, 2), dtype=np.int8),
        "W_g1": rng.integers(-8, 8, size=(3, 3, 2, 4), dtype=np.int8),
        "BIAS_g1": rng.integers(-50, 50, size=4, dtype=np.int32),
        "W_g4": rng.integers(-8, 8, size=(4, 5), dtype=np.int8),
        "GOLDEN": np.arange(5, dtype=np.float32),
    }
    steps = [
        {"kind": "conv2d", "group": 1, "in": "IMAGE", "out": "B_g1", "weight": "W_g1", "bias": "BIAS_g1", "relu": True,
         "scale": 0.05, "in_dim": 6, "ci": 2, "n": 4, "out_dim": 6, "stride": 1, "padding": 1, "kernel": 3,
         "pool": {"size": 0, "stride": 0, "padding": 0}},
        {"kind": "sum", "group": 2, "lhs": "B_g1", "rhs": "B_g1", "out": "B_g2", "rows": 36, "cols": 4,
         "lhs_load": 0.37, "rhs_load": 0.61, "readout": 1.0, "relu": False, "bound_lsb": 1},
        {"kind": "mean", "group": 3, "in": "B_g2", "out": "B_g3", "rows": 4, "window": 36, "multiplier": 0.04},
        {"kind": "matmul", "group": 4, "in": "B_g3", "out": "B_g4", "weight": "W_g4", "bias": None, "relu": False,
         "scale": None, "dequantize": 0.5, "m": 1, "k": 4, "n": 5},
    ]  # fmt: skip
    buffers = [
        {"name": "B_g1", "elements": 144, "ctype": "elem_t"},
        {"name": "B_g2", "elements": 144, "ctype": "elem_t"},
        {"name": "B_g3", "elements": 4, "ctype": "elem_t"},
        {"name": "B_g4", "elements": 5, "ctype": "acc_t"},
    ]
    return {"steps": steps, "buffers": buffers, "arrays": arrays, "classes": 5, "groups": 5, "device_groups": 4}


def test_the_emulation_is_the_arithmetic_each_library_call_documents() -> None:
    script, model = _script(), _model()
    values = script.emulate(model)["values"]
    # The convolution, written out the slow way with the library's own weight index.
    x, w = np.pad(model["arrays"]["IMAGE_DATA"][0].astype(int), ((1, 1), (1, 1), (0, 0))), model["arrays"]["W_g1"]
    out = np.zeros((6, 6, 4))
    for r in range(6):
        for c in range(6):
            for n in range(4):
                acc = int((x[r : r + 3, c : c + 3, :] * w[:, :, :, n]).sum()) + int(model["arrays"]["BIAS_g1"][n])
                out[r, c, n] = max(min(np.rint(np.float32(acc) * np.float32(0.05)), 127), 0)
    assert (values["B_g1"] == out.reshape(-1)).all()
    # A sum rounds each operand; computed the capture's way it may differ, by no more than its bound.
    once = script.emulate(model, single_rounding_sums=True)["values"]["B_g2"]
    difference = np.abs(values["B_g2"] - once)
    assert 0 < difference.max() <= model["steps"][1]["bound_lsb"]
    # The mean reads positions by channel: one sum per channel, scaled once.
    channels = values["B_g2"].reshape(36, 4).sum(axis=0)
    assert (values["B_g3"] == np.clip(np.rint(channels.astype(np.float32) * np.float32(0.04)), -128, 127)).all()


def test_every_group_buffer_is_marked_used_so_the_optimizer_cannot_drop_it() -> None:
    """A group's committed buffer is a property of the PROGRAM (the model statement declares it for
    every device group, whoever answers it), never of what the compiler can prove is read. Measured:
    routing a buffer's sole consumer to a vendor library call the compiler could fully see through (a
    large inline routine specialized on that call's literal shape/flag arguments) let -O2 prove the
    argument was never dereferenced for that specialization, which made the producing group's write a
    dead store too and dropped the buffer from the linked program -- `memory_map` then found no sized
    symbol for a group the model statement still declares. `__attribute__((used))` makes retention
    independent of any one call site's provable behavior."""
    script = _script()
    source = script.render(_model())
    for buffer in _model()["buffers"]:
        name = buffer["name"]
        assert f"static {buffer['ctype']} {name}[{buffer['elements']}] row_align(1) __attribute__((used));" in source


def test_each_group_is_one_timed_call_and_nothing_else_is_inside_the_window() -> None:
    script = _script()
    source = script.render(_model())
    assert source.count("t0 = read_cycles();") == 4 and source.count("tiled_conv_auto(") == 1
    assert "tiled_resadd_auto(36, 4, 0.37f, 0.61f, 1.0f, B_g1, B_g1, B_g2, false, WS);" in source
    # The mean is the library's matmul reading its operand transposed, against a constant one.
    assert "tiled_matmul_auto(4, 1, 36, B_g2, ONES, NULL, B_g3," in source and "false, true, false, false" in source
    # The classifier leaves as the full accumulator, and its checksum is the accumulator's.
    assert (
        "ACC_SCALE_IDENTITY, 0, true, false, false, true, false, 0, WS);" in source
        and "checksum_acc(B_g4, 5)" in source
    )
    for line in source.splitlines():
        if "t0 = read_cycles();" in line:
            # The accounting that follows is AFTER `dt` is taken, so it is outside the timed
            # region; what must never appear inside it is a checksum or a printf.
            assert "checksum" not in line and "printf" not in line
            timed = line.partition("t0 = read_cycles();")[2].partition("dt = read_cycles()")[0]
            assert "read_cycles" not in timed
            assert line.rstrip().endswith("++;") or line.rstrip().endswith("total += dt;")
    assert 'BLOB(W_g1, "W_g1.bin", elem_t)' in source and 'BLOB(BIAS_g1, "BIAS_g1.bin", acc_t)' in source


# --- the measurement protocol, pinned to the lines a real FPGA run printed ---
#
# Until 2026-09-19 this program published its measured total as `GROUP_MODEL_TOTAL cycles: %llu`
# and no `MERLIN_INVOCATIONS` line at all, so `firesim_receipt._verify_uart` rejected its UART
# outright: no run of this shape could ever have been sealed, independently of anything the queue
# did.  These tests hold the emission to the constants the receipt requires, and hold those
# constants to `merlin/tests/data/firesim_queue/job610_uart_marker_skeleton.log` -- the protocol
# lines one real FireSim run actually emitted -- so the harness can only be wrong if those bytes
# are wrong.

_OBSERVED_UART_LOG = merlin_dir() / "tests/data/firesim_queue/job610_uart_marker_skeleton.log"


def _observed(prefix: str) -> list[str]:
    return [
        line
        for line in _OBSERVED_UART_LOG.read_text(encoding="utf-8").splitlines()
        if line.startswith(prefix) and not line.startswith("#")
    ]


def test_the_harness_spells_its_metric_the_way_a_real_run_spelled_it() -> None:
    script = _script()
    observed = _observed("METRIC")

    assert observed == ["METRIC cycles 33085199302"]
    assert script.UART["metric"].format(cycles=33085199302) == observed[0]
    assert script.UART["invocations"] == _observed("MERLIN_INVOCATIONS")[0]
    assert [script.UART[key] for key in ("warm_begin", "warm_end", "measured_begin", "measured_end")] == _observed(
        "MERLIN_PROFILE"
    )


def test_the_rendered_program_publishes_the_protocol_in_the_required_order() -> None:
    source = _script().render(_model())
    published = [
        line.strip().partition('printf("')[2].partition('\\n"')[0]
        for line in source.splitlines()
        if line.strip().startswith("printf(")
    ]

    assert published == [
        "MERLIN_INVOCATIONS warmup=1 measured=1",
        "MERLIN_WINDOW begin label=group_model",
        "MERLIN_PROFILE warmup begin",
        "MERLIN_PROFILE warmup end rc=0",
        "MERLIN_PROFILE measured begin",
        # THE PER-GROUP DIGESTS COME AFTER THE WINDOW CLOSES, NOT INSIDE THE MEASURED PASS.
        # Measured on hardware: with the checksums and these prints inside the window, 98.9% of the
        # whole-model figure was this instrumentation rather than the model. The buffers are
        # retained, so the digests are byte-identical; only their timing moved.
        "GM_GROUP 1 conv2d %llu sum=%lld fnv1a=%lld",
        "GM_GROUP 2 sum %llu sum=%lld fnv1a=%lld",
        "GM_GROUP 3 mean %llu sum=%lld fnv1a=%lld",
        "GM_GROUP 4 matmul %llu sum=%lld fnv1a=%lld",
        # The quantity a tolerance-graded op is actually graded on. `sum` is permutation-blind and
        # `fnv1a` is exact, so neither can answer "is every element within +/- bound_lsb?" -- the
        # question `compare: bounded_int` asks. Emitted only for the groups that declare a bound.
        "GM_BOUND 2 max_abs=%lld over=%lld bound=1",
        # And the one element that deviates most, with the operands the device actually had, so the
        # arithmetic can be redone by hand rather than hypothesised about.
        "GM_WITNESS 2 i=%lld lhs=%lld rhs=%lld device=%lld reference=%lld",
        # And where the out-of-bound elements fall, which distinguishes a tile boundary from a
        # scattered fault. The row/col pairs follow on the same line.
        "GM_WHERE 2 cols=4 first=%lld %lld %lld %lld",
        # The three the published band is quoted in. `full model` is the whole-window delta the
        # reference artifact publishes, `bracketed compute only sum` is the additive total this
        # program has always published, and the difference is what the brackets never saw. They
        # precede GM_ARGMAX because the window closes before the golden comparison runs.
        "FM full model cycles: %llu",
        "FM bracketed compute only sum: %llu",
        # The split, never one fused number: two arms differing by a single total invite the reader
        # to attribute all of the difference to the kernels.
        "FM split authored=%llu groups=%u | im2col=%llu groups=%u (inside authored; harness code, "
        "a floor not a verdict) | vendor=%llu groups=%u",
        "FM uncounted delta: %llu",
        "GM_ARGMAX got=%d want=%d agrees=%d",
        "GM_COSINE_PPM %d",
        "METRIC cycles %llu",
        "MERLIN_PROFILE measured end rc=0",
        "MERLIN_WINDOW end label=group_model",
    ]
    # The spelling nothing could parse, and the reason no run of this shape had ever been sealed.
    assert "GROUP_MODEL_TOTAL" not in source, "one measurement has one spelling"


def test_the_uart_this_program_would_print_is_one_the_receipt_parser_accepts() -> None:
    """The end the harness owns: its own output, fed to the production parser, unedited.

    `_verify_uart` is the function that rejected every run of this program before today.  Feeding
    it the harness's own rendering closes the loop without hardware -- and it is the harness's
    rendering, built from the same format strings its C `printf` calls are, not a transcript
    written beside it.
    """
    from merlin.perf.firesim_receipt import _verify_uart

    script, model = _script(), _model()
    emulated = script.emulate(model)
    steps = script.program_steps(model)
    checksums = {str(step["group"]): int(emulated["values"][step["out"]].sum()) for step in model["steps"]}
    got, want = emulated["argmax"], emulated["want"]
    markers = (
        f"GM_ARGMAX got={got} want={want} agrees={int(got == want)}",
        f"GM_COSINE_PPM {int(emulated['cosine'] * 1000000.0)}",
    )
    text = (
        "\n".join(
            script.uart_lines(
                steps,
                cycles=987654,
                checksums=checksums,
                argmax=got,
                want=want,
                cosine_ppm=int(emulated["cosine"] * 1000000.0),
            )
        )
        + "\n"
    )

    _invocation, _profiles, _metric_line, cycles, correctness = _verify_uart(text, markers)

    assert cycles == 987654
    assert len(correctness) == len(markers)


def test_the_whole_model_window_holds_the_measured_pass_and_not_the_warm_up() -> None:
    """The window the published band is quoted in, and what falls inside it.

    The reference artifact opens its window at the first statement of ``main`` because its ``main``
    holds one model pass. This program runs a warm-up first, so the same placement would time two
    passes and publish roughly twice the work under a name claiming one.
    """
    source = _script().render(_model())
    body = source.partition("int main(void)")[2]
    opened = body.index("fm_start = read_cycles();")
    closed = body.index("fm_end = read_cycles();")

    # The warm-up is before the window; the measured pass is inside it.
    assert body.index("run(0);") < opened < body.index("run(1);") < closed
    # The runtime dequantize and the argmax are inside, as they are in the reference.
    window = body[opened:closed]
    assert "the runtime dequantize" in window and "GOLDEN[i] > GOLDEN[want]" in window
    # The golden/cosine comparison is this program's own check; the reference has no equivalent,
    # so timing it would add work to a number meant to be comparable.
    assert "dot += y * GOLDEN[i]" not in window
    # No printf inside the window: the reference's own banner keeps them out.
    assert "printf(" not in window


def test_the_three_published_quantities_are_one_arithmetic_identity() -> None:
    """`uncounted` is a difference, not a third measurement: it must be stated as one."""
    source = _script().render(_model())
    assert "(fm_end - fm_start) - total" in source
    assert "(unsigned long long)(fm_end - fm_start)" in source
    # The additive sum keeps its old spelling too, because seven FPGA runs on record published it.
    assert 'printf("METRIC cycles %llu' in source.replace("{", "").replace("}", "") or "METRIC cycles" in source


def test_an_object_named_but_absent_stops_the_build_rather_than_linking_without_it(tmp_path) -> None:
    """A missing submission kernel must not silently become a vendor call.

    The arm's whole claim is that these groups ran the submission's own code. Linking without an
    object that was named would produce a program that still runs -- the vendor call is already
    there -- and a number attributed to a compiler that contributed nothing.
    """
    import pytest

    script = _script()
    with pytest.raises(SystemExit, match="named but not present"):
        script.build(
            _model(),
            Path("/nonexistent-vendor"),
            Path("/nonexistent-compiler"),
            tmp_path / "build",
            extra_objects=[Path("/nonexistent/g7.o")],
        )


def test_the_receipt_names_linked_objects_by_content() -> None:
    """A path is the one thing a rebuild can keep while the bytes change underneath it."""
    source = Path(_script().__file__).read_text(encoding="utf-8")
    assert '"linked_objects"' in source and "_sha256(o)" in source


def test_no_digest_or_print_happens_inside_the_measured_window() -> None:
    """MEASURED on hardware: with the per-group checksums and prints inside the window, 98.9% of the
    whole-model figure was this harness's own instrumentation -- 71 scalar passes over every
    intermediate tensor plus 71 UART writes -- against a published band quoted in whole-window
    cycles. The reference driver's banner says it outright: no UART traffic inside the window.

    What must NOT move out is anything a deployment performs. The quantization, the layout changes,
    the buffer handling, the residual adds, the runtime dequantize and the argmax all stay inside.
    """
    source = _script().render(_model())
    body = source.partition("int main(void)")[2]
    # THE WINDOW IS WHAT EXECUTES BETWEEN THE TWO CLOCK READS, NOT WHAT IS PRINTED BETWEEN THEM IN
    # THIS FILE. `run(1)` is called inside it and its body is defined ABOVE main, so a check that
    # only slices main's text cannot see the per-group calls -- which is precisely where the
    # instrumentation used to sit. Verified by mutation: a digest put back into `run` passed a
    # main-only check untouched.
    run_body = source.partition("static uint64_t run(int measured)")[2].partition("return total;")[0]
    window = body[body.index("fm_start = read_cycles();") : body.index("fm_end = read_cycles();")] + run_body

    assert "printf(" not in window, "a UART write inside the window is measured as model time"
    assert "checksum_" not in window and "sum_elem(" not in window and "sum_acc(" not in window
    # The work a deployment still performs stays inside.
    assert "run(1)" in window
    assert "GOLDEN[i] > GOLDEN[want]" in window, "the argmax is the model's own answer"
    assert "the runtime dequantize" in window
    # And the digests are still produced, after the clock stops, over the retained buffers.
    after = body[body.index("fm_end = read_cycles();") :]
    assert after.count(chr(34) + "GM_GROUP") == len(_model()["steps"]), "one digest line per group"
    assert "checksum_" in after and "fm_dt[" in after


def test_the_bracketed_cycles_are_still_taken_inside_each_group() -> None:
    """`dt` is a register read, not a print. Moving it out would change the bracket sum itself and
    break comparability with every run already on record."""
    source = _script().render(_model())
    for line in source.splitlines():
        if "t0 = read_cycles();" in line:
            assert "dt = read_cycles() - t0;" in line and "fm_dt[" in line
            assert "printf" not in line and "checksum" not in line


def test_a_gathered_group_reports_its_gather_and_its_kernel_apart() -> None:
    """MEASURED on the FPGA: arm B's convolutions were 6.03x the vendor's and 95% of its whole gap,
    with the worst 3x3s at 23.8x. The bracket fused a caller-materialized im2col gather with the
    kernel that reads it, so one number could not say which half was slow -- and the gather is this
    harness's code while the kernel is the submission's. Both stay inside the window; only the
    attribution changes.
    """
    script = _script()
    gathered = {
        "definitions": "",
        "calls": {
            1: "{ uint64_t i0 = read_cycles(); g1(); fm_g = read_cycles() - i0; fm_im2col += fm_g; fm_im2col_groups++; k1(); }"
        },
    }
    source = script.render(_model(), sched_kernels=gathered)
    assert "static uint64_t fm_gather[" in source and "static uint64_t fm_g;" in source
    # Reset before every group, so a group with no gather cannot inherit the previous one's.
    for line in source.splitlines():
        if "t0 = read_cycles();" in line:
            assert line.strip().startswith("fm_g = 0;"), "a stale gather would be charged to the wrong group"
    assert "GM_SPLIT" in source and "fm_dt[0] - fm_gather[0]" in source
    # Reported on its own line, so a receipt parser that knows GM_GROUP keeps working unchanged.
    assert "GM_GROUP" in source


# ------------------------------------------------------- one accessor per fact, not two spellings


def test_the_elementwise_and_mean_steps_read_their_extents_off_the_stated_entry() -> None:
    """A group is stated ONCE, in the entry vocabulary the capsule corpus is built from, and this
    harness reads its numbers off that entry.

    The sum step used to flatten the capture's own result type inline (``_product(shape[:-1])`` /
    ``int(shape[-1])``) and the mean step used to read ``group.window_mean`` directly -- each a second
    statement of a fact `group_command` already states, in a file that is required to agree with it.

    MUTATION THIS CATCHES: put either inline derivation back and this fails, because the harness is
    once again the second place the same arithmetic is written down.
    """
    from selected_driver import driver_file

    source = driver_file("gemmini", "group_model_program.py").read_text(encoding="utf-8")
    steps = source[
        source.index("if group.operand_sum is not None:") : source.index("activation = list(group.root.operands)")
    ]
    assert "_product(shape[:-1])" not in source, "the harness restates the entry's own flattening"
    assert steps.count("GC.device_output_shape(entry)") == 2, "both steps take their extents from the entry"
    assert 'group.operand_sum["lhs_scale"]' not in source and 'group.operand_sum["relu"]' not in source
    assert 'group.window_mean["multiplier"]' not in source and 'group.window_mean["window"]' not in source


def test_a_stated_elementwise_sum_has_a_device_output_shape_at_all() -> None:
    """The accessor the harness now depends on has to answer for the op it is asked about.

    MUTATION THIS CATCHES: narrow ``device_output_shape`` to contractions and convolutions -- the two
    ops that had callers when it was written -- and every elementwise group of a model loses its
    buffer size. The rule is fail closed on an UNDERIVABLE fact, not on an unfamiliar op: this entry
    states its own extents.
    """
    from merlin.xdsl_dialects.lowering import group_command as GC

    entry = {
        "op": "residual_add",
        "M": 3584,
        "N": 56,
        "lhs_scale": 1.0,
        "rhs_scale": 1.0,
        "bound_lsb": 0,
        "epilogue": ["relu"],
    }
    assert GC.device_output_shape(entry) == [3584, 56]
    with pytest.raises(GC.NoDeviceShape):
        GC.device_output_shape({"op": "residual_add", "epilogue": []})


def test_the_bounded_check_recomputes_the_reference_the_contract_describes(tmp_path) -> None:
    """COMPILED AND RUN against numpy. The residual-add contract says the reference rounds ONCE --
    dequantize both, add in f32, quantize -- while a scaled rounding load rounds each operand. The
    device's own digest cannot express that difference, so the check recomputes the reference here
    and reports the max absolute deviation. If this arithmetic is wrong the gate is wrong.
    """
    import shutil
    import subprocess

    import numpy as np

    if shutil.which("cc") is None:
        pytest.skip("no host C compiler")
    source = _script().render(_model())
    block = source.partition("GM_BOUND")[0].rpartition("{\n        long long worst")[0]
    body = "{\n        long long worst" + source.partition("{\n        long long worst")[2]
    body = body[: body.index("GM_BOUND")]
    body = body[: body.rindex("printf(")]
    step = next(s for s in _model()["steps"] if s["kind"] == "sum")
    n = 64
    rng = np.random.default_rng(1)
    lhs = rng.integers(-128, 127, n).astype(np.int64)
    rhs = rng.integers(-128, 127, n).astype(np.int64)
    ref = (lhs * step["lhs_load"] + rhs * step["rhs_load"]) * step["readout"]
    want = np.where(ref < 0, -np.floor(0.5 - ref), np.floor(ref + 0.5))
    if step.get("relu"):
        want = np.maximum(want, 0)
    want = np.clip(want, -128, 127)
    # Feed the device buffer a deliberate off-by-one on one element.
    out = want.copy()
    out[7] += 1
    prog = tmp_path / "b.c"
    prog.write_text(
        "#include <stdio.h>\n#include <stdint.h>\n#include <stddef.h>\n#include <math.h>\n"
        "typedef int8_t elem_t;\n"
        f"static elem_t L[{n}], R[{n}], O[{n}];\n"
        "int main(void){\n"
        f'  for (int i=0;i<{n};i++) {{ if (scanf("%hhd %hhd %hhd", &L[i], &R[i], &O[i]) != 3) return 2; }}\n'
        "  long long worst = 0, over = 0;\n"
        f"  for (size_t i = 0; i < {n}; i++) {{\n"
        f"    double ref = ((double)L[i] * {step['lhs_load']!r} + (double)R[i] * {step['rhs_load']!r}) * {step['readout']!r};\n"
        "    long long w = (long long)(ref < 0 ? ref - 0.5 : ref + 0.5);\n"
        + ("    if (w < 0) w = 0;\n" if step.get("relu") else "")
        + "    if (w > 127) w = 127; else if (w < -128) w = -128;\n"
        "    long long d = (long long)O[i] - w; if (d < 0) d = -d;\n"
        f"    if (d > worst) worst = d; if (d > {step['bound_lsb']}) over++;\n"
        "  }\n"
        '  printf("%lld %lld\\n", worst, over);\n  return 0;\n}\n',
        encoding="utf-8",
    )
    subprocess.run(["cc", "-O2", "-o", str(tmp_path / "b"), str(prog), "-lm"], check=True, capture_output=True)
    feed = "\n".join(f"{int(a)} {int(b)} {int(c)}" for a, b, c in zip(lhs, rhs, out, strict=True))
    got = subprocess.run([str(tmp_path / "b")], input=feed, capture_output=True, text=True, check=True).stdout
    worst, over = (int(x) for x in got.split())
    assert worst == 1, "the injected one-step error must be seen"
    assert over == (1 if step["bound_lsb"] < 1 else 0), "and counted only when it exceeds the bound"


# ------------------------------------------------------------------------------ the local check


def _c_array(name: str, ctype: str, values) -> str:
    flat = [int(v) for v in np.asarray(values).reshape(-1)]
    return f"static {ctype} {name}[{len(flat)}] = {{{','.join(str(v) for v in flat)}}};"


def _local_run(tmp_path: Path, model: dict, outputs: dict) -> dict[str, dict[str, str]]:
    """Compile the program's LOCAL checks on the host against ``outputs`` as the device buffers."""
    import shutil
    import subprocess

    if shutil.which("cc") is None:
        pytest.skip("no host C compiler")
    script = _script()
    sizes = {b["name"]: b["elements"] for b in model["buffers"]}
    helpers, calls = script._local_checks(model, sizes)
    ctypes = {b["name"]: ("int32_t" if b["ctype"] == "acc_t" else "int8_t") for b in model["buffers"]}
    arrays = [
        _c_array(name, "const int32_t" if name.startswith("BIAS_") else "const int8_t", value)
        for name, value in model["arrays"].items()
        if name != "GOLDEN"
    ]
    buffers = [_c_array(name, ctypes[name], outputs[name]) for name in sizes]
    source = "\n".join(
        [
            "#include <stdio.h>\n#include <stdint.h>\n#include <stddef.h>",
            "typedef int8_t elem_t;\ntypedef int32_t acc_t;",
            "static const elem_t elem_t_max = 127;\nstatic const elem_t elem_t_min = -128;",
            *arrays,
            "#define IMAGE ((const elem_t *)IMAGE_DATA)",
            *buffers,
            helpers,
            "int main(void) {",
            calls,
            "  return 0;\n}",
        ]
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "local.c").write_text(source, encoding="utf-8")
    subprocess.run(
        ["cc", "-O2", "-o", str(tmp_path / "local"), str(tmp_path / "local.c")], check=True, capture_output=True
    )
    printed = subprocess.run([str(tmp_path / "local")], capture_output=True, text=True, check=True).stdout
    lines = {}
    for line in printed.splitlines():
        parts = line.split()
        assert parts[0] == "GM_LOCAL"
        lines[parts[1]] = dict(p.split("=", 1) for p in parts[2:])
    return lines


def _pooled_model() -> dict:
    model = _model()
    conv = model["steps"][0]
    conv["pool"] = {"size": 3, "stride": 2, "padding": 1}
    model["steps"][1].update({"rows": 9})
    model["steps"][2].update({"window": 9})
    for buffer in model["buffers"][:2]:
        buffer["elements"] = 36
    return model


@pytest.mark.parametrize("build", [_model, _pooled_model], ids=["conv", "conv_maxpool"])
def test_the_local_check_passes_a_correct_run_and_names_the_wrong_element(build, tmp_path) -> None:
    """COMPILED AND RUN. Every exact group (conv, mean, matmul) is recomputed from the buffers the
    device holds; a correct run prints zero mismatches, and one wrong element is found at its index."""
    model = build()
    values = {k: v for k, v in _script().emulate(model)["values"].items() if k.startswith("B_")}
    got = _local_run(tmp_path / "ok", model, values)
    assert set(got) == {"1", "3", "4"}, "one local line per exact group; the sum has its bounded check"
    assert all(v["mismatches"] == "0" and v["first"] == "-1" for v in got.values()), got
    broken = dict(values)
    broken["B_g1"] = values["B_g1"].copy()
    broken["B_g1"][5] += 1
    got = _local_run(tmp_path / "bad", model, broken)
    assert (got["1"]["mismatches"], got["1"]["first"]) == ("1", "5")


def test_an_upstream_bounded_difference_leaves_the_groups_below_it_locally_right(tmp_path) -> None:
    """The sum lands one step off the chained value on every element of one channel -- inside its
    declared bound, as a unit that rounds each operand legitimately does. The mean and the matmul
    below it are then computed FROM THAT VALUE, so they differ from the chained emulation, and they
    are right on the inputs they were given, which is all the local check asks."""
    model = _model()
    chained = _script().emulate(model)["values"]
    values = {k: v.copy() for k, v in chained.items() if k.startswith("B_")}
    rows = values["B_g2"].reshape(36, 4)
    rows[:, 0] = np.clip(rows[:, 0] + 1, -128, 127)
    assert np.abs(values["B_g2"] - chained["B_g2"]).max() <= model["steps"][1]["bound_lsb"]
    sums = rows.sum(axis=0).astype(np.float32) * np.float32(model["steps"][2]["multiplier"])
    values["B_g3"] = np.clip(np.rint(sums), -128, 127).astype(np.int64)
    values["B_g4"] = values["B_g3"] @ model["arrays"]["W_g4"].astype(np.int64)
    assert (values["B_g3"] != chained["B_g3"]).any() and (values["B_g4"] != chained["B_g4"]).any()
    got = _local_run(tmp_path, model, values)
    assert all(v["mismatches"] == "0" for v in got.values()), got


def _mapped_console(tmp_path: Path, model: dict, outputs: dict) -> str:
    """Compile the ``local_map`` checks on the host against ``outputs``; return what they print."""
    import shutil
    import subprocess

    if shutil.which("cc") is None:
        pytest.skip("no host C compiler")
    script = _script()
    sizes = {b["name"]: b["elements"] for b in model["buffers"]}
    helpers, calls = script._local_checks(model, sizes, mapped=True)
    ctypes = {b["name"]: ("int32_t" if b["ctype"] == "acc_t" else "int8_t") for b in model["buffers"]}
    arrays = [
        _c_array(name, "const int32_t" if name.startswith("BIAS_") else "const int8_t", value)
        for name, value in model["arrays"].items()
        if name != "GOLDEN"
    ]
    buffers = [_c_array(name, ctypes[name], outputs[name]) for name in sizes]
    source = "\n".join(
        [
            "#include <stdio.h>\n#include <stdint.h>\n#include <stddef.h>",
            "typedef int8_t elem_t;\ntypedef int32_t acc_t;",
            "static const elem_t elem_t_max = 127;\nstatic const elem_t elem_t_min = -128;",
            *arrays,
            "#define IMAGE ((const elem_t *)IMAGE_DATA)",
            *buffers,
            helpers,
            "int main(void) {",
            calls,
            "  return 0;\n}",
        ]
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "local.c").write_text(source, encoding="utf-8")
    subprocess.run(
        ["cc", "-O2", "-o", str(tmp_path / "local"), str(tmp_path / "local.c")], check=True, capture_output=True
    )
    return subprocess.run([str(tmp_path / "local")], capture_output=True, text=True, check=True).stdout


def test_a_mapped_local_check_says_where_a_group_is_wrong(tmp_path) -> None:
    """COMPILED AND RUN. Two wrong elements of the pooled convolution's 3x3x4 output are reported by
    row, column and channel, with their values and sign; a correct group maps to nothing."""
    # The console parser is the measurement service's (phase-2 port); run once it is present.
    FAST = pytest.importorskip("merlin.perf.whole_model_fast")

    model = _pooled_model()
    values = {k: v for k, v in _script().emulate(model)["values"].items() if k.startswith("B_")}
    clean = FAST.parse_group_check(_mapped_console(tmp_path / "ok", model, values), 1, target="gemmini")
    assert clean["status"] == "correct" and clean["maps"]["channel"] == {"extent": 4, "wrong": {}}
    broken = dict(values)
    broken["B_g1"] = values["B_g1"].copy()
    width, channels = 3, 4
    for row, col, channel, delta in ((0, 2, 1, -3), (2, 2, 3, 2)):
        index = (row * width + col) * channels + channel
        assert -128 <= int(values["B_g1"][index]) + delta <= 127, "the fixture's value leaves room for the edit"
        broken["B_g1"][index] += delta
    console = _mapped_console(tmp_path / "bad", model, broken)
    found = FAST.parse_group_check(console, 1, target="gemmini")
    assert found["status"] == "incorrect" and found["local"]["mismatches"] == 2
    assert found["maps"]["row"]["wrong"] == {0: 1, 2: 1} and found["maps"]["col"]["wrong"] == {2: 2}
    assert found["maps"]["channel"] == {"extent": 4, "wrong": {1: 1, 3: 1}}
    assert [(s["row"], s["col"], s["channel"], s["got"] - s["want"]) for s in found["samples"]] == [
        (0, 2, 1, -3),
        (2, 2, 3, 2),
    ]
    assert found["sign"] == {"over": 1, "under": 1, "max_abs": 3}
    text = FAST.render_group_check({"group": 1, **found})
    assert "INCORRECT: 2 of 36" in text and "channel (extent 4): 2 of 4" in text
    assert "row 2, col 2, channel 3" in text
    # The other exact groups of the same program are graded, and a group's lines are its own.
    assert FAST.parse_group_check(console, 3, target="gemmini")["status"] == "correct"


def test_a_local_build_is_unchanged_by_the_mapped_mode() -> None:
    script = _script()
    local = script.render(_model(), verify="local")
    mapped = script.render(_model(), verify="local_map")
    assert "GM_LOCAL_MAP" not in local and "lc_map_note" not in local
    assert mapped.count('"GM_LOCAL_SIGN') == 3
    assert script.program_without_verification(local) == script.program_without_verification(mapped)


def test_the_local_checks_run_after_the_window_and_only_in_local_mode() -> None:
    script, model = _script(), _model()
    assert "GM_LOCAL" not in script.render(model)
    source = script.render(model, verify="local")
    body = source.partition("int main(void)")[2]
    after = body[body.index("fm_end = read_cycles();") :]
    assert after.count(chr(34) + "GM_LOCAL") == 3 and body.count("GM_LOCAL") == 3
    with pytest.raises(ValueError, match="verify is one of"):
        script.render(model, verify="chained")


def test_a_words_build_and_a_local_build_differ_only_inside_the_verification_fence() -> None:
    script = _script()
    words = script.render(_model(), verify="words")
    local = script.render(_model(), verify="local")
    assert words != local
    assert script.program_without_verification(words) == script.program_without_verification(local)
    # Both print one GM_WORDS line per group; the timing build pays for no byte-wise digest pass.
    steps = len(_model()["steps"])
    assert words.count('printf("GM_WORDS') == steps and local.count('printf("GM_WORDS') == steps
    assert "checksum_elem(" not in words.partition("int main")[2]


def test_a_change_outside_the_fence_is_visible_to_the_comparison() -> None:
    script = _script()
    words = script.render(_model(), verify="words")
    tampered = words.replace("uint64_t fm_start = read_cycles();", "uint64_t fm_start = read_cycles() + 1;")
    assert tampered != words
    assert script.program_without_verification(tampered) != script.program_without_verification(words)


# ------------------------------------------------------------------ full-width accumulator readout


def _header(tmp_path: Path, name: str, *, full_width: bool) -> Path:
    path = tmp_path / name
    body = "#define ACC_READ_SMALL_WIDTH\n" + ("#define ACC_READ_FULL_WIDTH\n" if full_width else "")
    path.write_text("#ifndef P\n#define P\n" + body + "#endif\n", encoding="utf-8")
    return path


def test_the_readout_port_is_read_off_the_machines_own_header(tmp_path) -> None:
    """Derived from the header the program compiles against, and UNKNOWN when there is none. The
    repo's two generated gemmini-family headers are one of each."""
    script = _script()
    assert script.full_width_readout(_header(tmp_path, "a.h", full_width=True)) is True
    assert script.full_width_readout(_header(tmp_path, "b.h", full_width=False)) is False
    assert script.full_width_readout(tmp_path / "absent.h") is None
    assert script.full_width_readout(None) is None
    curated = (
        merlin_dir()
        / "experiments/capsule_bench/targets/gemmini/contracts/harness_curated/gemmini-rocc-tests/include/gemmini_params.h"
    )
    narrow = merlin_dir() / "targets/gemmini_universal/contracts/abi/gemmini_params.h"
    assert script.full_width_readout(curated) is True
    assert script.full_width_readout(narrow) is False


def test_a_raw_accumulator_group_is_routed_to_the_core_only_where_the_port_is_missing() -> None:
    script, model = _script(), _model()
    assert [s["group"] for s in script.accumulator_readout_groups(model)] == [4]
    assert script.readout_routing(model, True) == []
    (routed,) = script.readout_routing(model, False)
    assert (routed["group"], routed["cause"]) == (4, script.ACCUMULATOR_READOUT_UNAVAILABLE)
    with pytest.raises(SystemExit, match="UNKNOWN"):
        script.readout_routing(model, None)
    # A model with no raw-accumulator group needs no fact at all.
    closed = _model()
    closed["steps"] = closed["steps"][:3]
    assert script.readout_routing(closed, None) == []


def test_a_loop_free_library_host_path_sends_the_raw_accumulator_group_to_the_core() -> None:
    """The library's host matmul writes narrow elements only, so a full-width group cannot go there even
    on a machine that HAS the read port; with the library's own loops allowed nothing changes."""
    script, model = _script(), _model()
    host_library = script.library_path_without_loops("#define DIM 16\n")
    (routed,) = script.core_readout_routing(model, True, host_library)
    assert (routed["group"], routed["cause"]) == (4, script.LIBRARY_HOST_NARROW_OUTPUT)
    assert script.core_readout_routing(model, True, None) == []
    output_stationary = script.library_path_without_loops("#define GEMMINI_OS_DATAFLOW 1\n")
    assert script.core_readout_routing(model, True, output_stationary) == []
    (narrow,) = script.core_readout_routing(model, False, host_library)
    assert narrow["cause"] == script.ACCUMULATOR_READOUT_UNAVAILABLE, "the missing port is the first cause"


def test_a_package_may_answer_the_classifier_where_the_machine_has_the_full_width_readout() -> None:
    """The loop-free library's narrow host matmul is a fact about the LIBRARY: it routes the groups the
    library answers, never one whose kernel is the package's. On a machine whose readout facts establish
    the full-width port, the package's own classifier kernel stays on the device."""
    script, model = _script(), _model()
    host_library = script.library_path_without_loops("#define DIM 16\n")
    assert script.core_readout_routing(model, True, host_library, answered={4}) == []
    # The library's own answer to the same group still goes to the core, exactly as before.
    (routed,) = script.core_readout_routing(model, True, host_library, answered=())
    assert routed["cause"] == script.LIBRARY_HOST_NARROW_OUTPUT
    package = {4: "merlin_kernel_g4((void *)B_g3, (void *)W_g4, (void *)B_g4);"}
    rendered = script.render(
        model,
        sched_kernels={"calls": package},
        host_readout=[r["group"] for r in script.core_readout_routing(model, True, host_library, answered={4})],
        library=host_library,
    )
    assert package[4] in rendered and "hr_acc_matmul(B_g3" not in rendered


def test_without_the_full_width_readout_the_classifier_stays_on_the_core_with_the_reason() -> None:
    """A machine whose readout facts do NOT establish the port (a lean board) keeps routing the group to
    the core whoever answers it -- a machine fact, unchanged -- and says why."""
    script, model = _script(), _model()
    host_library = script.library_path_without_loops("#define DIM 16\n")
    for library in (host_library, None):
        (routed,) = script.core_readout_routing(model, False, library, answered={4})
        assert (routed["group"], routed["cause"]) == (4, script.ACCUMULATOR_READOUT_UNAVAILABLE)
        assert "ACC_READ_FULL_WIDTH" in routed["why"]
    package = {4: "merlin_kernel_g4((void *)B_g3, (void *)W_g4, (void *)B_g4);"}
    rendered = script.render(model, sched_kernels={"calls": package}, host_readout=[4], library=host_library)
    assert package[4] not in rendered and "hr_acc_matmul(B_g3" in rendered
    with pytest.raises(SystemExit, match="UNKNOWN"):
        script.core_readout_routing(model, None, host_library, answered={4})


@pytest.mark.parametrize("full_width", [True, False], ids=["port", "no_port"])
def test_the_program_build_routes_by_the_headers_readout_and_the_packages_groups(
    tmp_path, monkeypatch, full_width
) -> None:
    """End to end through the driver's build: the header it compiles against decides the port, and the
    package's own calls decide which groups the library rule may route."""
    import subprocess

    script, model = _script(), _model()
    vendor = tmp_path / "vendor"
    (vendor / "include").mkdir(parents=True)
    (vendor / "include" / "gemmini_params.h").write_text(
        "#define DIM 16\n" + ("#define ACC_READ_FULL_WIDTH\n" if full_width else ""), encoding="utf-8"
    )

    def compiled(argv, **kwargs):
        if "-o" in argv:
            Path(argv[argv.index("-o") + 1]).write_bytes(b"\x7fELF")
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(script.subprocess, "run", compiled)
    package = {4: "merlin_kernel_g4((void *)B_g3, (void *)W_g4, (void *)B_g4);"}
    receipt = script.build(
        model, vendor, tmp_path / "gcc", tmp_path / "out", sched_kernels={"calls": package}, library_loops=False
    )
    source = Path(receipt["program_source"]).read_text(encoding="utf-8")
    if full_width:
        assert receipt["host_routed"] == [] and package[4] in source
    else:
        assert [r["cause"] for r in receipt["host_routed"]] == [script.ACCUMULATOR_READOUT_UNAVAILABLE]
        assert package[4] not in source


def test_a_machine_with_the_port_renders_exactly_what_it_rendered_before() -> None:
    script, model = _script(), _model()
    assert script.render(model, host_readout=[]) == script.render(model)
    routed = script.render(model, host_readout=[4])
    assert "hr_acc_matmul(B_g3, (const elem_t *)W_g4, NULL, B_g4, 1, 4, 5, 0);" in routed
    assert "tiled_matmul_auto(1, 5, 4, B_g3" not in routed
    assert "tiled_matmul_auto(1, 5, 4, B_g3" in script.render(model)


def test_the_on_core_readout_is_the_int32_accumulator(tmp_path) -> None:
    """COMPILED AND RUN against numpy, with a bias and a tail narrower than the column block."""
    import shutil
    import subprocess

    if shutil.which("cc") is None:
        pytest.skip("no host C compiler")
    rng = np.random.default_rng(7)
    m, k, n = 3, 37, 13
    a = rng.integers(-128, 128, size=(m, k))
    w = rng.integers(-128, 128, size=(k, n))
    bias = rng.integers(-100000, 100000, size=n)
    want = a @ w + bias
    source = "\n".join(
        [
            "#include <stdio.h>\n#include <stdint.h>\n#include <stddef.h>",
            "typedef int8_t elem_t;\ntypedef int32_t acc_t;",
            _c_array("A", "const int8_t", a),
            _c_array("Wt", "const int8_t", w),
            _c_array("BIAS", "const int32_t", bias),
            f"static acc_t OUT[{m * n}];",
            _script()._HOST_ACC_MATMUL,
            "int main(void) {",
            f"  hr_acc_matmul(A, Wt, BIAS, OUT, {m}, {k}, {n}, 0);",
            f'  for (int i = 0; i < {m * n}; i++) printf("%d\\n", OUT[i]);',
            "  return 0;\n}",
        ]
    )
    (tmp_path / "h.c").write_text(source, encoding="utf-8")
    subprocess.run(["cc", "-O2", "-o", str(tmp_path / "h"), str(tmp_path / "h.c")], check=True, capture_output=True)
    got = subprocess.run([str(tmp_path / "h")], capture_output=True, text=True, check=True).stdout.split()
    assert [int(x) for x in got] == want.reshape(-1).tolist()


# ------------------------------------------------------------------------------ host-routed conv2d


def _conv_only_model(*, pooled: bool) -> dict:
    """One conv2d group, standalone -- small enough to build and run on spike in seconds."""
    rng = np.random.default_rng(11)
    dim = 8 if pooled else 6
    arrays = {
        "IMAGE_DATA": rng.integers(-128, 128, size=(1, dim, dim, 2), dtype=np.int8),
        "W_g1": rng.integers(-8, 8, size=(3, 3, 2, 4), dtype=np.int8),
        "BIAS_g1": rng.integers(-50, 50, size=4, dtype=np.int32),
        "GOLDEN": np.arange(4, dtype=np.float32),
    }
    pool = {"size": 3, "stride": 2, "padding": 1} if pooled else {"size": 0, "stride": 0, "padding": 0}
    out_dim = dim
    step = {
        "kind": "conv2d",
        "group": 1,
        "in": "IMAGE",
        "out": "B_g1",
        "weight": "W_g1",
        "bias": "BIAS_g1",
        "relu": True,
        "scale": 0.05,
        "in_dim": dim,
        "ci": 2,
        "n": 4,
        "out_dim": out_dim,
        "stride": 1,
        "padding": 1,
        "kernel": 3,
        "pool": pool,
        # No classifier is being modeled here; the runtime argmax/dequantize just needs a value.
        "dequantize": 1.0,
    }
    pooled_dim = (out_dim + 2 * pool["padding"] - pool["size"]) // pool["stride"] + 1 if pooled else out_dim
    buffer = {"name": "B_g1", "elements": pooled_dim * pooled_dim * 4, "ctype": "elem_t"}
    return {"steps": [step], "buffers": [buffer], "arrays": arrays, "classes": 4, "groups": 1, "device_groups": 1}


@pytest.mark.parametrize("pooled", [False, True], ids=["conv", "conv_maxpool"])
def test_a_host_routed_convolution_runs_loop_free_on_spike_and_matches_local(tmp_path: Path, pooled: bool) -> None:
    """BUILT AND RUN ON SPIKE. Under the no-FSM prohibition the library's own "CPU" convolution path
    still reaches ``gemmini_loop_conv_ws`` (measured: a real 38-group whole-model build with this
    routing aborted on spike with "merlin: gemmini_loop_ws issues a prohibited instruction", tohost=1,
    before ``hr_conv2d`` existed). ``hr_conv2d`` (see ``_HOST_CONV2D``) never calls into gemmini.h at
    all, so a group routed to it must both (a) leave no prohibited instruction in the linked program and
    (b) actually complete a real run and match the on-core local reference exactly -- a clean static
    scan alone would not catch a hang or a wrong answer that merely avoids the trapped macros."""
    import os

    from merlin.perf import isa_prohibition as ISA
    from merlin.perf.whole_model_build import _prohibited, _with_header

    # The spike machine is the measurement service's (phase-2 port); run once it is present.
    SpikeMachine = pytest.importorskip("merlin.perf.whole_model_machine").SpikeMachine
    from merlin.runtime.backends import base as backends
    from merlin.runtime.backends.base import get_backend

    recipe = backends.harness_build_recipe("gemmini")
    if not Path(recipe.compiler).is_file():
        pytest.skip("configured gemmini cross-compiler unavailable")
    gemmini_module = __import__("importlib").import_module(get_backend("gemmini").__package__ + ".gemmini")
    spike = gemmini_module.spike_path()
    if not spike.is_file():
        pytest.skip("spike unavailable")
    default_header = (
        merlin_dir()
        / "experiments/capsule_bench/targets/gemmini/contracts/harness_curated/gemmini-rocc-tests/include/gemmini_params.h"
    )
    if not default_header.is_file():
        pytest.skip("no default gemmini_params.h in the harness-curated tree")

    private_recipe = _with_header(recipe, default_header, tmp_path / "harness")
    model = _conv_only_model(pooled=pooled)
    receipt = _script().build(
        model,
        None,
        None,
        tmp_path / "build",
        recipe=private_recipe,
        verify="local",
        library_loops=False,
        prohibited_selectors=sorted(_prohibited("gemmini", ["loop_descriptor"])),
    )
    assert receipt["library_paths"]["conv2d"]["host"] is True, "the fixture must exercise the host path"

    report = ISA.check_program(
        Path(receipt["elf"]),
        target="gemmini",
        roles=["loop_descriptor"],
        compiler=receipt["compiler"],
        group_objects={},
        library_groups=["1"],
    )
    assert report["clean"], f"no-FSM scan must stay clean: {report['summary']}"

    flags, library_dir = gemmini_module.spike_extension()
    env = {"LD_LIBRARY_PATH": f"{library_dir}:{os.environ.get('LD_LIBRARY_PATH', '')}"}
    machine = SpikeMachine("gemmini", command=[str(spike), *flags], environment=env)
    run = machine.run(Path(receipt["elf"]), tmp_path / "run", timeout_s=120)
    assert run["completed"], f"spike run did not complete: {run.get('incomplete_reason')}"

    uart = Path(run["uart_log"]).read_text(encoding="utf-8")
    (local_line,) = (line for line in uart.splitlines() if line.startswith("GM_LOCAL 1 "))
    fields = dict(part.split("=", 1) for part in local_line.split()[2:])
    assert fields["mismatches"] == "0" and fields["first"] == "-1", uart


# --------------------------------------------------------------------------------- FUSED REGIONS


def _checks_run(tmp_path: Path, model: dict, outputs: dict, *, bounded: bool = False) -> dict[str, dict[str, str]]:
    """Compile the program's on-core checks against ``outputs`` as the device buffers: every GM_LOCAL
    line, and with ``bounded`` every fused region's sum check (``_region_bounded``) too."""
    import shutil
    import subprocess

    if shutil.which("cc") is None:
        pytest.skip("no host C compiler")
    script = _script()
    sizes = {b["name"]: b["elements"] for b in model["buffers"]}
    helpers, calls = script._local_checks(model, sizes)
    if bounded:
        calls += "\n" + "\n".join(script._region_bounded(s, sizes) for s in model["steps"] if s["kind"] == "region")
    ctypes = {b["name"]: ("int32_t" if b["ctype"] == "acc_t" else "int8_t") for b in model["buffers"]}
    arrays = [
        _c_array(name, "const int32_t" if name.startswith("BIAS_") else "const int8_t", value)
        for name, value in model["arrays"].items()
        if name != "GOLDEN"
    ]
    source = "\n".join(
        [
            "#include <stdio.h>\n#include <stdint.h>\n#include <stddef.h>",
            "typedef int8_t elem_t;\ntypedef int32_t acc_t;",
            "static const elem_t elem_t_max = 127;\nstatic const elem_t elem_t_min = -128;",
            *arrays,
            "#define IMAGE ((const elem_t *)IMAGE_DATA)",
            *[_c_array(name, ctypes[name], outputs[name]) for name in sizes],
            helpers,
            "int main(void) {",
            calls,
            "  return 0;\n}",
        ]
    )
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "checks.c").write_text(source, encoding="utf-8")
    subprocess.run(
        ["cc", "-O2", "-o", str(tmp_path / "checks"), str(tmp_path / "checks.c")], check=True, capture_output=True
    )
    printed = subprocess.run([str(tmp_path / "checks")], capture_output=True, text=True, check=True).stdout
    lines: dict[str, dict[str, str]] = {}
    for line in printed.splitlines():
        parts = line.split()
        if parts[0] in ("GM_LOCAL", "GM_BOUND"):
            lines[f"{parts[0]} {parts[1]}"] = dict(p.split("=", 1) for p in parts[2:])
    return lines


def _device(model: dict) -> dict:
    """What a correct device holds after the program: every group's output (the emulation)."""
    return {k: v.copy() for k, v in _script().emulate(model)["values"].items() if k.startswith("B_")}


def test_a_fused_region_is_one_step_graded_at_its_boundary() -> None:
    """The window mean and the classifier it feeds (a gap -> fc region) become ONE step under the
    classifier's group: its readout and dequantize are the boundary's, its fallback is the members'
    library calls in order, and it prints ONE group line -- the boundary's."""
    script, model = _script(), _model()
    merged, refused = script.region_steps(model, [{"members": [3, 4], "boundary": 4}])
    assert refused == {}
    assert [(s["group"], s["kind"]) for s in merged["steps"]] == [(1, "conv2d"), (2, "sum"), (4, "region")]
    region = merged["steps"][-1]
    assert (region["out"], region["dequantize"], region["boundary_kind"]) == ("B_g4", 0.5, "matmul")
    assert region["reads"] == ["B_g2"], "a region reads only what its members read from outside it"
    assert [m["group"] for m in region["members"]] == [3, 4]
    source = script.render(merged, verify="local")
    assert source.count('"GM_GROUP 4 region') == 1 and '"GM_GROUP 3 ' not in source
    # No kernel answers it here, so the region runs its members' own library calls, in order.
    call_line = next(line for line in source.splitlines() if "fm_dt[2] = dt" in line)
    assert call_line.index("B_g2") < call_line.index("tiled_matmul_auto(1, 5, 4, B_g3")
    # With the region's kernel linked, it is ONE call in its members' place.
    linked = script.render(merged, sched_kernels={"calls": {4: "merlin_kernel_g4((void *)B_g2, (void *)B_g4);"}})
    assert "merlin_kernel_g4((void *)B_g2, (void *)B_g4);" in linked and "tiled_matmul_auto(1, 5, 4" not in linked
    # The integer program is the same model: the emulation and the digests do not move.
    assert (
        script.group_checksums(merged, script.emulate(merged))["4"]
        == script.group_checksums(model, script.emulate(model))["4"]
    )


def test_a_region_the_driver_cannot_restate_is_refused_by_group() -> None:
    """Every fact is checked against the driver's OWN steps: a sum is never an internal member (its
    check is a bound, not a value to restate), members must be consecutive, and a refused region's
    members stay ordinary steps."""
    script, model = _script(), _model()
    merged, refused = script.region_steps(model, [{"members": [2, 3], "boundary": 3}])
    assert merged is model and "cannot restate" in refused[3]
    _merged, refused = script.region_steps(model, [{"members": [1, 3], "boundary": 3}])
    assert "consecutive" in refused[3]
    accumulator = _model()
    accumulator["steps"][0]["scale"] = None  # the convolution now leaves as the accumulator
    _merged, refused = script.region_steps(accumulator, [{"members": [1, 2], "boundary": 2}])
    assert "accumulator" in refused[2]


def test_an_answered_fused_region_passes_where_its_internal_value_never_existed(tmp_path) -> None:
    """COMPILED AND RUN. The region's kernel never writes the mean's buffer (garbage there), and the
    region is still graded right: the mean is restated on the core from the inputs the device held,
    and the classifier is checked from it. ONE local line, under the boundary's group."""
    script, model = _script(), _model()
    merged, _ = script.region_steps(model, [{"members": [3, 4], "boundary": 4}])
    device = _device(model)
    device["B_g3"] = np.full_like(device["B_g3"], 99)  # never formed by the region's kernel
    got = _checks_run(tmp_path / "ok", merged, device)
    assert set(got) == {"GM_LOCAL 1", "GM_LOCAL 4"}, "no line for the internal member; one for the region"
    assert got["GM_LOCAL 4"]["mismatches"] == "0"


def test_a_fused_region_answered_wrongly_is_caught_at_its_boundary(tmp_path) -> None:
    """COMPILED AND RUN. A region kernel that gets its boundary wrong by one element is caught by the
    region's own check, wherever the error came from inside it."""
    script, model = _script(), _model()
    merged, _ = script.region_steps(model, [{"members": [3, 4], "boundary": 4}])
    device = _device(model)
    device["B_g3"] = np.zeros_like(device["B_g3"])
    device["B_g4"][2] += 1
    got = _checks_run(tmp_path / "bad", merged, device)
    assert (got["GM_LOCAL 4"]["mismatches"], got["GM_LOCAL 4"]["first"]) == ("1", "2")


def test_a_fused_region_ending_in_a_sum_is_held_to_the_sums_bound(tmp_path) -> None:
    """COMPILED AND RUN. A convolution and the residual sum it feeds, as one region: the convolution
    is restated on the core, then the sum is checked within its declared bound -- passing for the
    device's own rounding, caught for a wrong element."""
    script, model = _script(), _model()
    merged, refused = script.region_steps(model, [{"members": [1, 2], "boundary": 2}])
    assert refused == {} and merged["steps"][0]["kind"] == "region"
    device = _device(model)
    device["B_g1"] = np.full_like(device["B_g1"], -7)  # the region's kernel kept the conv to itself
    got = _checks_run(tmp_path / "ok", merged, device, bounded=True)
    assert got["GM_BOUND 2"]["over"] == "0"
    device["B_g2"][9] = np.clip(device["B_g2"][9] + 5, -128, 127)
    got = _checks_run(tmp_path / "bad", merged, device, bounded=True)
    assert int(got["GM_BOUND 2"]["over"]) >= 1
