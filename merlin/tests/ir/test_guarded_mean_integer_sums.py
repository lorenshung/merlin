"""Independent source arithmetic checks for an external exact sum producer."""

import ctypes
import hashlib
import shutil
import subprocess

import numpy as np
import pytest

from merlin.llvmlower.guarded_quantized_mean import derive, emit_integer_sum_finish


def original(source, proof):
    total = np.zeros(source.shape[:2], dtype=np.float32)
    for index in range(proof["count"]):
        product = source[:, :, index].astype(np.float32) * np.float32(proof["input_scale"])
        total = product + total
    mean = total / np.float32(proof["count"])
    scaled = mean * np.float32(proof["reciprocal"])
    return np.clip(np.rint(scaled), -128, 127).astype(np.int8)


def execute(tmp_path, source, proof, layout, *, misalignment=0):
    batches, channels, _ = source.shape
    code = emit_integer_sum_finish(proof, "finish", batches, channels, input_layout=layout)
    # Passing pytest temp dirs can be removed/reused in the same process; dlopen
    # retains loaded images. Distinct generated code needs a distinct path.
    digest = hashlib.sha256(code.encode()).hexdigest()[:16]
    c, library = tmp_path / f"finish_{digest}.c", tmp_path / f"finish_{digest}.so"
    c.write_text(code)
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("native C compiler unavailable")
    subprocess.run(
        [compiler, "-O2", "-fno-fast-math", "-ffp-contract=off", "-shared", "-fPIC", str(c), "-o", str(library)],
        check=True,
        capture_output=True,
    )
    logical = source if layout == "channel_major" else source.transpose(0, 2, 1)
    flattened = np.ascontiguousarray(logical).reshape(-1)
    storage = np.full(flattened.size + misalignment + 16, 91, dtype=np.int8)
    storage[misalignment : misalignment + flattened.size] = flattened
    before = storage.copy()
    sums = source.astype(np.int32).sum(axis=2, dtype=np.int32)
    sums_before = sums.copy()
    output = np.full(batches * channels + 16, 73, dtype=np.int8)
    function = ctypes.CDLL(str(library)).finish
    function.argtypes = [ctypes.c_void_p] * 3
    function.restype = None
    function(storage.ctypes.data + misalignment, sums.ctypes.data, output.ctypes.data)
    assert np.array_equal(storage, before)
    assert np.array_equal(sums, sums_before)
    assert np.all(output[batches * channels :] == 73)
    assert np.array_equal(output[: batches * channels].reshape(batches, channels), original(source, proof))


@pytest.mark.parametrize("layout", ["channel_major", "channel_minor"])
def test_complete_two_input_domain_matches_original_ordered_source(tmp_path, layout):
    values = np.arange(-128, 128, dtype=np.int8)
    first, second = np.meshgrid(values, values, indexing="ij")
    source = np.stack((first.reshape(-1), second.reshape(-1)), axis=1)[None, :, :]
    proof = derive(2, 0.7, 0.2)
    assert proof["ambiguous_sums"]
    execute(tmp_path, source, proof, layout, misalignment=1)


@pytest.mark.parametrize("count", [1, 7, 49, 128])
@pytest.mark.parametrize("layout", ["channel_major", "channel_minor"])
def test_batches_channel_tails_extremes_and_immutable_inputs(tmp_path, count, layout):
    source = np.random.default_rng(913 + count).integers(-128, 128, (3, 17, count), dtype=np.int8)
    source[0, 0, :] = -128
    source[0, 1, :] = 127
    source[0, 2, :] = 0
    execute(tmp_path, source, derive(count, 1.7433754205703735, 1.3525974750518799), layout)


def test_refuse_changed_certificate_unknown_layout_or_extents():
    proof = derive(7, 0.4, 0.9)
    with pytest.raises(ValueError, match="certificate"):
        emit_integer_sum_finish(proof | {"sum_max": 0}, "finish", 1, 3, input_layout="channel_minor")
    with pytest.raises(ValueError, match="explicit"):
        emit_integer_sum_finish(proof, "finish", 1, 3, input_layout="guessed")
    with pytest.raises(ValueError, match="positive"):
        emit_integer_sum_finish(proof, "finish", True, 3, input_layout="channel_minor")
