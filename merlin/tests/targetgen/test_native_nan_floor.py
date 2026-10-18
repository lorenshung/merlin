"""Independent native NaNs may refuse a candidate; they cannot supply its readout."""

import pytest

from merlin.targetgen.core_aten_host_floor import native_boundary_has_nan


@pytest.mark.parametrize(
    "dtype,width,words,nan",
    [
        ("f16", 2, [0xFC00, 0x7C00, 0x8000], False),
        ("f16", 2, [0x7E00], True),
        ("bf16", 2, [0x7F80, 0xFF80], False),
        ("bf16", 2, [0xFFFF], True),
        ("f32", 4, [0xFF800000], False),
        ("f32", 4, [0x7F800001], True),
        ("f64", 8, [0x7FF8000000000000], True),
        ("complex<f32>", 4, [0, 0xFFC00000], True),
        ("complex<f64>", 8, [0x7FF0000000000000, 0], False),
        ("i32", 4, [0xFFFFFFFF], False),
    ],
)
def test_native_nan_qualification_uses_declared_word_formats(dtype, width, words, nan):
    record = dict(
        output_abi=[dict(dtype=dtype)],
        native_eager_boundary=dict(
            authority="untransformed_native_cpu_eager",
            output_bytes=[b"".join(v.to_bytes(width, "little") for v in words).hex()],
        ),
    )
    assert native_boundary_has_nan(dict(cases=[record])) is nan
    record["native_eager_boundary"]["authority"] = "untrusted"
    assert not native_boundary_has_nan(dict(cases=[record]))
