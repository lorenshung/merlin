"""Host ISA selection remains independent of accelerator preparation."""

import pytest

from merlin.runtime.backends.spike_model import RVV_CFLAGS, _select_host_vectorize


@pytest.mark.parametrize(
    "march,expected",
    [
        ("rv64gc", False),
        ("rv64gcv", True),
        ("rv64gcv1p0_zvl256b", True),
        ("rv64gc_zvl256b", False),
        ("rv64gc_zve64x", False),
        ("rv64gc_zve32f", True),
        ("rv64gc_zve64d_zvl256b", True),
        ("rv32imafcv", True),
    ],
)
def test_declared_float_vector_capability(march, expected):
    assert _select_host_vectorize(["-march=" + march], None, None) is expected


def test_default_rvv_and_last_march_selection():
    assert _select_host_vectorize(RVV_CFLAGS, None, None)
    assert not _select_host_vectorize(["-march=rv64gcv", "-march=rv64gc"], None, None)
    assert not _select_host_vectorize(["-O2"], None, None)


def test_caller_can_explicitly_select_scalarization_or_scalar_lowering():
    assert _select_host_vectorize(["-march=rv64gc"], None, True)
    assert _select_host_vectorize(["-march=rv64gc"], "custom.mlir", None)
    assert not _select_host_vectorize(RVV_CFLAGS, None, False)
    with pytest.raises(ValueError, match="conflicts"):
        _select_host_vectorize(RVV_CFLAGS, "custom.mlir", False)


@pytest.mark.parametrize("invalid", [0, 1, "false", []])
def test_invalid_override_refuses(invalid):
    with pytest.raises(ValueError, match="bool or None"):
        _select_host_vectorize(RVV_CFLAGS, None, invalid)
