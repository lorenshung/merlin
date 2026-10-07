"""Authority records cannot hide duplicate claims or non-finite values."""

import json

import pytest

from merlin.common.strict_json import loads


@pytest.mark.parametrize("raw", [b'{"claim":false,"claim":true}', b'{"nested":{"x":1,"x":2}}'])
def test_duplicate_keys_refuse(raw):
    with pytest.raises(ValueError, match="duplicate JSON key"):
        loads(raw)


@pytest.mark.parametrize("token", ["NaN", "Infinity", "-Infinity", "1e999", "-1e999"])
def test_nonfinite_numbers_refuse_in_nested_record(token):
    with pytest.raises(ValueError, match="non-finite JSON value"):
        loads('{"ignored":[{"number":' + token + "}]}")


def test_valid_records_preserve_json_types_and_values():
    raw = '{"text":"NaN", "integer":123456789123456789, "negative":-17, "finite":1e-300, "array":[null,true]}'
    assert loads(raw) == json.loads(raw)
    assert loads(raw.encode()) == json.loads(raw)
    assert loads("0", max_bytes=1) == 0


def test_bound_is_encoded_bytes_and_invalid_limits_refuse():
    with pytest.raises(ValueError, match="bounded reader"):
        loads('"é"', max_bytes=3)
    with pytest.raises(ValueError, match="bounded reader"):
        loads(b"[] ", max_bytes=2)
    for limit in [-1, True, 1.5]:
        with pytest.raises(ValueError, match="reader limit"):
            loads(b"0", max_bytes=limit)
    with pytest.raises(TypeError, match="bytes or text"):
        loads({})


def test_invalid_encoding_or_json_refuses():
    for raw in [b'"\xff"', b'{"x":', b"[1,]"]:
        with pytest.raises(ValueError):
            loads(raw)
