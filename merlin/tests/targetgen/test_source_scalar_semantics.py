"""Bit-exact typed scalar semantics, independent of the target implementation."""

import numpy as np
import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_w8a8_reference import evaluate


def source(arguments, result, body):
    types = [f"tensor<1x{t}>" for t in arguments]
    declaration = ",".join(f"%x{i}:{t}" for i, t in enumerate(types))
    inputs = ",".join(f"%x{i}" for i in range(len(types)))
    block = ",".join(f"%a{i}:{t}" for i, t in enumerate(arguments))
    maps = ",".join(["affine_map<(d0)->(d0)>"] * (len(types) + 1))
    return parse_mlir_text(f"""module {{
      func.func @forward({declaration}) -> tensor<1x{result}> {{
        %e=tensor.empty() : tensor<1x{result}>
        %r=linalg.generic {{indexing_maps=[{maps}],iterator_types=["parallel"]}}
          ins({inputs}:{",".join(types)}) outs(%e:tensor<1x{result}>) {{
          ^bb0({block},%unused:{result}):
          {body}
          linalg.yield %v : {result}
        }} -> tensor<1x{result}>
        return %r : tensor<1x{result}>
      }} }}""")


def run(module, arrays, audit=None):
    pytest.importorskip("torch")
    values, _ = evaluate(module, arrays, [], quant_min=-128, quant_max=127, eps=2**-12, ieee_audit=audit)
    return values[0].reshape(-1)


@pytest.mark.parametrize(
    "token,np_type,word",
    [
        ("f16", np.float16, 0xFE15),
        ("bf16", np.float32, 0xFFA1),
        ("f32", np.float32, 0xFFA12345),
        ("f64", np.float64, 0xFFF0123456789ABC),
    ],
)
def test_bitcast_preserves_signaling_nan_storage(token, np_type, word):
    width = 16 if token in {"f16", "bf16"} else 32 if token == "f32" else 64
    signed = word - (1 << width)
    integer = f"i{width}"
    module = source(
        [integer], integer, f"%fp=arith.bitcast %a0 : {integer} to {token}\n%v=arith.bitcast %fp : {token} to {integer}"
    )
    value = np.array([signed], dtype={16: np.int16, 32: np.int32, 64: np.int64}[width])
    assert run(module, [value]).item() == signed


@pytest.mark.parametrize("width", [16, 32, 64])
def test_width_correct_logical_shifts_and_bits(width):
    integer = f"i{width}"
    module = source(
        [integer],
        integer,
        f"""%one=arith.constant 1 : {integer}
      %hi=arith.shrui %a0,%one : {integer}
      %lo=arith.shli %a0,%one : {integer}
      %bits=arith.ori %hi,%lo : {integer}
      %v=arith.andi %bits,%hi : {integer}""",
    )
    value = np.array([-1], dtype={16: np.int16, 32: np.int32, 64: np.int64}[width])
    assert run(module, [value]).item() == (1 << (width - 1)) - 1


@pytest.mark.parametrize("predicate,expected", [("ult", False), ("ule", False), ("ugt", True), ("uge", True)])
def test_unsigned_i64_predicates(predicate, expected):
    module = source(["i64", "i64"], "i1", f"%v=arith.cmpi {predicate},%a0,%a1 : i64")
    assert run(module, [np.array([-1], np.int64), np.array([0], np.int64)]).item() is expected


@pytest.mark.parametrize(
    "predicate,expected",
    [
        ("false", False),
        ("oeq", False),
        ("ord", False),
        ("ueq", True),
        ("ugt", True),
        ("uge", True),
        ("ult", True),
        ("ule", True),
        ("une", True),
        ("uno", True),
        ("true", True),
    ],
)
def test_unordered_float_predicates(predicate, expected):
    module = source(["f32", "f32"], "i1", f"%v=arith.cmpf {predicate},%a0,%a1 : f32")
    assert run(module, [np.array([np.nan], np.float32), np.array([1], np.float32)]).item() is expected


@pytest.mark.parametrize("token,p", [("f32", 24), ("f64", 53)])
def test_fma_has_one_rounding_and_preserves_negative_zero(token, p):
    dtype = np.float32 if token == "f32" else np.float64
    module = source([token] * 3, token, f"%v=math.fma %a0,%a1,%a2 : {token}")
    a, b, c = 1 + 2.0 ** (1 - p), 1 - 2.0 ** (1 - p), -1.0
    expected = -(2.0 ** (2 - 2 * p))
    assert run(module, [np.array([v], dtype) for v in [a, b, c]]).item() == expected
    actual = run(module, [np.array([v], dtype) for v in [-0.0, 1.0, -0.0]])
    assert actual.view(__import__("torch").int32 if token == "f32" else __import__("torch").int64).item() < 0


@pytest.mark.parametrize("token,p", [("f32", 24), ("f64", 53)])
def test_unsigned_i64_conversion_avoids_double_rounding(token, p):
    dtype = np.float32 if token == "f32" else np.float64
    module = source(["i64"], token, f"%v=arith.uitofp %a0 : i64 to {token}")
    midpoint = (1 << 63) + (1 << (63 - p))
    for value in [midpoint - 1, midpoint, midpoint + 1, (1 << 64) - 1]:
        signed = value - (1 << 64)
        actual = run(module, [np.array([signed], np.int64)]).item()
        expected = (1 << 63) if value <= midpoint else (1 << 63) + (1 << (64 - p))
        if value == (1 << 64) - 1:
            expected = 1 << 64
        assert actual == float(expected)
        assert actual == dtype(expected)


@pytest.mark.parametrize("masked", [False, True])
def test_nan_audit_observes_only_selected_outputs(masked):
    suffix = "%v=arith.select %nan,%a0,%sum : f32" if masked else "%v=arith.addf %a0,%a1 : f32"
    module = source(
        ["f32", "f32"],
        "f32",
        f"""%sum=arith.addf %a0,%a1 : f32
      %nan=arith.cmpf uno,%sum,%sum : f32
      {suffix}""",
    )
    words = [np.array([word], np.uint32).view(np.float32) for word in [0x7FC00001, 0xFFC00002]]
    audit = {}
    result = run(module, words, audit)
    assert audit["ambiguous_nan_payload"] is (not masked)
    if masked:
        assert result.view(__import__("torch").int32).item() == 0x7FC00001


def test_nan_payload_bits_cannot_become_an_unaudited_float_predicate():
    module = source(
        ["f32", "f32"],
        "i1",
        """%sum=arith.addf %a0,%a1 : f32
      %word=arith.bitcast %sum : f32 to i32
      %one=arith.constant 1 : i32
      %payload=arith.andi %word,%one : i32
      %fp=arith.bitcast %payload : i32 to f32
      %zero=arith.constant 0.0 : f32
      %v=arith.cmpf oeq,%fp,%zero : f32""",
    )
    words = [np.array([word], np.uint32).view(np.float32) for word in [0x7FC00001, 0x7FC00002]]
    audit = {}
    run(module, words, audit)
    assert audit["ambiguous_nan_payload"] is True


def test_nan_payload_audit_follows_masks_across_tensor_regions():
    module = parse_mlir_text("""module {
      func.func @forward(%x:tensor<1xf32>,%y:tensor<1xf32>) -> tensor<1xf32> {
        %e=tensor.empty() : tensor<1xf32>
        %sum=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,
          affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%x,%y:tensor<1xf32>,tensor<1xf32>) outs(%e:tensor<1xf32>) {
          ^bb0(%a:f32,%b:f32,%unused:f32):
            %s=arith.addf %a,%b : f32
            linalg.yield %s : f32
        } -> tensor<1xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,
          affine_map<(d0)->(d0)>],iterator_types=["parallel"]}
          ins(%sum,%x:tensor<1xf32>,tensor<1xf32>) outs(%e:tensor<1xf32>) {
          ^bb1(%s:f32,%a:f32,%unused:f32):
            %nan=arith.cmpf uno,%s,%s : f32
            %v=arith.select %nan,%a,%s : f32
            linalg.yield %v : f32
        } -> tensor<1xf32>
        return %r : tensor<1xf32>
      }}""")
    words = [np.array([word], np.uint32).view(np.float32) for word in [0x7FC00001, 0xFFC00002]]
    audit = {}
    result = run(module, words, audit)
    assert audit["ambiguous_nan_payload"] is False
    assert result.view(__import__("torch").int32).item() == 0x7FC00001
