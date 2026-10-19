"""Full-domain admission rejects incomplete guards and runtime products."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.native_cpu_layout import input_layouts, profile
from merlin.llvmlower.significand_products import rewrite_significand_products
from merlin.llvmlower.uniform_numeric_literals import compile_time_operand
from merlin.targetgen.native_typed_semantics import prove


def guarded(*, missing=False, escape=False):
    return parse_mlir_text(
        """module {
      func.func @forward(%x:tensor<4xf32>,%y:tensor<4xf32>)->tensor<4xf32> {
        %e=tensor.empty():tensor<4xf32>
        %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]}
          ins(%x,%y:tensor<4xf32>,tensor<4xf32>) outs(%e:tensor<4xf32>) {
        ^bb0(%a:f32,%b:f32,%unused:f32):
          %q=arith.addf %a,%b:f32
          %n=arith.constant -4194304:i32
          %nan=arith.bitcast %n:i32 to f32
          %mask=arith.constant 4194304:i32
          %an=arith.cmpf uno,%a,%a:f32
          %bn=arith.cmpf uno,%b,%b:f32
          %qn=arith.cmpf uno,%q,%q:f32
          %ab=arith.bitcast %a:f32 to i32
          %bb=arith.bitcast %b:f32 to i32
          %aq=arith.ori %ab,%mask:i32
          %bq=arith.ori %bb,%mask:i32
          %av=arith.bitcast %aq:i32 to f32
          %bv=arith.bitcast %bq:i32 to f32
          %invalid=arith.select %qn,%nan,%q:f32
          %first=arith.select %an,%av,%invalid:f32
          %last=arith.select %bn,%bv,%first:f32
          linalg.yield RESULT:f32
        }->tensor<4xf32>
        return %r:tensor<4xf32>
      }}""".replace("RESULT", "%q" if escape else "%first" if missing else "%last")
    )


@pytest.fixture
def certificate(monkeypatch):
    monkeypatch.setattr(
        "merlin.targetgen.native_ieee_certificate.certificate",
        lambda: {
            "arithmetic_checks": [],
            "content_sha256": "test-structural-proof",
            "conversions": [],
            "loop_profile": {},
        },
    )


def test_complete_guard_composition_is_proven(certificate):
    assert prove(guarded())["status"] == "proven"


@pytest.mark.parametrize("options", [dict(missing=True), dict(escape=True)])
def test_partial_nan_guard_or_unprotected_escape_refuses(certificate, options):
    result = prove(guarded(**options))
    assert result["code"] == "native_competing_nan_priority_unproved"


def test_layout_metadata_does_not_read_payloads():
    class NoPayload(dict):
        def values(self):
            raise AssertionError("tensor payload visited")

    tensor = NoPayload(kind="tensor", shape=[33], stride=[1], values="unread", expected="unread")
    args = [object()]
    report = {"cases": [{"case": {"arguments": [tensor]}, "input_indices": [0]}]}
    assert input_layouts(report, args) == {args[0]: {"shape": [33], "stride": [1]}}


def test_vector_tail_and_unknown_layout_domain():
    assert profile({"shape": [33], "stride": [1]}, block=32, grain=32768)["vector_prefix"] == 32
    assert profile({"shape": [33], "stride": [2]}, block=32, grain=32768)["vector_prefix"] == 0
    assert profile({"shape": [2, 3], "stride": [10, 2]}, block=32, grain=32768) is None
    assert profile({"shape": [32769], "stride": [1]}, block=32, grain=32768) is None


def test_runtime_tensor_product_stays_unlowered():
    module = guarded()
    arithmetic = next(op for op in module.walk() if op.name == "arith.addf")
    assert not any(compile_time_operand(v) for v in arithmetic.operands)
    # A closed, unguarded product is numerically lowerable but ineligible.
    module = parse_mlir_text("""module {func.func @forward(%x:tensor<4xf16>,%y:tensor<4xf16>)->tensor<4xf16>{
      %e=tensor.empty():tensor<4xf16>
      %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
          iterator_types=["parallel"]}
        ins(%x,%y:tensor<4xf16>,tensor<4xf16>) outs(%e:tensor<4xf16>){
      ^bb0(%a:f16,%b:f16,%unused:f16):
        %p=arith.mulf %a,%b:f16
        linalg.yield %p:f16
      }->tensor<4xf16>
      return %r:tensor<4xf16>
    }}""")
    original = str(module)
    assert rewrite_significand_products(module) == ({}, [])
    assert str(module) == original


@pytest.mark.parametrize(
    "operation,left,right,expected",
    [
        ("arith.mulf", "0.5", "2.0", 1.0),
        ("arith.addf", "1.0", "0.00048828125", 1.0),
        ("arith.subf", "-0.0", "0.0", -0.0),
    ],
)
def test_source_literal_folding_preserves_typed_rne_and_zero_sign(operation, left, right, expected):
    import math

    from merlin.llvmlower.uniform_numeric_literals import fold_uniform_numeric_literals
    from merlin.xdsl_dialects.lowering.group_numerics import _constant_of

    module = parse_mlir_text(
        """module {func.func @forward()->tensor<4xf16>{
      %ca=arith.constant LEFT:f16
      %cb=arith.constant RIGHT:f16
      %x=tensor.splat %ca:tensor<4xf16>
      %y=tensor.splat %cb:tensor<4xf16>
      %e=tensor.empty():tensor<4xf16>
      %r=linalg.generic {indexing_maps=[affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>,affine_map<(d0)->(d0)>],
        iterator_types=["parallel"]} ins(%x,%y:tensor<4xf16>,tensor<4xf16>) outs(%e:tensor<4xf16>){
      ^bb0(%a:f16,%b:f16,%unused:f16):
        %p=OP %a,%b:f16
        linalg.yield %p:f16
      }->tensor<4xf16>
      return %r:tensor<4xf16>
    }}""".replace("LEFT", left)
        .replace("RIGHT", right)
        .replace("OP", operation)
    )
    assert fold_uniform_numeric_literals(module) == 1
    returned = next(op for op in module.walk() if op.name == "func.return").operands[0]
    actual = _constant_of(returned)
    assert actual == expected
    assert math.copysign(1.0, actual) == math.copysign(1.0, expected)


def test_corrupted_double_rounding_certificate_refuses(tmp_path, monkeypatch):
    import json
    from pathlib import Path

    import merlin.targetgen.native_ieee_certificate as checker

    document = json.loads(Path(checker.__file__).with_suffix(".json").read_text())
    document["double_rounding"]["f16"]["intermediate_precision"] = 23
    document.pop("content_sha256")
    document["content_sha256"] = checker.digest(document)
    fixture = tmp_path / "certificate.json"
    fixture.write_text(json.dumps(document))
    monkeypatch.setattr(checker, "__file__", str(fixture.with_suffix(".py")))
    monkeypatch.setattr(checker, "proof_sources", lambda: document["sources"])
    with pytest.raises(ValueError, match="double rounding precision"):
        checker.certificate()
