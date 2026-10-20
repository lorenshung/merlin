"""Uniform quantized zero needs a complete integral image and an exact divisor."""

from pathlib import Path

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.targetgen.source_integer_remainder import specialize

SPEC = Path(__file__).resolve().parents[3] / "examples/gemmini/target/software-spec.yaml"


def fixture(tmp_path, divisor, *, integer="i64", floating="f32", foreign=False):
    tmp_path.joinpath("model.mlir").write_text(f"""module {{func.func @forward(%x:tensor<4x{integer}>)
      -> (tensor<4x{floating}>,tensor<4x{integer}>) attributes {{"prov.aten"="aten.fmod.Scalar"}} {{
      {"func.call @foreign() : () -> ()" if foreign else ""}
      %e=tensor.empty() : tensor<4x{floating}>
      return %e,%x : tensor<4x{floating}>,tensor<4x{integer}>
    }} }}""")
    return dict(
        cases=[
            dict(
                status="bundled",
                case=dict(
                    overload="aten.fmod.Scalar",
                    arguments=dict(args=dict(items=[dict(kind="tensor", values="NEVER READ"), divisor])),
                ),
                output_indices=[0],
                semantic_boundary=dict(post_indices=[1]),
            )
        ]
    )


@pytest.mark.parametrize("integer", ["i1", "i8", "i32", "i64"])
@pytest.mark.parametrize("divisor", [1.0, -1.0, 0.5, -0.25])
def test_whole_integer_image_has_a_canonical_zero_after_quantization(tmp_path, integer, divisor):
    report = fixture(tmp_path, divisor, integer=integer)
    proof, original = specialize(tmp_path, report, SPEC)
    assert proof["input_signed_storage_bits"] == int(integer[1:])
    assert proof["quantized_contract"]["schema"] == "merlin.quantized_native.v1"
    module = parse_mlir_file(tmp_path / "model.mlir")
    module.verify()
    assert next(op for op in module.walk() if op.name == "arith.constant").value.value.data == 0
    assert original != tmp_path.joinpath("model.mlir").read_bytes()


@pytest.mark.parametrize("divisor", [0.0, 0.3, 0.3333333333333333, 2.0, 2.0**-200, float("inf")])
def test_zero_nonintegral_reciprocal_and_ieee_special_divisors_refuse(tmp_path, divisor):
    report = fixture(tmp_path, divisor)
    before = tmp_path.joinpath("model.mlir").read_bytes()
    assert specialize(tmp_path, report, SPEC) is None
    assert tmp_path.joinpath("model.mlir").read_bytes() == before


@pytest.mark.parametrize("options", [dict(integer="f32"), dict(floating="f16"), dict(foreign=True)])
def test_unproved_domain_or_effect_refuses(tmp_path, options):
    report = fixture(tmp_path, 0.5, **options)
    assert specialize(tmp_path, report, SPEC) is None


def test_oracle_qualification_failure_does_not_turn_into_a_payload_dependent_host_choice(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from merlin.targetgen.quantized_literal_reference import qualify

    monkeypatch.setattr(
        "merlin.targetgen.core_aten_numeric._source_reference",
        lambda *a: SimpleNamespace(returncode=1, stderr="oracle unavailable"),
    )
    with pytest.raises(ArithmeticError, match="oracle unavailable"):
        qualify(tmp_path, dict(cases=[], output_count=1), dict(exact_dtype_sites=1), {})
