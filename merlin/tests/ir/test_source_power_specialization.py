"""Source power fill/copy dispatch preserves NaNs without consulting payloads."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.targetgen.source_power_specialization import specialize


def fixture(tmp_path, exponent, *, source="f32", output="f32", shape="4", complex_output=False):
    tmp_path.joinpath("model.mlir").write_text(f"""module {{func.func @forward(%x:tensor<{shape}x{source}>)
      -> (tensor<{shape}x{output}>,tensor<{shape}x{source}>) attributes {{"prov.aten"="aten.pow.Tensor_Scalar"}} {{
      %e=tensor.empty() : tensor<{shape}x{output}>
      return %e,%x : tensor<{shape}x{output}>,tensor<{shape}x{source}>
    }} }}""")
    return dict(
        cases=[
            dict(
                status="bundled",
                case=dict(
                    overload="aten.pow.Tensor_Scalar",
                    arguments=dict(args=dict(items=[dict(kind="tensor", values="DO NOT READ"), exponent])),
                ),
                output_indices=[0],
                output_abi=[dict(dtype="complex<f32>" if complex_output else output)],
                semantic_boundary=dict(post_indices=[1]),
            )
        ]
    )


@pytest.mark.parametrize("element", ["f16", "bf16", "f32", "f64", "i8", "i32", "i64", "i1"])
@pytest.mark.parametrize("exponent", [0, 1])
def test_literal_power_selects_a_complete_uniform_fill_or_copy(tmp_path, element, exponent):
    report = fixture(tmp_path, exponent, source=element, output=element)
    proof = specialize(tmp_path, report)
    assert proof and "no tensor payload read" in proof["parameters"]
    module = parse_mlir_file(tmp_path / "model.mlir")
    module.verify()
    returned = next(op for op in module.walk() if op.name == "func.return")
    assert returned.operands[1] is next(op for op in module.walk() if op.name == "func.func").body.block.args[0]
    if exponent == 0:
        assert returned.operands[0].owner.name == "tensor.splat"
    else:
        assert (
            returned.operands[0].owner.body.block.last_op.operands[0] is returned.operands[0].owner.body.block.args[0]
        )


def test_complex_zero_power_fills_exactly_one_plus_positive_zero_i(tmp_path):
    report = fixture(tmp_path, 0, shape="4x2", complex_output=True)
    assert specialize(tmp_path, report)
    module = parse_mlir_file(tmp_path / "model.mlir")
    literal = next(op for op in module.walk() if op.name == "arith.constant")
    assert list(literal.value.get_values()) == [1.0, 0.0] * 4


@pytest.mark.parametrize("exponent", [0.5, 2, 3])
def test_boolean_positive_power_has_only_two_possible_values(tmp_path, exponent):
    report = fixture(tmp_path, exponent, source="i1")
    assert specialize(tmp_path, report)
    module = parse_mlir_file(tmp_path / "model.mlir")
    assert any(op.name == "arith.uitofp" for op in module.walk())


@pytest.mark.parametrize("failure", ["non_identity", "origin", "foreign_call", "runtime_exponent"])
def test_incomplete_semantic_binding_is_not_a_source_specialization(tmp_path, failure):
    report = fixture(tmp_path, 2 if failure == "non_identity" else 1)
    p = tmp_path / "model.mlir"
    original = p.read_text()
    if failure == "origin":
        p.write_text(original.replace("aten.pow.Tensor_Scalar", "aten.mul.Tensor"))
    elif failure == "foreign_call":
        p.write_text(original.replace("%e=tensor.empty()", "func.call @unknown() : () -> ()\n%e=tensor.empty()"))
    elif failure == "runtime_exponent":
        report["cases"][0]["case"]["arguments"]["args"]["items"][1] = dict(kind="tensor")
    before = p.read_bytes()
    assert specialize(tmp_path, report) is None
    assert p.read_bytes() == before
    assert not (tmp_path / "native-source-specialization.json").exists()


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float32", "float64", "complex64"])
@pytest.mark.parametrize("exponent", [0.0, 1.0])
def test_source_dispatch_agrees_on_storage_words_including_signaling_nans(dtype, exponent):
    torch = pytest.importorskip("torch")
    if dtype in {"float16", "bfloat16"}:
        values = torch.arange(65536, dtype=torch.int32).to(torch.uint16).view(getattr(torch, dtype))
    elif dtype == "float64":
        words = [
            0,
            0x8000000000000000,
            0x7FF0000000000000,
            0xFFF0000000000000,
            0x7FF0000000000001,
            0x7FF8000000000001,
            0xFFF8000000000001,
            0x3FF0000000000000,
        ]
        values = torch.tensor(words, dtype=torch.uint64).view(torch.float64)
    else:
        words = [0, 0x80000000, 0x7F800000, 0xFF800000, 0x7F800001, 0x7FC00001, 0xFFC00001, 0x3F800000]
        values = torch.tensor(words, dtype=torch.uint32).view(torch.float32)
        if dtype == "complex64":
            values = torch.view_as_complex(values.reshape(-1, 2))
    actual = torch.ops.aten.pow.Tensor_Scalar(values, exponent)
    expected = torch.ones_like(values) if exponent == 0 else values.clone()
    assert torch.equal(actual.contiguous().view(torch.uint8), expected.contiguous().view(torch.uint8))
