"""Opt-in W8A8 numeric contracts, reference independence and honest grading."""

import numpy as np
import pytest
from xdsl.parser import Parser

from merlin.frontends.linalg_mlir import make_context
from merlin.llvmlower.contraction_quantization import SPEC, quantize_module, selected_contractions


def module(body=None, init="0.0"):
    contraction = (
        body
        or """%r = linalg.matmul ins(%a, %b : tensor<2x3xf32>, tensor<3x2xf32>)
        outs(%z : tensor<2x2xf32>) -> tensor<2x2xf32>"""
    )
    return Parser(
        make_context(),
        f"""builtin.module {{
      func.func @forward(%a: tensor<2x3xf32>, %b: tensor<3x2xf32>) -> tensor<2x2xf32> {{
        %c = arith.constant {init} : f32
        %e = tensor.empty() : tensor<2x2xf32>
        %z = linalg.fill ins(%c : f32) outs(%e : tensor<2x2xf32>) -> tensor<2x2xf32>
        {contraction}
        func.return %r : tensor<2x2xf32>
      }}
    }}""",
    ).parse_module()


def test_opt_in_preserves_float_abi_and_uses_existing_lowering():
    m = module()
    original_type = next(op for op in m.walk() if op.name == "func.func").function_type
    chosen = list(selected_contractions(m, [("i8", "i8", "i32")], (2,)))
    assert len(chosen) == 1
    assert quantize_module(m, chosen) == 1
    assert next(op for op in m.walk() if op.name == "func.func").function_type == original_type
    names = [op.name for op in m.walk()]
    assert names.count("math.roundeven") == 2
    assert names.count("arith.extsi") == 2
    assert names.count("arith.sitofp") >= 1
    m.verify()


@pytest.mark.parametrize(
    "triples,ranks,init",
    [
        ([], (2,), "0.0"),
        ([("f32", "f32", "f32")], (2,), "0.0"),
        ([("i8", "i8", "i32")], (3,), "0.0"),
        ([("i8", "i8", "i32")], (2,), "1.0"),
    ],
)
def test_refuse_unsupported_capability_or_live_initializer(triples, ranks, init):
    m = module(init=init)
    before = str(m)
    assert list(selected_contractions(m, triples, ranks)) == []
    assert str(m) == before


def test_reference_is_integer_pytorch_and_dequantizes_in_order():
    torch = pytest.importorskip("torch")
    from merlin.targetgen.core_aten_w8a8_reference import evaluate

    m = module()
    chosen = list(selected_contractions(m, [("i8", "i8", "i32")], (2,)))
    a = np.array([[0, 0.5, -1], [1, 0.25, 0]], np.float32)
    b = np.array([[1, -1], [0.5, 0.25], [0, -0.5]], np.float32)
    outputs, scales = evaluate(m, [a, b], chosen, quant_min=SPEC.quant_min, quant_max=SPEC.quant_max, eps=SPEC.eps)
    s = torch.tensor(1 / 127, dtype=torch.float32)
    qa, qb = [torch.round(torch.from_numpy(x) * (1 / s)).to(torch.int32) for x in (a, b)]
    expected = ((qa @ qb).float() * s) * s
    assert torch.equal(outputs[0], expected)
    assert scales == [{"lhs": s.item(), "rhs": s.item(), "granularity": "per_tensor"}]


def test_exact_reference_grade_does_not_use_fp32_error():
    from merlin.targetgen.core_aten_batch_grade import grade_core_aten_batch

    document = dict(kind="tensor", dtype="float32", shape=[1], values=[1.0])
    record = dict(
        overload="aten.mm.default",
        status="bundled",
        output_indices=[0],
        output_abi=[dict(dtype="f32", shape=[1])],
        numeric_policy="int8_w8a8",
        scales=[dict(lhs=0.1, rhs=0.2)],
        case=dict(expected=document, comparison_parameters=dict(rtol=0, atol=0)),
        fp32_case=dict(expected={**document, "values": [2.0]}),
    )
    batch = dict(cases=[record], provenance={"test": True})
    verdict = grade_core_aten_batch(batch, [np.array([1.0], np.float32).tobytes()])["cases"]["aten.mm.default"]
    assert verdict["status"] == "pass"
    assert verdict["fp32_error"][0]["max_abs"] == 1
    assert verdict["numeric_policy"] == "int8_w8a8"
    raw = np.array([np.nextafter(np.float32(1), np.float32(2))], np.float32).tobytes()
    assert grade_core_aten_batch(batch, [raw])["cases"]["aten.mm.default"]["status"] == "mismatch"


@pytest.mark.parametrize("element", ["f16", "f64"])
def test_float_storage_policy_restores_original_dtype(element):
    m = Parser(make_context(), str(module()).replace("f32", element)).parse_module()
    original_type = next(op for op in m.walk() if op.name == "func.func").function_type
    selected = list(selected_contractions(m, [("i8", "i8", "i32")], (2,)))
    assert quantize_module(m, selected) == 1
    assert next(op for op in m.walk() if op.name == "func.func").function_type == original_type
    m.verify()


def test_non_mac_reduction_and_overflow_are_refused():
    # Keep matmul maps but replace the product with a difference: geometry is insufficient.
    m = module()
    from xdsl.dialects import arith
    from xdsl.rewriter import Rewriter

    mul = next(op for op in m.walk() if isinstance(op, arith.MulfOp))
    Rewriter.replace_op(mul, arith.SubfOp(*mul.operands))
    assert not list(selected_contractions(m, [("i8", "i8", "i32")], (2,)))
    huge = Parser(
        make_context(), str(module()).replace("2x3x", "2x200000x").replace("3x2x", "200000x2x")
    ).parse_module()
    assert not list(selected_contractions(huge, [("i8", "i8", "i32")], (2,)))


def test_zero_tensors_use_existing_epsilon_policy():
    torch = pytest.importorskip("torch")
    from merlin.targetgen.core_aten_w8a8_reference import evaluate

    m = module()
    selected = list(selected_contractions(m, [("i8", "i8", "i32")], (2,)))
    output, scales = evaluate(
        m,
        [np.zeros((2, 3), np.float32), np.zeros((3, 2), np.float32)],
        selected,
        quant_min=SPEC.quant_min,
        quant_max=SPEC.quant_max,
        eps=SPEC.eps,
    )
    assert torch.equal(output[0], torch.zeros((2, 2)))
    assert scales[0]["lhs"] == np.float32(SPEC.eps)
    assert scales[0]["rhs"] == np.float32(SPEC.eps)


def test_half_operands_with_float_accumulation_are_quantized():
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import FunctionType, TensorType, f16, f32

    m = module()
    op = next(op for op in m.walk() if op.name == "linalg.matmul")
    # Match the frontend's mixed-precision body: extend inputs, then f32 MAC.
    for src in op.operands[:2]:
        src._type = TensorType(f16, src.type.get_shape())
    block = op.body.block
    mul = next(inner for inner in block.ops if isinstance(inner, arith.MulfOp))
    for argument in block.args[:2]:
        argument._type = f16
    casts = [arith.ExtFOp(arg, f32) for arg in block.args[:2]]
    for cast in casts:
        block.insert_op_before(cast, mul)
    mul.operands = tuple(cast.result for cast in casts)
    function = next(inner for inner in m.walk() if inner.name == "func.func")
    function.function_type = FunctionType.from_lists(
        [arg.type for arg in function.body.block.args], [op.results[0].type]
    )
    chosen = list(selected_contractions(m, [("i8", "i8", "i32")], (2,)))
    assert chosen == [op]
    assert quantize_module(m, chosen) == 1
