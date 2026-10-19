"""Native default A/table readouts, frozen biases and callable-ABI guards."""

import hashlib
import json
import subprocess
from fractions import Fraction as F

import numpy as np
import pytest
from test_exact_readout_lowering import fixture

from merlin.llvmlower.exact_readout import lower_exact_readouts


def freeze(module, reader, tmp_path, values, literal):
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, StringAttr

    values = np.asarray(values, np.float32)
    if literal:
        old = reader.inputs[1]
        const = arith.ConstantOp(DenseIntOrFPElementsAttr.from_list(old.type, values.tolist()))
        reader.parent_block().insert_op_before(const, reader)
        reader.operands = tuple(const.result if v is old else v for v in reader.operands)
    else:
        data = values.tobytes()
        header = json.dumps({"bias": {"dtype": "F32", "shape": [len(values)], "data_offsets": [0, len(data)]}}).encode()
        path = tmp_path / "weights.safetensors"
        path.write_bytes(len(header).to_bytes(8, "little") + header + data)
        path.with_suffix(".safetensors.manifest.json").write_text(
            json.dumps({"1": dict(weight="bias", kind="param", dtype="float32", shape=[len(values)])})
        )
        module.attributes["prov.weights_file"] = StringAttr(str(path))


def native(module, arrays, tmp_path):
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.pipeline import lower_to_llvm_ir
    from merlin.llvmlower.toolchain import clang
    from merlin.xdsl_dialects._common import text

    llvm = lower_to_llvm_ir(text(module, generic=True), workdir=tmp_path / "lower", vectorize=False)
    src = tmp_path / "model.ll"
    src.write_text(llvm)
    obj = src.with_suffix(".o")
    lib = tmp_path / ("model_" + hashlib.sha256(llvm.encode()).hexdigest() + ".so")
    subprocess.run([str(clang()), "-O2", "-fPIC", "-c", str(src), "-o", str(obj)], check=True, capture_output=True)
    subprocess.run(
        ["cc", "-fPIC", "-shared", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(lib)],
        check=True,
        capture_output=True,
    )
    fn = HostModel.load(str(lib))

    def run():
        fn([(a.ctypes.data, a.shape) for a in arrays])

    return run


@pytest.mark.parametrize("floating", [False, True])
@pytest.mark.parametrize("literal", [False, True])
def test_frozen_per_channel_bias_native_and_changed_parameter(tmp_path, floating, literal):
    from merlin.targetgen.exact_arithmetic import round_binary

    scales = [F(1, 2), F(17, 31) if not floating else F(float(np.float32(0.03))), F(-3, 8)]
    module, reader = fixture(scales, size=129, bias=True, floating=floating)
    biases = np.array([0.0, -0.25, 1.0], np.float32)
    freeze(module, reader, tmp_path, biases, literal)
    assert lower_exact_readouts(module) == 1
    module.verify()
    proof = json.loads(reader.attributes["merlin.exact_readout_proof"].data)
    assert proof["frozen_bias"]["guarded"] is not literal
    if not floating:
        assert proof["schema"] == "requant-A-v1"
        assert {r["algorithm"] for r in proof["channels"]} == {"threshold_table", "clamp_mul_add_shift"}
    rng = np.random.default_rng(443)
    x = rng.integers(-(1 << 31), 1 << 31, (129, 3), dtype=np.int32)
    x[:7] = np.asarray([-(1 << 31), (1 << 31) - 1, -1, 0, 1, 2, 3], np.int32)[:, None]
    output = np.zeros(x.shape, dtype=np.float32 if floating else np.int8)
    run = native(module, [x, biases, output], tmp_path)
    output_scale = F(1) if floating else F(31)
    for mutate in (False, True):
        if mutate:
            biases[:] = [0.125, 8.0, -1.0]
        run()
        effective = np.array([0.0, -0.25, 1.0], np.float32) if literal else biases
        expected = np.empty_like(output)
        for ix in np.ndindex(x.shape):
            real = int(x[ix]) * scales[ix[-1]] + F(float(effective[ix[-1]])) / output_scale
            expected[ix] = round_binary(real) if floating else min(127, max(-128, round(real)))
        assert output.tobytes() == expected.tobytes()


def test_frozen_bias_manifest_mismatch_is_refused(tmp_path):
    module, reader = fixture([F(1, 2)], bias=True)
    freeze(module, reader, tmp_path, [0.5], False)
    path = tmp_path / "weights.safetensors"
    header = json.dumps({"bias": {"dtype": "F32", "shape": [1], "data_offsets": [0, 5]}}).encode()
    path.write_bytes(len(header).to_bytes(8, "little") + header + bytes(5))
    with pytest.raises(ValueError, match="byte interval"):
        lower_exact_readouts(module)


@pytest.mark.parametrize("bias", [False, True])
def test_imported_proof_and_body_are_revalidated(tmp_path, bias):
    from xdsl.dialects.builtin import StringAttr

    module, reader = fixture([F(17, 31)], bias=bias)
    if bias:
        freeze(module, reader, tmp_path, [0.25], False)
    assert lower_exact_readouts(module) == 1
    assert lower_exact_readouts(module) == 0
    proof = json.loads(reader.attributes["merlin.exact_readout_proof"].data)
    proof["channels"][0]["constants"]["B"] = str(int(proof["channels"][0]["constants"]["B"]) + 123)
    reader.attributes["merlin.exact_readout_proof"] = StringAttr(json.dumps(proof))
    with pytest.raises(ValueError, match="proof|boundary"):
        lower_exact_readouts(module)


def test_imported_valid_proof_does_not_authenticate_modified_body():
    from xdsl.dialects import arith
    from xdsl.dialects.builtin import IntegerAttr, i64

    module, reader = fixture([F(17, 31)])
    assert lower_exact_readouts(module) == 1
    constants = [op for op in reader.body.walk() if isinstance(op, arith.ConstantOp) and op.result.type == i64]
    constants[0].properties["value"] = IntegerAttr(123456, i64)
    with pytest.raises(ValueError, match="emitted body"):
        lower_exact_readouts(module)


def test_zero_coefficient_is_an_exact_constant():
    from test_exact_readout_lowering import scalar
    from xdsl.dialects.builtin import i8, i32
    from xdsl.dialects.linalg.ops import YieldOp
    from xdsl.ir import Block

    from merlin.llvmlower.exact_readout_source import Scalar
    from merlin.llvmlower.requant_a import emit, records

    proof = records([F(0)], [F(5, 2)], dict(zero_point=1, quant_min=-128, quant_max=127, relu=False))
    block = Block(arg_types=[i32, i8])
    builder = Scalar()
    result = emit(builder, block.args[0], proof)
    block.add_ops([*builder.ops, YieldOp(result)])
    for x in [-(1 << 31), -1, 0, 1, (1 << 31) - 1]:
        assert scalar(block, [x & 0xFFFFFFFF, 0]) == 3
