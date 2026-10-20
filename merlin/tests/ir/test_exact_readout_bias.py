"""Plain exact readouts preserve literal biases and changing runtime parameters."""

import hashlib
import json
import subprocess
from fractions import Fraction as F

import numpy as np
import pytest
from test_exact_readout_lowering import fixture



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
    module.verify()
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


