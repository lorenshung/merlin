"""Explicit pinned BF16 LayerNorm: refusal and actual emitted native arithmetic."""

import os
import subprocess

import numpy as np
import pytest
from xdsl.dialects import arith, builtin, func, tensor
from xdsl.dialects.linalg import ops as linalg
from xdsl.ir import Block, Region
from xdsl.ir.affine import AffineMap

from merlin.llvmlower import toolchain
from merlin.llvmlower.cpu_layer_norm import CPU_LAYER_NORM_BF16_AVX2_POLICY, build_cpu_layer_norm


def _options(**kwargs):
    return dict(
        epsilon=1e-6,
        backend_policy=CPU_LAYER_NORM_BF16_AVX2_POLICY,
        assume_finite_intermediates=True,
        allow_reassociation=True,
        **kwargs,
    )


def _args(shape=(1, 4, 768), dtype=builtin.bf16, param=builtin.bf16):
    return Block(arg_types=[builtin.TensorType(dtype, shape)] + [builtin.TensorType(param, [shape[-1]])] * 2)


@pytest.mark.parametrize(
    "key,value",
    [
        ("backend_policy", "math"),
        ("assume_finite_intermediates", False),
        ("allow_reassociation", False),
        ("epsilon", 0),
        ("epsilon", -1),
        ("epsilon", float("nan")),
        ("epsilon", float("inf")),
        ("epsilon", 1e300),
        ("epsilon", 1e-300),
    ],
)
def test_explicit_policy_and_domain_required_without_mutation(key, value):
    b = _args()
    options = _options()
    options[key] = value
    with pytest.raises(ValueError):
        build_cpu_layer_norm(*b.args, **options)
    assert all(not tuple(a.uses) for a in b.args)


@pytest.mark.parametrize("shape", [(768,), (4, 384), (4, 769), (-1, 768), (0, 768)])
def test_refuse_unaudited_shapes(shape):
    with pytest.raises(ValueError):
        build_cpu_layer_norm(*_args(shape).args, **_options())


@pytest.mark.parametrize("dtype", [builtin.f16, builtin.f32, builtin.i8])
def test_refuse_other_input_dtypes(dtype):
    with pytest.raises(TypeError):
        build_cpu_layer_norm(*_args(dtype=dtype).args, **_options())


def test_refuse_mismatched_affine_dtypes():
    b = Block(
        arg_types=[
            builtin.TensorType(builtin.bf16, [4, 768]),
            builtin.TensorType(builtin.bf16, [768]),
            builtin.TensorType(builtin.f32, [768]),
        ]
    )
    with pytest.raises(TypeError):
        build_cpu_layer_norm(*b.args, **_options())


def _native_module(shape, mixed):
    t = builtin.TensorType(builtin.f32, shape)
    p = builtin.TensorType(builtin.f32, [768])
    block = Block(arg_types=[t, p, p])

    def cast(value, dtype, cls):
        shp = list(value.type.get_shape())
        rank = len(shp)
        result_type = builtin.TensorType(dtype, shp)
        empty = tensor.EmptyOp((), result_type)
        bb = Block(arg_types=[value.type.element_type, dtype])
        op = cls(bb.args[0], dtype)
        bb.add_ops([op, linalg.YieldOp(op.result)])
        gen = linalg.GenericOp(
            inputs=(value,),
            outputs=(empty.tensor,),
            body=Region(bb),
            indexing_maps=builtin.ArrayAttr([builtin.AffineMapAttr(AffineMap.identity(rank))] * 2),
            iterator_types=builtin.ArrayAttr([linalg.IteratorTypeAttr(linalg.IteratorType.PARALLEL)] * rank),
            result_types=(result_type,),
        )
        block.add_ops([empty, gen])
        return gen.results[0]

    x = cast(block.args[0], builtin.bf16, arith.TruncFOp)
    g, b = block.args[1:] if mixed else [cast(a, builtin.bf16, arith.TruncFOp) for a in block.args[1:]]
    ops, result = build_cpu_layer_norm(x, g, b, **_options())
    block.add_ops(ops)
    result = cast(result, builtin.f32, arith.ExtFOp)
    block.add_op(func.ReturnOp(result))
    module = builtin.ModuleOp([func.FuncOp("forward", ([t, p, p], [t]), Region(block))])
    module.verify()
    return module


@pytest.mark.parametrize("optimization", ["-O0", "-O2"])
@pytest.mark.parametrize("mixed", [False, True])
def test_native_matches_pinned_torch_random_shifted_and_constant_rows(tmp_path, mixed, optimization):
    if not toolchain.available():
        pytest.skip("native compiler toolchain unavailable")
    from merlin.llvmlower.abi import HostModel
    from merlin.llvmlower.codegen import mlir_runtime_c
    from merlin.llvmlower.lower import lower_model_file

    # Unique library names avoid dlopen reusing a preceding parametrized case.
    root = tmp_path / (("f32affine" if mixed else "bf16affine") + optimization)
    root.mkdir()
    reference = r"""
import json,sys,torch,numpy as np
from pathlib import Path
p=Path(sys.argv[1]); mixed=sys.argv[2]=='1';torch.set_num_threads(2)
if not torch.__version__.startswith('2.10.') or torch.backends.cpu.get_cpu_capability()!='AVX2':sys.exit(77)
torch.manual_seed(918); x=torch.randn(1,8,768);x[:,0]=3;x[:,1]+=64;x=x.to(torch.bfloat16)
g=torch.randn(768);b=torch.randn(768)
if not mixed:g=g.to(torch.bfloat16);b=b.to(torch.bfloat16)
y=torch.nn.functional.layer_norm(x,(768,),g,b,1e-6)
for n,t in [('input',x),('gamma',g),('beta',b),('golden',y)]:np.save(p/(n+'.npy'),t.float().numpy())
(p/'runtime.json').write_text(json.dumps({'torch':torch.__version__,'capability':torch.backends.cpu.get_cpu_capability()}))
"""
    run = subprocess.run(
        [str(toolchain.compiler_python()), "-c", reference, str(root), "1" if mixed else "0"],
        env={**os.environ, "ATEN_CPU_CAPABILITY": "avx2"},
        capture_output=True,
        text=True,
    )
    if run.returncode == 77:
        pytest.skip("reference requires pinned Torch2.10 AVX2")
    assert run.returncode == 0, run.stderr
    module = _native_module([1, 8, 768], mixed)
    source = root / "model.mlir"
    source.write_text(str(module))
    lowered = lower_model_file(source, root / "lower", targets=(), textual=True)
    obj = root / "model.o"
    shared = root / "model.so"
    subprocess.run(
        [str(toolchain.clang()), optimization, "-fPIC", "-c", str(lowered.ll_path), "-o", str(obj)],
        check=True,
        capture_output=True,
        text=True,
    )
    subprocess.run(
        ["cc", "-shared", "-fPIC", str(obj), str(mlir_runtime_c()), "-lm", "-o", str(shared)],
        check=True,
        capture_output=True,
        text=True,
    )
    args = [np.ascontiguousarray(np.load(root / (n + ".npy"))) for n in ["input", "gamma", "beta"]]
    gold = np.load(root / "golden.npy")
    actual = np.zeros(gold.shape, dtype=np.float32, order="C")
    HostModel.load(str(shared))([(a.ctypes.data, a.shape) for a in args] + [(actual.ctypes.data, actual.shape)])
    assert actual.tobytes() == np.ascontiguousarray(gold).tobytes()
    assert np.isfinite(actual).all()
