"""Independent PyTorch W8A8 oracle for captured source layouts and host epilogues.

This walks the ORIGINAL source, never the prepared integer body or a device catalog.
Unsupported source operations refuse qualification rather than using compiled answers.
"""

from __future__ import annotations

import itertools


def evaluate(module, arrays, selected, *, quant_min, quant_max, eps):
    import torch
    from xdsl.dialects.builtin import DenseIntOrFPElementsAttr, TensorType
    from xdsl.ir.affine import AffineDimExpr

    dtypes = {
        "f16": torch.float16,
        "f32": torch.float32,
        "f64": torch.float64,
        "i8": torch.int8,
        "i32": torch.int32,
        "i64": torch.int64,
    }
    function = next(op for op in module.walk() if op.name == "func.func" and op.sym_name.data == "forward")
    env = {arg: torch.from_numpy(array.copy()) for arg, array in zip(function.body.block.args, arrays)}
    scales = []
    selected = set(selected)

    def dtype(t):
        return dtypes[str(t.element_type if isinstance(t, TensorType) else t)]

    def attr(op, key):
        return op.properties.get(key) or op.attributes[key]

    def constant(op):
        value = attr(op, "value")
        if isinstance(value, DenseIntOrFPElementsAttr):
            return torch.tensor(list(value.get_values()), dtype=dtype(op.results[0].type)).reshape(
                op.results[0].type.get_shape()
            )
        return torch.tensor(value.value.data, dtype=dtype(op.results[0].type))

    def scalar_body(block, operands):
        local = dict(zip(block.args, operands))
        for inner in block.ops:
            name = inner.name
            xs = [local[v] for v in inner.operands]
            if name == "linalg.yield":
                return xs[0]
            if name == "arith.constant":
                y = constant(inner)
            elif name in {"arith.extf", "arith.truncf"}:
                y = xs[0].to(dtype(inner.results[0].type))
            elif name in {"arith.addf", "arith.addi"}:
                y = xs[0] + xs[1]
            elif name in {"arith.mulf", "arith.muli"}:
                y = xs[0] * xs[1]
            elif name == "arith.subf":
                y = xs[0] - xs[1]
            elif name == "arith.divf":
                y = xs[0] / xs[1]
            else:
                raise ValueError(f"W8A8 reference refuses scalar op {name}")
            local[inner.results[0]] = y
        raise ValueError("source region lacks yield")

    def generic(op, xs):
        maps = [a.data for a in op.get_indexing_maps()]
        if op in selected:
            q, ss = [], []
            for original in xs[:2]:
                x = original.float()
                if not bool(torch.isfinite(x).all()):
                    raise ValueError("W8A8 reference refuses nonfinite contraction input")
                scale = torch.clamp(x.abs().amax() / ((quant_max - quant_min) / 2.0), min=eps)
                q.append(torch.clamp(torch.round(x * (1.0 / scale)), quant_min, quant_max).to(torch.int32))
                ss.append(scale)
            labels = []
            for amap in maps:
                if any(not isinstance(expr, AffineDimExpr) for expr in amap.results):
                    raise ValueError("integer oracle requires projected contraction maps")
                labels.append("".join(chr(ord("a") + expr.position) for expr in amap.results))
            # PyTorch integer einsum uses integer mm/bmm; independently of emitted MAC loops.
            acc = torch.einsum(labels[0] + "," + labels[1] + "->" + labels[2], *q)
            scales.append({"lhs": ss[0].item(), "rhs": ss[1].item(), "granularity": "per_tensor"})
            return ((acc.to(torch.float32) * ss[0]) * ss[1]).to(xs[-1].dtype)
        extents = [None] * maps[0].num_dims
        for amap, x in zip(maps, xs):
            for expr, size in zip(amap.results, x.shape):
                if isinstance(expr, AffineDimExpr):
                    old = extents[expr.position]
                    if old is not None and old != size:
                        raise ValueError("inconsistent source loop extent")
                    extents[expr.position] = size
        if any(x is None for x in extents):
            raise ValueError("source loop extent is underivable")
        result = xs[-1].clone()
        for point in itertools.product(*(range(n) for n in extents)):
            indices = [amap.eval(point, ()) for amap in maps]
            values = [x[index] for x, index in zip(xs[:-1], indices[:-1])]
            values.append(result[indices[-1]])
            result[indices[-1]] = scalar_body(op.body.block, values)
        return result

    for op in function.body.block.ops:
        xs = [env[v] for v in op.operands]
        name = op.name
        if name == "func.return":
            return xs, scales
        if name == "arith.constant":
            y = constant(op)
        elif name == "tensor.empty":
            y = torch.empty(op.results[0].type.get_shape(), dtype=dtype(op.results[0].type))
        elif name in {"tensor.expand_shape", "tensor.collapse_shape"}:
            y = xs[0].reshape(op.results[0].type.get_shape())
        elif name == "tensor.splat":
            y = xs[0].expand(op.results[0].type.get_shape()).clone()
        elif name == "linalg.fill":
            y = xs[0].expand(xs[1].shape).clone()
        elif name == "linalg.transpose":
            y = xs[0].permute(tuple(attr(op, "permutation").get_values())).contiguous()
        elif name in {"tensor.extract_slice", "tensor.insert_slice"}:
            offsets, sizes, strides = [
                list(attr(op, k).get_values()) for k in ("static_offsets", "static_sizes", "static_strides")
            ]
            if any(n < 0 for ns in (offsets, sizes, strides) for n in ns):
                raise ValueError("dynamic slice not supported by reference")
            indices = tuple(slice(o, o + n * s, s) for o, n, s in zip(offsets, sizes, strides))
            if name == "tensor.extract_slice":
                y = xs[0][indices].reshape(op.results[0].type.get_shape())
            else:
                y = xs[1].clone()
                y[indices] = xs[0].reshape(y[indices].shape)
        elif name in {"linalg.generic", "linalg.matmul", "linalg.batch_matmul"}:
            y = generic(op, xs)
        elif name == "linalg.reduce":
            # Keep source reduction order, instead of Torch's tree-sum order.
            dimensions = tuple(attr(op, "dimensions").get_values())
            y = xs[1].clone()
            for point in itertools.product(*(range(n) for n in xs[0].shape)):
                index = tuple(v for i, v in enumerate(point) if i not in dimensions)
                y[index] = scalar_body(op.regions[0].block, [xs[0][point], y[index]])
        else:
            raise ValueError(f"W8A8 reference refuses source op {name}")
        if len(op.results) != 1:
            raise ValueError("reference requires single-result source operations")
        env[op.results[0]] = y
    raise ValueError("source function lacks return")


def main():
    import json
    import sys
    from pathlib import Path

    import numpy as np

    from merlin.frontends.linalg_mlir import parse_mlir_file

    directory = Path(sys.argv[1])
    request = json.loads((directory / "w8a8-request.json").read_text())
    module = parse_mlir_file(directory / "model.before_w8a8.mlir")
    operations = list(module.walk())
    with np.load(directory / "inputs.npz") as archive:
        arrays = [archive[f"in{i}"] for i in range(len(archive.files))]
    values, scales = evaluate(
        module,
        arrays,
        [operations[i] for i in request["selected_ordinals"]],
        quant_min=request["quant_min"],
        quant_max=request["quant_max"],
        eps=request["eps"],
    )
    document = {
        "outputs": [v.tolist() for v in values],
        "scales": scales,
        "engine": "independent_pytorch_integer_einsum",
    }
    (directory / "w8a8-reference.json").write_text(json.dumps(document, indent=2) + "\n")


if __name__ == "__main__":
    main()
