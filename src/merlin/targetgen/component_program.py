"""Closed independent tensor programs, emitted as standard Linalg MLIR.

This generic builder describes observable tensor semantics. Logical aliases and
epoch updates are functionalized into SSA values; physical buffer aliasing,
allocation lifetime and host/device synchronization need execution witnesses.
It contains no target schedule, instruction, golden or workload information.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass


@dataclass(frozen=True)
class ValueType:
    shape: tuple[int, int]
    dtype: str

    @property
    def mlir(self):
        return "tensor<" + "x".join(str(value) for value in self.shape) + "x" + self.dtype + ">"


def _name(value):
    return isinstance(value, str) and value and value[0].isalpha() and all(c.isalnum() or c == "_" for c in value)


def analyze(program, *, operand_dtype, accumulator_dtype):
    """Check a finite typed DAG and derive source effects, uses and live outputs."""
    if not isinstance(program, dict) or set(program) != {"inputs", "nodes", "outputs"}:
        raise ValueError("component program requires exactly inputs, nodes and outputs")
    if not isinstance(program["inputs"], list) or not program["inputs"]:
        raise ValueError("component program requires explicit inputs")
    types, inputs, aliases, storage, current, uses = {}, [], {}, {}, {}, {}
    for row in program["inputs"]:
        if not isinstance(row, dict) or set(row) != {"name", "role", "shape", "dtype"}:
            raise ValueError("component input requires exactly name, role, shape and dtype")
        name, shape = row["name"], row["shape"]
        dtype = {"operand": operand_dtype, "accumulator": accumulator_dtype}.get(row["dtype"], row["dtype"])
        if not _name(name) or name in types or row["role"] not in {"input", "weight", "bias"}:
            raise ValueError("component input name/role is invalid or duplicated")
        if not isinstance(shape, list) or len(shape) != 2 or any(type(dim) is not int or dim < 1 for dim in shape):
            raise ValueError("component programs require positive rank-two input shapes")
        if not isinstance(dtype, str) or not dtype.startswith("i") or not dtype[1:].isdigit() or int(dtype[1:]) < 1:
            raise ValueError("independent component DAG currently requires signed integer types")
        types[name] = ValueType(tuple(shape), dtype)
        storage[name] = name
        current[name] = name
        uses[name] = 0
        inputs.append(dict(row, dtype=dtype))
    nodes, effects, versions = [], set(), {}
    if not isinstance(program["nodes"], list):
        raise ValueError("component nodes must be an explicit ordered list")
    for ordinal, node in enumerate(program["nodes"]):
        if not isinstance(node, dict) or not _name(node.get("name")) or node["name"] in types:
            raise ValueError("component node needs a fresh safe name")
        name, op, args = node["name"], node.get("op"), node.get("inputs")
        if set(node) != {"name", "op", "inputs"} or op not in {"matmul", "add", "copy", "transpose", "alias", "update"}:
            raise ValueError("unsupported component node; use explicit supported generic tensor operations")
        arity = 2 if op in {"matmul", "add", "update"} else 1
        if not isinstance(args, list) or len(args) != arity or any(arg not in types for arg in args):
            raise ValueError("component node operand is absent or its arity differs")
        actual = [current[storage[arg]] for arg in args]
        selected = [types[arg] for arg in args]
        for arg in actual:
            uses[arg] = uses.get(arg, 0) + 1
        lhs = selected[0]
        if op == "matmul":
            rhs = selected[1]
            if lhs.shape[1] != rhs.shape[0]:
                raise ValueError("component matmul reduction extents differ")
            if max(int(lhs.dtype[1:]), int(rhs.dtype[1:])) > int(accumulator_dtype[1:]):
                raise ValueError("component matmul operands exceed selected accumulator width")
            result = ValueType((lhs.shape[0], rhs.shape[1]), accumulator_dtype)
        elif op in {"add", "update"}:
            if selected[1] != lhs:
                raise ValueError("component add/update requires matching shapes and dtypes")
            result = lhs
        elif op == "transpose":
            result = ValueType((lhs.shape[1], lhs.shape[0]), lhs.dtype)
        else:
            result = lhs
        types[name] = result
        uses[name] = 0
        if op in {"alias", "update"}:
            owner = storage[args[0]]
            storage[name] = owner
            if op == "alias":
                aliases[name] = args[0]
                effects.add("alias")
            else:
                current[owner] = name
                versions[owner] = versions.get(owner, 0) + 1
                effects.add("epoch_mutation")
        else:
            storage[name] = name
            current[name] = name
        nodes.append(
            {**node, "ordinal": ordinal, "actual_inputs": actual, "shape": list(result.shape), "dtype": result.dtype}
        )
    outputs = program["outputs"]
    if not isinstance(outputs, list) or not outputs:
        raise ValueError("component program needs at least one published output")
    output_names, out = set(), []
    for row in outputs:
        if (
            not isinstance(row, dict)
            or set(row) != {"name", "value"}
            or not _name(row["name"])
            or row["name"] in output_names
            or row["value"] not in types
        ):
            raise ValueError("component publication needs a unique safe name and a known value")
        output_names.add(row["name"])
        name = current[storage[row["value"]]]
        value_type = types[name]
        out.append({**row, "actual_value": name, "shape": list(value_type.shape), "dtype": value_type.dtype})
        if uses.get(name, 0) and name not in {input["name"] for input in inputs}:
            effects.add("escaped_use")
    if any(count > 1 for name, count in uses.items()):
        effects.add("multiple_consumers")
    if any(count > 1 for name, count in uses.items() if name not in {input["name"] for input in inputs}):
        effects.add("shared_producer")
    if any(uses[row["name"]] > 1 and row["name"] not in versions for row in inputs if row["role"] == "weight"):
        effects.add("immutable_reuse")
    effects.add("output_publication")
    return {
        "inputs": inputs,
        "nodes": nodes,
        "outputs": out,
        "effects": sorted(effects),
        "uses": uses,
        "logical_epochs": versions,
        "logical_aliases": aliases,
        "selected_storage": {"operand": operand_dtype, "accumulator": accumulator_dtype},
        "scope": "functionalized tensor semantics; physical alias, lifetime and synchronization unverified",
    }


def render(program, *, operand_dtype, accumulator_dtype):
    """Emit the original generic source without corpus, data or target geometry.

    This fixed frontend construction establishes typed source semantics only.
    Numerical domains, target legality and candidate lowering remain separate.
    """
    accumulator = accumulator_dtype
    if not isinstance(accumulator, str) or not accumulator.startswith("i") or not accumulator[1:].isdigit():
        raise ValueError("component DAG requires a signed integer accumulator")
    program = analyze(program, operand_dtype=operand_dtype, accumulator_dtype=accumulator)
    types = {row["name"]: ValueType(tuple(row["shape"]), row["dtype"]) for row in program["inputs"] + program["nodes"]}
    values = {row["name"]: "%" + row["name"] for row in program["inputs"]}
    args = ", ".join(values[row["name"]] + ": " + types[row["name"]].mlir for row in program["inputs"])
    returns = ", ".join(ValueType(tuple(row["shape"]), row["dtype"]).mlir for row in program["outputs"])
    lines = [
        'builtin.module attributes {prov.level = "linalg-on-tensors"} {',
        f"  func.func @forward({args}) -> ({returns}) {{",
    ]
    identity = "affine_map<(d0, d1) -> (d0, d1)>"
    for node in program["nodes"]:
        name, op = node["name"], node["op"]
        actual = node["actual_inputs"]
        result = types[name]
        if op == "alias":
            values[name] = values[actual[0]]
            continue
        values[name] = "%" + name
        lines.append(f"    %{name}_empty = tensor.empty() : {result.mlir}")
        if op in {"copy", "transpose"}:
            source = types[actual[0]]
            if op == "copy":
                lines.append(
                    f"    %{name} = linalg.copy ins({values[actual[0]]} : {source.mlir}) "
                    f"outs(%{name}_empty : {result.mlir}) -> {result.mlir}"
                )
            else:
                lines.append(
                    f"    %{name} = linalg.transpose ins({values[actual[0]]} : {source.mlir}) "
                    f"outs(%{name}_empty : {result.mlir}) permutation = [1, 0]"
                )
            continue
        if op == "matmul":
            maps = [
                "affine_map<(d0, d1, d2) -> (d0, d2)>",
                "affine_map<(d0, d1, d2) -> (d2, d1)>",
                "affine_map<(d0, d1, d2) -> (d0, d1)>",
            ]
            iterators = '["parallel", "parallel", "reduction"]'
            # Keep initialization symbolic: parsers may expand tensor splats
            # into one host value per element even in source-only compilation.
            lines.append(f"    %{name}_zero = arith.constant 0 : {result.dtype}")
            lines.append(
                f"    %{name}_init = linalg.fill ins(%{name}_zero : {result.dtype}) "
                f"outs(%{name}_empty : {result.mlir}) -> {result.mlir}"
            )
            initial = f"%{name}_init"
            tagged_op, family = "matmul", "contraction"
        else:
            maps, iterators, initial = [identity] * 3, '["parallel", "parallel"]', f"%{name}_empty"
            tagged_op, family = "add", "elementwise"
        lhs, rhs = (types[arg] for arg in actual)
        lines.append(
            f"    %{name} = linalg.generic {{indexing_maps = [{', '.join(maps)}], iterator_types = {iterators}}} "
            f"ins({values[actual[0]]}, {values[actual[1]]} : {lhs.mlir}, {rhs.mlir}) outs({initial} : {result.mlir}) "
            f'attrs = {{prov.region_id = "component_{node["ordinal"]}", prov.op = "{tagged_op}", '
            f'prov.family = "{family}"}} {{'
        )
        lines.append(f"      ^bb{node['ordinal']}(%lhs: {lhs.dtype}, %rhs: {rhs.dtype}, %old: {result.dtype}):")
        if op == "matmul":
            operands = []
            for side, value_type in (("lhs", lhs), ("rhs", rhs)):
                if value_type.dtype != result.dtype:
                    lines.append(f"        %{side}_wide = arith.extsi %{side} : {value_type.dtype} to {result.dtype}")
                    operands.append(f"%{side}_wide")
                else:
                    operands.append(f"%{side}")
            lines.append(f"        %product = arith.muli {operands[0]}, {operands[1]} : {result.dtype}")
            lines.append(f"        %sum = arith.addi %old, %product : {result.dtype}")
        else:
            lines.append(f"        %sum = arith.addi %lhs, %rhs : {result.dtype}")
        lines.extend([f"        linalg.yield %sum : {result.dtype}", f"    }} -> {result.mlir}"])
    lines.extend(
        [
            "    func.return " + ", ".join(values[row["actual_value"]] for row in program["outputs"]) + " : " + returns,
            "  }",
            "}",
        ]
    )
    return program, "\n".join(lines) + "\n"


def build(entry, binding):
    from . import corpus_spec

    program, source = render(
        entry.get("program"),
        operand_dtype=binding.mlir_dtype(binding.operand_dtype),
        accumulator_dtype=binding.mlir_dtype(binding.accum_dtype),
    )
    cap = {
        "name": entry["name"],
        "kind": entry["kind"],
        "source_role": entry["source_role"],
        "source_reference": entry["source_reference"],
        "label": entry.get("label", "public"),
        "interface_mlir": "capsule.interface.mlir",
        "linalg_mlir": "capsule.interface.mlir",
        "inputs": program["inputs"],
        "operation": {
            "op": "component_program",
            "attributes": {
                "program": copy.deepcopy(entry["program"]),
                "out": program["outputs"][0]["name"],
                "outs": [row["name"] for row in program["outputs"]],
                "arg_order": [row["name"] for row in program["inputs"] + program["outputs"]],
            },
        },
        "component_program": program,
        "numeric_policy": corpus_spec._numeric_policy(binding, binding.accum_dtype, None),
        "expected": {
            "instruction_classes": binding.classes_for(op="matmul", output_dtype=binding.cap_dtype(binding.accum_dtype))
            if any(node["op"] == "matmul" for node in program["nodes"])
            else [],
            "modes": {},
        },
        "required_oracle_tiers": list(binding.tiers),
        "vcs": "optional",
        "firesim": "optional",
    }
    return cap, source
