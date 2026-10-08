"""Reconstruct full portable observations solely from compiled boundary readback."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from merlin.targetgen.core_aten_batch_grade import _DTYPES, _EXPECTED_DTYPES


def tensor_paths(value, path):
    if isinstance(value, Mapping) and value.get("kind") == "tensor":
        return [(path, value)]
    if isinstance(value, Mapping) and value.get("kind") == "tuple":
        return [leaf for i, item in enumerate(value["items"]) for leaf in tensor_paths(item, f"{path}[{i}]")]
    if isinstance(value, Mapping):
        return [leaf for key, item in value.items() for leaf in tensor_paths(item, f"{path}.{key}" if path else key)]
    if isinstance(value, list):
        return [leaf for i, item in enumerate(value) for leaf in tensor_paths(item, f"{path}[{i}]")]
    return []


def full_observation(record, raw, readback):
    def _encode(value):
        if isinstance(value, list):
            return [_encode(v) for v in value]
        if isinstance(value, complex):
            return dict(kind="complex", real=_encode(value.real), imag=_encode(value.imag))
        if isinstance(value, float) and (not math.isfinite(value) or (value == 0 and math.copysign(1, value) < 0)):
            return dict(kind="float", value=repr(value))
        return value

    boundary = record["semantic_boundary"]
    metadata = readback["metadata"]
    aliases = readback["aliases"]
    before = readback["pre_bytes"]

    def document(index, abi):
        layout = metadata[index]
        shape = layout["shape"]
        dtype = abi["dtype"]
        if dtype.startswith("complex<"):
            ndtype = np.complex64 if dtype == "complex<f32>" else np.complex128
            logical = np.dtype(ndtype).name
        else:
            ndtype, logical = _DTYPES[dtype], _EXPECTED_DTYPES[dtype]
        values = np.frombuffer(raw[index], ndtype).reshape(shape)
        return dict(
            kind="tensor",
            dtype=logical,
            shape=shape,
            stride=layout["stride"],
            storage_offset=layout["storage_offset"],
            requires_grad=layout["requires_grad"],
            values=_encode(values.tolist()),
        )

    # Only the argument/output TREE is reused. Tensor metadata and values come from readback.
    def replace(value, documents):
        if isinstance(value, dict) and value.get("kind") == "tensor":
            return next(documents)
        if isinstance(value, dict):
            return {key: replace(item, documents) for key, item in value.items()}
        if isinstance(value, list):
            return [replace(item, documents) for item in value]
        return value

    post_docs = [document(index, abi) for index, abi in zip(boundary["post_indices"], boundary["post_abi"])]
    post = replace(record["case"]["arguments"], iter(post_docs))
    output_docs = [document(index, abi) for index, abi in zip(record["output_indices"], record["output_abi"])]

    # Scalar return wrappers have one rank-zero tensor ABI. Preserve the source pytree kind.
    def output_tree(value, documents):
        if isinstance(value, dict) and value.get("kind") == "tensor":
            return next(documents)
        if isinstance(value, dict) and value.get("kind") == "tuple":
            return {**value, "items": [output_tree(item, documents) for item in value["items"]]}
        if isinstance(value, list):
            return [output_tree(item, documents) for item in value]
        tensor = next(documents)
        number = tensor["values"]
        if isinstance(value, bool):
            return bool(number)
        if isinstance(value, int):
            return int(number)
        return number

    output = output_tree(record["case"]["expected"], iter(output_docs))
    paths = tensor_paths(record["case"]["arguments"], "")
    mutated = []
    for (path, initial), observed, source_index, result_index in zip(
        paths, post_docs, record["input_indices"], boundary["post_indices"]
    ):
        # The corpus defines mutation by byte equality and dtype/shape/stride (not offset).
        if bytes.fromhex(before[str(source_index)]) != raw[result_index] or any(
            initial.get(key) != observed.get(key) for key in ("dtype", "shape", "stride")
        ):
            mutated.append(path)
    output_paths = tensor_paths(output, "result")
    relationships = []
    for (path, _), index in zip(output_paths, record["output_indices"]):
        for (input_path, _), post_index in zip(paths, boundary["post_indices"]):
            if post_index in aliases[index]:
                relationships.append(f"{path}->{input_path}")
    return dict(
        status="executed",
        execution_kind="simulator",
        output=output,
        post_arguments=post,
        mutated_arguments=sorted(mutated),
        output_input_aliases=sorted(relationships),
        evidence={"substrate": "spike_functional", "boundary": "same_conversion_exported_program"},
    )


def validate_full(record, raw, readback):
    from merlin.targetgen.core_aten_batch_grade import _grade_output, _leaves
    from merlin.targetgen.core_aten_eval import _without_values

    if not record.get("semantic_boundary"):
        return None
    if readback is None:
        return dict(status="ungradable", semantic_scope="full", reason="exported boundary lacks semantic readback")
    try:
        observation = full_observation(record, raw, readback)
        case = record["case"]
        for key, expected in [("output", case["expected"]), ("post_arguments", case["post_arguments"])]:
            if _without_values(observation[key]) != _without_values(expected):
                raise AssertionError(f"{key} metadata mismatch")
        if case["comparison"] not in ("metadata", "torch_close"):
            raise AssertionError("unknown comparison policy")
        if case["comparison"] == "torch_close":
            for indices, abis, expected in [
                (record["output_indices"], record["output_abi"], _leaves(case["expected"])),
                (
                    record["semantic_boundary"]["post_indices"],
                    record["semantic_boundary"]["post_abi"],
                    [doc for _, doc in tensor_paths(case["post_arguments"], "")],
                ),
            ]:
                for index, abi, doc in zip(indices, abis, expected):
                    abi = {**abi, "shape": readback["metadata"][index]["shape"]}
                    verdict = _grade_output(raw[index], abi, doc, case)
                    if verdict["status"] != "pass":
                        raise AssertionError(f"boundary value mismatch: {verdict}")
        for key in ("mutated_arguments", "output_input_aliases"):
            if sorted(observation[key]) != sorted(case[key]):
                raise AssertionError(f"{key} mismatch: expected {case[key]}, got {observation[key]}")
        return {"status": "pass", "semantic_scope": "full", "observation": observation}
    except AssertionError as exc:
        return {"status": "mismatch", "semantic_scope": "full", "reason": str(exc), "observation": observation}
    except (KeyError, IndexError, ValueError, TypeError, StopIteration) as exc:
        return {"status": "ungradable", "semantic_scope": "full", "reason": f"incomplete boundary readback: {exc}"}
