"""Opt-in Core ATen numeric policy preparation and separate approximation metrics."""

from __future__ import annotations

import copy
import json
import os
import subprocess
from pathlib import Path

import numpy as np


def prepare_w8a8(directory: Path, report: dict, *, triples, ranks) -> dict:
    from merlin.common.paths import python_import_roots
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.llvmlower.contraction_quantization import SPEC, quantize_module, selected_contractions
    from merlin.llvmlower.toolchain import m2m_python
    from merlin.targetgen import core_aten_w8a8_reference
    from merlin.targetgen.core_aten_batch import _tensor_documents

    module = parse_mlir_file(directory / "model.mlir")
    selected = list(selected_contractions(module, triples, ranks))
    if not selected:
        return {"quantized_contractions": 0, "scales": []}
    # Single-case attribution is mandatory for both oracle replacement and device evidence.
    records = [r for r in report["cases"] if r["status"] == "bundled"]
    if len(records) != 1:
        raise ValueError("W8A8 execution requires one case per shard")
    original = directory / "model.before_w8a8.mlir"
    original.write_bytes((directory / "model.mlir").read_bytes())
    operations = list(module.walk())
    request = dict(
        selected_ordinals=[operations.index(op) for op in selected],
        quant_min=SPEC.quant_min,
        quant_max=SPEC.quant_max,
        eps=SPEC.eps,
    )
    (directory / "w8a8-request.json").write_text(json.dumps(request, indent=2) + "\n")
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(str(p) for p in python_import_roots())
    process = subprocess.run(
        [str(m2m_python()), "-m", core_aten_w8a8_reference.__name__, str(directory.resolve())],
        env=env,
        capture_output=True,
        text=True,
        timeout=120,
    )
    (directory / "w8a8-reference.log").write_text(process.stdout + process.stderr)
    if process.returncode:
        raise ValueError(f"independent W8A8 reference failed: {process.stderr[-2000:]}")
    reference = json.loads((directory / "w8a8-reference.json").read_text())
    record = records[0]
    record["fp32_case"] = copy.deepcopy(record["case"])
    record["case"] = copy.deepcopy(record["case"])
    record["scales"] = reference["scales"]
    record["reference_engine"] = reference["engine"]
    expected = _tensor_documents(record["case"]["expected"])
    if len(expected) != len(record["output_indices"]):
        raise ValueError("W8A8 oracle needs tensor output documents")

    def reference_values(document, index):
        values = reference["outputs"][index]
        if document["dtype"] in {"complex64", "complex128"}:
            packed = np.asarray(values)
            if packed.shape != (*document["shape"], 2):
                raise ValueError("complex oracle output does not match the physical ABI")

            def encode(value):
                if len(value) == 2 and all(isinstance(v, (float, int)) for v in value):
                    return dict(kind="complex", real=value[0], imag=value[1])
                return [encode(v) for v in value]

            values = encode(values)
        return values

    for document, index in zip(expected, record["output_indices"], strict=True):
        document["values"] = reference_values(document, index)
    if record.get("semantic_boundary"):
        posts = _tensor_documents(record["case"]["post_arguments"])
        for document, index in zip(posts, record["semantic_boundary"]["post_indices"], strict=True):
            document["values"] = reference_values(document, index)
    record["case"]["comparison_parameters"] = dict(rtol=0.0, atol=0.0, equal_nan=True)
    count = quantize_module(module, selected)
    (directory / "model.mlir").write_text(str(module) + "\n")
    return dict(quantized_contractions=count, scales=reference["scales"], reference_engine=reference["engine"])


def fp32_error(record, outputs) -> list[dict]:
    """Report compiled-output error vs original golden, without affecting correctness."""
    from merlin.targetgen.core_aten_batch_grade import _DTYPES, _complex_values, _leaves

    errors = []
    for index, abi, expected in zip(
        record["output_indices"], record["output_abi"], _leaves(record["fp32_case"]["expected"]), strict=True
    ):
        complex_type = {"complex<f32>": np.complex64, "complex<f64>": np.complex128}.get(abi["dtype"])
        compute_type = np.complex128 if complex_type else np.float64
        actual = np.frombuffer(outputs[index], complex_type or _DTYPES[abi["dtype"]]).astype(compute_type)
        golden = np.asarray(_complex_values(expected["values"]), dtype=compute_type).reshape(-1)
        difference = np.abs(actual - golden)
        relative = np.divide(difference, np.abs(golden), out=np.where(difference == 0, 0.0, np.inf), where=golden != 0)
        norm = float(np.linalg.norm(actual) * np.linalg.norm(golden))
        cosine = float(np.real(np.vdot(actual, golden)) / norm) if norm else float(np.array_equal(actual, golden))
        errors.append(
            dict(
                max_abs=float(np.max(difference, initial=0)), max_rel=float(np.max(relative, initial=0)), cosine=cosine
            )
        )
    return errors
