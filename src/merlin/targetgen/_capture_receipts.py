"""Receipts a capture records about its own transformations, beside the capture worker.

Runs inside the capture interpreter (the one with torch) and is imported by the worker as a
sibling, so it must not import Merlin package code. Every function here observes or compares
values the worker already produced; none of them transforms the model.
"""

from __future__ import annotations


def mlir_dtype(torch_dtype) -> str:
    """Canonical MLIR element spelling for a captured torch tensor dtype.

    Do not infer this from JSON values: the worker converts tensors through float64 for
    serialization, so integral-looking values and genuinely integral tensors are
    indistinguishable afterwards.
    """
    import torch

    spelling = {
        torch.bool: "i1",
        torch.int8: "i8",
        torch.uint8: "ui8",
        torch.int16: "i16",
        torch.int32: "i32",
        torch.int64: "i64",
        torch.float16: "f16",
        torch.bfloat16: "bf16",
        torch.float32: "f32",
        torch.float64: "f64",
    }
    for name, mlir in (("float8_e4m3fn", "f8E4M3FN"), ("float8_e5m2", "f8E5M2")):
        dtype = getattr(torch, name, None)
        if dtype is not None:
            spelling[dtype] = mlir
    if torch_dtype not in spelling:
        raise RuntimeError(f"unsupported captured input dtype: {torch_dtype}")
    return spelling[torch_dtype]


def output_abi(outputs):
    """Flatten model results and preserve the ABI of every tensor result.

    Whole-model captures historically recorded only the nested JSON values.  A list-shaped tensor and
    a tuple of tensors are indistinguishable in that representation, which made the parent silently
    retain only result zero.  The pytree flattening performed while torch still owns the values is the
    authoritative result cardinality and dtype record.
    """
    import torch

    leaves, _spec = torch.utils._pytree.tree_flatten(outputs)
    bad = [type(x).__name__ for x in leaves if not isinstance(x, torch.Tensor)]
    if bad:
        raise RuntimeError(f"model outputs must have only tensor leaves; got {bad}")
    return leaves, [{"shape": list(x.shape), "dtype": mlir_dtype(x.dtype)} for x in leaves]


def integerized_agreement(before, after, *, atol: float, rtol: float) -> dict:
    """Compare two independently supplied executions on the capture input."""
    import torch

    left, left_abi = output_abi(before)
    right, right_abi = output_abi(after)
    rows = []
    compatible = left_abi == right_abi and bool(left)
    for original, rewritten in zip(left, right):
        lhs = original.detach().to(torch.float64)
        rhs = rewritten.detach().to(torch.float64)
        same_shape = lhs.shape == rhs.shape
        finite = bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all()) if same_shape else False
        if same_shape and finite and lhs.numel():
            delta = (lhs - rhs).abs()
            max_abs = float(delta.max())
            max_rel = float((delta / lhs.abs().clamp_min(1e-12)).max())
            within = bool(torch.all(delta <= atol + rtol * lhs.abs()))
        elif same_shape and finite:
            max_abs = max_rel = 0.0
            within = True
        else:
            max_abs = max_rel = 0.0
            within = False
        rows.append(
            {
                "max_abs": max_abs,
                "max_rel": max_rel,
                "within_tolerance": within,
                "finite": finite,
                "atol": atol,
                "rtol": rtol,
                "shape": list(original.shape),
            }
        )
    passed = compatible and len(rows) == len(left) and all(row["within_tolerance"] for row in rows)
    return {
        "status": "passed" if passed else "failed",
        "samples": 1,
        "atol": atol,
        "rtol": rtol,
        "max_abs": max((row["max_abs"] for row in rows), default=0.0),
        "max_rel": max((row["max_rel"] for row in rows), default=0.0),
        "outputs": rows,
        "finite": compatible and all(row["finite"] for row in rows),
    }


def fp32_stage_observation(
    original: dict, staged_leaves, original_input_abi: list[dict], staged_input_abi: list[dict]
) -> dict:
    """Measure a selected precision change; this is not an accuracy or quantization gate."""
    import torch

    before = original["leaves"]
    _, after_abi = output_abi(tuple(staged_leaves))
    before_abi = original["output_abi"]
    if len(before) != len(staged_leaves) or not before:
        raise RuntimeError("FP32 staging changed output cardinality")
    rows = []
    for left, right, old, new in zip(before, staged_leaves, before_abi, after_abi, strict=True):
        if left.shape != right.shape:
            raise RuntimeError("FP32 staging changed output shape")
        if not left.is_floating_point():
            if right.dtype != left.dtype or not torch.equal(left, right):
                raise RuntimeError("FP32 staging changed an exact integer/bool result")
            rows.append(
                {
                    "shape": list(left.shape),
                    "original_dtype": old["dtype"],
                    "staged_dtype": new["dtype"],
                    "comparison": "exact_nonfloating",
                    "max_abs": 0.0,
                    "max_rel": 0.0,
                }
            )
            continue
        if right.dtype != torch.float32:
            raise RuntimeError("FP32 staging retained a non-FP32 floating output")
        lhs, rhs = left.detach().to(torch.float64), right.detach().to(torch.float64)
        if not bool(torch.isfinite(lhs).all() and torch.isfinite(rhs).all()):
            raise RuntimeError("FP32 staging output comparison is nonfinite")
        delta = (lhs - rhs).abs()
        rows.append(
            {
                "shape": list(left.shape),
                "original_dtype": old["dtype"],
                "staged_dtype": new["dtype"],
                "comparison": "observed_floating",
                "max_abs": float(delta.max()) if delta.numel() else 0.0,
                "max_rel": float((delta / lhs.abs().clamp_min(1e-12)).max()) if delta.numel() else 0.0,
            }
        )
    return {
        "status": "observed",
        "scope": "original computation versus staged FP32; not accuracy equivalence",
        "original_input_abi": original_input_abi,
        "staged_input_abi": staged_input_abi,
        "original_output_abi": before_abi,
        "staged_output_abi": after_abi,
        "output_cardinality": len(rows),
        "output_metrics": rows,
    }


def exported_integer_mm_count(module) -> int:
    """Count proven i8×i8→i32 contractions in the emitted, unnormalized MLIR."""
    if module is None:
        return 0
    from xdsl.dialects.builtin import IntegerType, TensorType

    def width(value) -> int | None:
        typ = getattr(value, "type", None)
        if not isinstance(typ, TensorType) or not isinstance(typ.element_type, IntegerType):
            return None
        return int(typ.element_type.width.data)

    count = 0
    for op in module.walk():
        if op.name != "linalg.generic" or len(op.operands) < 3 or len(op.results) != 1:
            continue
        prov = op.attributes.get("prov.op")
        if (
            str(getattr(prov, "data", "")) == "int_matmul"
            and [width(value) for value in op.operands[:2]] == [8, 8]
            and width(op.results[0]) == 32
        ):
            count += 1
    return count


def realized_scheme(
    *,
    requested: str | None,
    recipe_selected: bool,
    already_quantized: bool,
    applied_scheme: str | None,
    quant_stats: dict | None,
    integerization_ok: bool,
) -> str | None:
    """The quantization scheme the applied transform and its emitted arithmetic establish.

    A recipe requests a format; only the applied transform and its emitted arithmetic establish
    the captured scheme. In particular, an already-integer graph that skipped observers must not
    inherit the request's label. ``requested`` is the scheme a scheme-named capture asked for;
    ``applied_scheme`` is the provenance scheme the recipe path handed to conversion;
    ``integerization_ok`` is true only when a static integer receipt exists and every one of
    its checks passed.
    """
    if already_quantized:
        return None
    if not recipe_selected:
        return requested
    if applied_scheme is None or not isinstance(quant_stats, dict):
        return None
    api = quant_stats.get("api")
    if applied_scheme == "int8_static_act_int8_weight":
        return applied_scheme if api == "pt2e" and integerization_ok else None
    if applied_scheme == "int8_dyn_act_int8_weight":
        return applied_scheme if api == "quantize_" and quant_stats.get("layers_quantized", 0) > 0 else None
    if not applied_scheme.startswith("fp8_"):
        return None
    if api == "pt2e" and quant_stats.get("annotated_contractions", 0) > 0:
        return applied_scheme
    # A dynamic float8 recipe goes through quantize_, not PT2E; without this row its capture
    # recorded `scheme: None` even when every planned Linear had been transformed.
    if (
        applied_scheme.endswith("_dynamic_act_weight")
        and api == "quantize_"
        and quant_stats.get("layers_quantized", 0) > 0
    ):
        return applied_scheme
    return None
