#!/usr/bin/env python3
"""Build TorchAO's own quantization objects from a recipe — runs INSIDE the capture venv.

The capture venv is the only interpreter with torch; the recipe is derived on the other side, from
a hardware target, and crosses as JSON (:mod:`merlin.targetgen.quant_recipe`). This module is the
generic half that turns that data into the framework's extension points:

* a static recipe becomes a PT2E :class:`Quantizer` whose ``QuantizationSpec`` s (dtype, range,
  symmetry, per-tensor or per-channel, observer) are read from the recipe, annotating the aten
  contractions;
* a dynamic recipe becomes a ``quantize_`` configuration with the recipe's granularity — built
  PER LAYER, through TorchAO's own ``AOBaseConfig`` subclass for that (:func:`build_fqn_config`),
  so a layer the target cannot run in its numeric form is mapped to ``None`` rather than being
  quantized because it happened to be a Linear.

There is one quantizer for every target. A target changes the recipe, never this file.
Scoped recipes reuse Merlin's shared SW admission screen; no target-name dispatch
or framework source patch is involved.

WHOSE LIMIT REFUSED A LAYER. :mod:`quant_layer_plan` decides what the TARGET absorbs; this file
adds what this FRAMEWORK BUILD can express of that decision, and the two are recorded under
different codes. A Conv2d the hardware absorbs perfectly well is still not something ``quantize_``
transforms, and filing that as a hardware gap would send someone to read RTL about a Python
dispatch table.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from typing import Any

from merlin.targetgen._recipe_integer import _per_tensor_scale as _per_tensor_scale
from merlin.targetgen._recipe_integer import integer_recipes as integer_recipes

RECIPE_SCHEMA = "quant_recipe_v1"
DEFAULT_OBSERVER_EPSILON = 2**-12

#: Recipe dtype -> the torch dtype a spec is built with. A dtype absent here is refused, not mapped
#: to a neighbour: the recipe states the hardware's element format and a near miss is a different
#: datapath.
_TORCH_DTYPE = {"int8": "int8", "uint8": "uint8", "fp8_e4m3": "float8_e4m3fn", "fp8_e5m2": "float8_e5m2"}


class RecipeError(ValueError):
    """The recipe cannot be realised with this framework build; says which field and why."""


class NoEligibleRecipeOperations(RecipeError):
    """The frontend proved that no operation is eligible for this recipe."""


class FFTAdmissionRefused(RecipeError):
    """The complete prospective FFT chain lacks operation admission."""

    code = "fft_admission_refused"


class CalibrationRefused(RecipeError):
    """The calibration sample cannot be observed by the declared static recipe."""

    code = "nonfinite_calibration"

    def __init__(self, reason, *, details=None):
        super().__init__(reason)
        self.details = details or {}


def require_finite_observer_state(model, observer_base, histogram_observer, *, calibration_sample):
    """Reject nonfinite statistics produced internally from finite observer inputs."""
    import torch

    for path, observer in model.named_modules():
        if not isinstance(observer, observer_base):
            continue
        buffers = list(observer.named_buffers(recurse=False))
        if isinstance(observer, histogram_observer):
            # HistogramObserver searches over an affine range in the buffers'
            # own precision. Finite endpoints can still overflow that span.
            buffers.append(("max_val-min_val", observer.max_val - observer.min_val))
        for name, buffer in buffers:
            if buffer.is_floating_point() and not bool(torch.isfinite(buffer).all()):
                location = path or "<root>"
                raise CalibrationRefused(
                    f"static int8 calibration observer {location} buffer {name} is nonfinite",
                    details=dict(
                        observer_path=location,
                        observer_type=type(observer).__qualname__,
                        observer_buffer=name,
                        calibration_sample=calibration_sample,
                        nonfinite_count=int((~torch.isfinite(buffer)).sum()),
                    ),
                )


def require_finite_calibration(value):
    import torch

    if isinstance(value, torch.Tensor):
        if value.is_complex():
            error = CalibrationRefused("static int8 calibration does not admit complex tensors")
            error.code = "complex_calibration"
            raise error
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise CalibrationRefused("static int8 calibration input or observer value is nonfinite")
    elif isinstance(value, Mapping):
        for item in value.values():
            require_finite_calibration(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            require_finite_calibration(item)
    elif isinstance(value, complex):
        error = CalibrationRefused("static int8 calibration does not admit complex scalar arguments")
        error.code = "complex_calibration"
        raise error


def _require(recipe: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    if recipe.get("schema") != RECIPE_SCHEMA:
        raise RecipeError(f"unsupported recipe schema {recipe.get('schema')!r}")
    if recipe.get("status") != "derived":
        raise RecipeError(f"the recipe is {recipe.get('status')!r}, not derived: {recipe.get('underivable')}")
    weight, activation = recipe.get("weight"), recipe.get("activation")
    if not isinstance(weight, Mapping) or not isinstance(activation, Mapping):
        raise RecipeError("a derived recipe carries a weight and an activation spec")
    return weight, activation


def _torch_dtype(name: str):
    import torch

    spelled = _TORCH_DTYPE.get(str(name))
    dtype = getattr(torch, spelled, None) if spelled else None
    if dtype is None:
        raise RecipeError(f"recipe dtype {name!r} has no torch dtype in this build")
    return dtype


def _observer_name(tensor: Mapping[str, Any], *, is_weight: bool) -> str:
    floating_format = str(tensor.get("dtype", "")).startswith("fp8_")
    default = "minmax" if is_weight or floating_format else "histogram"
    return str(tensor.get("observer") or default)


def _tensor_observer_class(tensor, *, is_weight):
    """One observer choice shared by PT2E annotations and static case calibration."""
    try:
        from torchao.quantization.pt2e import HistogramObserver, MinMaxObserver
    except ImportError:
        from torch.ao.quantization.observer import HistogramObserver, MinMaxObserver
    name = _observer_name(tensor, is_weight=is_weight)
    if str(tensor.get("dtype", "")).startswith("fp8_") and name == "histogram":
        raise RecipeError("FP8 histogram observation is unavailable in TorchAO PT2E; use minmax")
    observer = {"minmax": MinMaxObserver, "histogram": HistogramObserver}.get(name)
    if observer is None:
        raise RecipeError(f"recipe observer {tensor.get('observer')!r} is not one this quantizer builds")
    return observer


def observe_static_tensor(value, tensor_spec, *, eps=DEFAULT_OBSERVER_EPSILON):
    """Reuse the static recipe observer rule for an explicitly observed tensor input.

    ATen contraction operands are both activations; neither is asserted to be a
    stored model weight. PyTorch's observer classes implement the same PT2E rule
    in capture environments without the optional TorchAO package.
    """
    import torch

    if (
        tensor_spec.get("mode") != "static"
        or tensor_spec.get("granularity") != "tensor"
        or not tensor_spec.get("symmetric")
    ):
        raise RecipeError("requires static symmetric per-tensor recipe")
    if not torch.isfinite(value).all():
        raise CalibrationRefused("calibration input is nonfinite")
    name = _observer_name(tensor_spec, is_weight=False)
    observer_class = _tensor_observer_class(tensor_spec, is_weight=False)
    observer = observer_class(
        dtype=_torch_dtype(tensor_spec["dtype"]),
        qscheme=torch.per_tensor_symmetric,
        quant_min=tensor_spec["quant_min"],
        quant_max=tensor_spec["quant_max"],
        eps=eps,
    )
    observer(value)
    scale, zero_point = observer.calculate_qparams()
    if zero_point.item() != 0 or not torch.isfinite(scale).all() or scale.item() <= 0:
        raise RecipeError("observer did not produce finite positive symmetric qparams")
    return scale.to(torch.float32).reshape(()), name


def _spec(tensor: Mapping[str, Any], *, is_weight: bool, eps: float):
    """One ``QuantizationSpec`` from one recipe tensor spec."""
    import torch
    from torchao.quantization.pt2e import PerChannelMinMaxObserver
    from torchao.quantization.pt2e.quantizer import QuantizationSpec

    granularity, symmetric = tensor.get("granularity"), bool(tensor.get("symmetric"))
    dtype = _torch_dtype(tensor.get("dtype"))
    if granularity == "tensor":
        qscheme = torch.per_tensor_symmetric if symmetric else torch.per_tensor_affine
        # A weight is observed once, so its range is its min and max; an activation is observed
        # over a calibration stream, where a histogram is robust to a rare outlier. TorchAO's
        # HistogramObserver uses torch.iinfo and cannot observe an FP8 dtype; its MinMaxObserver
        # explicitly supports float8. Never silently replace an explicitly requested observer.
        observer = _tensor_observer_class(tensor, is_weight=is_weight)
        axis = None
    elif granularity == "channel" and is_weight:
        qscheme = torch.per_channel_symmetric if symmetric else torch.per_channel_affine
        observer, axis = PerChannelMinMaxObserver, 0
    else:
        raise RecipeError(
            f"a static {'weight' if is_weight else 'activation'} scale per "
            f"{granularity!r} has no PT2E spec; the recipe asks for something this "
            f"path cannot express"
        )
    return QuantizationSpec(
        dtype=dtype,
        observer_or_fake_quant_ctr=observer.with_args(eps=eps),
        quant_min=tensor.get("quant_min"),
        quant_max=tensor.get("quant_max"),
        qscheme=qscheme,
        ch_axis=axis,
    )


def _recipe_operators(torch: Any) -> dict[str, tuple[Any, ...]]:
    """ATen spellings covered by each recipe family, shared by the skip check and PT2E."""
    #: The aten operators that ARE the recipe's families. A family the recipe does not list is not
    #: annotated and stays in floating point, which is what "the target cannot absorb it" means.
    operators = {
        # A MATMUL IS A CONTRACTION. This listed `conv2d` and `linear` only, which are the spellings a
        # module built from `nn.Conv2d`/`nn.Linear` exports to -- and a model that calls
        # `torch.matmul` directly exports to `aten.matmul.default` and matched nothing, so the
        # quantizer annotated zero contractions and `validate` rejected the model as containing no
        # operator the recipe covers. Measured on `M3_host_island_seam_gemmini`, whose exported graph
        # is exactly `2x aten.matmul.default`: it is the accelerator/host-island/accelerator seam
        # capsule, so the one capsule built to prove that composition could not be captured at all.
        #
        # `mm` and `bmm` are listed beside it because which of the three survives export is a property
        # of the input ranks and of whatever decomposition ran, not of the model's intent -- matching
        # one spelling and not its siblings is the same bug with a different rank.
        "contraction": (
            torch.ops.aten.conv2d.default,
            torch.ops.aten.linear.default,
            torch.ops.aten.matmul.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
            torch.ops.aten.convolution.default,
            # The backward of a convolution is three contractions over the same three int8 tensors
            # (grad_input, grad_weight and the grad_bias sum); the frontend integerizes it as such.
            torch.ops.aten.convolution_backward.default,
        ),
        "operand_sum": (torch.ops.aten.add.Tensor, torch.ops.aten.add_.Tensor),
        "window_mean": (
            torch.ops.aten.adaptive_avg_pool2d.default,
            torch.ops.aten.avg_pool2d.default,
            torch.ops.aten.mean.dim,
            torch.ops.aten.avg_pool1d.default,
            torch.ops.aten.avg_pool3d.default,
            torch.ops.aten.adaptive_avg_pool1d.default,
            torch.ops.aten.adaptive_avg_pool3d.default,
            torch.ops.aten._adaptive_avg_pool2d.default,
            torch.ops.aten._adaptive_avg_pool3d.default,
            # Their transposes spread each window's (or patch's) value over the positions it covers:
            # the same sums, read out once (runtime.integer_ops.window_sum).
            torch.ops.aten.avg_pool2d_backward.default,
            torch.ops.aten._adaptive_avg_pool2d_backward.default,
            torch.ops.aten.col2im.default,
        ),
        # An affine map of one or two int8 tensors: integer weights and an int32 constant applied in
        # the accumulator, then ONE readout (merlin.targetgen.recipe_numerics). Its result has its
        # own scale, so its output edge is quantized here.
        "affine_map": (
            torch.ops.aten.add.Scalar,
            torch.ops.aten.sub.Scalar,
            torch.ops.aten.mul.Scalar,
            torch.ops.aten.div.Scalar,
            torch.ops.aten.neg.default,
            torch.ops.aten.sub.Tensor,
            torch.ops.aten.relu.default,
            # Export keeps a python scalar operand on the .Tensor overload until decomposition.
            torch.ops.aten.add.Tensor,
            torch.ops.aten.mul.Tensor,
            torch.ops.aten.div.Tensor,
        ),
        # Data motion: no arithmetic, so (by default) its output shares its input's quantization
        # parameters; a fill writes a constant code on its own output grid.
        "movement": (
            torch.ops.aten.clone.default,
            torch.ops.aten.permute.default,
            torch.ops.aten.cat.default,
            torch.ops.aten.repeat.default,
            torch.ops.aten.constant_pad_nd.default,
            torch.ops.aten.pad.default,
            torch.ops.aten.reflection_pad1d.default,
            torch.ops.aten.reflection_pad2d.default,
            torch.ops.aten.reflection_pad3d.default,
            torch.ops.aten.replication_pad2d.default,
            torch.ops.aten.replication_pad3d.default,
            torch.ops.aten.slice_scatter.default,
            torch.ops.aten.select_scatter.default,
            torch.ops.aten.full_like.default,
            torch.ops.aten.fill.Scalar,
            torch.ops.aten.full.default,
        ),
    }
    return operators


#: Movement operators whose output is a constant, not a copy of an input.
_CONSTANT_MOVEMENT = ("full_like.default", "fill.Scalar", "full.default")


#: Floating widths a quantized graph reads a result out at, other than the recipe's own float32 grid.
_READOUT_WIDTHS = ("float64", "float16", "bfloat16")


def read_out_floating_widths(graph_module: Any, moves: set[Any]) -> list[dict[str, Any]]:
    """Move a float dtype conversion or a non-float32 fill to the readout of a quantized graph.

    On the int8 grid a value is its code and its scale; its floating WIDTH is chosen only when it is
    dequantized. So a ``_to_copy`` that changes nothing but the float dtype is, in the quantized
    domain, a copy of the codes, and a ``full``/``full_like`` of a float64 constant is a constant fill
    of codes on its output grid. PT2E quantizes float32 edges only, so each such operation is spelt
    at float32 here -- a conversion as a ``clone`` (the copy it is on codes), a fill with
    ``dtype=float32``. After PT2E conversion, the requested width is put on the final
    dequantization itself; a float32 dequantization followed by widening would lose bits.

    Only operations the recipe's movement family covers move, and only when every tensor input is
    float32 (a float64 INPUT cannot be quantized). Returns one record per rewritten node, which the
    quantization stats carry as the rewrite's provenance (a re-export drops node metadata).
    """
    import torch

    aten = torch.ops.aten
    widths = {getattr(torch, name) for name in _READOUT_WIDTHS}
    graph = graph_module.graph
    rewritten = []
    for node in list(graph.nodes):
        value = node.meta.get("val")
        if node.op != "call_function" or not isinstance(value, torch.Tensor) or value.dtype not in widths:
            continue
        if node.target in (aten._to_copy.default, aten.to.dtype) and aten.clone.default in moves:
            # _to_copy(x, *, dtype, ...) after decomposition; to.dtype(x, dtype, ...) before it.
            source = node.args[0] if node.args else None
            positional = node.args[2:] if node.target == aten.to.dtype else node.args[1:]
            extra = [v for key, v in node.kwargs.items() if key != "dtype"]
            if (
                not isinstance(source, torch.fx.Node)
                or not isinstance(source.meta.get("val"), torch.Tensor)
                or source.meta["val"].dtype != torch.float32
                or any(v not in (None, False, torch.preserve_format, torch.strided) for v in [*positional, *extra])
            ):
                continue
            with graph.inserting_before(node):
                computed = graph.call_function(aten.clone.default, args=(source,))
            computed.meta = {**node.meta, "val": source.meta["val"].clone()}
        elif node.target in moves and str(node.target).endswith(("full_like.default", "full.default")):
            if node.kwargs.get("dtype") != value.dtype:
                continue
            kwargs = {**node.kwargs, "dtype": torch.float32}
            with graph.inserting_before(node):
                computed = graph.call_function(node.target, args=node.args, kwargs=kwargs)
            args = torch.fx.map_arg(node.args, lambda n: n.meta["val"])
            computed.meta = {**node.meta, "val": node.target(*args, **kwargs)}
        else:
            continue
        computed.meta["_m2m_source_aten"] = str(node.target)
        with graph.inserting_after(computed):
            readout = graph.call_function(aten._to_copy.default, args=(computed,), kwargs={"dtype": value.dtype})
        readout.meta = {**node.meta, "val": computed.meta["val"].to(value.dtype), "merlin_readout_width": True}
        node.replace_all_uses_with(readout)
        graph.erase_node(node)
        rewritten.append(
            {
                "node": node.name,
                "operator": str(node.target),
                "quantized_as": str(computed.target),
                "readout_dtype": str(value.dtype).removeprefix("torch."),
            }
        )
    if rewritten:
        graph.lint()
        graph_module.recompile()
    return rewritten


def restore_floating_readout_widths(graph_module: Any) -> None:
    """Dequantize directly at a requested result width, without an intermediate f32 rounding.

    Clone the boundary for this reader: other users may still require the f32 grid.
    This applies only to readout conversions introduced by this recipe.
    """
    import torch

    graph = graph_module.graph
    for node in list(graph.nodes):
        if not node.meta.get("merlin_readout_width"):
            continue
        source = node.args[0]
        if not isinstance(source, torch.fx.Node) or not str(source.target).startswith(
            "quantized_decomposed.dequantize_per_"
        ):
            raise NoEligibleRecipeOperations("floating readout width requires a direct dequantization boundary")
        kwargs = {**source.kwargs, "out_dtype": node.kwargs["dtype"]}
        with graph.inserting_before(node):
            readout = graph.call_function(source.target, args=source.args, kwargs=kwargs)
        readout.meta = {**node.meta}
        readout.meta.pop("tensor_meta", None)
        node.replace_all_uses_with(readout)
        graph.erase_node(node)
    graph.eliminate_dead_code()
    graph.lint()
    graph_module.recompile()


def _has_floating_recipe_work(exported: Any, recipe: Mapping[str, Any]) -> bool:
    """Whether an integral-input graph still computes a floating recipe operation inside."""
    import torch

    operators = _recipe_operators(torch)
    targets = {op for family in recipe.get("families") or () for op in operators.get(family, ())}
    return any(
        node.op == "call_function"
        and node.target in targets
        and isinstance((value := node.meta.get("val")), torch.Tensor)
        and value.is_floating_point()
        for node in exported.graph.nodes
    )


def _source_weight_geometry_matches(
    source_layers: list[dict[str, Any]], owner: str, observed_shape: list[int] | None
) -> bool:
    """A staged weight can inherit only its exact, unique source module's placement."""
    matching = [row for row in source_layers if row.get("fqn") == owner]
    return bool(
        len(matching) == 1 and isinstance(observed_shape, list) and matching[0].get("weight_shape") == observed_shape
    )


def build_quantizer(
    recipe: Mapping[str, Any],
    *,
    layer_plan: Mapping[str, Any],
    source_layer_inventory: list[dict[str, Any]] | None = None,
    eps: float = DEFAULT_OBSERVER_EPSILON,
    fold_candidates: list | None = None,
):
    """The PT2E quantizer a static recipe describes, limited to placed operations.

    ATen operator names alone cannot license quantization: an unsupported module may call
    ``aten.linear`` internally. Export's ``nn_module_stack`` identifies the owning module.
    Leaf operations use their module plan; container/root arithmetic uses its own
    functional operation plan. Missing ownership or unknown leaves are refused.
    """
    import torch
    from torchao.quantization.pt2e.quantizer import QuantizationAnnotation, Quantizer

    weight, activation = _require(recipe)
    if activation.get("mode") != "static":
        raise RecipeError("build_quantizer realises static recipes; a dynamic one goes through quantize_config")
    operators = _recipe_operators(torch)
    listed = tuple(recipe.get("families") or ())
    targets = {op for family in listed if family == "contraction" for op in operators.get(family, ())}
    sums = set(operators["operand_sum"]) if "operand_sum" in listed else set()
    means = set(operators["window_mean"]) if "window_mean" in listed else set()
    affine = set(operators["affine_map"]) if "affine_map" in listed else set()
    moves = set(operators["movement"]) if "movement" in listed else set()
    #: shared (default): a movement's output and every input share one scale, as PT2E does for cat;
    #: independent: each input keeps its own scale and the output gets its own (one load rounding each).
    movement_qparams = recipe.get("movement_qparams", "shared")
    if movement_qparams not in {"shared", "independent"}:
        raise RecipeError(f"movement_qparams must be shared or independent, got {movement_qparams!r}")
    #: narrow: the input edge of a negation uses [-qmax, qmax] so -(-128) never occurs.
    negation_range = recipe.get("negation_input_range", "full")
    if negation_range not in {"full", "narrow"}:
        raise RecipeError(f"negation_input_range must be full or narrow, got {negation_range!r}")
    if not targets:
        raise RecipeError(f"the recipe quantizes {recipe.get('families')}, none of which this quantizer can annotate")
    #: What a value may pass through on its way to a reader without ceasing to be the same integers:
    #: an activation a readout applies, and shape changes that touch no element.
    transparent = {
        torch.ops.aten.relu.default,
        torch.ops.aten.relu_.default,
        torch.ops.aten.flatten.using_ints,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.contiguous.default,
        torch.ops.aten.dropout.default,
    }
    # Shape-only views preserve a stored parameter's identity. An arbitrary producer does not:
    # quantizing the second input of an activation x activation matmul as a weight would use the
    # wrong observer and imply a constant operand that the model never supplied.
    weight_views = {
        torch.ops.aten.t.default,
        torch.ops.aten.transpose.int,
        torch.ops.aten.permute.default,
        torch.ops.aten.view.default,
        torch.ops.aten.reshape.default,
        torch.ops.aten._unsafe_view.default,
        torch.ops.aten.contiguous.default,
    }

    decisions = {str(d.get("fqn")): d for d in layer_plan.get("layers") or ()}
    software_decisions = []

    def contraction_operands(node: Any):
        if node.target == torch.ops.aten.convolution_backward.default:
            return node.args[:3]  # grad_output, input, weight
        start = 1 if node.target == torch.ops.aten.addmm.default else 0
        return node.args[start : start + 2]

    def software_allows(node: Any, family: str) -> bool:
        selected = recipe.get("software_admission")
        if selected is None:
            return True  # Compatibility recipes retain their recorded policy.
        from merlin.targetgen.software_spec import admit_contraction_operation, admit_operation

        operands = contraction_operands(node) if family == "contraction" else node.args
        value = operands[0].meta.get("val") if operands and isinstance(operands[0], torch.fx.Node) else None
        prospective = {
            "family": family,
            "numerical_family": "F3",
            "operand_dtype": activation["dtype"],
            "accum_dtype": recipe["accumulator_dtype"],
            "rank": len(value.shape) if isinstance(value, torch.Tensor) else None,
            "layout": "row_major_contiguous" if isinstance(value, torch.Tensor) and value.is_contiguous() else None,
            "scale_granularity": activation["granularity"],
        }
        if family == "contraction":
            prospective.update(
                ordered_operand_dtypes=[
                    activation["dtype"],
                    weight["dtype"] if stored_floating_weight(operands[-1]) else activation["dtype"],
                    recipe["accumulator_dtype"],
                ],
                ordered_result_dtypes=[recipe["accumulator_dtype"]],
                readout_dtype=recipe["accumulator_dtype"],
            )
            decision = admit_contraction_operation(selected, str(node.target), prospective, "accelerator")
        else:
            decision = admit_operation(selected, str(node.target), prospective, "accelerator")
        software_decisions.append(
            {
                "node": node.name,
                "source_operator": str(node.target),
                "prospective_operator": str(node.target),
                "prospective_signature": prospective,
                **decision,
                "scope": "prospective quantization screen; not realized precision or target admission",
            }
        )
        if decision.get("applicable_exact_source_declaration"):
            return decision["status"] == "admitted"
        return decision["status"] != "unsupported"

    def owner(node: Any) -> str | None:
        stack = node.meta.get("nn_module_stack")
        if not isinstance(stack, Mapping) or not stack:
            return None
        paths = [entry[0] for entry in stack.values() if isinstance(entry, (tuple, list)) and entry]
        return str(paths[-1]) if paths else None

    def stored_weight_view(node: Any) -> bool:
        return node.target in weight_views or (
            node.target == torch.ops.aten.select.int
            and len(node.args) == 3
            and all(type(index) is int for index in node.args[1:])
            and not node.kwargs
        )

    def stored_weight_tensor(value: Any):
        while isinstance(value, torch.fx.Node) and value.op == "call_function" and stored_weight_view(value):
            value = value.args[0] if value.args else None
        tensor = value.meta.get("val") if isinstance(value, torch.fx.Node) and value.op == "get_attr" else None
        return tensor if isinstance(tensor, torch.Tensor) else None

    def stored_floating_weight(value: Any) -> bool:
        tensor = stored_weight_tensor(value)
        return tensor is not None and tensor.is_floating_point()

    def stored_weight_shape(value: Any) -> list[int] | None:
        tensor = stored_weight_tensor(value)
        return list(tensor.shape) if isinstance(tensor, torch.Tensor) else None

    def eligible(node: Any, family: str) -> bool:
        if family == "operand_sum" and node.meta.get("merlin_contraction_epilogue"):
            return False
        if not software_allows(node, family):
            return False
        if family == "contraction":
            operands = contraction_operands(node)
            if len(operands) not in (2, 3) or any(
                not isinstance(operand, torch.fx.Node)
                or not isinstance((value := operand.meta.get("val")), torch.Tensor)
                or not value.is_floating_point()
                for operand in operands
            ):
                return False
        path = owner(node)
        if path is None:
            return False
        if path:
            decision = decisions.get(path)
            if (
                source_layer_inventory is not None
                and family == "contraction"
                and stored_floating_weight(contraction_operands(node)[-1])
            ):
                if not _source_weight_geometry_matches(
                    source_layer_inventory, path, stored_weight_shape(contraction_operands(node)[-1])
                ):
                    return False
            if decision is not None:
                return bool(decision.get("placement") == "device" and decision.get("family") == family)
            # The inventory contains leaves only. A container still owns functional
            # arithmetic between its children (for example, a residual add). Prove
            # it is a container in this plan before using the root operation rule.
            if not any(fqn.startswith(path + ".") for fqn in decisions):
                return False
        # Root/container arithmetic has no leaf decision. Check the actual
        # operand geometry with the same functional placement rule.
        if family == "contraction":
            kernel_node = contraction_operands(node)[-1]
            kernel = kernel_node.meta.get("val")
            shape = list(kernel.shape) if kernel is not None and hasattr(kernel, "shape") else None
        else:
            shape = None
        import quant_layer_plan as QLP

        direct = QLP.plan(
            recipe, [{"fqn": "<root>", "kind": "functional", "family": family, "weight_shape": shape}]
        ).decisions[0]
        return direct.placement == QLP.DEVICE

    def family_of(node: Any) -> str | None:
        if node.target in targets:
            return "contraction"
        if node.target in sums and len(tensor_operands(node)) == 2:
            return "operand_sum"
        if node.target in means:
            return "window_mean"
        if node.target in affine and affine_operands_ok(node):
            return "affine_map"
        if node.target in moves:
            return "movement"
        return None

    def tensor_operands(node: Any) -> list[Any]:
        return [a for a in node.args[:2] if isinstance(a, torch.fx.Node) and a.op != "get_attr"]

    def affine_operands_ok(node: Any) -> bool:
        """One tensor and a python scalar, or (a difference) two tensors; a stored operand is not."""
        tensors, scalars = tensor_operands(node), [a for a in node.args[1:2] if isinstance(a, (int, float))]
        if node.target in (torch.ops.aten.neg.default, torch.ops.aten.relu.default):
            return len(tensors) == 1
        if node.target in (torch.ops.aten.sub.Tensor, torch.ops.aten.add.Tensor) and len(tensors) == 2:
            # A two-tensor sum or difference in its exact form: signed rational integer weights in the
            # accumulator and one readout (recipe_numerics.linear_combination).
            return True
        return len(tensors) == 1 and len(scalars) == 1

    def tensor_inputs(node: Any) -> list[Any]:
        """The activation tensors an affine map or movement reads, in argument order."""
        from merlin.targetgen._recipe_integer import data_arguments

        return [
            value
            for value in data_arguments(node)
            if isinstance(value, torch.fx.Node)
            and value.op != "get_attr"
            and isinstance(value.meta.get("val"), torch.Tensor)
            and value.meta["val"].is_floating_point()
        ]

    def fused_behind_contraction(node: Any) -> bool:
        """An epilogue of a contraction is group formation's to absorb, not a standalone site."""
        if node.meta.get("merlin_contraction_epilogue"):
            return True
        return any(isinstance(a, torch.fx.Node) and a.target in targets for a in node.args)

    def closable(nodes: Any) -> set[Any]:
        """The sums and means whose result only quantized readers ever see.

        Putting a sum on the integer grid is worth its accuracy cost only when the result is
        requantized, because that is what closes it into one integer region a unit can take. A sum
        read by a normalization, or returned, stays a float add whose operands were quantized for
        nothing (measured on a two-layer decoder: 8 sums and 5 means annotated, none placeable,
        17 more host regions). A reader may itself be a candidate, so this is a fixpoint: drop
        every candidate with a reader that is neither a contraction nor a surviving candidate.
        """
        candidates = set()
        for node in nodes:
            if node.target in sums and eligible(node, "operand_sum"):
                operands = [a for a in node.args[:2] if isinstance(a, torch.fx.Node) and a.op != "get_attr"]
                if len(operands) == 2:
                    candidates.add(node)
            elif (
                node.target in means
                and eligible(node, "window_mean")
                and node.args
                and isinstance(node.args[0], torch.fx.Node)
            ):
                candidates.add(node)

        def readers(node: Any) -> list[Any]:
            found, pending = [], list(node.users)
            while pending:
                user = pending.pop()
                if user.op == "call_function" and user.target in transparent:
                    pending.extend(user.users)
                else:
                    found.append(user)
            return found

        while True:
            dropped = {
                node
                for node in candidates
                if not (seen := readers(node))
                or any(
                    (r.target not in targets or not eligible(r, "contraction")) and r not in candidates for r in seen
                )
            }
            if not dropped:
                return candidates
            candidates -= dropped

    class CapabilityQuantizer(Quantizer):
        def __init__(self) -> None:
            self.activation = _spec(activation, is_weight=False, eps=eps)
            self.weight = _spec(weight, is_weight=True, eps=eps)
            self.annotated = 0
            self.annotated_sums = 0
            self.annotated_means = 0
            self.annotated_maps = 0
            self.annotated_moves = 0
            if negation_range == "narrow":
                from merlin.targetgen.recipe_numerics import narrow_range

                low, high = narrow_range(int(activation["quant_max"]))
                self.negation_input = _spec(
                    {**activation, "quant_min": low, "quant_max": high}, is_weight=False, eps=eps
                )
            else:
                self.negation_input = self.activation
            self.left_in_float = 0
            self.refused_by_placement: list[dict[str, str | None]] = []
            self.refused_rewrites: list[dict[str, Any]] = []
            self.software_decisions = software_decisions
            self.fold_candidates = fold_candidates or []
            self.readout_widths: list[dict[str, Any]] = []

        def transform_for_annotation(self, graph_module: Any) -> Any:
            # TorchAO calls this after folding Conv+BN and before inserting
            # observers, while the exact FX nodes and their rewired users are
            # still available to prove each folded source identity.
            if self.fold_candidates:
                from m2m.capture.trace import attach_pt2e_conv_bn_folds

                attach_pt2e_conv_bn_folds(graph_module, self.fold_candidates)
            if moves:
                self.readout_widths = read_out_floating_widths(graph_module, moves)
            # Preserve addmm as one exact readout boundary: decomposing its alpha/beta
            # epilogue would round the MAC and scaled bias before their final sum.
            # Normalize only the convolution spelling required by the frontend.
            graph = graph_module.graph
            for node in list(graph.nodes):
                if node.target != torch.ops.aten.convolution.default:
                    continue
                if not eligible(node, "contraction"):
                    continue
                previous = set(graph.nodes)
                with graph.inserting_before(node):
                    if node.target == torch.ops.aten.convolution.default:
                        input_value = node.args[0].meta.get("val")
                        if not isinstance(input_value, torch.Tensor) or input_value.dim() != 4:
                            self.refused_rewrites.append(
                                {
                                    "node": node.name,
                                    "operator": str(node.target),
                                    "reason": "convolution to conv2d requires a rank-4 input",
                                }
                            )
                            node.meta["quantization_refusal"] = self.refused_rewrites[-1]["reason"]
                            continue
                        if len(node.args) != 9 or node.args[6] or any(node.args[7]):
                            continue  # Transposed/output-padding semantics lack this rewrite.
                        value = graph.call_function(torch.ops.aten.conv2d.default, args=(*node.args[:6], node.args[8]))
                    # Compute metadata from each operation's own operands. A
                    # scaled bias keeps its broadcast shape and no module owner.
                    for created in graph.nodes:
                        if created in previous:
                            continue
                        if created.target in (torch.ops.aten.mm.default, torch.ops.aten.conv2d.default):
                            created.meta = dict(node.meta)
                        else:
                            created.meta = {"merlin_contraction_epilogue": True}
                        created.meta["merlin_normalized_from"] = node.name
                        args = torch.fx.map_arg(created.args, lambda n: n.meta["val"])
                        kwargs = torch.fx.map_arg(created.kwargs, lambda n: n.meta["val"])
                        created.meta["val"] = created.target(*args, **kwargs)
                node.replace_all_uses_with(value)
                graph.erase_node(node)
            graph.lint()
            graph_module.recompile()
            return graph_module

        def annotate(self, graph_module: Any) -> Any:
            nodes = list(graph_module.graph.nodes)
            kept = closable(nodes)
            self.left_in_float = sum(1 for n in nodes if (n.target in sums or n.target in means) and n not in kept)
            for node in nodes:
                if node.meta.get("quantization_refusal"):
                    continue
                family = family_of(node)
                if family is not None and not eligible(node, family):
                    self.refused_by_placement.append(
                        {"node": node.name, "operator": str(node.target), "owner": owner(node), "family": family}
                    )
                    continue
                if (node.target in sums or node.target in means) and node not in kept:
                    # No quantized reader closes it. Where the recipe licenses a standalone readout
                    # (affine_map), the sum or mean reads itself out: its output edge is quantized.
                    if "affine_map" in listed:
                        self._annotate_standalone(node, "operand_sum" if node.target in sums else "window_mean")
                    continue
                if node.target in sums:
                    # A sum of two TENSORS, both operands on the recipe's activation grid. A scalar
                    # or a parameter operand is a different operation (a bias, a shift) and is left
                    # alone. The sum's result stays in floating point like a contraction's: whoever
                    # reads it quantizes its own input edge, and group formation closes the region.
                    operands = [a for a in node.args[:2] if isinstance(a, torch.fx.Node) and a.op != "get_attr"]
                    if len(operands) == 2:
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={operand: self.activation for operand in operands}, _annotated=True
                        )
                        self.annotated_sums += 1
                    continue
                if node.target in means:
                    # Only the input: a mean has no stored operand, and its reader quantizes its own edge.
                    if node.args and isinstance(node.args[0], torch.fx.Node):
                        node.meta["quantization_annotation"] = QuantizationAnnotation(
                            input_qspec_map={node.args[0]: self.activation}, _annotated=True
                        )
                        node.meta["merlin_recipe_family"] = "window_mean"
                        self.annotated_means += 1
                    continue
                if family in {"affine_map", "movement"}:
                    self._annotate_map(node, family)
                    continue
                if node.target not in targets or len(node.args) < 2:
                    continue
                *operands, kernel = contraction_operands(node)
                # Both inputs of the contraction; its result stays in floating point and the next
                # contraction quantizes its own input edge. What follows a contraction is absorbed
                # by the compiler's group formation, not assumed here.
                node.meta["quantization_annotation"] = QuantizationAnnotation(
                    input_qspec_map={
                        **{operand: self.activation for operand in operands},
                        kernel: self.weight if stored_floating_weight(kernel) else self.activation,
                    },
                    _annotated=True,
                )
                self.annotated += 1
            # Closed sum/mean candidates form a set. Its fixpoint can revisit a
            # reader in any object-hash order, so keep every decision (including
            # repeated screens) but expose the complete report rows canonically.
            # Otherwise equivalent captures disagree only in metadata and fail
            # byte-exact independent replay.
            self.software_decisions.sort(
                key=lambda row: json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False)
            )
            return graph_module

        def _annotate_standalone(self, node: Any, family: str) -> None:
            """A sum or mean that reads itself out: its tensor inputs and its own output scale."""
            if family == "operand_sum":
                operands = [a for a in node.args[:2] if isinstance(a, torch.fx.Node) and a.op != "get_attr"]
                if len(operands) != 2:
                    return
            else:
                operands = [node.args[0]] if node.args and isinstance(node.args[0], torch.fx.Node) else []
                if not operands:
                    return
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map={operand: self.activation for operand in operands},
                output_qspec=self.activation,
                _annotated=True,
            )
            node.meta["merlin_recipe_family"] = family
            if family == "operand_sum":
                self.annotated_sums += 1
            else:
                self.annotated_means += 1
            self.left_in_float -= 1

        def _annotate_map(self, node: Any, family: str) -> None:
            """Input and output edges of a standalone affine map or movement."""
            from torchao.quantization.pt2e.quantizer import SharedQuantizationSpec

            if fused_behind_contraction(node):
                return
            inputs = tensor_inputs(node)
            constant = family == "movement" and str(node.target).endswith(_CONSTANT_MOVEMENT)
            if not inputs and not constant:
                return
            value = node.meta.get("val")
            if not isinstance(value, torch.Tensor) or value.dtype != torch.float32:
                # PT2E quantizes float32 edges only; a float64 (or integer) result stays as declared.
                return
            if family == "affine_map":
                spec = self.negation_input if node.target == torch.ops.aten.neg.default else self.activation
                input_map = {operand: spec for operand in inputs}
                output = self.activation
            elif constant:
                input_map, output = {}, self.activation
            elif movement_qparams == "independent":
                input_map = {operand: self.activation for operand in inputs}
                output = self.activation
            else:
                first = inputs[0]
                input_map = {first: self.activation}
                input_map.update({operand: SharedQuantizationSpec((first, node)) for operand in inputs[1:]})
                output = SharedQuantizationSpec((first, node))
            node.meta["quantization_annotation"] = QuantizationAnnotation(
                input_qspec_map=input_map, output_qspec=output, _annotated=True
            )
            node.meta["merlin_recipe_family"] = family
            if family == "affine_map":
                self.annotated_maps += 1
            else:
                self.annotated_moves += 1

        def validate(self, graph_module: Any) -> None:
            del graph_module
            # COUNT EVERY FAMILY THE RECIPE LISTS, not just contractions. Three counters are kept and
            # this read one, so a model carrying only the operand sums or window means a recipe also
            # covers was refused with a message saying its families were not covered -- which was
            # false, and pointed a reader at the model rather than at this table.
            if (
                self.annotated + self.annotated_sums + self.annotated_means + self.annotated_maps + self.annotated_moves
                == 0
            ):
                raise NoEligibleRecipeOperations(
                    f"the model has no operator the recipe's families cover "
                    f"(families={list(listed)}; annotated contractions=0, sums=0, means=0, maps=0, moves=0)"
                )

    return CapabilityQuantizer()


def quantize_config(recipe: Mapping[str, Any]):
    """The ``quantize_`` configuration a dynamic recipe describes."""
    from torchao.quantization import (
        Float8DynamicActivationFloat8WeightConfig,
        Int8DynamicActivationInt8WeightConfig,
        PerRow,
        PerTensor,
    )

    weight, activation = _require(recipe)
    if activation.get("mode") != "dynamic":
        raise RecipeError("quantize_config realises dynamic recipes")
    granularity = {"tensor": PerTensor, "channel": PerRow}.get(str(weight.get("granularity")))
    if granularity is None:
        raise RecipeError(
            f"a dynamic recipe with weights per {weight.get('granularity')!r} has no configuration in this build"
        )
    dtype = str(weight.get("dtype"))
    if dtype == "int8":
        return Int8DynamicActivationInt8WeightConfig(granularity=granularity())
    if dtype in ("fp8_e4m3", "fp8_e5m2"):
        return Float8DynamicActivationFloat8WeightConfig(granularity=granularity())
    raise RecipeError(f"no dynamic configuration for recipe dtype {dtype!r}")


#: The module kinds ``quantize_`` actually transforms under the configs :func:`quantize_config`
#: builds. This is a property of the FRAMEWORK BUILD, not of any target: TorchAO's handlers for the
#: dynamic-activation configs replace a Linear's weight parameter, and nothing else's. A device
#: layer of any other kind is mapped to ``None`` with that named as the reason, so the plan's
#: hardware verdict is never quietly rewritten into a framework one.
_QUANTIZE_TRANSFORMS = ("Linear", "LazyLinear")
FRAMEWORK_CANNOT_EXPRESS = "framework_cannot_express"


def layer_inventory(model: Any) -> list[dict[str, Any]]:
    """The layers of ``model`` as plain mappings :mod:`quant_layer_plan` can decide on.

    Leaf modules only, and no decision is taken here: this reports the model's own shape (name,
    class, stored-operand shape) and nothing else, so that the placement rule stays in the one
    module that is readable from both sides of the capture boundary and testable without torch.
    """
    layers: list[dict[str, Any]] = []
    for fqn, module in model.named_modules():
        if not fqn or any(True for _ in module.children()):
            continue  # container-owned functional arithmetic is planned per operation
        weight = getattr(module, "weight", None)
        layers.append(
            {
                "fqn": fqn,
                "kind": type(module).__name__,
                "weight_shape": (list(weight.shape) if weight is not None and hasattr(weight, "shape") else None),
                "stores_operand": any(True for _ in module.named_parameters(recurse=False))
                or any(True for _ in module.named_buffers(recurse=False)),
            }
        )
    return layers


def build_fqn_config(recipe: Mapping[str, Any], layer_plan: Mapping[str, Any]):
    """TorchAO's per-layer configuration for ``layer_plan``, as its own ``AOBaseConfig`` subclass.

    Returns ``(config, notes)``. ``config`` maps each layer's fully qualified name to the
    configuration the recipe describes, or to ``None`` where the layer is not quantized — which is
    TorchAO's own spelling for "leave this module alone", so nothing here forks its behaviour.
    ``notes`` is one entry per host layer saying which limit refused it and in whose words.

    Every layer the plan saw appears in the mapping, including the refused ones. An absent key and
    a ``None`` key behave the same way today; recording ``None`` explicitly is what makes the
    refusal reviewable instead of inferable from a gap in a dict.
    """
    from collections import OrderedDict

    try:  # its current name, with the one it shipped under before as the fallback
        from torchao.quantization import FqnToConfig as _FqnToConfig
    except ImportError:  # pragma: no cover - older builds
        from torchao.quantization import ModuleFqnToConfig as _FqnToConfig

    shared = quantize_config(recipe)
    mapping: OrderedDict[str, Any] = OrderedDict()
    notes: list[dict[str, Any]] = []
    for layer in layer_plan.get("layers") or ():
        fqn, kind = str(layer.get("fqn") or ""), str(layer.get("kind") or "")
        if layer.get("placement") != "device":
            mapping[fqn] = None
            notes.append(
                {"fqn": fqn, "kind": kind, "refusal": layer.get("refusal"), "why": layer.get("why"), "by": "target"}
            )
            continue
        if kind not in _QUANTIZE_TRANSFORMS:
            mapping[fqn] = None
            notes.append(
                {
                    "fqn": fqn,
                    "kind": kind,
                    "refusal": FRAMEWORK_CANNOT_EXPRESS,
                    "why": (
                        f"the target absorbs this layer ({layer.get('why')}), but quantize_ transforms "
                        f"{list(_QUANTIZE_TRANSFORMS)} under this configuration and not a {kind!r}"
                    ),
                    "by": "framework",
                }
            )
            continue
        mapping[fqn] = shared
    if not any(value is not None for value in mapping.values()):
        raise RecipeError(
            f"no layer of this model is both absorbed by the target and expressible by this framework build: {notes}"
        )
    return _FqnToConfig(mapping), notes


def _plan_layers(
    recipe: Mapping[str, Any], model: Any, *, source_layer_inventory: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """The per-layer plan for ``model`` under ``recipe``, decided by the rule both sides share.

    The rule lives in :mod:`quant_layer_plan`, which imports neither merlin nor a framework, so the
    capture venv reads the same module the derivation side tests. It sits beside this file and is
    imported as a sibling by bare name, the way this file is itself imported.
    """
    import sys
    from pathlib import Path

    here = str(Path(__file__).resolve().parent)
    if here not in sys.path:
        sys.path.insert(0, here)
    import quant_layer_plan as QLP

    return QLP.plan(
        recipe, layer_inventory(model) if source_layer_inventory is None else source_layer_inventory
    ).to_dict()


def _as_tuple(sample: Any) -> tuple:
    return tuple(sample) if isinstance(sample, (tuple, list)) else (sample,)


def _no_batch_norm_fold_needed(original_snapshot: Any, exported: Any) -> bool:
    """Prove the selected original and PT2E-input graphs contain no BN to fold.

    A missing upstream fold-lineage API is harmless only for complete, plain
    ATen graphs with no BatchNorm call. Unknown/custom calls cannot establish
    absence, even when their names do not mention BatchNorm.
    """
    if not isinstance(original_snapshot, Mapping):
        return False
    if original_snapshot.get("schema") != "m2m.frontend_graph.v1" or original_snapshot.get("status") != "complete":
        return False
    nodes = original_snapshot.get("nodes")
    if not isinstance(nodes, list):
        return False
    for node in nodes:
        if not isinstance(node, Mapping):
            return False
        if node.get("op") not in {"call_function", "call_method", "call_module"}:
            continue
        target = node.get("target")
        if node.get("classification") != "aten" or not isinstance(target, str):
            return False
        if "batch_norm" in target.lower() or "batchnorm" in target.lower():
            return False
    graph = getattr(exported, "graph", None)
    if graph is None:
        return False
    for node in graph.nodes:
        if node.op not in {"call_function", "call_method", "call_module"}:
            continue
        if node.op == "call_module" and node.target == "_guards_fn":
            # torch.export.module() injects a shape-guard submodule after the
            # original ATen snapshot. The selected framework's GuardsFn does
            # not perform model computation and cannot hide a BN call.
            get_submodule = getattr(exported, "get_submodule", None)
            guard = get_submodule("_guards_fn") if callable(get_submodule) else None
            guard_type = type(guard)
            if guard_type.__module__ == "torch.export._unlift" and guard_type.__qualname__ == "GuardsFn":
                continue
            return False
        target = str(node.target)
        if node.op != "call_function" or not target.startswith("aten."):
            return False
        if "batch_norm" in target.lower() or "batchnorm" in target.lower():
            return False
    return True


def _normalize_conv_padding(exported, padding_operator) -> list[dict]:
    """Require the frontend's optional rewrite only for a graph that needs it."""
    if not any(node.op == "call_function" and node.target == padding_operator for node in exported.graph.nodes):
        return []
    try:
        from m2m.capture.pt2e_padding import normalize_conv2d_padding
    except ModuleNotFoundError as exc:
        if exc.name != "m2m.capture.pt2e_padding":
            raise
        raise RecipeError("selected frontend cannot normalize Conv2d string padding") from exc
    return normalize_conv2d_padding(exported)


def _allow_exported_model_train_eval(quantized) -> None:
    """Use the public framework helper when the selected frontend predates its shim."""
    try:
        from m2m.capture.pt2e_compat import allow_exported_model_train_eval
    except (ImportError, AttributeError):
        try:
            from torchao.quantization.pt2e import allow_exported_model_train_eval
        except (ImportError, AttributeError):
            from torch.ao.quantization import allow_exported_model_train_eval
    allow_exported_model_train_eval(quantized)


def apply_recipe(
    model: Any,
    recipe: Mapping[str, Any],
    *,
    example_inputs: tuple,
    calibration_inputs: Iterable[Any] | None = None,
    calibration_samples: int = 100,
    original_frontend_snapshot: dict[str, Any] | None = None,
    source_layer_inventory: list[dict[str, Any]] | None = None,
) -> Any:
    """Quantize ``model`` under ``recipe`` and return the module the capture should lower."""
    import torch

    _weight, activation = _require(recipe)
    # Integral EXTERNAL inputs are not proof that a model is already quantized: a language model
    # accepts integer token IDs, then performs floating contractions after an embedding. Inspect
    # the operations covered by this recipe before skipping calibration. A graph with only integer
    # contractions (such as an authored integer seam) still needs no observer.
    tensors = [t for t in (example_inputs or ()) if isinstance(t, torch.Tensor)]
    integral_inputs = bool(tensors) and not any(t.is_floating_point() or t.is_complex() for t in tensors)

    def skip_already_integer() -> Any:
        model._recipe_quantization_stats = {  # type: ignore[attr-defined]
            "applied": False,
            "why": (
                f"example inputs are integral ({sorted({str(t.dtype) for t in tensors})}) and no "
                "floating-point operation covered by the recipe needs quantization"
            ),
        }
        return model

    if activation.get("mode") == "dynamic":
        if recipe.get("software_admission") is not None:
            raise RecipeError("scoped SW admission is currently supported by static PT2E recipes only")
        # Dynamic TorchAO transforms floating Linear modules rather than observing the external
        # input. Integer token IDs must not suppress those internal projections either.
        if integral_inputs and not any(
            isinstance(module, (torch.nn.Linear, torch.nn.LazyLinear))
            and (weight := getattr(module, "weight", None)) is not None
            and torch.is_floating_point(weight)
            for module in model.modules()
        ):
            return skip_already_integer()
        from torchao.quantization import quantize_

        layer_plan = _plan_layers(recipe, model)
        config, notes = build_fqn_config(recipe, layer_plan)
        # filter_fn=None is REQUIRED, not tidiness: quantize_ defaults it to "is this a Linear",
        # and TorchAO refuses to hold both that and a per-layer mapping. Which layers are
        # quantized is the plan's verdict, derived from the target; being a Linear is not a reason.
        quantize_(model, config, filter_fn=None)
        placed = sum(1 for value in config.fqn_to_config.values() if value is not None)
        model._recipe_quantization_stats = {  # type: ignore[attr-defined]
            "api": "quantize_",
            "recipe_sha256": recipe.get("recipe_sha256"),
            "plan_sha256": layer_plan.get("plan_sha256"),
            "layers_seen": len(config.fqn_to_config),
            "layers_quantized": placed,
            # Every layer that is NOT in the target's numeric form, and whose limit put it there.
            "layers_on_host": notes,
            "refusals": layer_plan.get("refusals"),
            "weight_granularity": recipe["weight"]["granularity"],
            "activation_granularity": recipe["activation"]["granularity"],
        }
        return model

    from torchao.quantization.pt2e.quantize_pt2e import convert_pt2e, prepare_pt2e

    if example_inputs is None:
        # An empty tuple is a model with no inputs (a constant fill): it exports and calibrates.
        raise RecipeError("a static recipe needs example inputs to export the model")
    # An ExportedProgram.module() GraphModule may return None from eval(); the
    # owned module, not that method's return value, remains the export input.
    model.eval()
    exported_program = torch.export.export(model, tuple(example_inputs))
    if original_frontend_snapshot is not None:
        from m2m.capture.trace import (
            attach_original_identity,
            snapshot_exported_program,
            tuple_selection_trace_program,
        )

        actual = snapshot_exported_program(exported_program, stage="quantization_input")
        attach_original_identity(exported_program, original_frontend_snapshot, actual)
        # PT2E replays this exact ExportedProgram. Its proxy interpreter may
        # alias tuple selections onto the producer's result proxies; preserve
        # the selecting call-site lineage at that replay, not by guessing from
        # node order or tensor shape after quantization.
        exported_program = tuple_selection_trace_program(exported_program)
    exported = exported_program.module()
    fft_selection = recipe.get("fft_admission_selection")
    if fft_selection is not None:
        from m2m.capture.fft_pt2e import FFT_TARGETS, fft_candidates

        from merlin.targetgen.fft_admission import screen_fft_candidates, screen_fft_component_dtype

        for node in exported_program.graph.nodes:
            if node.op == "call_function" and str(node.target) in FFT_TARGETS:
                source = node.args[0].meta["val"]
                reasons = screen_fft_component_dtype(str(source.dtype), source.ndim, fft_selection)
                if reasons:
                    raise FFTAdmissionRefused("FFT pre-PT2E admission refused: " + "; ".join(reasons))
        try:
            candidates = fft_candidates(exported_program)
        except ValueError as exc:
            raise FFTAdmissionRefused(f"FFT pre-PT2E admission refused: {exc}") from exc
        if candidates:
            sites = screen_fft_candidates(candidates, fft_selection)
            reasons = [reason for site in sites for reason in site["reasons"]]
            if not reasons:
                reasons = ["candidate replacement and source lineage are not qualified"]
            raise FFTAdmissionRefused("FFT pre-PT2E admission refused: " + "; ".join(dict.fromkeys(reasons)))
    # The selected frontend owns the semantic rewrite from string-padding
    # Conv2d to the numeric overload TorchAO observes. Apply it to the exact
    # exported graph before either the integral-input check or PT2E folding:
    # the recipe's operator table must never misread an unnormalized floating
    # Conv as absent. This does not rewrite the source model or its recipe.
    padding_normalizations = _normalize_conv_padding(exported, torch.ops.aten.conv2d.padding)
    if original_frontend_snapshot is not None:
        source_nodes = {row["id"]: row for row in original_frontend_snapshot.get("nodes", ())}
        source_padding_schema = torch.ops.aten.conv2d.padding._schema.arguments
        padding_index = next(
            index for index, argument in enumerate(source_padding_schema) if argument.name == "padding"
        )
        padding_argument = source_padding_schema[padding_index]
        for row in padding_normalizations:
            origins = set(row["source_ids"]).intersection(source_nodes)
            # Precision staging may add intermediate identities. Exactly one
            # original string-padding Conv must own this rewritten node. The
            # selected source spelling must also agree on the literal padding
            # mode: an unrelated same-shaped Conv is not semantic ancestry.
            source = source_nodes[next(iter(origins))] if len(origins) == 1 else {}
            args, kwargs = source.get("args"), source.get("kwargs")
            if isinstance(args, list) and isinstance(kwargs, Mapping) and "padding" not in kwargs:
                source_padding = (
                    args[padding_index]
                    if len(args) > padding_index
                    else padding_argument.default_value
                    if padding_argument.has_default_value()
                    else None
                )
            elif isinstance(args, list) and isinstance(kwargs, Mapping) and len(args) <= padding_index:
                source_padding = kwargs.get("padding")
            else:
                source_padding = None
            if source.get("target") != "aten.conv2d.padding" or source_padding != row["padding"]:
                raise RecipeError("Conv2d string-padding normalization lost original source ancestry")
    if integral_inputs and not _has_floating_recipe_work(exported, recipe):
        return skip_already_integer()
    layer_plan = _plan_layers(recipe, model, source_layer_inventory=source_layer_inventory)
    policy = recipe.get("framework_capture_policy") or {}
    epsilon = policy.get("observer_epsilon", DEFAULT_OBSERVER_EPSILON)
    if type(epsilon) not in (int, float) or not math.isfinite(epsilon) or epsilon <= 0:
        raise RecipeError("observer epsilon must be finite and positive")
    fold_candidates = []
    fold_provenance_api = "not_applicable"
    if original_frontend_snapshot is not None:
        from m2m.capture import trace as capture_trace

        fold_api = getattr(capture_trace, "pt2e_conv_bn_fold_candidates", None)
        if callable(fold_api):
            fold_candidates = fold_api(exported, original_frontend_snapshot)
            fold_provenance_api = "selected_model2mlir"
        elif _no_batch_norm_fold_needed(original_frontend_snapshot, exported):
            # There is no BatchNorm source identity for a PT2E fold to carry.
            # This is an exact graph absence proof, not a family-level alias.
            fold_provenance_api = "not_needed_no_batch_norm"
        else:
            raise RecipeError(
                "selected model2MLIR lacks pt2e_conv_bn_fold_candidates; "
                "cannot prove PT2E Conv+BatchNorm source identity for this graph"
            )
    quantizer = build_quantizer(
        recipe,
        layer_plan=layer_plan,
        source_layer_inventory=source_layer_inventory,
        eps=epsilon,
        fold_candidates=fold_candidates,
    )
    prepared = prepare_pt2e(exported, quantizer)
    # Validate observer inputs before HistogramObserver/MinMaxObserver perform
    # reductions, including nonfinite intermediates from finite caller inputs.
    try:
        from torchao.quantization.pt2e.observer import HistogramObserver, ObserverBase
    except ImportError:
        from torch.ao.quantization.observer import HistogramObserver, ObserverBase

    observer_checks = [
        observer.register_forward_pre_hook(lambda _module, args: require_finite_calibration(args))
        for observer in prepared.modules()
        if isinstance(observer, ObserverBase)
    ]
    samples = calibration_inputs if calibration_inputs is not None else (tuple(example_inputs),)
    calibrated = 0
    try:
        with torch.no_grad():
            for sample in samples:
                if calibrated >= calibration_samples:
                    break
                require_finite_calibration(sample)
                prepared(*_as_tuple(sample))
                calibrated += 1
                require_finite_observer_state(prepared, ObserverBase, HistogramObserver, calibration_sample=calibrated)
    finally:
        for check in observer_checks:
            check.remove()
    if calibrated == 0:
        raise RecipeError("static calibration saw no input sample")
    quantized = convert_pt2e(prepared, use_reference_representation=False, fold_quantize=True)
    scale_changes = []
    if recipe["activation"]["dtype"] in {"int8", "uint8"}:
        # References and integerization must observe the same canonical graph.
        # The frontend's idempotent owner retains every changed scale in its
        # ledger, before either the independent oracle or recipe extraction.
        from m2m.capture.quant_scales import canonicalize_pt2e_scales

        scale_changes = canonicalize_pt2e_scales(quantized)
    restore_floating_readout_widths(quantized)
    if original_frontend_snapshot is not None:
        from m2m.capture.trace import attach_quantization_boundaries

        operators = _recipe_operators(torch)
        selected = {op for family in recipe.get("families") or () for op in operators.get(family, ())}
        attach_quantization_boundaries(quantized, selected)
    pruned = None
    try:  # the capture library's own dead-state sweep, when present
        from m2m.capture.torchao_pipeline import _drop_unused_graph_state

        pruned = _drop_unused_graph_state(quantized)
    except Exception:  # noqa: BLE001 -- an optional tidy-up; the graph is correct without it
        pass
    _allow_exported_model_train_eval(quantized)
    quantized._recipe_quantization_stats = {  # type: ignore[attr-defined]
        "api": "pt2e",
        "recipe_sha256": recipe.get("recipe_sha256"),
        "plan_sha256": layer_plan.get("plan_sha256"),
        "layers_on_host": [d for d in layer_plan.get("layers") or () if d.get("placement") == "host"],
        "operators_refused_by_placement": quantizer.refused_by_placement,
        "rewrites_refused": quantizer.refused_rewrites,
        "annotated_contractions": quantizer.annotated,
        "annotated_sums": quantizer.annotated_sums,
        "annotated_means": quantizer.annotated_means,
        "annotated_affine_maps": quantizer.annotated_maps,
        "annotated_movements": quantizer.annotated_moves,
        # Float dtype conversions and fills spelt at float32 and read out at their requested width.
        "floating_readout_widths": quantizer.readout_widths,
        # The exact integer form of every quantized affine map and movement, from its observed scales.
        "integer_recipes": integer_recipes(quantized),
        "scale_canonicalization": {"dtype": "float32", "changes": scale_changes},
        # Sums and means the recipe's families cover and the graph does not requantize: left alone.
        "sums_and_means_left_in_float": quantizer.left_in_float,
        "calibration_samples": calibrated,
        "pruned_dead_state_tensors": pruned,
        "weight_granularity": recipe["weight"]["granularity"],
        "activation_granularity": recipe["activation"]["granularity"],
        "software_admission": quantizer.software_decisions,
        "fold_provenance_api": fold_provenance_api,
        "conv2d_padding_normalizations": padding_normalizations,
        "framework_capture_policy": {
            "activation_observer": _observer_name(recipe["activation"], is_weight=False),
            "weight_observer": _observer_name(recipe["weight"], is_weight=True),
            "observer_epsilon": epsilon,
            "calibration_source": "explicit_calibration_stream" if calibration_inputs is not None else "example_inputs",
            "calibration_count": calibrated,
        },
    }
    return quantized


def agreement(reference: Any, candidate: Any, samples: Iterable[Any], *, limit: int = 64) -> dict[str, Any]:
    """How a quantized model's outputs compare with the floating-point model's, on real samples.

    A contract change is never an optimisation: moving a weight scale from per-channel to
    per-tensor changes the numbers, and this is the receipt that says by how much. Reported per
    stream, not judged here; the consumer holds it to the workload's own bar.
    """
    import copy

    import torch

    count = agree = empty = 0
    cosines: list[float] = []
    worst = 0.0
    with torch.no_grad():
        for sample in samples:
            if count >= limit:
                break
            inputs = _as_tuple(sample)
            expected = torch.utils._pytree.tree_flatten(reference(*copy.deepcopy(inputs)))[0][0].float().flatten()
            actual = torch.utils._pytree.tree_flatten(candidate(*copy.deepcopy(inputs)))[0][0].float().flatten()
            if expected.shape != actual.shape:
                raise ValueError("agreement requires matching output shapes")
            count += 1
            if expected.numel() == 0:
                empty += 1
                continue
            cosines.append(float(torch.nn.functional.cosine_similarity(expected, actual, dim=0)))
            worst = max(worst, float((expected - actual).abs().max()))
            agree += int(int(expected.argmax()) == int(actual.argmax()))
    if not count:
        return {"samples": 0}
    if empty == count:
        return {"samples": count, "empty_samples": empty}
    return {
        "samples": count,
        "empty_samples": empty,
        "argmax_agreement": agree / (count - empty),
        "cosine_min": min(cosines),
        "cosine_mean": sum(cosines) / count,
        "max_abs_error": worst,
    }
