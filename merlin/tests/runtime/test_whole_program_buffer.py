"""A whole captured model as ONE command buffer, and the closure rule that makes it honest.

The gap this covers. Everything that asks "can this compiler emit THIS MODEL" reads a command
buffer, not a module, and a buffer built from only the layers that happened to route would be an
INCOMPLETE program presented as a whole one -- worse than a refusal, because a consumer cannot tell.
So the buffer is emitted only for a CLOSED model (its one permitted host region is the quantization
of a model argument onto the integer grid), the closure predicate has exactly one definition shared
with the group-model harness, and a model that is not closed is refused with the regions named.
"""

from __future__ import annotations

import json
import pathlib
import sys
from pathlib import Path
from types import SimpleNamespace

import jsonschema
import pytest

from merlin.common import mlir_query as mq
from merlin.common.ir_lock import IR_LOCK
from merlin.common.paths import repo_root
from merlin.llvmlower import whole_program as WP

pytestmark = pytest.mark.target("gemmini")

#: One real target, for the same reason the group-route tests name one: the code under test is
#: target-agnostic (the device is a parameter throughout) but "did this model become one buffer"
#: needs an oracle that actually admits its layers.
_TARGET = "gemmini"


def _binder():
    """The corpus binding every group put to a package is stated under (the example recipe's)."""
    from example_binder import binder

    return binder(_TARGET)


_ID = "affine_map<(d0, d1) -> (d0, d1)>"
_COL = "affine_map<(d0, d1) -> (d1)>"


def _quantize(src: str, out: str, m: int, n: int, dtype: str) -> list[str]:
    return [
        f'    %{out} = "quant_ext.quantize_per_tensor"(%{src}, %s, %z) <{{quant_min = -128 : i64, '
        f'quant_max = 127 : i64, output_dtype = "int8"}}> : (tensor<{m}x{n}x{dtype}>, tensor<f32>, '
        f"tensor<i64>) -> tensor<{m}x{n}xi8>",
    ]


def _layer(tag: str, activation: str, m: int, k: int, n: int, *, bias: str, close: bool) -> list[str]:
    """dq(x), dq(w) -> matmul -> bias -> [relu -> quantize], the fake-quantized form a capture has.

    ``close=False`` leaves the result as the accumulator -- a model's final classifier, whose
    dequantize is the program's readout rather than one of its commands.
    """
    t = tag
    rows = [
        f'    %wd{t} = "quant_ext.dequantize_per_tensor"(%w{t}, %s, %z) <{{quant_min = -127 : i64, '
        f"quant_max = 127 : i64}}> : (tensor<{k}x{n}xi8>, tensor<f32>, tensor<i64>) -> tensor<{k}x{n}xf32>",
        f'    %xd{t} = "quant_ext.dequantize_per_tensor"({activation}, %s, %z) <{{quant_min = -128 : i64, '
        f"quant_max = 127 : i64}}> : (tensor<{m}x{k}xi8>, tensor<f32>, tensor<i64>) -> tensor<{m}x{k}xf32>",
        f"    %e0{t} = tensor.empty() : tensor<{m}x{n}xf32>",
        f"    %f{t} = linalg.fill ins(%c0 : f32) outs(%e0{t} : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>",
        f"    %mm{t} = linalg.matmul ins(%xd{t}, %wd{t} : tensor<{m}x{k}xf32>, tensor<{k}x{n}xf32>) "
        f"outs(%f{t} : tensor<{m}x{n}xf32>) -> tensor<{m}x{n}xf32>",
        f"    %e1{t} = tensor.empty() : tensor<{m}x{n}xf32>",
        f"    %ba{t} = linalg.generic {{indexing_maps = [{_ID}, {_COL}, {_ID}], "
        f'iterator_types = ["parallel", "parallel"]}} ins(%mm{t}, {bias} : tensor<{m}x{n}xf32>, '
        f"tensor<{n}xf32>) outs(%e1{t} : tensor<{m}x{n}xf32>) {{",
        "    ^bb0(%p: f32, %q0: f32, %o: f32):",
        "      %r = arith.addf %p, %q0 : f32",
        "      linalg.yield %r : f32",
        f"    }} -> tensor<{m}x{n}xf32>",
    ]
    if not close:
        return rows
    return [
        *rows,
        f"    %e2{t} = tensor.empty() : tensor<{m}x{n}xf32>",
        f"    %relu{t} = linalg.generic {{indexing_maps = [{_ID}, {_ID}], "
        f'iterator_types = ["parallel", "parallel"]}} ins(%ba{t} : tensor<{m}x{n}xf32>) '
        f"outs(%e2{t} : tensor<{m}x{n}xf32>) {{",
        "    ^bb0(%p: f32, %o: f32):",
        "      %zero = arith.constant 0.000000e+00 : f32",
        "      %r = arith.maximumf %p, %zero : f32",
        "      linalg.yield %r : f32",
        f"    }} -> tensor<{m}x{n}xf32>",
        *_quantize(f"relu{t}", f"q{t}", m, n, "f32"),
    ]


def _closed_model() -> str:
    """A model in the shape a capture has: quantize the argument, two layers, logits out.

    The first region is the permitted host one (an ARGUMENT onto the integer grid); the last layer
    leaves as the accumulator, which is what a classifier does.
    """
    return "\n".join(
        [
            "builtin.module {",
            "  func.func @forward(%x: tensor<4x8xf32>, %wa: tensor<8x16xi8>, %biasa: tensor<16xf32>, "
            "%wb: tensor<16x32xi8>, %biasb: tensor<32xf32>) -> tensor<4x32xf32> {",
            "    %s = arith.constant dense<5.000000e-01> : tensor<f32>",
            "    %z = arith.constant dense<0> : tensor<i64>",
            "    %c0 = arith.constant 0.000000e+00 : f32",
            *_quantize("x", "qx", 4, 8, "f32"),
            *_layer("a", "%qx", 4, 8, 16, bias="%biasa", close=True),
            *_layer("b", "%qa", 4, 16, 32, bias="%biasb", close=False),
            "    func.return %bab : tensor<4x32xf32>",
            "  }",
            "}",
        ]
    )


#: What a weights manifest says about this model: argument 0 is the caller's, the rest are stored.
_MANIFEST = {
    "0": {"kind": "input", "name": "image"},
    "1": {"kind": "param", "weight": "layer_a.weight"},
    "2": {"kind": "param", "weight": "layer_a.bias"},
    "3": {"kind": "param", "weight": "layer_b.weight"},
    "4": {"kind": "param", "weight": "layer_b.bias"},
}
_WEIGHT_ARGS = {1, 2, 3, 4}


def _buffer(text: str) -> dict:
    with IR_LOCK:
        return WP.whole_program_buffer(
            mq.parse(text), _TARGET, weight_args=_WEIGHT_ARGS, manifest=_MANIFEST, model="two_layer"
        )


# ------------------------------------------------------------------ the buffer IS the whole model


def test_a_closed_model_becomes_one_command_buffer_in_its_own_order() -> None:
    buffer = _buffer(_closed_model())
    # A CONTRACTION AND ITS READOUT ARE TWO COMMANDS. MATMUL declares no attributes: the product
    # lands in the accumulator and a COMMIT applies the epilogue. One fused command per group
    # validated against the schema, which does not constrain attributes per opcode, and no engine
    # following the ABI would have applied the readout at all.
    assert [command["opcode"] for command in buffer["commands"]] == ["MATMUL", "COMMIT", "MATMUL", "COMMIT"]
    assert buffer["whole_program"]["groups"] == 3, "two device groups plus the permitted host one"
    assert buffer["whole_program"]["commands"] == 4


def test_each_command_reads_the_buffer_the_previous_one_wrote() -> None:
    """A concatenation that did not wire the dataflow would be two programs printed in one file."""
    first, commit, second, _ = _buffer(_closed_model())["commands"]
    # The contraction writes its accumulator, the commit reads exactly that and writes the
    # activation, and the next contraction reads the activation -- not the accumulator.
    assert commit["operands"]["src"] == first["operands"]["dst"]
    assert second["operands"]["lhs"] == commit["operands"]["dst"]
    assert first["operands"]["lhs"] == "image", "the first command reads the quantized argument"


def test_the_weights_and_biases_are_named_by_the_capture_not_by_position() -> None:
    buffer = _buffer(_closed_model())
    first, commit = buffer["commands"][0], buffer["commands"][1]
    # The weight is the contraction's; the bias belongs to the READOUT, which is where the ABI
    # declares it -- COMMIT takes it as an operand and MATMUL has no attributes to carry it in.
    assert (first["operands"]["rhs"], commit["operands"]["bias"]) == ("layer_a.weight", "layer_a.bias")
    roles = {name: spec["role"] for name, spec in buffer["tensors"].items()}
    assert roles["layer_a.weight"] == "weight" and roles["layer_a.bias"] == "bias"
    assert roles["image"] == "input"


def test_the_program_states_the_conversion_its_caller_applies_before_the_first_command() -> None:
    """The one permitted host region is RECORDED, never implied: the buffer's first operand is
    integers and nothing else says what they are integers of."""
    domain = _buffer(_closed_model())["whole_program"]["input_domain"]
    assert domain["tensor"] == "image" and domain["from_dtype"] == "f32"
    assert domain["scale"] == pytest.approx(0.5) and domain["zero_point"] == 0


def test_a_group_that_leaves_as_the_accumulator_commits_the_accumulator_and_states_its_readout() -> None:
    """A model's classifier does not requantize. Typing its result i8 would be wrong by 24 bits, and
    dropping the dequantize would make the buffer's integers mean nothing."""
    buffer = _buffer(_closed_model())
    # The readout is the COMMIT's, not the contraction's: MATMUL declares no attributes at all.
    _, closing, _, classifier = buffer["commands"]
    assert closing["attributes"]["output_dtype"] == "i8"
    assert closing["attributes"]["epilogue"] == ["bias_add", "acc_scale", "relu"]
    assert classifier["attributes"]["output_dtype"] == "i32"
    assert classifier["attributes"]["epilogue"] == ["bias_add"]
    readout = buffer["whole_program"]["readout"]
    assert readout["tensor"] == classifier["operands"]["dst"] and readout["dtype"] == "f32"
    assert readout["dequantize"] == pytest.approx(0.5 * 0.5)


def test_the_result_is_an_output_and_every_other_produced_buffer_is_an_intermediate() -> None:
    """`intermediate` is what the schema calls a runner-allocated buffer inside one whole-program
    kernel; calling them all outputs would make the benchmark read a dozen results."""
    buffer = _buffer(_closed_model())
    roles = [spec["role"] for spec in buffer["tensors"].values()]
    assert buffer["outputs"] == [buffer["commands"][-1]["operands"]["dst"]]
    # One activation between the two groups, plus one accumulator per contraction: a MATMUL's
    # product is a runner-allocated buffer exactly as its committed activation is, and a tensor a
    # command names that nothing declares is a buffer nobody allocated.
    assert roles.count("output") == 1
    assert roles.count("intermediate") == 3
    accumulators = [name for name, spec in buffer["tensors"].items() if name.startswith("ACC_")]
    assert len(accumulators) == 2 and all(buffer["tensors"][a]["role"] == "intermediate" for a in accumulators)


def test_the_buffer_validates_against_the_compiler_api_schema_the_harness_pins() -> None:
    """The gate validates the emitted buffer against this exact file, so a buffer this repo cannot
    validate itself is one the harness will reject with the compiler blamed."""
    schema = json.loads(
        (repo_root() / "merlin/contract/schemas/command_buffer.schema.json").read_text(encoding="utf-8")
    )
    errors = list(jsonschema.Draft202012Validator(schema).iter_errors(_buffer(_closed_model())))
    assert not errors, [f"/{'/'.join(map(str, e.path))}: {e.message}" for e in errors[:4]]


def test_the_kernel_abi_names_every_buffer_the_runner_must_bind() -> None:
    buffer = _buffer(_closed_model())
    abi = buffer["kernel_abi"]
    assert abi["kind"] == "whole_program" and abi["outputs"] == buffer["outputs"]
    named = {argument["tensor"] for argument in abi["args"]}
    assert named == {name for name, spec in buffer["tensors"].items() if spec["role"] != "intermediate"}


# ------------------------------------------------------------------ closure, and one definition of it


def test_a_model_with_host_work_between_its_groups_is_refused_with_the_regions_named() -> None:
    """A buffer of "the groups" for a model whose host also computes would report a fraction of a
    model under the model's name, and nothing in it would say so.

    Over a REAL public capsule rather than a synthetic one, because the thing being checked is that
    a model this target genuinely cannot close is refused: measured, `SY_micro_model` places 11 of
    its 15 regions on the host (its elementwise work is fp32 and this unit's map is int8-only).
    """
    capsule = repo_root() / "merlin/contract/capsules/model/SY_micro_model/capsule.interface.mlir"
    with pytest.raises(WP.WholeProgramError) as refusal:
        with IR_LOCK:
            WP.whole_program_buffer(mq.parse(capsule.read_text(encoding="utf-8")), _TARGET)
    assert "host region(s) compute between the groups" in str(refusal.value)
    assert "group " in str(refusal.value), "the refusal must name which regions, not count them"


def test_the_input_quantization_is_permitted_only_for_a_model_argument() -> None:
    """Both halves of the predicate carry weight: a requantize in the MIDDLE of a model is the same
    operation on a value the model produced, and leaving that on the host is the split the rule
    exists to catch."""
    from merlin.xdsl_dialects.lowering import compute_groups as CG
    from merlin.xdsl_dialects.lowering import model_closure as MC

    with IR_LOCK:
        groups = CG.form_groups(mq.parse(_closed_model()), _TARGET)
    host = [g for g in groups if g.placement == CG.HOST]
    assert len(host) == 1 and MC.quantizes_an_input(host[0])
    assert MC.open_host_regions(groups) == []
    # The device groups are not input quantizations however they are spelled.
    assert not any(MC.quantizes_an_input(g) for g in groups if g.placement != CG.HOST)


def test_a_packages_commands_are_renamed_into_the_programs_own_buffers() -> None:
    """The submission names its operands A0/W/B/Y0; this program names them after the capture. What
    has to agree is the ROLE, not the spelling -- and a scratch handle is PRIVATE to its group: two
    groups both calling their accumulator `acc0` would otherwise alias into one buffer."""
    emitted = {
        "tensors": {
            "A0": {"shape": [4, 8], "dtype": "i8", "role": "input"},
            "W": {"shape": [8, 16], "dtype": "i8", "role": "weight"},
            "B": {"shape": [16], "dtype": "i32", "role": "bias"},
            "Y0": {"shape": [4, 16], "dtype": "i8", "role": "output"},
        },
        "commands": [
            {"opcode": "RES_PACK", "operands": {"src": "W", "dst": "W_res"}},
            {"opcode": "MATMUL_RESIDENT", "operands": {"lhs": "A0", "rhs": "W_res", "dst": "acc0"}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "acc0", "bias": "B", "dst": "Y0"},
                "attributes": {"bias": "B", "epilogue": ["bias_add"]},
            },
        ],
    }
    bound = {"input": ("image",), "weight": ("layer_a.weight",), "bias": ("layer_a.bias",), "output": ("B_g1",)}
    commands, scratch, why, _cause = WP._spliced(emitted, bound, "g1")
    assert why == "" and len(commands) == 3
    assert commands[1]["operands"] == {"lhs": "image", "rhs": "g1_W_res", "dst": "g1_acc0"}
    assert commands[2]["operands"] == {"src": "g1_acc0", "bias": "layer_a.bias", "dst": "B_g1"}
    # A tensor NAME can live in an attribute too; leaving that one unrenamed points a command at
    # another group's buffer with nothing to notice it.
    assert commands[2]["attributes"]["bias"] == "layer_a.bias"
    assert set(scratch) == set(), "a declared tensor is bound, not scratch"


def test_a_packages_own_staging_beside_an_operand_is_bound_by_shape_and_kept_as_scratch() -> None:
    """A convolution capsule declares its im2col buffer as an `input` too. That is the package's own
    staging, not a second operand, so the one the group binds is found by SHAPE -- which both sides
    state -- and the rest becomes that group's scratch.

    The staging is named so that SORTED ORDER puts it first: a binding that fell back to position
    would pick it, and the kernel would read the im2col buffer as though it were the image.
    """
    emitted = {
        "tensors": {
            "IFM": {"shape": [1, 56, 56, 64], "dtype": "i8", "role": "input"},
            "AAA_im2col": {"shape": [3136, 576], "dtype": "i8", "role": "input"},
            "W": {"shape": [576, 64], "dtype": "i8", "role": "weight"},
            "Y0": {"shape": [3136, 64], "dtype": "i8", "role": "output"},
        },
        "commands": [{"opcode": "CONV2D", "operands": {"lhs": "IFM", "pack": "AAA_im2col", "rhs": "W", "dst": "Y0"}}],
    }
    bound = {"input": ("act",), "weight": ("w",), "output": ("out",)}
    shapes = {
        "act": {"shape": [1, 56, 56, 64]},
        "w": {"shape": [576, 64]},
        "out": {"shape": [3136, 64]},
    }
    commands, scratch, why, _cause = WP._spliced(emitted, bound, "g1", shapes)
    assert why == "" and commands[0]["operands"] == {"lhs": "act", "pack": "g1_AAA_im2col", "rhs": "w", "dst": "out"}
    assert set(scratch) == {"g1_AAA_im2col"}, "the package's staging is scratch, private to this group"


def test_two_same_role_tensors_of_the_same_shape_are_refused_rather_than_guessed() -> None:
    """Ambiguity is a decline. A mis-bound operand is a numerically wrong program that still passes,
    which is strictly worse than this group falling back."""
    emitted = {
        "tensors": {
            "A0": {"shape": [4, 8], "dtype": "i8", "role": "input"},
            "A1": {"shape": [4, 8], "dtype": "i8", "role": "input"},
            "Y0": {"shape": [4, 16], "dtype": "i8", "role": "output"},
        },
        "commands": [{"opcode": "MATMUL", "operands": {"lhs": "A0", "dst": "Y0"}}],
    }
    commands, _scratch, why, cause = WP._spliced(
        emitted, {"input": ("act",), "output": ("out",)}, "g1", {"act": {"shape": [4, 8]}, "out": {"shape": [4, 16]}}
    )
    assert cause == WP.ROLE_AMBIGUOUS, "the cause must say WHICH refusal this is, not only its sentence"
    assert not commands and "2 'input' tensor(s) of shape [4, 8]" in why and "would be a guess" in why


def test_a_package_that_emitted_nothing_for_a_group_is_refused_for_that_group_only() -> None:
    commands, _scratch, why, _cause = WP._spliced({"tensors": {}, "commands": []}, {}, "g1")
    assert not commands and "no command for this group" in why


def test_with_no_package_every_command_is_this_repos_own_statement_and_says_so() -> None:
    """`on: reference` and `on: submission` are DIFFERENT CLAIMS about different compilers, and a
    buffer that summed them would say nothing about which of a model's layers the compiler under test
    actually lowered."""
    census = _buffer(_closed_model())["whole_program"]
    assert census["on_reference"] == 2 and census["on_submission"] == 0
    assert [row["on"] for row in census["per_group"]] == [WP.FROM_REFERENCE, WP.FROM_REFERENCE]
    assert all("no package was named" in row["why"] for row in census["per_group"])


# ------------------------------------------------- a widening cast is silent, and says that it was


_RETYPE_MODEL = """
builtin.module {
  func.func @forward(%x: tensor<4x8xf32>, %w: tensor<8x16xi8>) -> tensor<4x16xf32> {
    %s = arith.constant dense<5.000000e-01> : tensor<f32>
    %z = arith.constant dense<0> : tensor<i64>
    %c0 = arith.constant 0 : i8
    %qx = "quant_ext.quantize_per_tensor"(%x, %s, %z) <{quant_min = -128 : i64, quant_max = 127 : i64,
      output_dtype = "int8"}> : (tensor<4x8xf32>, tensor<f32>, tensor<i64>) -> tensor<4x8xi8>
    %e0 = tensor.empty() : tensor<4x16xi8>
    %f = linalg.fill ins(%c0 : i8) outs(%e0 : tensor<4x16xi8>) -> tensor<4x16xi8>
    %mm = linalg.matmul ins(%qx, %w : tensor<4x8xi8>, tensor<8x16xi8>)
                        outs(%f : tensor<4x16xi8>) -> tensor<4x16xi8>
    %e1 = tensor.empty() : tensor<4x16xf32>
    %c = linalg.generic {indexing_maps = [affine_map<(d0, d1) -> (d0, d1)>,
      affine_map<(d0, d1) -> (d0, d1)>], iterator_types = ["parallel", "parallel"]}
      ins(%mm : tensor<4x16xi8>) outs(%e1 : tensor<4x16xf32>) {
    ^bb0(%p: i8, %o: f32):
      %r = arith.sitofp %p : i8 to f32
      linalg.yield %r : f32
    } -> tensor<4x16xf32>
    func.return %c : tensor<4x16xf32>
  }
}
"""


def test_an_unscaled_widening_cast_stays_on_the_host_and_the_model_is_not_closed() -> None:
    """Group formation keeps a conversion of a scaled store off a group with no output scale
    (``compute_groups.READOUT_REQUIRES_SCALE``): an unscaled integer-to-float cast is host work. So the
    statement never sees a pure retype inside a group, and a model whose only escape is such a cast
    is refused as not closed -- by the refusal's own name, not as a readout this program invents."""
    from merlin.xdsl_dialects.lowering import compute_groups as CG

    with IR_LOCK:
        groups = CG.form_groups(mq.parse(_RETYPE_MODEL), _TARGET)
    host = [g for g in groups if g.placement == CG.HOST and CG.CAST in g.stages]
    assert host and host[0].refusal == CG.READOUT_REQUIRES_SCALE
    assert all(WP._pure_retype(g) is None for g in groups if g.root is not None)
    with IR_LOCK, pytest.raises(WP.WholeProgramError, match="conversion of a scaled store"):
        WP.whole_program_buffer(
            mq.parse(_RETYPE_MODEL), _TARGET, weight_args={1}, manifest={"0": {"kind": "input", "name": "image"}}
        )


# ------------------------------------------- a shape mismatch is refused, however the roles pair


def test_the_equal_count_path_checks_shapes_instead_of_pairing_blind() -> None:
    """THE HOLE THIS CLOSED, and it was live for every spliced group. When a role's declared count
    equalled the count this program binds, the pairing was positional and the SHAPES were never
    looked at -- so a package declaring its weight transposed bound silently, with no view recorded
    and nothing to notice it. That is a correctly-sized buffer in the wrong element order, which
    links, runs and passes.

    Measured on a captured ResNet-50: every one of its nineteen convolutions wants NHWC where the
    program holds NCHW -- `[1,512,14,14]` against `[1,14,14,512]` -- equal element counts and a
    different order. A permutation, not a reshape, and now refused by name.
    """
    emitted = {
        "tensors": {
            "A0": {"shape": [1, 512, 14, 14], "dtype": "i8", "role": "input"},
            "Y0": {"shape": [4, 16], "dtype": "i8", "role": "output"},
        },
        "commands": [{"opcode": "CONV2D", "operands": {"lhs": "A0", "dst": "Y0"}}],
    }
    # One tensor per role on both sides, so the counts agree and the old path would have paired them.
    commands, _scratch, why, cause = WP._spliced(
        emitted,
        {"input": ("act",), "output": ("out",)},
        "g1",
        {"act": {"shape": [1, 14, 14, 512]}, "out": {"shape": [4, 16]}},
    )
    assert not commands, "a transposed operand must not bind because the counts happened to agree"
    assert cause == WP.ROLE_UNBOUND and "permutation rather than a reshape" in why


def test_a_provable_reshape_still_binds_and_is_stated() -> None:
    """The check is precise, not blanket: a tensor that IS the same bytes in the same order binds,
    and the buffer records the derivation that admitted it rather than leaving a reader to infer it."""
    emitted = {
        "tensors": {
            "A0": {"shape": [1, 56, 56, 64], "dtype": "i8", "role": "input"},
            "Y0": {"shape": [4, 16], "dtype": "i8", "role": "output"},
        },
        "commands": [{"opcode": "CONV2D", "operands": {"lhs": "A0", "dst": "Y0"}}],
    }
    views: list = []
    commands, _scratch, why, _cause = WP._spliced(
        emitted,
        {"input": ("act",), "output": ("out",)},
        "g1",
        {"act": {"shape": [3136, 64]}, "out": {"shape": [4, 16]}},
        views,
    )
    assert commands and why == ""
    assert views and views[0]["tensor"] == "act" and views[0]["read_as"] == [1, 56, 56, 64]
    assert "same elements in the same order" in views[0]["basis"]


def test_a_schema_valid_buffer_was_never_evidence_about_shapes() -> None:
    """Why the check had to exist at all. `command_buffer.schema.json` requires each command's
    operands to NAME declared tensors; it says nothing about their extents. A buffer carrying a
    transposed weight validated cleanly the whole time."""
    schema = json.loads(
        (repo_root() / "merlin/contract/schemas/command_buffer.schema.json").read_text(encoding="utf-8")
    )
    transposed = {
        "abi_version": "0.1",
        "target": _TARGET,
        "tensors": {
            "w": {"shape": [1, 14, 14, 512], "dtype": "i8", "role": "weight"},
            "out": {"shape": [4, 16], "dtype": "i8", "role": "output"},
        },
        "commands": [{"opcode": "CONV2D", "operands": {"rhs": "w", "dst": "out"}}],
    }
    assert not list(jsonschema.Draft202012Validator(schema).iter_errors(transposed))


def test_the_byte_order_rule_rejects_a_pair_that_is_not_even_the_same_size() -> None:
    """Both clauses of the rule are load-bearing, but not on the same path.

    `_record_binding` checks element count first, and given equal totals the trailing clause is
    IMPLIED by the leading one -- if the leading products agree the trailing extents must too. The
    match list in `_spliced` has no such precondition, and there the trailing clause is the only thing
    between a 32-element tensor and a 64-element one. This pins it directly, because the
    `_record_binding` tests above cannot: a mutation removing it survives them.
    """
    assert WP._same_bytes([4, 8], [4, 16]) == "", "equal leading extents do not make these the same bytes"
    assert WP._same_bytes([1, 14, 14, 512], [1, 512, 14, 14]) == "", "NCHW against NHWC is a permutation"
    assert WP._same_bytes([3136, 64], [1, 56, 56, 64]), "a genuine reshape must still be admitted"


def test_the_record_says_that_splits_measured_before_the_shape_check_are_void() -> None:
    """The sentence that stops the old number being quoted back. 35 of 71 was never real: of the 123
    bindings that split rested on, 2 were identical and 121 were permutations, so the groups counted
    as the submission's had kernels reading wrongly-ordered buffers. The honest baseline for "how much
    of this model does the submission author" was UNKNOWN all session, not 35."""
    assert "void rather than superseded" in WP.__doc__ or WP.SPLIT_UNMEASURED_BEFORE
    # Beside the code that COMPUTES the split, wherever that lives (it is re-exported by WP).
    computes = sys.modules[WP.attribution.__module__].__file__
    assert "UNKNOWN, not 35" in (pathlib.Path(computes).read_text(encoding="utf-8")), (
        "the correction must stay beside the code that computes the split"
    )


def _rows(*rows):
    return {
        "schema": "whole_program_v1",
        "commands": [],
        "tensors": {},
        "whole_program": {"per_group": list(rows)},
    }


def test_a_mixed_cause_bucket_quotes_the_sentence_most_of_it_gave():
    """A bucket quoting whichever refusal arrived first states the wrong reason over the right count."""
    from merlin.llvmlower import whole_program as WP

    buffer = _rows(
        {
            "group": 1,
            "op": "conv2d",
            "on": WP.FROM_REFERENCE,
            "cause": "package_declined",
            "why": "a capacity argument",
        },
        *(
            {
                "group": index,
                "op": "residual_add",
                "on": WP.FROM_REFERENCE,
                "cause": "package_declined",
                "why": "no lowering for this operation",
            }
            for index in range(2, 18)
        ),
    )
    bucket = WP.attribution(buffer)["fallback_by_reason"][0]
    assert bucket["groups"] == 17
    assert bucket["why"] == "no lowering for this operation"
    assert bucket["distinct_sentences"] == [
        {"count": 16, "why": "no lowering for this operation"},
        {"count": 1, "why": "a capacity argument"},
    ]


def test_a_single_sentence_bucket_still_carries_its_one_sentence():
    from merlin.llvmlower import whole_program as WP

    buffer = _rows(
        *(
            {"group": i, "op": "matmul", "on": WP.FROM_REFERENCE, "cause": "no_commands", "why": "it emitted nothing"}
            for i in range(3)
        )
    )
    bucket = WP.attribution(buffer)["fallback_by_reason"][0]
    assert bucket["why"] == "it emitted nothing"
    assert bucket["distinct_sentences"] == [{"count": 3, "why": "it emitted nothing"}]


def _abi_opcodes():
    import yaml

    from merlin.common.paths import merlin_dir

    declared = yaml.safe_load((merlin_dir() / "contract/command_buffer_abi.yaml").read_text(encoding="utf-8"))
    return (declared or {}).get("opcodes") or {}


def test_every_command_names_its_operands_the_way_the_contract_declares():
    """The JSON schema constrains STRUCTURE, not per-opcode names, so it said yes to a buffer no
    consumer following the ABI could read: CONV2D carried `lhs`/`rhs` where the contract declares
    `ifm`/`weight`, and the independent reference recomputation raised KeyError on its first
    command. A check that cannot fail is the defect; this is the one that can.
    """
    from merlin.llvmlower import whole_program as WP

    opcodes = _abi_opcodes()
    # The operand roles this module builds a statement from, before it renames them.
    built = {"lhs": "A", "rhs": "B", "dst": "Y", "bias": "C"}
    for opcode in ("CONV2D", "MATMUL", "RESIDUAL_ADD"):
        declared = set(opcodes[opcode].get("operands") or {}) | {"bias"}
        named = WP._as_declared(opcode, built)
        assert set(named) <= declared, (
            f"{opcode} emitted {sorted(set(named) - declared)}, which the contract does not declare"
        )
        assert set(named.values()) == set(built.values()), "renaming must not drop an operand"
    # The rename is real, not a no-op: the two contractions differ in exactly this.
    assert WP._as_declared("CONV2D", built)["ifm"] == "A"
    assert WP._as_declared("CONV2D", built)["weight"] == "B"
    assert WP._as_declared("MATMUL", built)["lhs"] == "A"


def test_a_convolution_states_the_kernel_the_contract_requires():
    from merlin.llvmlower import whole_program as WP

    entry = {"kh": 7, "kw": 7, "ci": 3, "N": 64, "Himg": 224, "Wimg": 224, "op": "conv2d"}
    stated = WP._as_declared_attributes(
        "CONV2D", entry, {"epilogue": ["relu"], "stride": [2, 2], "padding": [3, 3, 3, 3], **entry}
    )
    assert stated["kernel"] == [7, 7, 3, 64]
    assert stated["layout"] == "nhwc"
    # The entry's generator vocabulary must not survive: an engine that refuses an attribute it does
    # not implement is the only thing that ever noticed these.
    assert not ({"kh", "kw", "ci", "N", "Himg", "Wimg", "op"} & set(stated))


def test_a_non_convolution_keeps_the_vocabulary_it_already_spoke():
    from merlin.llvmlower import whole_program as WP

    attributes = {"epilogue": ["relu"], "bound_lsb": 2}
    assert WP._as_declared_attributes("RESIDUAL_ADD", {}, dict(attributes)) == attributes


def test_a_weight_is_declared_in_the_device_layout_even_with_no_package(tmp_path):
    """The prepack states the layout the TARGET reads, not a service to a splice.

    It was skipped whenever no package was named, so a reference-only buffer declared every
    convolution's weight as the capture holds it -- [cout, cin, kh, kw] against the
    [kh*kw*ci, cout] its own opcode defines. The buffer passed every structural check and the
    independent recomputation refused its first command.
    """
    import inspect

    from merlin.llvmlower import whole_program as WP

    source = inspect.getsource(WP.whole_program_buffer)
    head = source[: source.index("result: str | None = None")]
    assert "if capture is None" in head, "the prepack must be gated on the capture, not the package"
    assert "if package is None else _prepack_rows" not in head


def test_a_contraction_carries_no_attributes_because_its_opcode_declares_none():
    """MATMUL declares an EMPTY attribute list. A fused epilogue on it validated against the schema
    -- which constrains structure, not attributes per opcode -- and no engine following the ABI
    would have applied the readout, so the accumulator's integers were read as the committed
    output. The capsule cohort cannot expose this: a single-op capsule has no chained readout.
    """
    buffer = _buffer(_closed_model())
    declared = _abi_opcodes()
    for command in buffer["commands"]:
        allowed = set(declared[command["opcode"]].get("attributes") or {})
        carried = set(command["attributes"])
        assert carried <= allowed, (
            f"{command['opcode']} carries {sorted(carried - allowed)}, which its opcode does not declare"
        )


def test_every_commit_reads_an_accumulator_some_contraction_wrote():
    """A COMMIT whose src nothing produced would apply a readout to an unallocated buffer."""
    buffer = _buffer(_closed_model())
    written = {c["operands"]["dst"] for c in buffer["commands"] if c["opcode"] in ("MATMUL", "MATMUL_RESIDENT")}
    for command in buffer["commands"]:
        if command["opcode"] == "COMMIT":
            assert command["operands"]["src"] in written
            assert buffer["tensors"][command["operands"]["src"]]["role"] == "intermediate"


def test_a_contraction_missing_an_operand_its_opcode_declares_is_refused_by_name():
    """A filter that keeps whichever operands are present emits a one-input contraction, and the
    engine reading it fails on a KeyError naming the operand rather than the group that lacked it.
    A window mean is exactly this case: its stationary operand is a constant, not a model argument.
    """
    from merlin.llvmlower import whole_program as WP

    with pytest.raises(WP.WholeProgramError, match="states no rhs operand"):
        WP._readout_pair("MATMUL", {"lhs": "A", "dst": "Y"}, {}, "ACC", [4, 2], "i32")


def test_a_contraction_with_every_declared_operand_is_stated():
    from merlin.llvmlower import whole_program as WP

    (contraction, commit), staged = WP._readout_pair(
        "MATMUL", {"lhs": "A", "rhs": "B", "dst": "Y"}, {"output_dtype": "i8"}, "ACC", [4, 2], "i32"
    )
    assert contraction["operands"] == {"lhs": "A", "rhs": "B", "dst": "ACC"}
    assert commit["operands"] == {"src": "ACC", "dst": "Y"}
    assert staged["role"] == "intermediate"


# ------------------------------------------- one path: a model group reaches the package as a capsule


#: One compute group as `group_command.program` states it: the SAME entry vocabulary the capsule
#: corpus is built from, which is what makes the two routes the same job rather than two.
_GROUP_ENTRY = {
    "name": "group_1",
    "kind": "op",
    "op": "matmul",
    "M": 4,
    "K": 8,
    "N": 16,
    "epilogue": ["bias_add"],
    "scale_granularity": "tensor",
    "operand_dtype": "int8",
}
_GROUP_OPERANDS = {"lhs": "image", "rhs": "layer_a.weight", "bias": "layer_a.bias", "dst": "B_g1"}

#: What the WHOLE PROGRAM (never the package) already knows about each of ``_GROUP_OPERANDS``' own
#: buffers at the point it asks the package: the role it was declared under. This is the single
#: source of truth ``_ask_package`` binds a constant operand by -- never whether the group also
#: carries a bias, which only ever happened to agree with it in this fixture.
_GROUP_SHAPES = {
    "image": {"shape": [4, 8], "dtype": "i8", "role": "intermediate"},
    "layer_a.weight": {"shape": [8, 16], "dtype": "i8", "role": "weight"},
    "layer_a.bias": {"shape": [16], "dtype": "f32", "role": "bias"},
}


def _stub_package(commands=("parse", "lower_interface_to_target", "emit_command_buffer", "emit_target_artifact")):
    from types import SimpleNamespace

    return SimpleNamespace(
        manifest={"target": _TARGET, "commands": {name: {"argv": ["{tool}", "{input_mlir}"]} for name in commands}},
        tool=pathlib.Path("/nonexistent/tool"),
        directory=pathlib.Path("/nonexistent"),
    )


def _stub_buffer(**extra) -> dict:
    """What the package answers for this group: [4,8] x [8,16] + [16] -> [4,16]."""
    return {
        "abi_version": "0.1",
        "target": _TARGET,
        "tensors": {
            "A0": {"shape": [4, 8], "dtype": "i8", "role": "input"},
            "W": {"shape": [8, 16], "dtype": "i8", "role": "weight"},
            "B": {"shape": [16], "dtype": "i32", "role": "bias"},
            "Y0": {"shape": [4, 16], "dtype": "i8", "role": "output"},
        },
        "commands": [
            {"opcode": "MATMUL", "operands": {"lhs": "A0", "rhs": "W", "dst": "acc0"}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "acc0", "bias": "B", "dst": "Y0"},
                "attributes": {"epilogue": ["bias_add"]},
            },
        ],
        **extra,
    }


def _ask(tmp_path, buffer, *, package=None, operands=None, shapes=None):
    """Put one group to a stub package, recording every entrypoint it was asked for."""
    from types import SimpleNamespace

    asked: list[str] = []

    def invoke(pkg, name, source, destination=None, *, timeout):  # noqa: ARG001
        asked.append(name)
        if destination is not None:
            pathlib.Path(destination).write_text(json.dumps(buffer), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="; target artifact\n", stderr="")

    answered = WP._ask_package(
        package if package is not None else _stub_package(),
        dict(_GROUP_ENTRY),
        dict(operands if operands is not None else _GROUP_OPERANDS),
        _TARGET,
        tmp_path,
        "g1",
        timeout=30,
        run=invoke,
        shapes=_GROUP_SHAPES if shapes is None else shapes,
        binder=_binder(),
    )
    return asked, answered


def test_a_model_group_is_put_to_the_package_through_the_capsule_routes_own_walk(tmp_path) -> None:
    """A capsule, a kernel and a model group are three spellings of one job.

    MUTATION THIS CATCHES: give the model route its own single-entrypoint invocation back and the
    recorded walk collapses to ``["emit_command_buffer"]`` -- no ``parse``, no
    ``lower_interface_to_target``, and no target artifact for any package that declares no analysis
    bundle, which was all 76 in this tree. That divergence is why a group could be reported lowered
    on a buffer the grader had never seen.
    """
    asked, (spliced, _scratch, why, _cause) = _ask(tmp_path, _stub_buffer())
    assert asked == ["parse", "lower_interface_to_target", "emit_command_buffer", "emit_target_artifact"]
    assert spliced and why == "", why
    assert (tmp_path / "g1.artifact.txt").read_text(encoding="utf-8") == "; target artifact\n"


def test_a_package_that_declares_the_bundle_is_asked_for_it_once(tmp_path) -> None:
    """The package's own declaration decides, through the one accessor that reads it."""
    package = _stub_package(("parse", "lower_interface_to_target", "emit_analysis_bundle"))
    asked, (spliced, _scratch, why, _cause) = _ask(tmp_path, _stub_buffer(), package=package)
    assert asked == ["parse", "lower_interface_to_target", "emit_analysis_bundle"]
    assert spliced and why == ""


def test_a_group_whose_buffer_the_contract_refuses_falls_back_instead_of_splicing(tmp_path) -> None:
    """The model route validates what comes back exactly as the grader does, and names the refusal.

    MUTATION THIS CATCHES: drop the contract validation from the shared walk and this group splices a
    command whose opcode the ABI does not have -- a model reported as lowered on a buffer that would
    have failed certification.
    """
    _asked, (spliced, _scratch, why, cause) = _ask(
        tmp_path, _stub_buffer(commands=[{"opcode": "NOT_AN_OPCODE", "operands": {"src": "A0", "dst": "Y0"}}])
    )
    assert not spliced and cause == WP.MALFORMED, (cause, why)


def test_a_stated_decline_stays_a_decline_and_not_an_invocation_failure(tmp_path) -> None:
    """A backend that SAYS it cannot lower a shape is a different fact from one that crashed, and the
    token is what a reader groups by."""
    _asked, (spliced, _scratch, why, cause) = _ask(
        tmp_path, _stub_buffer(declined={"reason": "no lowering for this shape", "op": "matmul"})
    )
    assert not spliced and cause == WP.PACKAGE_DECLINED
    assert "no lowering for this shape" in why


# ------------------------------------------------ the binding a caller links the package's kernel by


def _matmul_reply(output_dtype: str = "i8") -> dict:
    return {
        "tensors": {
            "A0": {"shape": [4, 8], "dtype": "i8", "role": "input"},
            "W": {"shape": [8, 16], "dtype": "i8", "role": "weight"},
            "B": {"shape": [16], "dtype": "i32", "role": "bias"},
            "Y0": {"shape": [4, 16], "dtype": output_dtype, "role": "output"},
        },
        "commands": [
            {"opcode": "MATMUL", "operands": {"lhs": "A0", "rhs": "W", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "bias": "B", "dst": "Y0"}},
        ],
    }


_HELD = {
    "image": {"shape": [4, 8], "dtype": "i8"},
    "layer_a.weight": {"shape": [8, 16], "dtype": "i8"},
    "layer_a.bias": {"shape": [16], "dtype": "f32"},
    "B_g1": {"shape": [4, 16], "dtype": "i8"},
}
_BOUND = {"input": ("image",), "weight": ("layer_a.weight",), "bias": ("layer_a.bias",), "output": ("B_g1",)}


def test_the_splice_records_which_program_tensor_each_package_tensor_is() -> None:
    """The ONE statement of which buffer a kernel argument is. A caller that links the package's
    kernel reads it here; re-deriving it from roles and extents was a second spelling of the fact."""
    record: dict = {}
    _commands, _scratch, why, _cause = WP._spliced(_matmul_reply(), _BOUND, "g1", _HELD, [], record)
    assert why == ""
    binding = record["binding"]
    assert {name: row["program"] for name, row in binding.items()} == {
        "A0": "image",
        "W": "layer_a.weight",
        "B": "layer_a.bias",
        "Y0": "B_g1",
    }
    assert all(row["bound"] for row in binding.values())
    assert binding["A0"]["declared"] == [4, 8] and binding["A0"]["role"] == "input"


def test_a_data_tensor_committed_at_another_width_is_refused() -> None:
    """A count is blind to width. A kernel committing i8 into a buffer this program holds at i32 --
    a classifier that leaves as the accumulator -- writes a quarter of the bytes, and every shape
    check still agrees."""
    held = dict(_HELD, B_g1={"shape": [4, 16], "dtype": "i32"})
    spliced, _scratch, why, cause = WP._spliced(_matmul_reply("i8"), _BOUND, "g1", held, [])
    assert not spliced and cause == WP.DTYPE_MISMATCH
    assert "'Y0' as i8" in why and "'B_g1' as i32" in why


#: A bias-free contraction against a synthesized constant -- the shape a window mean states
#: (``ones[1, window] @ activation[window, features]``), and the same shape a per-row reduction or a
#: per-channel constant scale would state for any other target. Neither operand carries a bias, so
#: only the PROGRAM's own declared role (never bias presence) can tell ``_ask_package`` which of the
#: two is the stationary constant.
_MEAN_OPERANDS = {"lhs": "ONES_g70", "rhs": "pooled_window", "dst": "B_g70"}
_MEAN_SHAPES = {
    "ONES_g70": {"shape": [1, 3], "dtype": "i8", "role": "weight"},
    "pooled_window": {"shape": [3, 5], "dtype": "i8", "role": "intermediate"},
}


def _mean_reply(weight_shape: list[int]) -> dict:
    return {
        "abi_version": "0.1",
        "target": _TARGET,
        "tensors": {
            "K": {"shape": weight_shape, "dtype": "i8", "role": "weight"},
            "A0": {"shape": [3, 5], "dtype": "i8", "role": "input"},
            "Y0": {"shape": [1, 5], "dtype": "i8", "role": "output"},
        },
        "commands": [
            {"opcode": "MATMUL", "operands": {"lhs": "K", "rhs": "A0", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"}},
        ],
    }


def test_a_bias_free_constant_operand_binds_by_its_declared_role(tmp_path) -> None:
    """A window mean's ``ones`` is a stored constant with no bias beside it, exactly as ResNet-50's
    g70 states it. The package's kernel declares it distinctly (role ``weight``, shape ``[1, 3]``)
    from the real activation (role ``input``, shape ``[3, 5]``), and the program binds each by that
    declared role -- not by guessing from operand position, and not by the group having a bias.

    MUTATION THIS CATCHES: gate the input/weight split on bias presence again (the prior behavior)
    and this group's ``ones``/activation both fall under the generic ``input`` role. The package
    declares one ``input`` tensor (``A0``, shape ``[3, 5]``) and one ``weight`` tensor (``K``), so the
    role split then finds zero declared ``input`` tensors of ``ONES_g70``'s shape ``[1, 3]`` and
    refuses with exactly the message measured on ResNet-50's g70: `the package declares 0 'input'
    tensor(s) of shape [1, 3] for 'ONES_g70'`.
    """
    _asked, (spliced, _scratch, why, cause) = _ask(
        tmp_path, _mean_reply([1, 3]), operands=_MEAN_OPERANDS, shapes=_MEAN_SHAPES
    )
    assert spliced and why == "", (cause, why)


def test_a_constant_operand_of_the_wrong_shape_is_still_refused(tmp_path) -> None:
    """The role split never widens what shape agreement demands: a package that declares its
    constant under the right role but the wrong shape is refused by name, not silently bound."""
    _asked, (spliced, _scratch, why, cause) = _ask(
        tmp_path, _mean_reply([1, 4]), operands=_MEAN_OPERANDS, shapes=_MEAN_SHAPES
    )
    assert not spliced and cause == WP.ROLE_UNBOUND, (cause, why)
    assert "'K'" in why and "[1, 4]" in why and "[1, 3]" in why


def test_a_vector_read_across_its_unit_axis_is_the_same_bytes() -> None:
    """A unit axis holds one index, so [2048, 1] and [1, 2048] are one vector -- the pooled features
    a classifier reads as its row. A genuine transpose is still refused."""
    assert WP._same_bytes([2048, 1], [1, 2048])
    assert WP._same_bytes([1, 1, 2048], [2048])
    assert not WP._same_bytes([49, 2048], [2048, 49])


def test_the_interface_states_the_width_the_program_commits(tmp_path) -> None:
    """Left undeclared, the capsule builder derives the committed width from the epilogue alone, and
    a group leaving as the accumulator was asked for a narrow commit into a wide buffer."""
    from types import SimpleNamespace

    def invoke(pkg, name, source, destination=None, *, timeout):  # noqa: ARG001
        return SimpleNamespace(returncode=1, stdout="", stderr="not asked")

    WP._ask_package(
        _stub_package(),
        {**_GROUP_ENTRY, "epilogue": [], "output_dtype": "i32"},
        dict(_GROUP_OPERANDS),
        _TARGET,
        tmp_path,
        "g1",
        timeout=30,
        run=invoke,
        binder=_binder(),
    )
    assert 'output_dtype = "i32"' in (tmp_path / "g1.iface.mlir").read_text(encoding="utf-8")


# ----------------------------------------- a region is only ever offered to a package that asks for it


def _declining_run(pkg, name, source, destination=None, *, timeout):  # noqa: ARG001
    """Every entrypoint call answers as a stated decline: never a crash, never a splice -- so the
    whole model falls back to the reference statement for every group either way, and the only thing
    a test over this can observe is WHETHER a region was ever attempted."""
    if destination is not None:
        Path(destination).write_text(
            json.dumps({"declined": {"reason": "test stub declines everything", "op": name}}), encoding="utf-8"
        )
    return SimpleNamespace(returncode=0, stdout="", stderr="")


def test_a_region_is_never_offered_to_a_package_that_did_not_declare_it(tmp_path, monkeypatch) -> None:
    """``_closed_model()``'s two layers are exactly one legal region (layer a's quantized output has
    one consumer, layer b, and is not the model's own result) -- so this is the SAME model a region
    mechanism would offer a fused ask over, and the only variable is the package's own manifest."""
    calls: list[str] = []

    def spy(*args, **kwargs):
        calls.append("called")
        from merlin.llvmlower import whole_program as WP_

        return WP_._refuse(WP_.NOT_ASKED, "spy") + (None,)

    monkeypatch.setattr("merlin.llvmlower.region_capsule.ask_package_region", spy)

    package = _stub_package()
    assert "whole_model_regions" not in package.manifest
    with IR_LOCK:
        WP.whole_program_buffer(
            mq.parse(_closed_model()),
            _TARGET,
            weight_args=_WEIGHT_ARGS,
            manifest=_MANIFEST,
            model="two_layer",
            package=package,
            binder=_binder(),
            workdir=tmp_path / "no_opt_in",
            timeout=30,
            run=_declining_run,
            allow_regions=True,
        )
    assert calls == [], "a package that never declared whole_model_regions must never be offered one"


def _weights_beside(capture: Path) -> None:
    """A minimal REAL weights manifest + safetensors pair beside ``capture``, matching
    ``_MANIFEST``/``_closed_model()`` exactly -- so ``_prepack_rows`` finds a device layout for both
    layers' weights and neither ever falls into ``weight_refusal`` (the state a region offer already
    refuses to be tried under, since a group missing its device weight cannot be spliced at all)."""
    import struct

    import numpy as np

    tensors = {
        "layer_a.weight": np.arange(8 * 16, dtype=np.int8).reshape(8, 16),
        "layer_a.bias": np.zeros(16, dtype=np.float32),
        "layer_b.weight": np.arange(16 * 32, dtype=np.int8).reshape(16, 32),
        "layer_b.bias": np.zeros(32, dtype=np.float32),
    }
    header: dict = {}
    payload = b""
    for name, array in tensors.items():
        dtype = "I8" if array.dtype == np.int8 else "F32"
        chunk = array.tobytes()
        header[name] = {
            "dtype": dtype,
            "shape": list(array.shape),
            "data_offsets": [len(payload), len(payload) + len(chunk)],
        }
        payload += chunk
    header_bytes = json.dumps(header).encode("utf-8")
    (capture.parent / "weights.safetensors").write_bytes(struct.pack("<Q", len(header_bytes)) + header_bytes + payload)
    (capture.parent / "weights.manifest.json").write_text(json.dumps(_MANIFEST), encoding="utf-8")


def test_a_region_is_offered_to_a_package_that_declares_it(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def spy(*args, **kwargs):
        calls.append("called")
        from merlin.llvmlower import whole_program as WP_

        return WP_._refuse(WP_.NOT_ASKED, "spy") + (None,)

    monkeypatch.setattr("merlin.llvmlower.region_capsule.ask_package_region", spy)

    package = _stub_package()
    package.manifest["whole_model_regions"] = True
    capture = tmp_path / "capsule.interface.mlir"
    capture.write_text(_closed_model(), encoding="utf-8")
    _weights_beside(capture)
    with IR_LOCK:
        WP.whole_program_buffer(
            mq.parse(_closed_model()),
            _TARGET,
            weight_args=_WEIGHT_ARGS,
            manifest=_MANIFEST,
            model="two_layer",
            package=package,
            binder=_binder(),
            capture=capture,
            workdir=tmp_path / "opt_in",
            timeout=30,
            run=_declining_run,
            allow_regions=True,
            region_internal_ops=("matmul",),
        )
    assert calls == ["called"], "a package that declares whole_model_regions must be offered the legal window"


def test_allow_regions_off_never_computes_a_region_even_for_an_opted_in_package(tmp_path, monkeypatch) -> None:
    calls: list[str] = []

    def spy(*args, **kwargs):
        calls.append("called")
        from merlin.llvmlower import whole_program as WP_

        return WP_._refuse(WP_.NOT_ASKED, "spy") + (None,)

    monkeypatch.setattr("merlin.llvmlower.region_capsule.ask_package_region", spy)

    package = _stub_package()
    package.manifest["whole_model_regions"] = True
    with IR_LOCK:
        WP.whole_program_buffer(
            mq.parse(_closed_model()),
            _TARGET,
            weight_args=_WEIGHT_ARGS,
            manifest=_MANIFEST,
            model="two_layer",
            package=package,
            binder=_binder(),
            workdir=tmp_path / "flag_off",
            timeout=30,
            run=_declining_run,
            allow_regions=False,
        )
    assert calls == [], "allow_regions=False must never offer a region, whatever the package declares"


def test_a_declined_member_is_never_pulled_into_a_region_for_an_opted_in_package(tmp_path, monkeypatch) -> None:
    """The bug this guards: ``window_of`` is computed from the model's own dataflow alone, with no
    knowledge of ``decline`` -- so a caller-declined group inside an otherwise-legal window could still
    be bundled into a region ask with its (undeclined) neighbour. A region is never a way around a
    decline: the whole window must fall back to per-group, and the declined member must still show
    ``CALLER_DECLINED`` by name, exactly as a solo ask would."""
    calls: list[str] = []

    def spy(*args, **kwargs):
        calls.append("called")
        from merlin.llvmlower import whole_program as WP_

        return WP_._refuse(WP_.NOT_ASKED, "spy") + (None,)

    monkeypatch.setattr("merlin.llvmlower.region_capsule.ask_package_region", spy)

    package = _stub_package()
    package.manifest["whole_model_regions"] = True
    capture = tmp_path / "capsule.interface.mlir"
    capture.write_text(_closed_model(), encoding="utf-8")
    _weights_beside(capture)
    with IR_LOCK:
        census = WP.whole_program_buffer(
            mq.parse(_closed_model()),
            _TARGET,
            weight_args=_WEIGHT_ARGS,
            manifest=_MANIFEST,
            model="two_layer",
            package=package,
            binder=_binder(),
            capture=capture,
            workdir=tmp_path / "opt_in_but_declined",
            timeout=30,
            run=_declining_run,
            allow_regions=True,
            region_internal_ops=("matmul",),
            decline=[1],
        )["whole_program"]
    assert calls == [], "a region must never be attempted when the caller declined one of its members"
    rows = {row["group"]: row for row in census["per_group"]}
    assert rows[1]["on"] == WP.FROM_REFERENCE and rows[1]["cause"] == WP.CALLER_DECLINED


# ------------------------------------------------------------------------- FUSED REGIONS, answered


def _fused_reply() -> dict:
    """A package's answer to the region of ``_closed_model()``'s two layers as ONE kernel: it reads the
    image and both layers' weights and biases, and commits ONLY the boundary's output (layer b's
    accumulator). Layer a's value is never formed in any buffer -- the kernel keeps it to itself."""
    from merlin.llvmlower.region_capsule import region_output_name

    out = region_output_name(1)
    return {
        "abi_version": "0.1",
        "target": _TARGET,
        "tensors": {
            "A0": {"shape": [4, 8], "dtype": "i8", "role": "input"},
            "W0": {"shape": [8, 16], "dtype": "i8", "role": "weight"},
            "W1": {"shape": [16, 32], "dtype": "i8", "role": "weight"},
            "B0": {"shape": [16], "dtype": "i32", "role": "bias"},
            "B1": {"shape": [32], "dtype": "i32", "role": "bias"},
            out: {"shape": [4, 32], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {"opcode": "MATMUL", "operands": {"lhs": "A0", "rhs": "W0", "dst": "acc0"}},
            {"opcode": "MATMUL", "operands": {"lhs": "acc0", "rhs": "W1", "dst": "acc1"}},
            {"opcode": "COMMIT", "operands": {"src": "acc1", "bias": "B1", "dst": out}},
        ],
        "kernel_abi": {
            "kind": "whole_program",
            "args": [{"tensor": t, "access": "read"} for t in ("A0", "W0", "B0", "W1", "B1")]
            + [{"tensor": out, "access": "write"}],
            "outputs": [out],
        },
    }


def _region_run(*, region: dict | None, single: dict | None, asked: list[str]):
    """An entrypoint stub: a REGION capsule is answered with ``region`` (or declined), a single group's
    with ``single`` (or declined); every interface asked is recorded by its file name."""

    def run(pkg, name, source, destination=None, *, timeout):  # noqa: ARG001
        source_name = Path(str(source)).parent.name
        if name == "parse":
            asked.append(source_name)
        reply = region if source_name.startswith("region_") else single
        if destination is not None:
            body = reply if reply is not None else {"declined": {"reason": "stub declines", "op": name}}
            Path(destination).write_text(json.dumps(body), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="; target artifact\n" if reply else "", stderr="")

    return run


def _regioned(tmp_path, run, *, internal_ops=("matmul",)) -> dict:
    package = _stub_package()
    package.manifest["whole_model_regions"] = True
    capture = tmp_path / "capsule.interface.mlir"
    capture.write_text(_closed_model(), encoding="utf-8")
    _weights_beside(capture)
    with IR_LOCK:
        return WP.whole_program_buffer(
            mq.parse(_closed_model()),
            _TARGET,
            weight_args=_WEIGHT_ARGS,
            manifest=_MANIFEST,
            model="two_layer",
            package=package,
            binder=_binder(),
            capture=capture,
            workdir=tmp_path / "work",
            timeout=30,
            run=run,
            allow_regions=True,
            region_internal_ops=internal_ops,
        )


def test_a_fused_region_the_package_answers_is_one_kernel_and_both_groups_are_the_packages(tmp_path) -> None:
    """ANSWERED: the two layers are offered as ONE region; the package answers with one kernel that
    commits only the boundary. Both groups are the package's -- the internal one with no command of its
    own -- and the boundary carries the whole kernel and is where the region is graded."""
    from merlin.perf import whole_model_build as WMB

    asked: list[str] = []
    buffer = _regioned(tmp_path, _region_run(region=_fused_reply(), single=None, asked=asked))
    rows = {row["group"]: row for row in buffer["whole_program"]["per_group"]}
    assert asked == ["region_g1_g2.generated"], "the region answered, so no member was asked alone"
    assert rows[1]["on"] == rows[2]["on"] == WP.FROM_SUBMISSION
    assert (rows[1]["region"]["role"], rows[1]["commands"]) == ("internal", 0)
    assert (rows[2]["region"]["role"], rows[2]["region"]["graded_at"]) == ("boundary", "g2")
    assert rows[2]["commands"] == len(buffer["commands"]) == 3
    # The internal member's tensor is never committed: the one kernel computes through it.
    committed = {c["operands"]["dst"] for c in buffer["commands"] if "dst" in c.get("operands", {})}
    assert rows[2]["operands"]["dst"] in committed and rows[1]["operands"]["dst"] not in committed
    # COVERAGE: both groups are package-answered; the region's one kernel is bound at its boundary.
    internal, boundary = [r for r in WMB.bind_groups({**buffer, "target": _TARGET}) if r["group"] in (1, 2)]
    assert internal["on"] == boundary["on"] == WMB.ON_PACKAGE
    assert internal["graded_at"] == 2 and "args" not in internal
    assert [a["program"] for a in boundary["args"]] == [
        "image",
        "layer_a.weight",
        "layer_a.bias",
        "layer_b.weight",
        "layer_b.bias",
        rows[2]["operands"]["dst"],
    ]


def test_a_fused_region_the_package_declines_falls_back_to_asking_each_group_alone(tmp_path) -> None:
    """DECLINED: the region is offered and declined; every member is then asked ALONE, exactly as if
    the window had never been legal -- the package answering one of them is credited for that one."""
    asked: list[str] = []
    buffer = _regioned(tmp_path, _region_run(region=None, single=_stub_buffer(), asked=asked))
    assert asked[0] == "region_g1_g2.generated" and set(asked[1:]) == {"g1.generated", "g2.generated"}
    rows = {row["group"]: row for row in buffer["whole_program"]["per_group"]}
    assert not any("region" in row for row in rows.values())
    # Layer a's single-group reply fits it; layer b's shape does not, so it falls to the reference.
    assert rows[1]["on"] == WP.FROM_SUBMISSION and rows[2]["on"] == WP.FROM_REFERENCE


def test_a_region_is_never_offered_across_a_boundary_the_target_cannot_restate(tmp_path) -> None:
    """THE OFFER IS DERIVED FROM FACTS: with the target restating no matmul as an internal member, the
    legal window is never offered as a region at all, and each group is asked alone."""
    asked: list[str] = []
    _regioned(tmp_path, _region_run(region=_fused_reply(), single=None, asked=asked), internal_ops=("conv2d",))
    assert not any(name.startswith("region_") for name in asked)


# ------------------------------------------------------- a CALLER-declined group is never asked, and
# ------------------------------------------------------- is stated exactly as a package refusal would be


def _fake_splice(*_args, **_kwargs):
    """A package that would ALWAYS answer successfully, were it asked."""
    return [{"opcode": "FAKE", "operands": {}}], {}, "", ""


def _buffer_with_package(text: str, *, decline=(), ask=None, monkeypatch) -> dict:
    # A device-weight prepack normally needs a real capture + weights-manifest + safetensors triple
    # beside it; faked here to a trivial success so the fixture reaches the splice/decline decision
    # itself, which is what these tests are about -- not weight-layout derivation, covered elsewhere.
    monkeypatch.setattr(WP, "_device_weight", lambda _prepack_row, _rhs: ({"shape": [1, 1]}, ""))
    monkeypatch.setattr(WP, "_ask_package", ask or _fake_splice)
    with IR_LOCK:
        return WP.whole_program_buffer(
            mq.parse(text),
            _TARGET,
            weight_args=_WEIGHT_ARGS,
            manifest=_MANIFEST,
            model="two_layer",
            package=_stub_package(),
            binder=_binder(),
            decline=decline,
        )


def test_with_no_decline_and_a_package_that_answers_both_groups_splice(monkeypatch) -> None:
    census = _buffer_with_package(_closed_model(), monkeypatch=monkeypatch)["whole_program"]
    assert [row["on"] for row in census["per_group"]] == [WP.FROM_SUBMISSION, WP.FROM_SUBMISSION]


def test_a_declined_group_by_index_is_never_asked_and_falls_back_to_the_reference(monkeypatch) -> None:
    calls: list[int] = []

    def counting_ask(*args, **kwargs):
        calls.append(1)
        return _fake_splice(*args, **kwargs)

    census = _buffer_with_package(_closed_model(), decline=[1], ask=counting_ask, monkeypatch=monkeypatch)[
        "whole_program"
    ]
    rows = {row["group"]: row for row in census["per_group"]}
    assert rows[1]["on"] == WP.FROM_REFERENCE
    assert rows[1]["cause"] == WP.CALLER_DECLINED
    assert "caller" in rows[1]["why"].lower()
    assert rows[2]["on"] == WP.FROM_SUBMISSION
    # The package was asked exactly once -- for group 2, never for the declined group 1.
    assert len(calls) == 1


def test_decline_takes_priority_over_a_weight_prepack_refusal(monkeypatch) -> None:
    """A declined group is never asked, so whether the package COULD have been given a correctly
    device-laid-out weight must never matter for it -- that concern only gates an ask that will not
    happen. Real prepack data is withheld here (no ``_device_weight`` fake) so an un-declined group
    would show ``NO_PREPACK``; the declined one must show ``CALLER_DECLINED`` regardless."""
    with IR_LOCK:
        buffer = WP.whole_program_buffer(
            mq.parse(_closed_model()),
            _TARGET,
            weight_args=_WEIGHT_ARGS,
            manifest=_MANIFEST,
            model="two_layer",
            package=_stub_package(),
            binder=_binder(),
            decline=[1],
        )
    rows = {row["group"]: row for row in buffer["whole_program"]["per_group"]}
    assert rows[1]["on"] == WP.FROM_REFERENCE and rows[1]["cause"] == WP.CALLER_DECLINED
    assert rows[2]["cause"] == WP.NO_PREPACK, "the un-declined group's own refusal is unrelated to decline"


def test_a_declined_group_still_commits_a_real_buffer_the_reference_statement_produces(monkeypatch) -> None:
    """The bug this guards: a post-hoc attribution rewrite left a declined group with NEITHER a
    package kernel nor a library one, so its committed buffer (`B_g<N>`) never existed at all. Stating
    the decline INSIDE the buffer (this function, not a rewrite afterward) must always produce one."""
    document = _buffer_with_package(_closed_model(), decline=[1], monkeypatch=monkeypatch)
    tensors = document["tensors"]
    assert any(name.endswith("_g1") and tensors[name].get("role") == "intermediate" for name in tensors), (
        f"no committed intermediate buffer for the declined group among {sorted(tensors)}"
    )
    # It is CHECKABLE like any other command: a real MATMUL/COMMIT pair reads and writes real tensors.
    assert any(c["opcode"] in ("MATMUL", "MATMUL_RESIDENT") for c in document["commands"])


def test_declining_by_op_name_declines_every_group_of_that_op(monkeypatch) -> None:
    census = _buffer_with_package(_closed_model(), decline=["matmul"], monkeypatch=monkeypatch)["whole_program"]
    assert [row["on"] for row in census["per_group"]] == [WP.FROM_REFERENCE, WP.FROM_REFERENCE]
    assert all(row["cause"] == WP.CALLER_DECLINED for row in census["per_group"])


def test_a_group_not_named_in_decline_is_unaffected(monkeypatch) -> None:
    census = _buffer_with_package(_closed_model(), decline=[99], monkeypatch=monkeypatch)["whole_program"]
    assert [row["on"] for row in census["per_group"]] == [WP.FROM_SUBMISSION, WP.FROM_SUBMISSION]


class TestDeclineKeyMatchesTheSameSpellingsWholeModelBuildDoes:
    def test_a_bare_int_is_a_group(self):
        assert WP._decline_key(33) == ("group", 33)

    def test_a_g_prefixed_digit_string_is_a_group(self):
        assert WP._decline_key("g33") == ("group", 33)

    def test_a_plain_digit_string_is_a_group(self):
        assert WP._decline_key("33") == ("group", 33)

    def test_anything_else_is_an_op_name(self):
        assert WP._decline_key("residual_add") == ("op", "residual_add")

    def test_a_bool_is_refused_as_neither(self):
        import pytest

        with pytest.raises(ValueError):
            WP._decline_key(True)

    def test_is_declined_matches_by_group_or_by_op(self):
        assert WP._is_declined(7, "matmul", [7])
        assert WP._is_declined(7, "matmul", ["g7"])
        assert WP._is_declined(7, "matmul", ["matmul"])
        assert not WP._is_declined(7, "matmul", [8])
        assert not WP._is_declined(7, "matmul", ["conv2d"])
        assert not WP._is_declined(7, "matmul", [])


# ----------------------------------------- a window mean is asked for in the orientation the program holds


#: A window mean as the statement records it (``x[features, window] @ ones[window, 1]``: M features,
#: K the window, N one) and as the program holds its operands (``ones[1, window] @ x[window, features]``).
_STATED_MEAN = {
    "name": "mean_group",
    "kind": "op",
    "op": "matmul",
    "M": 5,
    "K": 3,
    "N": 1,
    "epilogue": ["acc_scale"],
    "acc_scale": 0.25,
    "scale_granularity": "tensor",
    "operand_dtype": "int8",
    "output_dtype": "i8",
}


def _oriented_reply(ones: list[int], activation: list[int], output: list[int]) -> dict:
    return {
        "abi_version": "0.1",
        "target": _TARGET,
        "tensors": {
            "A0": {"shape": ones, "dtype": "i8", "role": "input"},
            "W": {"shape": activation, "dtype": "i8", "role": "weight"},
            "Y0": {"shape": output, "dtype": "i8", "role": "output"},
        },
        "commands": [
            {"opcode": "MATMUL", "operands": {"lhs": "A0", "rhs": "W", "dst": "acc0"}},
            {"opcode": "COMMIT", "operands": {"src": "acc0", "dst": "Y0"}},
        ],
    }


def _ask_mean(tmp_path, reply):
    from types import SimpleNamespace

    group = SimpleNamespace(window_mean={"window": 3, "rows": 5})
    entry = WP._asked_orientation(dict(_STATED_MEAN), group, _MEAN_OPERANDS, _MEAN_SHAPES)

    def invoke(pkg, name, source, destination=None, *, timeout):  # noqa: ARG001
        if destination is not None:
            pathlib.Path(destination).write_text(json.dumps(reply), encoding="utf-8")
        return SimpleNamespace(returncode=0, stdout="; target artifact\n", stderr="")

    answered = WP._ask_package(
        _stub_package(), entry, dict(_MEAN_OPERANDS), _TARGET, tmp_path, "g70",
        timeout=30, run=invoke, shapes=_MEAN_SHAPES, binder=_binder(),
    )  # fmt: skip
    return entry, answered, (tmp_path / "g70.iface.mlir").read_text(encoding="utf-8")


def test_a_window_mean_is_asked_for_as_the_program_holds_its_operands(tmp_path) -> None:
    """The capsule a package is asked for states the ones on the left and the held activation as the
    stationary operand -- the tensors this program has -- so a kernel that answers it exactly binds:
    the ones by the capsule's input role, the activation where a weight goes. Asked in the statement's
    orientation instead, the capsule named [5, 3] and [3, 1], which no buffer of the program is, and the
    package's exact answer was refused at the splice (measured: ResNet-50's global mean ran as host
    code for it)."""
    entry, (spliced, _scratch, why, cause), interface = _ask_mean(tmp_path, _oriented_reply([1, 3], [3, 5], [1, 5]))
    assert (entry["M"], entry["K"], entry["N"]) == (1, 3, 5)
    assert "tensor<1x3xi8>" in interface and "tensor<3x5xi8>" in interface
    assert spliced and why == "", (cause, why)


def test_a_window_mean_answered_in_the_other_orientation_is_still_refused(tmp_path) -> None:
    """The orientation changes what is ASKED, never what binding accepts: a kernel declaring the
    transposed tensors (the statement's orientation) is refused by shape, by name."""
    _entry, (spliced, _scratch, why, cause), _interface = _ask_mean(tmp_path, _oriented_reply([5, 3], [3, 1], [5, 1]))
    assert not spliced and cause == WP.ROLE_UNBOUND, (cause, why)


def test_the_orientation_is_used_only_when_the_programs_operands_are_its_extents() -> None:
    """A group that is not a window mean, or whose held operands do not have the oriented extents, is
    asked for exactly as stated: the orientation is checked against the buffers, never assumed."""
    from types import SimpleNamespace

    mean = SimpleNamespace(window_mean={"window": 3, "rows": 5})
    assert (
        WP._asked_orientation(dict(_STATED_MEAN), SimpleNamespace(window_mean=None), _MEAN_OPERANDS, _MEAN_SHAPES)
        == _STATED_MEAN
    )
    other = {**_MEAN_SHAPES, "pooled_window": {"shape": [5, 3], "dtype": "i8", "role": "intermediate"}}
    assert WP._asked_orientation(dict(_STATED_MEAN), mean, _MEAN_OPERANDS, other) == _STATED_MEAN
