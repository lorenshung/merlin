"""Independent-reference coverage for schema-native movement, convolution, and attention-PV ops.

The command-buffer simulator already executes these whole-op forms.  The independent
reference must execute the same ABI rather than rejecting them as unmodelled or returning
an empty output map, because every hardware backend gates its result against that map.
"""

from __future__ import annotations

import pytest

from merlin.runtime.reference import MODELED_OPCODES, reference_outputs
from merlin.runtime.simulator import SimulationError, simulate


def _movement_cb(*, output_dtype: str = "i32") -> dict:
    return {
        "abi_version": "0.1",
        "target": "t",
        "tensors": {
            "X": {"shape": [2, 3], "dtype": "i8", "role": "input"},
            "Y": {"shape": [2, 3], "dtype": output_dtype, "role": "output"},
        },
        "commands": [
            {
                "opcode": "MOVEMENT",
                "operands": {"src": "X", "dst": "Y"},
                "attributes": {"output_dtype": output_dtype, "semantic": "mvin_mvout"},
            }
        ],
    }


def _conv_cb(
    *, resident_weight: bool = False, output_dtype: str = "i32", epilogue: list[str] | None = None, **attributes
) -> dict:
    commands: list[dict] = []
    weight = "W"
    if resident_weight:
        weight = "W_res"
        commands.append(
            {
                "opcode": "RES_PACK",
                "operands": {"src": "W", "dst": weight},
                "attributes": {"layout": "packed_rhs"},
            }
        )
    attrs = {
        "kernel": [2, 2, 1, 1],
        "stride": [1, 1],
        "padding": [0, 0, 0, 0],
        "dilation": [1, 1],
        "layout": "nhwc",
        "epilogue": list(epilogue or []),
        "output_dtype": output_dtype,
    }
    attrs.update(attributes)
    commands.append(
        {
            "opcode": "CONV2D",
            "operands": {"ifm": "IFM", "weight": weight, "dst": "Y"},
            "attributes": attrs,
        }
    )
    return {
        "abi_version": "0.1",
        "target": "t",
        "tensors": {
            "IFM": {"shape": [1, 3, 3, 1], "dtype": "i8", "role": "input"},
            "W": {"shape": [4, 1], "dtype": "i8", "role": "weight"},
            "Y": {"shape": [4, 1], "dtype": output_dtype, "role": "output"},
        },
        "commands": commands,
    }


def _attention_pv_cb(*, output_dtype: str = "i32", epilogue: list[str] | None = None, **attributes) -> dict:
    attrs = {"epilogue": list(epilogue or []), "output_dtype": output_dtype}
    attrs.update(attributes)
    return {
        "abi_version": "0.1",
        "target": "t",
        "tensors": {
            "P": {"shape": [2, 3], "dtype": "i8", "role": "input"},
            "V": {"shape": [3, 2], "dtype": "i8", "role": "input"},
            "Y": {"shape": [2, 2], "dtype": output_dtype, "role": "output"},
        },
        "commands": [
            {
                "opcode": "ATTENTION_PV",
                "operands": {"p": "P", "v": "V", "dst": "Y"},
                "attributes": attrs,
            }
        ],
    }


def test_native_movement_is_an_identity_copy_in_the_declared_container_dtype():
    cb = _movement_cb(output_dtype="i32")
    inputs = {"X": [[-128, -3, 0], [1, 42, 127]]}
    expected = {"Y": inputs["X"]}

    assert "MOVEMENT" in MODELED_OPCODES
    assert simulate(cb, inputs)["outputs"] == expected
    assert reference_outputs(cb, inputs) == expected


@pytest.mark.parametrize("resident_weight", [False, True])
def test_native_conv_uses_original_ifm_and_declared_weight_abi(resident_weight):
    """The weight operand may be a declared tensor or the handle produced by RES_PACK.

    Neither form introduces a derived im2col pointer into the external ABI: im2col is an
    internal semantic transform of IFM.
    """
    cb = _conv_cb(resident_weight=resident_weight)
    inputs = {
        "IFM": [1, 2, 3, 4, 5, 6, 7, 8, 9],
        "W": [[1], [2], [3], [4]],
    }
    expected = {"Y": [[37], [47], [67], [77]]}

    assert "CONV2D" in MODELED_OPCODES
    assert simulate(cb, inputs)["outputs"] == expected
    assert reference_outputs(cb, inputs) == expected


def test_native_conv_reference_matches_epilogue_and_saturating_readout():
    cb = _conv_cb(epilogue=["acc_scale", "requant", "relu"], output_dtype="i8", acc_scale=0.5, requant_shift=1)
    inputs = {
        "IFM": [-128, -64, 0, 32, 64, 96, 110, 120, 127],
        "W": [[3], [3], [3], [3]],
    }

    assert reference_outputs(cb, inputs) == simulate(cb, inputs)["outputs"]


def test_native_conv_reference_matches_fused_pooling_geometry_and_values():
    cb = _conv_cb(
        kernel=[1, 1, 1, 1],
        epilogue=["maxpool"],
        pool_in_dims=[4, 4],
        pool_size=[2, 2],
        pool_stride=[2, 2],
        pool_padding=[0, 0, 0, 0],
    )
    cb["tensors"]["IFM"]["shape"] = [1, 4, 4, 1]
    cb["tensors"]["W"]["shape"] = [1, 1]
    cb["tensors"]["Y"]["shape"] = [4, 1]
    inputs = {"IFM": list(range(1, 17)), "W": [[1]]}
    expected = {"Y": [[6], [8], [14], [16]]}

    assert simulate(cb, inputs)["outputs"] == expected
    assert reference_outputs(cb, inputs) == expected


def test_native_conv_reference_rejects_an_attribute_the_simulator_rejects():
    cb = _conv_cb(transpose_weight=True)

    with pytest.raises(SimulationError, match="transpose_weight"):
        simulate(cb)
    with pytest.raises(ValueError, match="transpose_weight"):
        reference_outputs(cb)


def test_attention_pv_reference_computes_p_at_v_without_a_transpose():
    cb = _attention_pv_cb()
    inputs = {"P": [[1, 2, 3], [4, 5, 6]], "V": [[1, 2], [3, 4], [5, 6]]}
    expected = {"Y": [[22, 28], [49, 64]]}

    assert "ATTENTION_PV" in MODELED_OPCODES
    assert simulate(cb, inputs)["outputs"] == expected
    assert reference_outputs(cb, inputs) == expected


@pytest.mark.parametrize(
    "epilogue, attributes, output_dtype",
    [
        (["relu"], {}, "i32"),
        (["acc_scale"], {"acc_scale": 0.5}, "i32"),
        (["requant"], {"requant_shift": 2}, "i8"),
    ],
)
def test_attention_pv_reference_matches_supported_epilogues(epilogue, attributes, output_dtype):
    cb = _attention_pv_cb(epilogue=epilogue, output_dtype=output_dtype, **attributes)
    inputs = {"P": [[-3, 2, 4], [8, -7, 6]], "V": [[2, -1], [3, 5], [-4, 7]]}

    assert reference_outputs(cb, inputs) == simulate(cb, inputs)["outputs"]


def test_attention_native_ops_fail_closed_on_unsupported_attributes():
    pv = _attention_pv_cb(transpose_v=True)
    with pytest.raises(SimulationError, match="transpose_v"):
        simulate(pv)
    with pytest.raises(ValueError, match="transpose_v"):
        reference_outputs(pv)


def _chained_conv_cb(declared_ifm: list[int]) -> dict:
    """A contraction, the COMMIT that reads its accumulator out flat, and a convolution consuming
    exactly that. The shape a whole model produces constantly and no capsule can: of the graded
    cohort's conv2d interfaces, not one has a second compute op to chain a commit from.
    """
    return {
        "abi_version": "0.1",
        "target": "t",
        "tensors": {
            "L": {"shape": [4, 1], "dtype": "i8", "role": "input"},
            "R": {"shape": [1, 2], "dtype": "i8", "role": "weight"},
            "ACC": {"shape": [4, 2], "dtype": "i32", "role": "intermediate"},
            "ACT": {"shape": declared_ifm, "dtype": "i8", "role": "intermediate"},
            "W": {"shape": [2 * 2 * 2, 3], "dtype": "i8", "role": "weight"},
            "Y": {"shape": [1, 3], "dtype": "i32", "role": "output"},
        },
        "commands": [
            {"opcode": "MATMUL", "operands": {"lhs": "L", "rhs": "R", "dst": "ACC"}, "attributes": {}},
            {
                "opcode": "COMMIT",
                "operands": {"src": "ACC", "dst": "ACT"},
                "attributes": {"epilogue": [], "output_dtype": "i8"},
            },
            {
                "opcode": "CONV2D",
                "operands": {"ifm": "ACT", "weight": "W", "dst": "Y"},
                "attributes": {"kernel": [2, 2, 2, 3], "layout": "nhwc", "output_dtype": "i32"},
            },
        ],
    }


_CHAINED_INPUTS = {"L": [[1], [2], [3], [4]], "R": [[1, 1]], "W": [[1, 0, 0]] * 8}


def test_a_convolution_reads_a_committed_activation_in_the_flat_form_its_own_semantics_describe():
    """CONV2D's contract already gathers the activation into [N*Ho*Wo, Kh*Kw*Ci], so [N*H*W, C] is
    the same statement without the spatial split -- the same bytes, not a second layout."""
    outputs = reference_outputs(
        _chained_conv_cb([1, 2, 2, 2]),
        _CHAINED_INPUTS,
    )
    assert "Y" in outputs


def test_a_declaration_that_does_not_fold_to_the_flat_extents_is_refused_by_name():
    """The load-bearing half. 4 rows is 2x2 or 4x1 or 1x4 and the flat form distinguishes none of
    them, so a declaration that does not multiply out is a permutation nobody can perform."""
    with pytest.raises(ValueError, match="is not the same bytes"):
        reference_outputs(
            _chained_conv_cb([1, 3, 3, 2]),
            _CHAINED_INPUTS,
        )


def test_a_flat_activation_with_no_declaration_is_still_refused():
    """Absent a declared shape there is nothing to prove the fold against, so it stays a refusal."""
    cb = _chained_conv_cb([1, 2, 2, 2])
    del cb["tensors"]["ACT"]["shape"]
    with pytest.raises((ValueError, KeyError)):
        reference_outputs(cb, _CHAINED_INPUTS)
