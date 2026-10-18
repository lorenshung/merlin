"""Native rounding floors use independent exact math, with no tolerance."""

import copy
import struct

import pytest

from merlin.targetgen.core_aten_host_floor import grade_native_bits
from merlin.targetgen.native_rational_reference import Gaussian, rational_outputs


def matrix(values):
    return dict(kind="tensor", dtype="float32", shape=[1, len(values)], values=[values])


def record():
    # Exact dot = 2**24 + 1 - 2**24 = 1. The sealed CPU order yields 0.
    a = matrix([float(2**24), 1.0, -float(2**24)])
    b = dict(kind="tensor", dtype="float32", shape=[3, 1], values=[[1.0], [1.0], [1.0]])
    return dict(
        output_indices=[0],
        output_abi=[dict(dtype="f32", shape=[1, 1])],
        native_eager_indices=dict(outputs=[0], posts=[1]),
        native_eager_boundary=dict(
            authority="untransformed_native_cpu_eager",
            output_bytes=[struct.pack("<f", 0.0).hex()],
            post_input_bytes=["a5"],
        ),
        numeric_admission=dict(host_floor=dict(order_dependent_reduction=True)),
        case=dict(overload="aten.mm.default", arguments=dict(args=[a, b], kwargs={})),
    )


def test_better_exact_result_passes_but_larger_error_and_changed_posts_fail():
    r = record()
    assert rational_outputs(r["case"])[0].item() == 1
    assert grade_native_bits(r, [struct.pack("<f", 1.0), b"\xa5"])["status"] == "pass"
    assert grade_native_bits(r, [struct.pack("<f", 2.0), b"\xa5"])["status"] == "pass"  # equal error
    assert grade_native_bits(r, [struct.pack("<f", 2.000000238418579), b"\xa5"])["status"] == "mismatch"
    assert grade_native_bits(r, [struct.pack("<f", 1.0), b"\xa4"])["status"] == "mismatch"
    r["numeric_admission"]["host_floor"]["order_dependent_reduction"] = False
    assert grade_native_bits(r, [struct.pack("<f", 1.0), b"\xa5"])["status"] == "mismatch"


def test_gaussian_arithmetic_keeps_real_and_imaginary_errors_separate():
    a, b = Gaussian.of(2) + Gaussian(0, 3), Gaussian(4, -5)
    assert a * b == Gaussian(23, 2)
    assert a / b * b == a


def test_native_policy_keeps_sealed_metadata_aliases_and_mutation():
    pytest.importorskip("torch")
    from merlin.targetgen.core_aten_eval import _validate_observation

    original = dict(
        kind="tensor", dtype="float32", shape=[1], stride=[1], storage_offset=0, requires_grad=False, values=[0.0]
    )
    case = dict(
        case_id="independent",
        expected=original,
        post_arguments={},
        comparison="torch_close",
        comparison_parameters=dict(rtol=0.0, atol=0.0),
        mutated_arguments=[],
        output_input_aliases=[],
    )
    observation = dict(
        status="executed",
        execution_kind="simulator",
        output={**original, "values": [1.0]},
        post_arguments={},
        mutated_arguments=[],
        output_input_aliases=[],
        numeric_reference=dict(
            policy="native_host_exact",
            admission_status="admitted",
            gate=dict(
                case_id="independent",
                status="pass",
                results=[dict(status="pass")],
                comparison="exact_rational_error_not_greater_than_native_per_component",
            ),
        ),
    )
    _validate_observation(case, observation)
    for field, value in [("stride", [2]), ("storage_offset", 1), ("requires_grad", True)]:
        changed = copy.deepcopy(observation)
        changed["output"][field] = value
        with pytest.raises(AssertionError):
            _validate_observation(case, changed)
    for field, value in [("case_id", "other"), ("status", "mismatch"), ("comparison", "torch_close")]:
        changed = copy.deepcopy(observation)
        changed["numeric_reference"]["gate"][field] = value
        with pytest.raises(AssertionError, match="independent exactness gate"):
            _validate_observation(case, changed)
    for field in ["mutated_arguments", "output_input_aliases"]:
        changed = copy.deepcopy(observation)
        changed[field] = [0]
        with pytest.raises(AssertionError):
            _validate_observation(case, changed)


def test_complex_addmm_coefficients_broadcast_as_exact_gaussian_scalars():
    from fractions import Fraction

    def z(real, imag):
        return dict(kind="complex", real=real, imag=imag)

    def tensor(values, shape):
        return dict(kind="tensor", dtype="complex64", values=values, shape=shape)

    case = dict(
        overload="aten.addmm.default",
        arguments=dict(
            args=dict(
                kind="tuple",
                items=[
                    tensor([[z(2, -1)]], [1, 1]),
                    tensor([[z(1, 1), z(2, -1)]], [1, 2]),
                    tensor([[z(3, -1)], [z(1, 2)]], [2, 1]),
                ],
            ),
            kwargs=dict(alpha=z(0.5, -1), beta=z(1, 1)),
        ),
    )
    assert rational_outputs(case)[0].item() == Gaussian(Fraction(12), Fraction(-9, 2))


def test_empty_mean_dimension_list_reduces_all_axes():
    from fractions import Fraction

    case = dict(
        overload="aten.mean.dim",
        arguments=dict(
            args=dict(
                kind="tuple",
                items=[dict(kind="tensor", dtype="float32", shape=[2, 3], values=[[1, 2, 3], [4, 5, 6]]), [], False],
            ),
            kwargs={},
        ),
    )
    result = rational_outputs(case)[0]
    assert result.shape == () and result.item() == Fraction(7, 2)


@pytest.mark.parametrize("axes", [0, -1, [0], [-1]])
def test_scalar_mean_accepts_torch_scalar_axis_conventions(axes):
    from fractions import Fraction

    case = dict(
        overload="aten.mean.dim",
        arguments=dict(
            args=dict(kind="tuple", items=[dict(kind="tensor", dtype="float32", shape=[], values=1.25), axes, True]),
            kwargs={},
        ),
    )
    result = rational_outputs(case)[0]
    assert result.shape == () and result.item() == Fraction(5, 4)


@pytest.mark.parametrize("encoded", [False, True])
@pytest.mark.parametrize("alpha,proved", [(0.5, True), (0.1, False)])
def test_native_scalar_opmath_proof_covers_sealed_float_encoding(encoded, alpha, proved):
    from fractions import Fraction

    tensor = dict(kind="tensor", dtype="float32", shape=[1, 1], values=[[1.0]])
    case = dict(
        overload="aten.addmm.default",
        arguments=dict(
            args=dict(kind="tuple", items=[tensor, tensor, tensor]),
            kwargs=dict(alpha=dict(kind="float", value=str(alpha)) if encoded else alpha),
        ),
    )
    if proved:
        assert rational_outputs(case)[0].item() == Fraction(3, 2)
    else:
        with pytest.raises(ValueError, match="opmath representation proof"):
            rational_outputs(case)
