from __future__ import annotations

import json

import pytest

from merlin.targetgen._aten_opset_worker import core_opset
from merlin.targetgen.core_aten_capture import loader_source
from merlin.targetgen.core_aten_cases import (
    _aliases,
    _changed,
    _tensor_snapshot,
    case_document,
    core_aten_cases,
    corpus_document,
    decode_arguments,
    decode_value,
    encode_value,
    resolve_overload,
    validate_result,
)

torch = pytest.importorskip("torch")


@pytest.fixture(scope="module")
def opset() -> dict:
    return core_opset()


@pytest.fixture(scope="module")
def corpus(opset: dict) -> dict:
    return corpus_document(opset["ops"], pytorch_version=opset["torch"])


def test_registry_is_exact_bijection_with_core_tagged_overloads(opset: dict) -> None:
    assert list(core_aten_cases()) == opset["ops"]
    assert len(opset["ops"]) == 193
    assert opset["sha256"] == "2195c829a9f600e91247ee168374180c5b34367a10d5a6243723d0f75f82af65"


def test_every_case_resolves_and_executes_its_exact_overload(corpus: dict) -> None:
    assert corpus["complete"] is True
    assert corpus["case_count"] == corpus["overload_count"]
    for case in corpus["cases"]:
        overload = resolve_overload(case["overload"])
        assert str(overload) == case["overload"]
        assert str(overload._schema) == case["schema"]


def test_portable_arguments_round_trip_and_oracles_validate(corpus: dict) -> None:
    for case in corpus["cases"]:
        torch.manual_seed(case["rng_seed"])
        args, kwargs = decode_arguments(case["arguments"])
        actual = resolve_overload(case["overload"])(*args, **kwargs)
        validate_result(case, actual)


def test_corpus_is_byte_deterministic(opset: dict) -> None:
    first = corpus_document(opset["ops"], pytorch_version=opset["torch"])
    second = corpus_document(reversed(opset["ops"]), pytorch_version=opset["torch"])
    assert json.dumps(first, sort_keys=True, separators=(",", ":")) == json.dumps(
        second, sort_keys=True, separators=(",", ":")
    )


def test_factories_return_fresh_mutable_arguments() -> None:
    case = core_aten_cases()["aten._local_scalar_dense.default"]
    first_args, _ = case.make_arguments()
    second_args, _ = case.make_arguments()
    assert first_args[0] is not second_args[0]
    assert first_args[0].untyped_storage().data_ptr() != second_args[0].untyped_storage().data_ptr()


def test_portable_tensor_round_trip_preserves_nonzero_storage_offset() -> None:
    tensor = torch.arange(12, dtype=torch.float32).reshape(3, 4)[1:, 1:]
    assert tensor.storage_offset() != 0
    rebuilt = decode_value(encode_value(tensor))
    torch.testing.assert_close(rebuilt, tensor)
    assert rebuilt.shape == tensor.shape
    assert rebuilt.stride() == tensor.stride()
    assert rebuilt.storage_offset() == tensor.storage_offset()


def test_nonfinite_complex_and_zero_stride_nan_round_trip_as_strict_json() -> None:
    complex_document = encode_value(complex(float("nan"), float("inf")))
    json.dumps(complex_document, allow_nan=False)
    rebuilt_complex = decode_value(complex_document)
    assert torch.isnan(torch.tensor(rebuilt_complex.real))
    assert rebuilt_complex.imag == float("inf")

    expanded = torch.tensor([float("nan")]).expand(3)
    expanded_document = encode_value(expanded)
    json.dumps(expanded_document, allow_nan=False)
    rebuilt_expanded = decode_value(expanded_document)
    assert rebuilt_expanded.stride() == (0,)
    assert torch.isnan(rebuilt_expanded).all()


def test_mutation_and_alias_checks_handle_nan_signed_zero_and_empty_views() -> None:
    nan = torch.tensor(float("nan"))
    assert not _changed(_tensor_snapshot(nan), nan.clone())
    positive_zero = torch.tensor(0.0)
    negative_zero = torch.tensor(-0.0)
    assert _changed(_tensor_snapshot(positive_zero), negative_zero)

    base = torch.empty(1)
    empty_view = base[:0]
    unrelated = torch.empty(0)
    assert _aliases(base, empty_view)
    assert not _aliases(empty_view, unrelated)


def test_mutation_out_and_alias_contracts_are_recorded() -> None:
    resize = case_document(core_aten_cases()["aten.resize_.default"])
    out = case_document(core_aten_cases()["aten.atan2.out"])
    alias = case_document(core_aten_cases()["aten.alias.default"])
    assert resize["mutated_arguments"] == ["args[0]"]
    assert "kwargs.out" in out["mutated_arguments"]
    assert "result->kwargs.out" in out["output_input_aliases"]
    assert "result->args[0]" in alias["output_input_aliases"]


def test_backward_rng_symbolic_and_overload_families_are_not_collapsed(corpus: dict) -> None:
    names = {case["overload"] for case in corpus["cases"]}
    assert {
        "aten.convolution_backward.default",
        "aten.rand.default",
        "aten.sym_size.int",
        "aten.add.Tensor",
        "aten.add.Scalar",
    } <= names
    assert len(names) == corpus["case_count"]


@pytest.mark.parametrize(
    "name",
    [
        "aten.add.Tensor",
        "aten.arange.start_step",
        "aten.atan2.out",
        "aten.index.Tensor",
    ],
)
def test_compiler_adapter_loader_is_self_contained_and_preserves_oracle(name: str) -> None:
    case = case_document(core_aten_cases()[name])
    namespace: dict = {}
    exec(compile(loader_source(case), f"<{name}>", "exec"), namespace)  # noqa: S102
    model, inputs = namespace["get_model_and_inputs"]()
    torch.manual_seed(case["rng_seed"])
    validate_result(case, model(*inputs))
