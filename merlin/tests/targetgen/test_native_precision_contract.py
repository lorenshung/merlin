"""Native eager proofs cannot silently substitute for declared quantized bits."""

import copy

import pytest

from merlin.targetgen import functional_truth
from merlin.targetgen.native_precision_contract import proves_quantized_model, requires_quantized_model


def software():
    return dict(
        numerical_semantics=dict(
            quantized_native_fallback=dict(admission="after_native_exact_failure"),
            calibration_domain=dict(
                minimum=-1, maximum=1, evaluation_inputs="never_read", scale_policy="enclosing_power_of_two"
            ),
            quantization_nonfinite="byte_storage_zero_nan_saturating_infinity",
        )
    )


@pytest.mark.parametrize(
    "dtype,precision,wanted",
    [
        ("f16", "quantized_exact", True),
        ("bf16", "quantized_exact", True),
        ("f32", "quantized_exact", True),
        ("f64", "quantized_exact", True),
        ("complex<f32>", "quantized_exact", True),
        ("i64", "quantized_exact", False),
        ("f32", "bit_exact_vs_eager", False),
    ],
)
def test_source_storage_and_requested_model_control_obligation(monkeypatch, dtype, precision, wanted):
    monkeypatch.setattr(functional_truth, "expectation", lambda _: dict(precision_class=precision))
    record = dict(status="bundled", case=dict(schema=1, arguments={}, expected={}), input_abi=[dict(dtype=dtype)])
    assert requires_quantized_model(dict(cases=[record])) is wanted
    record["status"] = "not_in_shard"
    assert not requires_quantized_model(dict(cases=[record]))


def test_byte_model_requires_matching_parameters_and_independent_reference():
    from merlin.llvmlower.quantized_native import parameters

    spec = software()
    plan = dict(
        quantized_contract=parameters(spec),
        integer_reference=dict(engine="independent_source_quantized_exact_rational"),
    )
    assert proves_quantized_model(plan, spec)
    for change in (dict(scale_numerator=3), dict(zero_point=1)):
        bad = copy.deepcopy(plan)
        bad["quantized_contract"].update(change)
        assert not proves_quantized_model(bad, spec)
    for engine in (None, "eager", "native_typed_reference"):
        bad = copy.deepcopy(plan)
        bad["integer_reference"]["engine"] = engine
        assert not proves_quantized_model(bad, spec)


def test_legacy_model_requires_declared_calibration_and_independent_rational_engine():
    spec = software()
    recipe = copy.deepcopy(spec["numerical_semantics"])
    plan = dict(capture_recipe=recipe, integer_reference=dict(engine="independent_pt2e_exact_rational"))
    assert proves_quantized_model(plan, spec)
    bad = copy.deepcopy(plan)
    bad["capture_recipe"]["calibration_domain"]["minimum"] = -2
    assert not proves_quantized_model(bad, spec)
    bad = copy.deepcopy(plan)
    bad["capture_recipe"]["calibration_domain"]["evaluation_inputs"] = "observed"
    assert not proves_quantized_model(bad, spec)
    assert not proves_quantized_model(dict(exact_dtype_sites=10), spec)


def test_wrong_model_rolls_back_source_abi_and_retained_record_identity(tmp_path, monkeypatch):
    import yaml

    from merlin.targetgen import native_precision_contract, operator_review
    from merlin.targetgen.core_aten_admission import prepare

    original = "module { func.func @forward(%x: tensor<2xf32>) -> tensor<2xf32> { return %x : tensor<2xf32> } }"
    (tmp_path / "model.mlir").write_text(original)
    (tmp_path / "inputs.npz").write_bytes(b"original input storage")
    spec = software()
    spec["numerical_semantics"]["native_unit_contract"] = dict(excluded_constructions=[])
    path = tmp_path / "spec.yaml"
    path.write_text(yaml.safe_dump(spec))
    record = dict(status="bundled", input_abi=[dict(dtype="f32", shape=[2])])
    report = dict(cases=[record], output_count=1)
    monkeypatch.setattr(native_precision_contract, "requires_quantized_model", lambda _: True)
    monkeypatch.setattr(operator_review, "review_record", lambda _: dict(operator_accepted=True))

    def candidate(directory, report, **kwargs):
        (directory / "model.mlir").write_text(original + "\n// prepared eager candidate")
        (directory / "inputs.npz").write_bytes(b"changed storage")
        report["cases"][0]["input_abi"] = [dict(dtype="i8", shape=[8])]
        report["output_count"] = 99
        return dict(selection=dict(identities={}), exact_dtype_sites=1)

    plan = prepare(tmp_path, report, candidate, software_spec=path)
    assert plan["refusal_code"] == "declared_quantized_precision_requires_quantized_model"
    assert (tmp_path / "model.mlir").read_text() == original
    assert (tmp_path / "inputs.npz").read_bytes() == b"original input storage"
    assert report["output_count"] == 1
    assert report["cases"][0] is record
    assert record["input_abi"] == [dict(dtype="f32", shape=[2])]
