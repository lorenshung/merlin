"""Independent measurement role admission cannot use pass labels or CPU counts."""
from dataclasses import fields
from types import SimpleNamespace

import pytest
from merlin_experiments.phase2 import component_measurement_qualification as Q
from merlin_experiments.phase2.component_measurement_protocol import MEASUREMENT_CASES, MEASUREMENT_MECHANISMS
from merlin_experiments.phase2.component_runtime_authority import IndependentRuntimeServices
from merlin_experiments.phase2.component_variants import ComponentVariantSnapshot, capture_component_variant
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, exact_tree_record, sha256_file
from test_component_observer import actual_records
from test_edit_authority import package as package

from merlin.common import invocation_record
from merlin.perf.component_cost import ComponentCostScope


def test_measurement_controls_cover_both_directions_of_every_complete_scope_mechanism():
    assert len(MEASUREMENT_CASES) == 14
    assert set(MEASUREMENT_CASES) == {mechanism + "." + direction for mechanism in MEASUREMENT_MECHANISMS
                                      for direction in ("positive", "negative")}
    assert "cold_warm_regime.negative" in MEASUREMENT_CASES
    assert "hardware_timer_scope.negative" in MEASUREMENT_CASES
    assert "applicability_scope.negative" in MEASUREMENT_CASES


def test_serialized_measurement_report_and_constructed_variant_cannot_issue_roles(tmp_path):
    values = {field.name: None for field in fields(Q.IndependentMeasurementQualification)}
    with pytest.raises(StageGateError, match="live evaluated controls"):
        Q.IndependentMeasurementQualification(**values).verify()
    values = {field.name: None for field in fields(ComponentVariantSnapshot)}
    with pytest.raises(StageGateError, match="live controller-owned source capture"):
        ComponentVariantSnapshot(**values).verify()
    with pytest.raises(StageGateError, match="independent normal execution authority"):
        capture_component_variant(execution=SimpleNamespace(status="qualified"), output=tmp_path / "imported")
    with pytest.raises(StageGateError, match="live functional/fresh source"):
        Q.qualify_independent_component_measurement(
            functional_runtime=SimpleNamespace(status="qualified"), baseline_admission=None,
            scope=None, calibration_adapter=None, variants=(), evidence_root=tmp_path / "private",
        )


def actual_unqualified_role(**_arguments):
    pytest.fail("an unqualified or missing measurement producer was invoked")


def test_native_functional_transport_cannot_supply_missing_physical_timer_producer():
    from merlin_experiments.phase2.component_runtime_authority import _callback_identity

    owner = _callback_identity(actual_unqualified_role)
    runtime = SimpleNamespace(
        services=IndependentRuntimeServices(actual_unqualified_role, actual_unqualified_role,
                                            actual_unqualified_role, actual_unqualified_role, actual_unqualified_role),
        source_pins=((owner[2], owner[3]),), qualification=SimpleNamespace(context=SimpleNamespace()),
    )
    with pytest.raises(StageGateError, match="physical measurement producer is unavailable"):
        Q._identity_sources(runtime)


def diagnostic_product(path):
    path.write_text('{"status":"DIAGNOSTIC_ONLY","cycles":"UNKNOWN"}\n')


def test_actual_native_product_join_rejects_unobserved_timer_json(tmp_path):
    from merlin_experiments.phase2.component_measurement_evidence import actual_measurement_product

    _selected, _source, root, paths = actual_records(tmp_path)
    native = [invocation_record.verify(path) for path in paths]
    elf_sha = native[0]["outputs"][0]["sha256"]
    product = root / "diagnostic.json"
    with invocation_record.observe_call(
        root, stage="diagnostic_observation", function=diagnostic_product,
        arguments={"path": str(product)}, inputs=(root / "program.elf",), outputs=(product,),
    ) as call:
        diagnostic_product(product)
        call.returned()
    invocations = [invocation_record.verify(path) for path in (root / "invocations").glob("*/invocation.json")]
    actual_measurement_product(invocations, measurement=product,
                               measurement_sha256=sha256_file(product), executable_sha256=elf_sha)
    assert '"cycles":"UNKNOWN"' in product.read_text()
    forged = root / "unobserved.json"
    forged.write_bytes(product.read_bytes())
    with pytest.raises(StageGateError, match="not produced at an actual invoked"):
        actual_measurement_product(invocations, measurement=forged,
                                   measurement_sha256=sha256_file(forged), executable_sha256=elf_sha)
    with pytest.raises(StageGateError, match="not consumed by an actual runtime"):
        actual_measurement_product(invocations, measurement=product,
                                   measurement_sha256=sha256_file(product), executable_sha256="f" * 64)


def test_actual_timer_producer_cannot_join_an_unrelated_runtime_execution(tmp_path):
    from merlin_experiments.phase2.component_measurement_evidence import actual_measurement_product

    _selected, _source, root, paths = actual_records(tmp_path)
    native = [invocation_record.verify(path) for path in paths]
    elf_sha = native[0]["outputs"][0]["sha256"]
    product = root / "disconnected.json"
    with invocation_record.observe_call(
        root, stage="disconnected_observation", function=diagnostic_product,
        arguments={"path": str(product)}, inputs=(), outputs=(product,),
    ) as call:
        diagnostic_product(product)
        call.returned()
    records = [invocation_record.verify(path) for path in (root / "invocations").glob("*/invocation.json")]
    assert any(pin["sha256"] == elf_sha for row in records if row["kind"] == "subprocess" for pin in row["inputs"])
    with pytest.raises(StageGateError, match="timer producer did not consume the exact executed ELF"):
        actual_measurement_product(records, measurement=product,
                                   measurement_sha256=sha256_file(product), executable_sha256=elf_sha)


def test_diagnostic_capture_retains_allowed_variants_without_granting_experimental_execution(package, monkeypatch):
    from merlin_experiments.phase2.component_execution import ComponentIndependentExecution

    authority, candidate, contract, root = package
    authority.freeze(candidate, contract, has_iterations=False)
    identity = document_sha256("synthetic controller only")
    execution = ComponentIndependentExecution(
        baseline_admission=SimpleNamespace(sha256=identity), independent_runtime=SimpleNamespace(sha256=identity),
        candidate=candidate, edit_authority=authority, edit_authority_sha256=document_sha256(authority.binding),
        view=SimpleNamespace(manifest_sha256=identity), runtime=(), contract_root=root,
        contract_sha256=identity, scope=ComponentCostScope(identity, identity, identity), output=root / "diagnostics",
        _issuer=object(),
    )
    with monkeypatch.context() as diagnostic:
        diagnostic.setattr(ComponentIndependentExecution, "verify", lambda self: self.sha256)
        original = exact_tree_record(candidate)["sha256"]
        first = capture_component_variant(execution=execution, output=execution.output / "first")
        assert first.membership_sha256 == original
        source = candidate / "compiler.py"
        source.write_text(source.read_text().replace("return 1", "return 3"))
        second = capture_component_variant(execution=execution, output=execution.output / "second")
        assert first.membership_sha256 != second.membership_sha256
        assert first.verify() and second.verify()
        assert not any(path.stat().st_mode & 0o222 for path in first.compiler.rglob("*"))
    with pytest.raises(StageGateError, match="issued independent normal service"):
        first.verify()
