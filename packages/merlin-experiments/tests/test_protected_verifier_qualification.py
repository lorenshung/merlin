"""Frozen control lifecycle fixtures; none claims actual hardware authority."""

from dataclasses import replace

import pytest
from merlin_experiments.phase2 import protected_final_evaluation as F
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from test_numerical_readback import native_case as _native_case
from test_protected_final_evaluation import _request

native_case = _native_case


def test_json_receipt_and_reconstructed_record_never_mint_capability(native_case):
    verifier = _request(native_case).verifier
    capability = verifier.qualification
    with pytest.raises(StageGateError, match="issued qualification capability"):
        replace(verifier, qualification=capability.receipt).validate()
    with pytest.raises(StageGateError, match="not independently issued"):
        replace(verifier, qualification=replace(capability, _issuer=None)).validate()
    record = F._mapping(capability.receipt)
    assert len(record["controls"]) == 2 * len(F.REQUIRED_WITNESSES)
    assert "synthetic" in record["scope"]
    assert record["status"] == "observation_qualified"
    assert record["qualified_roles"] == ["private_witness_observation"]
    assert {row["status"] for row in record["controls"]} == {"observed", "refused"}


def test_negative_controls_must_actually_refuse(native_case):
    with pytest.raises(StageGateError, match="accepted an independently declared negative"):
        _request(native_case, control_fault="accept")


def test_verifier_crash_is_not_negative_control_success(native_case):
    with pytest.raises(RuntimeError, match="no diagnosed refusal"):
        _request(native_case, control_fault="crash")


def test_each_witness_requires_both_independent_control_polarities(native_case):
    with pytest.raises(StageGateError, match="positive and negative cases for all witnesses"):
        _request(native_case, omit_control=True)


def test_control_plan_cannot_change_after_issuance(native_case):
    verifier = _request(native_case).verifier
    plan = verifier.qualification.original_plan
    plan.write_bytes(plan.read_bytes() + b"\n")
    with pytest.raises(StageGateError, match="absent, indirect or changed"):
        verifier.validate()


def test_non_authorizing_observation_does_not_admit_selection(native_case):
    request = _request(native_case)
    selection = replace(request.verifier, qualification=None, qualification_sha256=None)
    witness = F.observe_arm(selection, request.binding, "candidate", request.candidate, request.original_reference)
    assert type(witness) is F.FinalExecutionWitness
    with pytest.raises(StageGateError, match="issued qualification capability"):
        selection.validate()


def test_issued_verifier_cannot_authorize_a_different_execution_domain(native_case):
    request = _request(native_case)
    changed = replace(request.binding, hardware_sha256="a" * 64)
    with pytest.raises(StageGateError, match="qualified runtime/toolchain/hardware/timer domain"):
        F._arm(request.verifier, changed, "candidate", request.candidate, request.original_reference)


@pytest.mark.parametrize("ancestor", [False, True])
def test_indirect_protected_records_rejected_even_with_matching_bytes(tmp_path, ancestor):
    owner = tmp_path / "owner"
    owner.mkdir()
    actual = owner / "record.json"
    actual.write_text('{"status":"qualified"}')
    alias = tmp_path / "alias"
    alias.symlink_to(owner if ancestor else actual, target_is_directory=ancestor)
    selected = alias / actual.name if ancestor else alias
    with pytest.raises(StageGateError, match="absent or indirect"):
        F._mapping(selected)
    with pytest.raises(StageGateError, match="absent or indirect"):
        F._pins(((selected, sha256_file(actual)),))


def test_code_identity_is_stable_across_actual_exception_execution():
    def rejects():
        raise RuntimeError("an actual exception")

    before = F.callable_code_sha256(rejects)
    for _ in range(3):
        with pytest.raises(RuntimeError):
            rejects()
        assert F.callable_code_sha256(rejects) == before
