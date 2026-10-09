"""Live original source selection and mandatory qualification refusal controls.

These tests use actual native RTL/SW/source replay. They do not author or seed a
compiler, issue a functional runtime or infer static legality from compilation.
"""

from dataclasses import fields, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase0 import component_compile_sources as S
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal
from merlin_experiments.phase1.component_compile_admission import qualification_compile_roles, verify_compile_roster
from merlin_experiments.phase1.component_compile_roles import (
    ComponentCompileRoleEvaluation,
    evaluate_component_compile_roles,
)
from merlin_experiments.phase2.contracts import StageGateError
from test_component_compile_sources import (  # noqa: F401 -- actual independent source roster fixtures
    configured,
    member,
    plan,
    selected,
    selection,
)


def test_live_source_only_selection_reopens_complete_original_required_roster(configured):  # noqa: F811
    plan(configured, [member(), member("large-unseen", "withheld_transfer", M={"kind": "integer", "value": 1 << 30},
                                     N={"kind": "integer", "value": 7})])
    roster = S.issue_independent_compile_only_roster(**configured)
    arguments = dict(hardware=roster.hardware, software=roster.software, descriptor=roster.target_descriptor)
    assert verify_compile_roster(roster, **arguments) == roster.sha256
    assert len(roster.members) == 2
    assert roster.members[1].original_abi.inputs[0].shape == (1 << 30, 7)
    for claim in (None, roster.public_summary(), SimpleNamespace(**roster.public_summary())):
        with pytest.raises(StageGateError, match="live independent source-only roster"):
            verify_compile_roster(claim, **arguments)
    for forged in (replace(roster), replace(roster, members=roster.members[:1])):
        with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
            verify_compile_roster(forged, **arguments)
    with pytest.raises(StageGateError, match="hardware/software/target selection"):
        verify_compile_roster(roster, **{**arguments, "hardware": object()})
    with pytest.raises(StageGateError, match="hardware/software/target selection"):
        verify_compile_roster(roster, **{**arguments, "descriptor": roster.target_descriptor.parent / "different.yaml"})
    roster.members[1].source.write_text("module {}\n")
    with pytest.raises(RtlIntakeRefusal, match="source changed: original-source-only-member"):
        verify_compile_roster(roster, **arguments)


def test_issued_guard_only_source_roster_cannot_replace_required_private_scale_transfer(configured):  # noqa: F811
    plan(configured, [member()])
    roster = S.issue_independent_compile_only_roster(**configured)
    roster.verify()
    with pytest.raises(StageGateError, match="guards and withheld transfer"):
        verify_compile_roster(
            roster, hardware=roster.hardware, software=roster.software, descriptor=roster.target_descriptor
        )


@pytest.mark.parametrize("claim", [None, {"status": "qualified", "static_denominator": {"proved": 4}}, True])
def test_missing_or_saved_role_flags_refuse_before_any_compiler_invocation(claim, tmp_path):
    with pytest.raises(StageGateError, match="evaluated original source-only roles"):
        qualification_compile_roles(
            claim, origin=None, lineage=None, candidate=tmp_path / "candidate", contract_root=tmp_path / "contract"
        )
    assert not list(tmp_path.iterdir())


def test_reconstructing_a_transport_handle_does_not_issue_evaluated_role_authority(tmp_path):
    values = {item.name: None for item in fields(ComponentCompileRoleEvaluation)}
    values["_issuer"] = object()
    forged = ComponentCompileRoleEvaluation(**values)
    with pytest.raises(StageGateError, match="not issued around actual original member compilation"):
        forged.verify(candidate=tmp_path / "candidate")
    assert not list(tmp_path.iterdir())


def test_compile_evaluator_refuses_unbounded_or_declaration_only_build_inputs_before_mutation(tmp_path):
    for timeout, build in ((601, None), (20, {"source_pins": []})):
        with pytest.raises(StageGateError, match="bounded ordinary build services"):
            evaluate_component_compile_roles(
                roster=None, compiler_origin=None, candidate=tmp_path / "candidate",
                contract_root=tmp_path / "contract", build_service=build, instruction_check=None,
                readelf=Path("/missing/readelf"), evidence_root=tmp_path / "evidence", timeout_s=timeout,
            )
    assert not list(tmp_path.iterdir())
