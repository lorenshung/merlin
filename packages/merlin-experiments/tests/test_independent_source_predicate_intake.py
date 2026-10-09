"""Actual tracked source predicates, original command binding and drift controls."""

import json
import subprocess
from dataclasses import replace

import pytest
from merlin_experiments.phase0.command_intake import issue_independent_command_intake
from merlin_experiments.phase0.rtl_intake import RtlIntakeRefusal
from merlin_experiments.phase0.source_predicate_intake import (
    SourcePredicateSelection,
    issue_independent_source_predicate_intake,
)
from test_independent_command_intake import command_selection  # noqa: F401 -- shared actual native replay fixture
from test_independent_rtl_intake import selected  # noqa: F401 -- shared actual native replay fixture


@pytest.fixture
def selection(command_selection, tmp_path):  # noqa: F811 -- shared registered fixture
    checkout = command_selection["checkout"]
    route = checkout / "Routing.scala"
    route.write_text(
        "object Routing {\n val run = cmd.code === MOVE\n val config = cmd.code >= EXECUTE && cmd.code <= MOVE\n}\n"
    )
    subprocess.run(["git", "-C", str(checkout), "add", route.name], check=True)
    subprocess.run(["git", "-C", str(checkout), "commit", "-qm", "independent predicate source"], check=True)
    commit = subprocess.check_output(["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True).strip()
    command_selection["commit"] = commit
    command = issue_independent_command_intake(**command_selection)
    return dict(
        command_intake=command,
        public_checkout=checkout,
        commit=commit,
        selections=(
            SourcePredicateSelection(route, "run", "cmd.code"),
            SourcePredicateSelection(route, "config", "cmd.code"),
        ),
        forbidden_roots=command_selection["forbidden_roots"],
        output_root=tmp_path / "predicates",
    )


def test_real_selected_source_predicates_and_original_command_authority(selection):
    actual = issue_independent_source_predicate_intake(**selection)
    actual.verify()
    assert actual.command_intake is selection["command_intake"]
    facts = actual.public_facts()
    assert facts["command_intake_sha256"] == selection["command_intake"].sha256
    assert facts["predicates"][0]["selected_source_values"] == [17]
    assert facts["predicates"][1]["selected_source_values"] == [3, 17]
    assert facts["predicates"][0]["source"] == "Routing.scala"
    assert "source_to_selected_rtl_elaboration_correspondence" in facts["unknowns"]
    actual.verify_public_facts(selection["output_root"] / "facts.json")
    facts["predicates"][0]["selected_source_values"] = [3]
    assert actual.public_facts()["predicates"][0]["selected_source_values"] == [17]


def test_copy_or_saved_record_cannot_authorize_source_membership(selection):
    actual = issue_independent_source_predicate_intake(**selection)
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        replace(actual).verify()
    object.__setattr__(actual, "facts_json", b"{}")
    with pytest.raises(RtlIntakeRefusal, match="live independently issued"):
        actual.verify()


def test_source_drift_and_public_policy_value_injection_refuse(selection):
    actual = issue_independent_source_predicate_intake(**selection)
    injected = selection["output_root"] / "extra.json"
    facts = actual.public_facts()
    facts["prohibited_values"] = [3]
    injected.write_text(json.dumps(facts))
    with pytest.raises(RtlIntakeRefusal, match="complete issued"):
        actual.verify_public_facts(injected)
    source = selection["selections"][0].source
    source.write_text(source.read_text() + "// changed")
    with pytest.raises(RtlIntakeRefusal, match="tracked modifications"):
        actual.verify()


def test_outside_declaration_alias_does_not_become_instruction_identity(selection):
    selection["selections"] = (SourcePredicateSelection(selection["selections"][0].source, "run", "different.code"),)
    with pytest.raises(RtlIntakeRefusal, match="exact selected operand"):
        issue_independent_source_predicate_intake(**selection)


def test_source_selection_outside_public_checkout_and_protected_path_refuse(selection):
    selection["public_checkout"] = selection["public_checkout"].parent
    with pytest.raises(RtlIntakeRefusal, match="exact selected public"):
        issue_independent_source_predicate_intake(**selection)
    protected = selection["forbidden_roots"][0]
    selection["public_checkout"] = protected
    with pytest.raises(RtlIntakeRefusal, match="protected implementation"):
        issue_independent_source_predicate_intake(**selection)
