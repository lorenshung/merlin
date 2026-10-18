"""Bounded definition generation keeps the denominator and strict lane obligations."""

import json
import runpy
from types import SimpleNamespace

import pytest
import yaml


@pytest.fixture
def definition_inputs(tmp_path):
    from merlin.common.paths import repo_root

    root = repo_root()
    generate = runpy.run_path(str(root / "build_tools/scripts/make_bounded_phase0.py"))["generate"]
    repository = tmp_path / "repository"
    target = repository / "examples/gemmini/target"
    target.mkdir(parents=True)
    (target / "descriptor.yaml").write_text("target: gemmini\nworkload_spec: {applications: [example]}\n")
    cases = [dict(case_id=f"support::{n}", overload="support") for n in range(4)]
    cases += [dict(case_id=f"other::{n}", overload=f"other{n % 3}") for n in range(6)]
    suite = tmp_path / "suite.json"
    suite.write_text(json.dumps({"selected_cases": cases, "denominator_sha256": "original"}))
    support = tmp_path / "support.txt"
    support.write_text("support\n")
    args = SimpleNamespace(
        suite=suite,
        support=support,
        captures=tmp_path / "captures",
        host_exceptions=None,
        other_count=3,
        output=tmp_path / "definition",
        descriptor=target / "descriptor.yaml",
        software_spec=target / "software-spec.yaml",
        host_capabilities=target / "host-capabilities.yaml",
        execution_provider=tmp_path / "provider.py",
        capability_contract=target / "contract.yaml",
        performance_template=tmp_path / "performance.yaml",
        device_package=tmp_path / "provider",
        facts=tmp_path / "facts.json",
        overlay=tmp_path / "overlay.json",
        workers=2,
    )
    return generate, args


def test_strict_definition_retains_every_support_case_and_disjoint_hidden_call(definition_inputs):
    generate, args = definition_inputs
    generate(args)
    recipe = yaml.safe_load((args.output / "recipe.yaml").read_bytes())
    cohorts = recipe["core_aten"]["cohorts"]
    assert cohorts["public"][0]["lane"] == "device"
    assert set(cohorts["public"][0]["select"]) == {f"support::{n}" for n in range(4)}
    all_ids = [key for rows in cohorts.values() for row in rows for key in row["select"]]
    assert len(all_ids) == len(set(all_ids)) == 7
    assert recipe["core_aten"]["workers"] == 2
    cases = json.loads((args.output / "cases.json").read_bytes())
    assert {case["overload"] for case in cases["cases"]} == {"support", "other0", "other1", "other2"}
    assert cases["denominator_sha256"] == "original"


def test_exceptions_require_evidence_and_remain_in_denominator(definition_inputs):
    generate, args = definition_inputs
    args.host_exceptions = args.suite.parent / "exceptions.json"
    row = dict(case_id="support::0", category="empty", reason="no device work", evidence="capture receipt")
    args.host_exceptions.write_text(json.dumps([row]))
    generate(args)
    definition = json.loads((args.output / "definition.json").read_bytes())
    assert definition["support_count"] == 4 and definition["device_count"] == 3
    assert definition["host_exceptions"] == [row]
    assert definition["selected_count"] == 7


@pytest.mark.parametrize(
    "rows",
    [[{"case_id": "support::0"}], [dict(case_id="absent", category="empty", reason="empty", evidence="receipt")]],
)
def test_invalid_exception_is_refused_before_creating_output(definition_inputs, rows):
    generate, args = definition_inputs
    args.host_exceptions = args.suite.parent / "exceptions.json"
    args.host_exceptions.write_text(json.dumps(rows))
    with pytest.raises(ValueError, match="host exception"):
        generate(args)
    assert not args.output.exists()
