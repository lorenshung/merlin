from __future__ import annotations

import json
import sys
from copy import deepcopy
from dataclasses import dataclass

import pytest

from merlin.targetgen.core_aten_bounded import (
    BoundedCoverageProfile,
    _case_id,
    _obligations,
    bounded_suite_digest,
)
from merlin.targetgen.core_aten_cases import case_document, core_aten_cases, eager_case_observation
from merlin.targetgen.core_aten_cover import exact_minimum_cover, overload_digest
from merlin.targetgen.core_aten_eqsat import quotient_observational_equivalents
from merlin.targetgen.core_aten_eval import (
    BoundedEagerSuiteAdapter,
    EagerSuiteAdapter,
    JsonCommandSuiteAdapter,
    evaluate_bounded_core_aten_suite,
    evaluate_core_aten_suite,
)

torch = pytest.importorskip("torch")


def _corpus(*names: str) -> dict:
    cases = [case_document(core_aten_cases()[name]) for name in sorted(names)]
    return {
        "schema_version": 1,
        "pytorch_version": torch.__version__,
        "overload_count": len(cases),
        "case_count": len(cases),
        "complete": True,
        "cases": cases,
    }


@dataclass
class RecordingHardwareAdapter:
    name: str = "recording-hardware"
    calls: int = 0

    def execute(self, corpus):
        self.calls += 1
        results = {}
        for case in corpus["cases"]:
            observation = eager_case_observation(case)
            observation["execution_kind"] = "hardware"
            observation["evidence"] = {
                "device": "unit-test-device",
                "decomposed": True,
                "receipt": f"unit-test::{case['overload']}",
            }
            results[case["overload"]] = observation
        return results

    def verify_hardware_receipt(self, case, observation):
        return observation["evidence"]["receipt"] == f"unit-test::{case['overload']}"


def test_one_function_call_runs_the_complete_suite_once() -> None:
    adapter = RecordingHardwareAdapter()
    result = evaluate_core_aten_suite(adapter)
    assert adapter.calls == 1
    assert result["status"] == "pass"
    assert result["passed_count"] == result["case_count"] == 193


def test_decomposition_and_provenance_do_not_affect_semantic_pass() -> None:
    adapter = RecordingHardwareAdapter()
    result = evaluate_core_aten_suite(adapter, corpus=_corpus("aten.add.Tensor"), validate_live_denominator=False)
    assert result["cases"]["aten.add.Tensor"]["status"] == "pass"
    assert result["cases"]["aten.add.Tensor"]["evidence"]["decomposed"] is True


def test_requested_hardware_rejects_host_execution() -> None:
    result = evaluate_core_aten_suite(
        EagerSuiteAdapter(),
        corpus=_corpus("aten.add.Tensor"),
        required_execution_kind="hardware",
        validate_live_denominator=False,
    )
    assert result["status"] == "fail"
    assert result["status_counts"] == {"execution_kind_mismatch": 1}


def test_missing_result_fails_closed() -> None:
    class MissingAdapter:
        name = "missing"

        def execute(self, corpus):
            return {}

    result = evaluate_core_aten_suite(
        MissingAdapter(), corpus=_corpus("aten.add.Tensor"), validate_live_denominator=False
    )
    assert result["status_counts"] == {"missing_result": 1}


def test_compiler_failure_is_preserved() -> None:
    class CompileFailureAdapter:
        name = "compile-failure"

        def execute(self, corpus):
            return {
                corpus["cases"][0]["overload"]: {
                    "status": "compile_failure",
                    "reason": "unsupported lowering",
                    "evidence": {"stage": "merlin"},
                }
            }

    result = evaluate_core_aten_suite(
        CompileFailureAdapter(),
        corpus=_corpus("aten.add.Tensor"),
        validate_live_denominator=False,
    )
    verdict = result["cases"]["aten.add.Tensor"]
    assert verdict["status"] == "compile_failure"
    assert verdict["reason"] == "unsupported lowering"


def test_output_mutation_and_alias_mismatches_fail() -> None:
    corpus = _corpus("aten.alias.default", "aten.resize_.default")
    adapter = RecordingHardwareAdapter()
    observations = adapter.execute(corpus)
    observations["aten.alias.default"]["output_input_aliases"] = []
    observations["aten.resize_.default"]["mutated_arguments"] = []

    class FixedAdapter:
        name = "fixed"

        def execute(self, corpus):
            return observations

    result = evaluate_core_aten_suite(FixedAdapter(), corpus=corpus, validate_live_denominator=False)
    assert result["status_counts"] == {"semantic_mismatch": 2}


def test_batch_adapter_crash_marks_every_case() -> None:
    class CrashingAdapter:
        name = "crashing"

        def execute(self, corpus):
            raise RuntimeError("queue unavailable")

    result = evaluate_core_aten_suite(
        CrashingAdapter(),
        corpus=_corpus("aten.add.Scalar", "aten.add.Tensor"),
        validate_live_denominator=False,
    )
    assert result["status_counts"] == {"adapter_error": 2}


def test_json_command_adapter_invokes_one_external_batch(tmp_path) -> None:
    corpus = _corpus("aten.add.Tensor")
    observation = eager_case_observation(corpus["cases"][0])
    observation["execution_kind"] = "hardware"
    observation["evidence"]["receipt"] = "external-test-receipt"
    response_json = json.dumps({"schema_version": 1, "results": {"aten.add.Tensor": observation}}, sort_keys=True)
    runner = tmp_path / "runner.py"
    runner.write_text(
        """\
import json
import pathlib
import sys

request = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
assert request["pytorch_version"]
assert request["complete"] is True
assert [case["overload"] for case in request["cases"]] == ["aten.add.Tensor"]
pathlib.Path(sys.argv[2]).write_text(RESPONSE + "\\n", encoding="utf-8")
""".replace("RESPONSE", repr(response_json)),
        encoding="utf-8",
    )
    adapter = JsonCommandSuiteAdapter(
        argv=(sys.executable, str(runner), "{request}", "{response}"),
        name="external-hardware",
        receipt_verifier=lambda _case, observed: observed["evidence"]["receipt"] == "external-test-receipt",
    )

    result = evaluate_core_aten_suite(adapter, corpus=corpus, validate_live_denominator=False)

    assert result["status"] == "pass"
    assert result["status_counts"] == {"pass": 1}


def test_malformed_observation_fails_closed() -> None:
    class MalformedAdapter:
        name = "malformed"

        def execute(self, corpus):
            return {corpus["cases"][0]["overload"]: ["not", "a", "mapping"]}

    result = evaluate_core_aten_suite(
        MalformedAdapter(), corpus=_corpus("aten.add.Tensor"), validate_live_denominator=False
    )
    assert result["status_counts"] == {"invalid_result": 1}


def test_default_api_rejects_a_declared_complete_subset() -> None:
    with pytest.raises(ValueError, match="live Core ATen denominator"):
        evaluate_core_aten_suite(EagerSuiteAdapter(), corpus=_corpus("aten.add.Tensor"))


def test_one_call_bounded_evaluator_uses_case_ids() -> None:
    assignment = {
        "control:alpha": "2",
        "dtype": "float32",
        "layout": "contiguous",
        "shape": "rectangular",
        "values": "ordinary",
    }
    case_id = _case_id("aten.add.Tensor", assignment)
    obligations = sorted(_obligations("aten.add.Tensor", assignment))
    case = {
        **case_document(core_aten_cases()["aten.add.Tensor"]),
        "case_id": case_id,
        "partition_assignment": assignment,
        "covered_obligations": obligations,
    }
    quotient, eqsat = quotient_observational_equivalents({case_id: case})
    case = quotient[case_id]
    cover = exact_minimum_cover(obligations, {case_id: obligations}).to_dict()
    profile = BoundedCoverageProfile()
    corpus = {
        "schema_version": 1,
        "scope": "bounded unit test",
        "pytorch_version": torch.__version__,
        "profile": profile.to_dict(),
        "profile_sha256": profile.sha256,
        "generator_sha256": "a" * 64,
        "engines": {
            "set_cover": cover["solver"],
            "set_cover_version": cover["solver_version"],
            "minimum_certificate": "unit-test certificate",
            "observational_quotient": "xdsl-eqsat",
            "observational_quotient_version": eqsat["xdsl_version"],
            "eqsat_rewrite_rules": [],
        },
        "overload_count": 1,
        "denominator_sha256": "unit-test-denominator",
        "selected_count": 1,
        "witnessed_obligation_count": len(obligations),
        "witnessed_obligations": obligations,
        "selected_union_sha256": overload_digest(obligations),
        "complete": True,
        "selected_cases": [case],
        "overloads": {
            "aten.add.Tensor": {
                "baseline": assignment,
                "witnessed_obligations": obligations,
                "selected_case_ids": [case_id],
                "cover": cover,
                "eqsat": eqsat,
                "candidates": {case_id: case},
            }
        },
    }
    corpus["suite_sha256"] = bounded_suite_digest(corpus)
    result = evaluate_bounded_core_aten_suite(
        BoundedEagerSuiteAdapter(),
        corpus,
        required_execution_kind="host",
        validate_live_denominator=False,
        validate_solver_certificate=False,
    )
    assert result["status"] == "pass"
    assert result["status_counts"] == {"pass": 1}
    tampered = deepcopy(corpus)
    tampered_case = tampered["selected_cases"][0]
    tampered_case["eqsat_eclass"]["observational_fingerprint"] = "0" * 64
    tampered["overloads"]["aten.add.Tensor"]["candidates"][case_id] = tampered_case
    tampered["suite_sha256"] = bounded_suite_digest(tampered)
    with pytest.raises(ValueError, match="eqsat fingerprint mismatch"):
        evaluate_bounded_core_aten_suite(
            BoundedEagerSuiteAdapter(),
            tampered,
            required_execution_kind="host",
            validate_live_denominator=False,
            validate_solver_certificate=False,
        )
    tampered = deepcopy(corpus)
    tampered["overloads"]["aten.add.Tensor"]["cover"]["minimum_cardinality_lower_bound"] = 0
    tampered["suite_sha256"] = bounded_suite_digest(tampered)
    with pytest.raises(ValueError, match="invalid stored SMT certificate"):
        evaluate_bounded_core_aten_suite(
            BoundedEagerSuiteAdapter(),
            tampered,
            required_execution_kind="host",
            validate_live_denominator=False,
            validate_solver_certificate=False,
        )
    with pytest.raises(ValueError, match="live Core ATen denominator"):
        evaluate_bounded_core_aten_suite(
            BoundedEagerSuiteAdapter(),
            corpus,
            required_execution_kind="host",
            validate_solver_certificate=False,
        )
