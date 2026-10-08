"""Static capture demand stays separate from measured tuning-corpus priority."""

import hashlib
import json
from pathlib import Path

import pytest
from merlin_experiments.phase2 import bottleneck_priority as BP
from merlin_experiments.phase2 import measurement_evidence as ME
from merlin_experiments.phase2 import measurement_support as MS

RTL_SHA = "f" * 64


def _sources():
    basis = {
        "schema": "merlin.phase0.performance_basis.v1",
        "target": "fixture",
        "status": "observed",
        "sources": {"raw_rtl_facts_sha256": RTL_SHA},
        "applications": {
            "selected_capture": {
                "rows": [
                    {
                        "application": "selected_capture",
                        "operation": "contract",
                        "count": 1,
                        "independent_compute_demand": True,
                        "macs": {"total": 1000, "reason": None},
                    },
                    {
                        "application": "selected_capture",
                        "operation": "unpriced_map",
                        "count": 2,
                        "independent_compute_demand": True,
                        "macs": {"total": None, "reason": "semantic_form_has_no_defined_mac_count"},
                    },
                ]
            }
        },
    }
    raw = json.dumps(basis, sort_keys=True).encode()
    justification = {
        "schema": "merlin.phase2.test_justification.v1",
        "target": "fixture",
        "source": {"performance_basis_sha256": hashlib.sha256(raw).hexdigest(), "raw_rtl_facts_sha256": RTL_SHA},
        "members": [
            {
                "family": "f",
                "capsule": "a",
                "relative_path": "f/a",
                "capsule_tree_sha256": "a" * 64,
                "workload_need": {
                    "matched_demands": [
                        {
                            "application": "selected_capture",
                            "signature_index": 0,
                            "form_match": {"status": "exact_arithmetic_form"},
                        }
                    ]
                },
            },
            {
                "family": "f",
                "capsule": "b",
                "relative_path": "f/b",
                "capsule_tree_sha256": "b" * 64,
                "workload_need": {"matched_demands": []},
            },
        ],
    }
    scope = {
        "required": [{"signature": "movement -> contraction", "occurrences": 1}],
        "typed_required_instances": {
            "schema": "merlin.phase0.typed_scope_instances.v1",
            "instances": [{"instance_id": "chain-1", "signature": "movement -> contraction"}],
        },
        "performance": {
            "schema": "merlin.phase0.performance_scope.v1",
            "status": "no_eligible_chain",
            "required": [],
            "unresolved": [],
            "excluded": [
                {
                    "instance_id": "chain-1",
                    "signature": "movement -> contraction",
                    "status": "software_refused",
                    "reason": "mixed-lane precision refused",
                }
            ],
        },
    }
    return raw, justification, scope


def _counter_measurement(cycles: int):
    identity = {
        "program": {"kind": "compiler_command_buffer", "sha256": "a" * 64},
        "inputs": {"kind": "frozen_capsule_tree", "sha256": "b" * 64},
        "toolchain": {
            "target": "fixture",
            "frozen_submission_sha256": "c" * 64,
            "recorded_revisions": {"compiler": "revision-1"},
        },
    }
    common = {
        "discovery": {"status": "derived", "header_sha256": "d" * 64},
        "capacity": {"status": "derived", "slots": 8, "provenance": {"source": "rtl", "sha256": "e" * 64}},
        "measured_header_sha256": "d" * 64,
    }
    occupancy = {
        "measurement_identity": identity,
        "measurement_identity_refusals": [],
        "per_sim": {
            "gsim": {
                "correct": True,
                "cycles": cycles,
                "measurement_conditions": {"cycle_window": "region"},
                "counters": {
                    **common,
                    "selection": {"kind": "joint_occupancy", "unit": None},
                    "occupancy": {"by_combination": {"busy": "BUSY"}},
                    "readings": {"BUSY": cycles // 2},
                },
            }
        },
    }
    physical = {
        "measurement_identity": identity,
        "measurement_identity_refusals": [],
        "per_sim": {
            "gsim": {
                "correct": True,
                "cycles": cycles,
                "measurement_conditions": {"cycle_window": "region"},
                "counters": {
                    **common,
                    "selection": {"kind": "unit", "unit": "BYTES"},
                    "selected_counters": {"READ": 1, "WRITE": 2},
                    "readings": {"READ": 32, "WRITE": 8},
                },
            }
        },
    }
    facts = [
        {
            "fact_kind": "counter_byte_binding",
            "artifact_sha256": RTL_SHA,
            "counter_field": field,
            "direction": direction,
            "unit_bytes": 1,
            "derived_from_rtl": True,
            "provenance": "RTL counter proof",
        }
        for field, direction in (("READ", "read"), ("WRITE", "write"))
    ]
    linked = MS.link_counter_passes(
        occupancy,
        physical,
        physical_unit="BYTES",
        timing_simulator="gsim",
        rtl_facts_sha256=RTL_SHA,
        counter_binding={"status": "exact", "rtl_facts_sha256": RTL_SHA, "counter_facts": facts},
    )
    assert linked["status"] == "linked"
    return {
        **occupancy,
        "counter_passes": {"occupancy": occupancy, "physical_bytes": physical},
        "linked_counter_evidence": linked,
    }


def _verified(tmp_path: Path, *, tamper_counter: bool = False, capsules=("a", "b")):
    store = tmp_path / "raw_results" / "sha256"
    store.mkdir(parents=True)
    rows = []
    for capsule, baseline_samples, candidate_samples in (
        ("a", (100, 120), (80, 100)),
        ("b", (25, 60), (20, 50)),
    ):
        if capsule not in capsules:
            continue
        for arm, samples in (("baseline", baseline_samples), ("candidate", candidate_samples)):
            for replicate, cycles in zip(ME.REPLICATES, samples, strict=True):
                measurement = (
                    _counter_measurement(cycles)
                    if arm == "candidate" and capsule == "a" and replicate == "r000"
                    else {"per_sim": {"gsim": {"cycles": cycles}}}
                )
                if tamper_counter and arm == "candidate" and capsule == "a" and replicate == "r000":
                    measurement["linked_counter_evidence"] = {
                        **measurement["linked_counter_evidence"],
                        "rtl_facts_sha256": "0" * 64,
                    }
                raw = {
                    "schema": "paired_arm4_raw_execution_v2",
                    "execution": {
                        "phase": "tuning",
                        "arm": arm,
                        "family": "f",
                        "capsule": capsule,
                        "replicate": replicate,
                    },
                    "measurement": measurement,
                }
                payload = json.dumps(raw, sort_keys=True).encode()
                digest = hashlib.sha256(payload).hexdigest()
                path = store / f"{digest}.json"
                path.write_bytes(payload)
                rows.append(
                    {
                        "phase": "tuning",
                        "arm": arm,
                        "family": "f",
                        "capsule": capsule,
                        "simulator": "gsim",
                        "replicate": replicate,
                        "correct": True,
                        "citable": True,
                        "cycles": cycles,
                        "qualification": {"admitted": True},
                        "provenance": {
                            "tier": "L3",
                            "simulator": "gsim",
                            "oracle_kind": "rtl_gsim",
                            "derived_from_rtl": True,
                            "cycle_accurate": True,
                            "elf_sha256": "e" * 64,
                        },
                        "raw_result_sha256": digest,
                        "raw_result_path": str(path),
                    }
                )
                rows.append(
                    {
                        "phase": "tuning",
                        "arm": arm,
                        "family": "f",
                        "capsule": capsule,
                        "simulator": "spike",
                        "replicate": replicate,
                        "correct": True,
                        "citable": False,
                        "cycles": None,
                    }
                )
    selected = {
        "schema_version": 1,
        "target": "fixture",
        "capsules_sha256": "9" * 64,
        "capsules": [
            {"family": "f", "capsule": capsule, "source_relative_path": f"f/{capsule}", "snapshot_sha256": capsule * 64}
            for capsule in capsules
        ],
    }
    selected_bytes = json.dumps(selected, sort_keys=True).encode()
    manifest = {
        "schema": "paired_arm4_performance_campaign_v2",
        "phase": "tuning",
        "status": "GO",
        "frozen_corpus": {
            "visibility": "tuning",
            "manifest_sha256": hashlib.sha256(selected_bytes).hexdigest(),
            "capsules_sha256": selected["capsules_sha256"],
        },
        "engine_policy": {"timing_authority": "gsim"},
        "rtl_identity": {"rtl_facts": {"sha256": RTL_SHA}},
        "measurement_plan": {
            "expected_results": [
                {key: row[key] for key in ("phase", "arm", "family", "capsule", "simulator", "replicate")}
                for row in rows
            ]
        },
    }
    expected = tuple(ME.ResultIdentity(**row) for row in manifest["measurement_plan"]["expected_results"])
    manifest["completion"] = ME.completion_report(rows, expected)
    assert manifest["completion"]["complete"] is True
    return ME.VerifiedPairedMeasurement(
        tmp_path / "campaign_manifest.json", "1" * 64, manifest, tuple(rows)
    ), selected_bytes


def test_static_demand_and_excluded_graph_chain_do_not_invent_model_priority():
    raw, justification, scope = _sources()
    report = BP.build_report(raw, justification, selected_scope=scope)
    assert report["static_demand"]["unmatched"] == 1
    assert report["static_demand"]["unknown_macs"] == 1
    assert report["connected_slice"]["status"] == "no_eligible_chain"
    assert report["connected_slice"]["excluded"][0]["status"] == "software_refused"
    assert report["measured_priority"]["status"] == "insufficient_measurement"
    assert report["whole_model_priority"]["status"] == "insufficient_measurement"


def test_measured_priority_uses_candidate_cycles_and_exact_counter_receipts(tmp_path):
    raw, justification, scope = _sources()
    verified, selected = _verified(tmp_path)
    report = BP.build_report(
        raw, justification, verified, selected_corpus_manifest_bytes=selected, selected_scope=scope
    )
    priority = report["measured_priority"]
    assert priority["status"] == "measured_tuning_corpus_only"
    assert priority["baseline_total_cycles_by_replicate"] == {"r000": 125, "r001": 180}
    assert priority["candidate_total_cycles_by_replicate"] == {"r000": 100, "r001": 150}
    assert [(row["capsule"], row["rank"]) for row in priority["ranked"]] == [("a", 1), ("b", 2)]
    assert priority["ranked"][0]["candidate_cycle_share_range"] == [0.666667, 0.8]
    assert priority["ranked"][0]["perfect_elimination_speedup_upper_bound"] == 5.0
    assert priority["ranked"][0]["paired_effect"]["saved_cycles_by_replicate"] == {"r000": 20, "r001": 20}
    assert priority["ranked"][0]["paired_effect"]["direction"] == "improved"
    assert priority["ranked"][0]["development_source_match"]["exact_demands"][0]["application"] == "selected_capture"
    counters = priority["ranked"][0]["receipt_evidence"]["r000"]
    assert counters["physical_traffic"]["total_bytes"] == 40
    assert counters["resource_bottleneck"]["status"] == "unknown"
    assert report["whole_model_priority"]["status"] == "insufficient_measurement"


def test_candidate_only_timing_plan_cannot_be_ranked_as_paired(tmp_path):
    raw, justification, _ = _sources()
    verified, selected = _verified(tmp_path)
    verified.manifest["measurement_plan"]["expected_results"] = [
        row for row in verified.manifest["measurement_plan"]["expected_results"] if row["arm"] == "candidate"
    ]
    with pytest.raises(ValueError, match="complete baseline/candidate"):
        BP.build_report(raw, justification, verified, selected_corpus_manifest_bytes=selected)


def test_incorrect_baseline_or_missing_correctness_stays_unmeasured(tmp_path):
    raw, justification, _ = _sources()
    verified, selected = _verified(tmp_path)
    rows = list(verified.rows)
    first = next(row for row in rows if row["arm"] == "baseline" and row["simulator"] == "spike")
    first["correct"] = False
    report = BP.build_report(raw, justification, verified, selected_corpus_manifest_bytes=selected)
    assert report["measured_priority"]["status"] == "insufficient_measurement"
    assert "matrix is incomplete" in report["measured_priority"]["reason"]


def test_regression_is_visible_but_not_called_a_whole_model_effect(tmp_path):
    raw, justification, _ = _sources()
    verified, selected = _verified(tmp_path)
    for row in verified.rows:
        if row["arm"] == "baseline" and row["capsule"] == "b" and row["simulator"] == "gsim":
            row["cycles"] = 10
    report = BP.build_report(raw, justification, verified, selected_corpus_manifest_bytes=selected)
    assert report["measured_priority"]["status"] == "measured_tuning_corpus_only"
    assert report["measured_priority"]["ranked"][1]["paired_effect"]["direction"] == "regressed"
    assert report["whole_model_priority"]["status"] == "insufficient_measurement"


def test_counter_claim_with_different_rtl_facts_stays_unknown(tmp_path):
    raw, justification, _ = _sources()
    verified, selected = _verified(tmp_path, tamper_counter=True)
    report = BP.build_report(raw, justification, verified, selected_corpus_manifest_bytes=selected)
    counter = report["measured_priority"]["ranked"][0]["receipt_evidence"]["r000"]
    assert counter["status"] == "counter_attribution_unknown"
    assert counter["physical_traffic"] is None


def test_selected_tuning_subset_lists_other_justified_members_as_unmeasured(tmp_path):
    raw, justification, _ = _sources()
    verified, selected = _verified(tmp_path, capsules=("a",))
    report = BP.build_report(raw, justification, verified, selected_corpus_manifest_bytes=selected)
    priority = report["measured_priority"]
    assert priority["status"] == "measured_tuning_corpus_only"
    assert priority["unmeasured_justified_members"] == [{"family": "f", "capsule": "b"}]
    assert priority["ranked"][0]["candidate_cycle_share_range"] == [1.0, 1.0]
    assert priority["ranked"][0]["perfect_elimination_speedup_upper_bound"] is None


def test_selected_capsule_hash_must_match_justification(tmp_path):
    raw, justification, _ = _sources()
    verified, selected = _verified(tmp_path)
    changed = json.loads(selected)
    changed["capsules"][0]["snapshot_sha256"] = "0" * 64
    changed_bytes = json.dumps(changed, sort_keys=True).encode()
    verified.manifest["frozen_corpus"]["manifest_sha256"] = hashlib.sha256(changed_bytes).hexdigest()
    with pytest.raises(ValueError, match="capsule differs"):
        BP.build_report(raw, justification, verified, selected_corpus_manifest_bytes=changed_bytes)


def test_absent_phase0_inventory_is_unknown_not_zero():
    raw, justification, _ = _sources()
    basis = json.loads(raw)
    basis["status"] = "not_available"
    basis["applications"] = {}
    raw = json.dumps(basis, sort_keys=True).encode()
    justification["source"]["performance_basis_sha256"] = hashlib.sha256(raw).hexdigest()
    justification["members"][0]["workload_need"]["matched_demands"] = []
    census = BP.build_report(raw, justification)["static_demand"]
    assert census["unmatched"] is None
    assert census["unknown_macs"] is None


def test_changed_basis_identity_is_refused():
    raw, justification, _ = _sources()
    with pytest.raises(ValueError, match="differs"):
        BP.build_report(raw + b" ", justification)


def test_publish_optional_report_uses_frozen_selected_context(tmp_path, monkeypatch):
    raw, justification, _ = _sources()
    verified, selected_bytes = _verified(tmp_path)
    selected = json.loads(selected_bytes)
    justification_bytes = json.dumps(justification).encode()
    selected["test_justification"] = {
        "path": "test_justification.json",
        "sha256": hashlib.sha256(justification_bytes).hexdigest(),
        "inputs": {"performance-basis.json": hashlib.sha256(raw).hexdigest()},
    }
    selected_bytes = json.dumps(selected, sort_keys=True).encode()
    selected_path = tmp_path / "performance_corpus_manifest.json"
    selected_path.write_bytes(selected_bytes)
    verified.manifest["frozen_corpus"]["manifest_sha256"] = hashlib.sha256(selected_bytes).hexdigest()
    (tmp_path / "test_justification.json").write_bytes(justification_bytes)
    inputs = tmp_path / "test_justification_inputs"
    inputs.mkdir()
    (inputs / "performance-basis.json").write_bytes(raw)
    binding = object()

    def verify(path, *, expected):
        assert path.resolve() == verified.manifest_path.resolve()
        assert expected is binding
        return verified

    monkeypatch.setattr(ME, "verify_paired_measurement", verify)
    receipt = BP.publish_optional_report(tmp_path, verified.manifest_path, expected=binding)
    report_path = tmp_path / "bottleneck_priority.json"
    assert receipt == {
        "status": "written",
        "path": str(report_path),
        "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
    }
    report = json.loads(report_path.read_bytes())
    assert report["source"]["paired_tuning_manifest_sha256"] == verified.manifest_sha256
    assert report["source"]["test_justification_sha256"] == hashlib.sha256(justification_bytes).hexdigest()
    assert report["measured_priority"]["status"] == "measured_tuning_corpus_only"
    assert report["connected_slice"]["status"] == "unknown"
    assert report["whole_model_priority"]["status"] == "insufficient_measurement"
    assert BP.publish_optional_report(tmp_path, verified.manifest_path, expected=binding)["status"] == "already_present"
    (tmp_path / "test_justification.json").write_text("{}")
    with pytest.raises(ValueError, match="frozen Phase 0 context"):
        BP.publish_optional_report(tmp_path, verified.manifest_path, expected=binding)
    (tmp_path / "test_justification.json").write_bytes(justification_bytes)
    report_path.write_text("different report")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        BP.publish_optional_report(tmp_path, verified.manifest_path, expected=binding)
    assert report_path.read_text() == "different report"
    alias = tmp_path / "run_alias"
    alias.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(ValueError, match="symlinked parent"):
        BP.publish_optional_report(tmp_path, alias / "campaign_manifest.json", expected=binding)


def test_optional_report_skips_older_frozen_corpus_without_changing_measurement(tmp_path, monkeypatch):
    (tmp_path / "performance_corpus_manifest.json").write_text(json.dumps({"schema_version": 1}))

    def fail_verify(*_args, **_kwargs):
        pytest.fail("absent optional context must not trigger measurement re-admission")

    monkeypatch.setattr(ME, "verify_paired_measurement", fail_verify)
    result = BP.publish_optional_report(tmp_path, tmp_path / "campaign_manifest.json", expected=object())
    assert result["status"] == "not_available"
    assert not (tmp_path / "bottleneck_priority.json").exists()


def test_published_priority_replays_bound_excluded_connected_slice(tmp_path, monkeypatch):
    raw, justification, scope = _sources()
    verified, selected_bytes = _verified(tmp_path)
    scope_bytes = json.dumps(scope, sort_keys=True).encode()
    scope_sha = hashlib.sha256(scope_bytes).hexdigest()
    requirement_sha = "d" * 64
    justification["source"].update({"selected_scope_sha256": scope_sha, "selected_requirement_sha256": requirement_sha})
    justification_bytes = json.dumps(justification).encode()
    selected = json.loads(selected_bytes)
    selected["test_justification"] = {
        "path": "test_justification.json",
        "sha256": hashlib.sha256(justification_bytes).hexdigest(),
        "selected_requirement_sha256": requirement_sha,
        "inputs": {
            "performance-basis.json": hashlib.sha256(raw).hexdigest(),
            "performance-scope.json": scope_sha,
        },
    }
    selected_bytes = json.dumps(selected, sort_keys=True).encode()
    verified.manifest["frozen_corpus"]["manifest_sha256"] = hashlib.sha256(selected_bytes).hexdigest()
    (tmp_path / "performance_corpus_manifest.json").write_bytes(selected_bytes)
    (tmp_path / "test_justification.json").write_bytes(justification_bytes)
    inputs = tmp_path / "test_justification_inputs"
    inputs.mkdir()
    (inputs / "performance-basis.json").write_bytes(raw)
    (inputs / "performance-scope.json").write_bytes(scope_bytes)
    monkeypatch.setattr(ME, "verify_paired_measurement", lambda *_args, **_kwargs: verified)

    BP.publish_optional_report(tmp_path, verified.manifest_path, expected=object())
    report = json.loads((tmp_path / "bottleneck_priority.json").read_bytes())
    assert report["connected_slice"]["status"] == "no_eligible_chain"
    assert report["connected_slice"]["excluded"][0]["status"] == "software_refused"
    assert report["source"]["selected_requirement_sha256"] == requirement_sha
    assert report["whole_model_priority"]["status"] == "insufficient_measurement"
