"""Installed provider reaches normal broker with exact member and cost identities."""

import json
import subprocess
import sys
import time
from dataclasses import replace
from functools import partial
from pathlib import Path
from types import SimpleNamespace

import pytest
from component_baseline_fixture import synthetic_component_controller as synthetic_component_controller
from component_baseline_fixture import unissued_runtime
from merlin_experiments.phase2 import broker
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import component_analytical as CA
from merlin_experiments.phase2 import component_workflow as CW
from merlin_experiments.phase2.contracts import StageGateError, document_sha256, sha256_file
from merlin_experiments.phase2.supervised_feedback import bounded_feedback
from test_component_workflow import context

from merlin.benchharness import hash_tree
from merlin.common.jsonio import canonical_sha256 as sha
from merlin.perf.component_cost import (
    COMPLETE_STAGES,
    ComponentCostRegion,
    ComponentCostScope,
    ComponentFeatureObservation,
)
from merlin.perf.component_screen import qualify_component_screen
from merlin.perf.fast_estimate_validation import Observation
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval

pytestmark = pytest.mark.usefixtures("synthetic_component_controller")


def feature_provider(*, baseline, candidate, member, corpus, target_descriptor, scope, workspace, timeout_s):
    assert timeout_s <= 600
    result = []
    for arm, compiler, count in (("baseline", baseline, 13), ("candidate", candidate, 7)):
        artifact = workspace / (arm + ".elf")
        artifact.write_bytes((compiler / "compiler").read_bytes())
        result.append(
            ComponentFeatureObservation(
                hash_tree(compiler)["sha256"],
                member.source_sha256,
                corpus.capsules_sha256,
                sha256_file(target_descriptor),
                scope.sha256,
                sha("domain"),
                sha256_file(artifact),
                sha("deps"),
                sha("inputs"),
                (sha("evidence"),),
                (ComponentCostRegion("cold", COMPLETE_STAGES, ("compute",), {"count": count + 3}),),
                (ComponentCostRegion("warm", COMPLETE_STAGES, ("compute",), {"count": count}),),
                "PASS",
                ((str(artifact), sha256_file(artifact)),),
                legality_status="PASS",
            )
        )
    return tuple(result)


def controlled_calibration(target):
    receipts = [sha(index) for index in range(4)]
    return {
        "schema": "phase2_host_analytical_calibration_v1",
        "target_sha256": sha256_file(target),
        "evidence_sha256s": receipts,
        "composition": {"operator": "sum", "eta": 0, "provenance_sha256": receipts[0]},
        "accelerator_compute_roles": ["execute"],
        "risk_score": 0,
        "features": [
            {
                "id": kind,
                "pointer": "/count",
                "resource": kind,
                "kind": kind,
                "cycles_per_unit": {"lo": 1, "hi": 1},
                "physical_bytes_per_unit": 4 if kind != "compute" else None,
                "commands_per_unit": 1 if kind != "compute" else None,
                "transitions_per_unit": 1 if kind == "encoding" else None,
                "provenance_sha256": receipts[i + 1],
            }
            for i, kind in enumerate(("compute", "movement", "encoding"))
        ],
    }


class Predictor:
    def predict(self, features, *, domain_sha256):
        return CycleInterval.point(features["/count"], "controlled held prediction")


def provider(tmp_path, monkeypatch, selected, *, qualified=True, feature_callback=feature_provider):
    # Reporting arithmetic only: this unissued controller is never a physical
    # applicability capability. The real guard is separately exercised below.
    monkeypatch.setattr(CA, "_applicability", lambda _binding, _observation, _workspace: (None, None))
    baseline = selected["baseline_admission"].baseline
    adapter = tmp_path / "calibration-adapter.json"
    adapter.write_text("{}")
    calibration = controlled_calibration(selected["target_experiment"].path)
    for module in (CA, CW):
        monkeypatch.setattr(
            module, "prepare_phase2_calibration", lambda path: {"status": "ready", "calibration": calibration}
        )
    qualification = None
    if qualified:
        rows = [
            Observation(sha([g, n]), sha(g), g, sha("domain"), {"/count": n}, n, (sha([g, n, "receipt"]),))
            for g in ("a", "b", "c")
            for n in range(1, 11)
        ]
        qualification = tmp_path / "held-validation.json"
        qualification.write_text(
            json.dumps(
                qualify_component_screen(
                    rows, lambda train: Predictor(), calibration_sha256=document_sha256(calibration)
                )
            )
        )
    pins = {Path(__file__): sha256_file(Path(__file__)), adapter: sha256_file(adapter)}
    pins.update(CA._feature_context_pins(feature_callback))
    if qualification is not None:
        pins[qualification] = sha256_file(qualification)
    runtime = unissued_runtime(target_descriptor=selected["target_experiment"].path,
                               feature_provider=feature_callback, source_pins=tuple(pins.items()))
    selected["independent_runtime"] = runtime
    return CA.build_component_analytical_provider(
        baseline=baseline,
        corpus=selected["component_corpus"],
        target_descriptor=selected["target_experiment"].path,
        feature_provider=feature_callback,
        calibration_adapter=adapter,
        scope=ComponentCostScope(sha("timer"), sha("accuracy"), sha("input policy")),
        output=tmp_path / "host-output",
        lease_path=tmp_path / "engine.lease",
        dependencies={Path(__file__): sha256_file(Path(__file__))},
        qualification=qualification,
        max_workers=2,
        memory_per_worker_bytes=1 << 20,
        engine_slots=2,
        baseline_admission=selected["baseline_admission"],
        independent_runtime=runtime,
    )


def test_calibration_and_held_files_require_independent_membership(tmp_path, monkeypatch):
    selected = context(tmp_path)
    own = provider(tmp_path, monkeypatch, selected)
    foreign = tmp_path / "foreign-coefficients.json"
    foreign.write_text("{}")
    with pytest.raises(StageGateError, match="outside independent runtime membership"):
        CA._independent_calibration(foreign, own.binding.independent_runtime, own.binding.qualification)
    with pytest.raises(StageGateError, match="outside independent runtime membership"):
        CA._independent_calibration(own.calibration_adapter, own.binding.independent_runtime, foreign)
    monkeypatch.setattr(CA, "prepare_phase2_calibration", lambda _: {
        "calibration": controlled_calibration(selected["target_experiment"].path),
        "evidence_files": [{"path": str(foreign), "sha256": sha256_file(foreign)}],
    })
    with pytest.raises(StageGateError, match="outside independently qualified membership"):
        CA._independent_calibration(own.calibration_adapter, own.binding.independent_runtime, own.binding.qualification)


def test_normal_broker_uses_bounded_factory_and_redacted_complete_cost(tmp_path, monkeypatch):
    selected = context(tmp_path)
    own = provider(tmp_path, monkeypatch, selected)
    policy = BP.select_workflow(BP.COMPONENT_ONLY_V1, **selected, component_analytical=own)
    engine = broker.Broker(
        SimpleNamespace(),
        selected["target_experiment"],
        selected["candidate"],
        policy.build_registry(),
        selected["receipt_path"],
        workflow=policy,
        deadline=time.monotonic() + 30,
        max_calls=3,
        max_tool_seconds=10,
    )
    result = engine.execute({"action": CW.ANALYTICAL_ACTION})
    assert result["returncode"] == 0, result
    document = json.loads(result["stdout"])
    row = document["evidence"][0]
    assert row["baseline"]["lo"] == 13 and row["candidate"]["lo"] == 7
    assert row["complete_cost"]["candidate"]["regimes"]["cold"]["total"]["lo"] == 10
    assert document["configuration_sha256"] == own.configuration_sha256
    assert str(own.binding.baseline) not in result["stdout"]
    assert str(own.binding.output) not in result["stdout"]
    assert document["promotion"] == "NO_FINAL_ACCEPTANCE"
    assert document["screening"]["order"] == [document_sha256([row["family"], row["capsule"]])]
    assert document["screening"]["evidence"][0]["conservative_saved_cycles"] == 6
    assert document["screening"]["evidence"][0]["legal"] is True
    assert document["screening"]["work_share_basis"] == "equal generated families and equal members within family"
    lifecycle = json.loads(next((selected["receipt_path"].parent / "component_workers").glob(
        "component_worker_*/lifecycle.json")).read_text())
    assert lifecycle["cleanup"]["status"] == "COMPLETE" and lifecycle["lease_release"]["status"] == "RELEASED"
    assert lifecycle["lease_release"]["path"] == str(own.binding.lease_path)
    CW.validate_component_feedback(document)
    own.binding.baseline.joinpath("compiler").write_text("changed baseline")
    with pytest.raises(StageGateError, match="freeze changed"):
        policy._validate()


def test_unqualified_domain_keeps_totals_unknown_and_region_prices_visible(tmp_path, monkeypatch):
    selected = context(tmp_path)
    own = provider(tmp_path, monkeypatch, selected, qualified=False)
    raw = bounded_feedback(
        own.evaluate,
        timeout_s=5,
        output=tmp_path / "workers",
        kwargs={"candidate": selected["candidate"], "corpus": selected["component_corpus"], "timeout_s": 5},
    )
    identity = next(iter(raw))
    assert not raw[identity][0].resolved
    assert raw.reports[identity]["baseline"]["regimes"]["warm"]["regions"][0]["cycles"]["resolved"]


def different_input_provider(**kwargs):
    baseline, candidate = feature_provider(**kwargs)
    return baseline, replace(candidate, inputs_sha256=sha("different inputs"))


def test_compiler_arms_refuse_different_input_bytes(tmp_path, monkeypatch):
    selected = context(tmp_path)
    own = provider(tmp_path, monkeypatch, selected, feature_callback=different_input_provider)
    with pytest.raises(StageGateError, match="identical admitted inputs"):
        own.evaluate(candidate=selected["candidate"], corpus=selected["component_corpus"], timeout_s=5)


def test_prepared_target_context_pins_bind_cache_and_revalidate(tmp_path, monkeypatch):
    selected = context(tmp_path)
    engine = tmp_path / "private-engine"
    engine.write_bytes(b"pinned execution engine")
    prepared = partial(feature_provider)
    prepared.component_source_pins = {engine: sha256_file(engine)}
    own = provider(tmp_path, monkeypatch, selected, feature_callback=prepared)
    assert (engine, sha256_file(engine)) in own.binding.dependencies
    engine.write_bytes(b"different execution engine")
    with pytest.raises(StageGateError, match="changed"):
        own.binding.validate(controlled_calibration(selected["target_experiment"].path))


def rogue_callback(*, pidfile):
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    pidfile.write_text(str(child.pid))
    time.sleep(30)


def test_worker_deadline_cancels_uncooperative_callback_and_new_session_descendant(tmp_path):
    pidfile = tmp_path / "child.pid"
    started = time.monotonic()
    with pytest.raises(StageGateError, match="deadline"):
        bounded_feedback(rogue_callback, timeout_s=0.3, output=tmp_path / "workers", kwargs={"pidfile": pidfile})
    assert time.monotonic() - started < 2
    pid = int(pidfile.read_text())
    stat = Path(f"/proc/{pid}/stat")
    if stat.exists():
        assert stat.read_text().split(")", 1)[1].split()[0] == "Z"


@pytest.mark.parametrize(
    "arm,status,expected",
    [(0, "UNKNOWN", "UNRESOLVED"), (0, "FAIL", "REFUSAL"), (1, "UNKNOWN", "UNRESOLVED"), (1, "FAIL", "REFUSAL")],
)
def test_paired_legality_unknown_differs_from_failure(tmp_path, monkeypatch, arm, status, expected):
    selected = context(tmp_path)

    def selected_features(**kwargs):
        pair = list(feature_provider(**kwargs))
        pair[arm] = replace(pair[arm], legality_status=status)
        return tuple(pair)

    own = provider(tmp_path, monkeypatch, selected, feature_callback=selected_features)
    raw = own.evaluate(candidate=selected["candidate"], corpus=selected["component_corpus"], timeout_s=5)
    assert all(row["status"] == expected for row in raw.screening["evidence"])


class LivePreparedFeatureOwner:
    def __init__(self, engine):
        self.engine = engine
        self.original_sha = sha256_file(engine)

    @property
    def component_source_pins(self):
        if sha256_file(self.engine) != self.original_sha:
            raise StageGateError("live selected execution context changed")
        return {self.engine: self.original_sha}

    def evaluate(self, **kwargs):
        return feature_provider(**kwargs)


def test_bound_prepared_owner_revalidates_live_context_before_cached_use(tmp_path, monkeypatch):
    selected = context(tmp_path)
    engine = tmp_path / "private-engine"
    engine.write_bytes(b"original selected context")
    prepared = LivePreparedFeatureOwner(engine)
    own = provider(tmp_path, monkeypatch, selected, feature_callback=prepared.evaluate)
    own.evaluate(candidate=selected["candidate"], corpus=selected["component_corpus"], timeout_s=5)
    assert list((own.binding.output / "cache").glob("*.pickle"))
    engine.write_bytes(b"changed selected context")
    with pytest.raises(StageGateError, match="changed"):
        own.evaluate(candidate=selected["candidate"], corpus=selected["component_corpus"], timeout_s=5)
