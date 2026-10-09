"""Real callback processes exercise bounded data and late-result refusals."""
import json
import os
import signal
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import pytest
from component_baseline_fixture import synthetic_component_controller as synthetic_component_controller
from merlin_experiments.phase2 import broker_policy as policy_owner
from merlin_experiments.phase2 import component_workflow as workflow
from merlin_experiments.phase2.component_analytical import ComponentAnalyticalResults
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from merlin_experiments.phase2.feedback_protocol import (
    FeedbackValueLimits,
    decode_feedback_value,
    encode_feedback_value,
    receive_feedback_value,
)
from merlin_experiments.phase2.portfolio_launch import acquire_host_resource_lease
from merlin_experiments.phase2.supervised_feedback import bounded_feedback
from test_component_workflow import analytical, context, provider

from merlin.benchharness import hash_tree
from merlin.xdsl_dialects.lowering.global_plan import CycleInterval


def return_plain_result():
    return {("family", "member"): (CycleInterval.point(12), CycleInterval.unknown("unmeasured"))}


def lifecycle(root):
    receipts = list(root.glob("component_worker_*/lifecycle.json"))
    assert len(receipts) == 1
    return json.loads(receipts[0].read_text())


def test_actual_callback_roundtrip_retains_owned_cleanup_and_unrequested_lease_scope(tmp_path):
    result = bounded_feedback(return_plain_result, timeout_s=2, output=tmp_path / "workers", kwargs={})
    assert result == return_plain_result()
    receipt = lifecycle(tmp_path / "workers")
    assert receipt["status"] == "returned" and receipt["worker"]["reaped"]
    assert receipt["received_bytes"] > 0 and receipt["deadline_exceeded"] is False
    assert receipt["cleanup"]["status"] == "COMPLETE" and receipt["lease_release"]["status"] == "NOT_REQUESTED"
    assert receipt["guardian"]["reaped"] and "OS hard deadline UNKNOWN" in receipt["scope"]
    assert receipt["sources"] and not list((tmp_path / "workers").glob("**/result.pickle"))


def test_closed_codec_preserves_existing_analytical_envelope():
    value = ComponentAnalyticalResults(return_plain_result(), {("family", "member"): {"stages": ["complete"]}},
                                       {"promotion": "SCREENING_ONLY"})
    limits = FeedbackValueLimits()
    observed = decode_feedback_value(encode_feedback_value(value, limits=limits), limits=limits)
    assert type(observed) is ComponentAnalyticalResults and observed == value
    assert observed.reports == value.reports and observed.screening == value.screening


def stalled_reconstruction(marker):
    Path(marker).write_text("ran arbitrary reconstruction")
    time.sleep(30)


class ReconstructionTrap:
    def __init__(self, marker):
        self.marker = marker

    def __reduce__(self):
        return stalled_reconstruction, (str(self.marker),)


def return_reconstruction_trap(*, marker):
    return ReconstructionTrap(marker)


def test_actual_worker_cannot_schedule_arbitrary_parent_deserialization(tmp_path):
    marker = tmp_path / "decoder-ran"
    started = time.monotonic()
    with pytest.raises(StageGateError, match="unsupported result type"):
        bounded_feedback(return_reconstruction_trap, timeout_s=1, output=tmp_path / "workers",
                         kwargs={"marker": marker})
    assert time.monotonic() - started < 2 and not marker.exists()
    assert lifecycle(tmp_path / "workers")["status"] == "refused"


def delayed_callback():
    time.sleep(30)
    return "late"


def test_real_stalled_worker_is_reaped_and_partial_results_are_retained(tmp_path):
    started = time.monotonic()
    with pytest.raises(StageGateError, match="deadline"):
        bounded_feedback(delayed_callback, timeout_s=0.15, output=tmp_path / "workers", kwargs={})
    assert time.monotonic() - started < 2
    receipt = lifecycle(tmp_path / "workers")
    assert receipt["worker"]["reaped"] and receipt["deadline_exceeded"]
    assert receipt["received_bytes"] is None and receipt["status"] == "refused"


def hold_lease_and_stall(*, lease_path, marker):
    lease = acquire_host_resource_lease(marker.parent, lease_path=lease_path)
    assert lease is not None
    marker.write_text("held in direct callback worker")
    time.sleep(30)
    lease.close()


def test_direct_worker_exit_releases_this_observed_lock_without_general_lease_authority(tmp_path):
    lease_path, marker = tmp_path / "host.lease", tmp_path / "held"
    with pytest.raises(StageGateError, match="deadline"):
        bounded_feedback(hold_lease_and_stall, timeout_s=0.3, output=tmp_path / "workers",
                         kwargs={"lease_path": lease_path, "marker": marker})
    assert marker.exists()
    handle = acquire_host_resource_lease(tmp_path / "reacquired", lease_path=lease_path)
    assert handle is not None
    handle.close()
    assert lifecycle(tmp_path / "workers")["lease_release"]["status"] == "NOT_REQUESTED"


def pause_coordinator_and_publish(*, coordinator):
    # Only an owned, separate test coordinator is stopped. Its independent child
    # resumes it after the deadline, while the feedback worker publishes in time.
    os.kill(coordinator, signal.SIGSTOP)
    return {"published": True}


def paused_coordinator_case(root):
    helper = subprocess.Popen([sys.executable, "-c",
                               "import os,signal,time,sys; time.sleep(0.7); os.kill(int(sys.argv[1]),signal.SIGCONT)",
                               str(os.getpid())], start_new_session=True)
    try:
        bounded_feedback(pause_coordinator_and_publish, timeout_s=0.3, output=root, kwargs={"coordinator": os.getpid()})
    except StageGateError as error:
        print(json.dumps({"status": "refused", "detail": str(error)}))
    else:
        print(json.dumps({"status": "accepted"}))
    finally:
        helper.wait(timeout=2)


def test_real_ready_file_cannot_bypass_parent_deadline_after_resume(tmp_path):
    script = ("import pathlib,runpy,sys; "
              "sys.path.insert(0,str(pathlib.Path(sys.argv[1]).parent)); "
              "runpy.run_path(sys.argv[1])['paused_coordinator_case'](pathlib.Path(sys.argv[2]))")
    root = tmp_path / "workers"
    result = subprocess.run([sys.executable, "-c", script, str(Path(__file__).resolve()), str(root)],
                            capture_output=True, text=True, timeout=8)
    assert result.returncode == 0, result.stderr
    observed = json.loads(result.stdout)
    assert observed["status"] == "refused" and "result deadline" in observed["detail"]
    assert len(list(root.glob("component_worker_*/result.json"))) == 1
    receipt = lifecycle(root)
    assert receipt["received_bytes"] is None and receipt["deadline_exceeded"] and receipt["worker"]["reaped"]


def oversized_result(*, kind):
    if kind == "bytes":
        return "x" * 4096
    if kind == "nodes":
        return list(range(100))
    result = None
    for _ in range(20):
        result = [result]
    return result


@pytest.mark.parametrize("kind,limits", [
    ("bytes", FeedbackValueLimits(max_bytes=2048)),
    ("nodes", FeedbackValueLimits(max_nodes=64)),
    ("depth", FeedbackValueLimits(max_depth=12)),
])
def test_actual_worker_excessive_results_refuse_before_parent_acceptance(tmp_path, kind, limits):
    with pytest.raises(StageGateError, match="limit"):
        bounded_feedback(oversized_result, timeout_s=2, output=tmp_path / "workers",
                         kwargs={"kind": kind}, limits=limits)
    assert lifecycle(tmp_path / "workers")["status"] == "refused"


def test_received_bytes_depth_and_nodes_are_bounded_before_decode(tmp_path):
    path = tmp_path / "oversized.json"
    path.write_bytes(b"x" * 4096)
    path.chmod(0o600)
    with pytest.raises(StageGateError, match="bounded plain file"):
        receive_feedback_value(path, limits=FeedbackValueLimits(max_bytes=1024), deadline=time.monotonic() + 1)
    with pytest.raises(StageGateError, match="syntax depth"):
        decode_feedback_value(b"[" * 1000 + b"]" * 1000, limits=FeedbackValueLimits())
    with pytest.raises(StageGateError, match="syntax node"):
        decode_feedback_value(b"[" + b"0," * 1000 + b"0]", limits=FeedbackValueLimits(max_nodes=20))
    os.mkfifo(tmp_path / "pipe")
    with pytest.raises(StageGateError, match="bounded plain file"):
        receive_feedback_value(tmp_path / "pipe", limits=FeedbackValueLimits(), deadline=time.monotonic() + 1)


def test_expired_absolute_budget_refuses_an_already_encoded_result():
    limits = FeedbackValueLimits()
    data = encode_feedback_value({"ready": True}, limits=limits)
    with pytest.raises(StageGateError, match="result deadline"):
        decode_feedback_value(data, limits=limits, deadline=time.monotonic() - 1)


def mutate_candidate(*, candidate, corpus, timeout_s):
    candidate.joinpath("compiler").write_text("mutated during native callback")
    return analytical(candidate, corpus, timeout_s)


def stall_without_mutation(**_kwargs):
    time.sleep(30)


@pytest.mark.usefixtures("synthetic_component_controller")
def test_normal_path_refuses_and_retains_actual_mutation_without_restoration(tmp_path, monkeypatch):
    selected = context(tmp_path)
    source = Path(__file__).resolve()
    original = hash_tree(selected["candidate"])["sha256"]
    selected["component_analytical"] = replace(
        provider(tmp_path, monkeypatch, selected["target_experiment"]), evaluate=mutate_candidate,
        implementation=source, implementation_sha256=sha256_file(source),
    )
    policy = policy_owner.select_workflow(policy_owner.COMPONENT_ONLY_V1, **selected)
    result, document = policy.execute({}, workflow.ANALYTICAL_ACTION, {}, 0, 5, time.monotonic())
    assert result["returncode"] == 125 and document is None
    assert hash_tree(selected["candidate"])["sha256"] != original
    paths = list((selected["receipt_path"].parent / "component_workers").glob("component_refusal_*/refusal.json"))
    assert len(paths) == 1
    refusal = json.loads(paths[0].read_text())
    assert refusal["candidate_changed"] is True and refusal["candidate_before_sha256"] == original
    assert refusal["descendant_cleanup"] == refusal["lease_release"] == "UNKNOWN"


@pytest.mark.usefixtures("synthetic_component_controller")
def test_normal_path_timeout_retains_original_candidate_and_private_refusal(tmp_path, monkeypatch):
    selected, source = context(tmp_path), Path(__file__).resolve()
    original = hash_tree(selected["candidate"])["sha256"]
    selected["component_analytical"] = replace(
        provider(tmp_path, monkeypatch, selected["target_experiment"]), evaluate=stall_without_mutation,
        implementation=source, implementation_sha256=sha256_file(source),
    )
    policy = policy_owner.select_workflow(policy_owner.COMPONENT_ONLY_V1, **selected)
    result, document = policy.execute({}, workflow.ANALYTICAL_ACTION, {}, 0, 0.3, time.monotonic())
    assert result["returncode"] == 125 and document is None
    assert hash_tree(selected["candidate"])["sha256"] == original
    root = selected["receipt_path"].parent / "component_workers"
    paths = list(root.glob("component_refusal_*/refusal.json"))
    assert len(paths) == 1 and json.loads(paths[0].read_text())["candidate_changed"] is False
    assert lifecycle(root)["worker"]["reaped"]
