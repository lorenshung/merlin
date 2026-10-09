"""Ordinary isolated component authoring and descendant requalification lifecycle.

The launch edge retains its stable public entrypoint. This executing controller
preserves exact original feedback owners, observed lineage and complete final
compile/static/numeric gates; sealing a refused candidate cannot promote it.
"""

from __future__ import annotations

import shutil
import time
from dataclasses import replace
from pathlib import Path

from merlin.benchharness import hash_tree
from merlin.common.digest import sha256_bytes
from merlin_experiments.phase1.component_lineage import run_component_origin_round

from . import broker as B
from . import component_launch as L
from . import contracts as C
from . import stage_inputs as SI
from . import stage_prompt as SP
from . import telemetry as TEL
from .component_experiment import strict_tool_policy
from .component_final_qualification import qualify_observed_component_descendant
from .component_workflow import ComponentOnlyPolicy
from .transcript_audit import audit_codex_transcript


def run_component_stage(
    launch: L.QualifiedComponentLaunch,
    *,
    model: str,
    effort: str,
    wall_budget_seconds: int,
    max_tool_calls: int,
    tool_timeout_seconds: int,
    suite: str,
) -> Path:
    """Author one fresh bounded session, audit receipts, and requalify final bytes."""
    if type(launch) is not L.QualifiedComponentLaunch:
        raise C.StageGateError("component authoring requires evaluated launch prerequisites")
    launch.verify()
    if model != launch.selected_model or effort != launch.selected_effort:
        raise C.StageGateError("component authoring cannot change the qualified model transport")
    if (
        any(
            type(value) is not int or value <= 0
            for value in (wall_budget_seconds, max_tool_calls, tool_timeout_seconds)
        )
        or tool_timeout_seconds > 600
    ):
        raise C.StageGateError("component authoring budgets require positive integers and a bounded tool ceiling")
    inputs = launch.inputs
    if not isinstance(suite, str) or not suite.strip():
        raise C.StageGateError("component authoring requires an explicit accounting suite")
    telemetry_preflight = TEL.prepare(
        model=model,
        authoring_stage=run_component_stage,
        price_table=inputs.price_table,
        codex_binary=inputs.codex_binary,
    )
    # Readiness receipts are retained as a separate pre-authoring attempt. Use a
    # new exact policy owner for authoring rather than adopting probe feedback.
    policy = ComponentOnlyPolicy(
        candidate=inputs.candidate,
        target_experiment=inputs.policy.target_experiment,
        receipt_path=inputs.stage_root / "control-authoring" / "receipts.jsonl",
        component_corpus=inputs.policy.component_corpus,
        component_analytical=inputs.policy.component_analytical,
        component_cca=inputs.policy.component_cca,
        component_rtl=inputs.policy.component_rtl,
        services=inputs.policy.services,
        baseline_admission=inputs.policy.baseline_admission,
        independent_runtime=inputs.policy.independent_runtime,
    )
    # Copy only the immutable launch configuration with this new receipt owner.
    authored_inputs = replace(inputs, policy=policy)
    tool = L.ComponentToolPolicy(
        authored_inputs,
        strict_tool_policy(
            inputs.view,
            inputs.candidate,
            runtime=inputs.runtime,
            candidate_destination=str(inputs.candidate),
            bwrap_binary=inputs.sandbox_binary,
        ),
    )
    actions = policy.build_registry()
    prompt_inputs = SI.prepare_component_prompt_inputs(
        policy,
        corpus_manifest_path="/component-inputs/contract/performance_corpus_manifest.json",
        candidate_path=str(inputs.candidate),
        allowed_paths=(
            "/component-inputs",
            "/component-inputs/contract/performance_corpus_manifest.json",
            str(inputs.candidate),
            B.BROKER_NAME,
            str(B.BROKER_RECEIPT_MOUNT),
        ),
        wall_budget_seconds=wall_budget_seconds,
        max_tool_calls=max_tool_calls,
        tool_timeout_seconds=tool_timeout_seconds,
        broker_path=B.BROKER_NAME,
        broker_receipt_path=str(B.BROKER_RECEIPT_MOUNT),
        public_manifest_sha256=sha256_bytes(L.public_component_manifest(policy.component_corpus)),
    )
    prompt = SP.render_component_prompt(prompt_inputs)
    broker = B.Broker(
        tool,
        policy.target_experiment,
        inputs.candidate,
        actions,
        policy.receipt_path,
        deadline=time.monotonic() + wall_budget_seconds,
        workflow=policy,
        max_calls=max_tool_calls,
        max_tool_seconds=tool_timeout_seconds,
        public_control_dir=authored_inputs.public_control_dir,
    )
    B.stage_broker_shim(
        authored_inputs.public_control_dir,
        host="",
        port=0,
        token=broker.token,
        actions=actions,
        tool_timeout_s=tool_timeout_seconds,
        socket_path="/perf-control/channel.sock",
    )
    refusals, audit, receipts, final_qualification, telemetry = [], None, None, None, None
    started = time.monotonic()
    with broker.serving(socket_path=authored_inputs.socket_path):
        rc, transcript, lineage = run_component_origin_round(
            launch,
            authored_inputs,
            model=model,
            effort=effort,
            wall_budget_seconds=wall_budget_seconds,
            prompt=prompt,
        )
    try:
        round_telemetry = TEL.collect_round(
            inputs.stage_root, 0, model=model, agent_exit_code=rc, preflight_record=telemetry_preflight
        )
        telemetry = TEL.finalize(
            inputs.stage_root,
            [{"round": 0, "telemetry": round_telemetry}],
            model=model,
            target=policy.target_experiment.target,
            suite=suite,
            run_id=inputs.stage_root.name,
            preflight_record=telemetry_preflight,
        )
        authored_inputs.verify()
        audit = audit_codex_transcript(transcript, policy.target_experiment, inputs.candidate, actions)
        if audit.get("clean") is not True:
            raise C.StageGateError("component authoring transcript audit failed")
        digest = str(hash_tree(inputs.candidate)["sha256"])
        receipts = policy.verify_receipts(policy.receipt_path, actions, audit, candidate_sha256=digest)
        if rc not in (0, C.ROUND_DEADLINE_EXIT):
            raise C.StageGateError("component model transport did not complete an admitted bounded session")
        final_qualification = qualify_observed_component_descendant(
            authored_inputs,
            lineage,
            timeout_s=tool_timeout_seconds,
        )
        final_qualification.verify()
    except Exception as exc:  # noqa: BLE001 - seal refused evidence without promotion
        refusals.append(type(exc).__name__ + ": " + str(exc)[:500])
    sealed = inputs.stage_root / "sealed_candidate"
    shutil.copytree(inputs.candidate, sealed)
    for path in (sealed, *sealed.rglob("*")):
        if path.is_symlink():
            raise C.StageGateError("component final candidate contains a symlink")
        path.chmod(path.stat().st_mode & ~0o222)
    record = inputs.stage_root / "candidate_record.json"
    C.write_json(
        record,
        {
            "schema": "merlin.component_authoring_candidate.v1",
            "workflow_id": policy.workflow_id,
            "candidate_path": str(sealed),
            "candidate_sha256": str(hash_tree(sealed)["sha256"]),
            "launch_qualification_sha256": launch.qualification_sha256,
            "functional_qualification_sha256": final_qualification.receipt_sha256 if final_qualification else None,
            "phase1_origin_sha256": inputs.qualification.compiler_origin.receipt_sha256,
            "phase2_lineage_sha256": lineage.receipt_sha256,
            "transcript": str(transcript),
            "transcript_sha256": C.sha256_file(transcript),
            "exit_code": rc,
            "audit": audit,
            "receipts": receipts,
            "wall_seconds": time.monotonic() - started,
            "telemetry": telemetry,
            "admission": {"consumable": not refusals, "refusal": "; ".join(refusals) or None},
            "final_acceptance": "NOT_ESTABLISHED",
        },
    )
    record.chmod(0o400)
    return record
