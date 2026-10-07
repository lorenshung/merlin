"""Selected full-model inputs and complete session roster stay fail-closed."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from merlin_experiments.capture_execution import sealed_m2m
from merlin_experiments.capture_execution.sealed_static import _file_digest
from merlin_experiments.phase0 import capture_execution_attestation as attestation
from merlin_experiments.phase0 import capture_selection


def test_issuer_checks_selected_destination_capacity_before_writing(tmp_path: Path, monkeypatch):
    plan = {
        "schema": sealed_m2m.SCHEMA_V3,
        "status": "plan_only",
        "m2m_root": str(tmp_path / "m2m"),
        "workload_root": str(tmp_path / "workload"),
        "worker": str(tmp_path / "worker.py"),
        "venv": str(tmp_path / "venv"),
        "schemas_root": str(tmp_path / "schemas"),
        "dtype": "fp32",
        "recipe": None,
        "max_snapshot_bytes": 15_000_000_000,
        "selected_inputs": [{"role": "checkpoint", "source": str(tmp_path / "weights"), "guest_member": "weights"}],
        "loader_env": {},
        "execution_timeout_seconds": 3600,
        "estimate_bytes": 1024,
    }
    monkeypatch.setattr(sealed_m2m, "prepare_plan", lambda **_kwargs: dict(plan))
    monkeypatch.setattr(sealed_m2m.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1023))
    run = tmp_path / "run"
    with pytest.raises(sealed_m2m.SealedM2MError, match="run filesystem free space"):
        sealed_m2m.issue(plan, run)
    assert not run.exists()


def test_v3_declares_every_loader_environment_read_and_rejects_dynamic_reads():
    source = 'import os\nx = os.environ.get("MODEL_MODE")\ny = os.getenv("MODEL_INPUT")\n'
    with pytest.raises(sealed_m2m.SealedM2MError, match="MODEL_INPUT"):
        sealed_m2m._declared_loader_env(source, {"MODEL_MODE": "complete"})
    env, reads = sealed_m2m._declared_loader_env(
        source, {"MODEL_MODE": "complete", "MODEL_INPUT": None, "HF_HUB_OFFLINE": "1"}
    )
    assert reads == ["MODEL_INPUT", "MODEL_MODE"]
    assert env["MODEL_INPUT"] is None
    with pytest.raises(sealed_m2m.SealedM2MError, match="dynamic"):
        sealed_m2m._declared_loader_env("import os\nx = os.environ.get(name)\n", {})
    with pytest.raises(sealed_m2m.SealedM2MError, match="unsafe name"):
        sealed_m2m._declared_loader_env(source, {"MODEL_MODE": "complete", "MODEL_INPUT": None, "PYTHONPATH": "x"})


def test_v3_selected_checkpoint_is_frozen_before_issue(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"real selected weights")
    library = tmp_path / "library.so"
    library.write_bytes(b"selected library")
    bwrap = tmp_path / "bwrap"
    bwrap.write_bytes(b"selected bwrap")
    roots = {name: tmp_path / name for name in ("m2m", "workload", "merlin", "schemas", "venv", "base")}
    for root in roots.values():
        root.mkdir()
    worker = roots["merlin"] / "targetgen/worker.py"
    worker.parent.mkdir()
    worker.write_bytes(b"worker")

    def plan(**_kwargs):
        return {
            "schema": sealed_m2m.SCHEMA_V3,
            "status": "plan_only",
            "m2m_root": str(roots["m2m"]),
            "workload_root": str(roots["workload"]),
            "merlin_root": str(roots["merlin"]),
            "schemas_root": str(roots["schemas"]),
            "venv": str(roots["venv"]),
            "base": str(roots["base"]),
            "worker": str(worker),
            "dtype": "fp32",
            "recipe": None,
            "system_libs": [str(library)],
            "selected_inputs": [
                {"role": "checkpoint", **sealed_m2m._input_selection(checkpoint, "weights/checkpoint.bin")}
            ],
            "loader_env": {"MODEL_MODE": "complete"},
            "execution_timeout_seconds": 3600,
            "max_snapshot_bytes": 15_000_000_000,
        }

    monkeypatch.setattr(sealed_m2m, "prepare_plan", plan)
    monkeypatch.setattr(capture_selection, "_bwrap_binary", lambda *_: bwrap)
    run, destination = tmp_path / "run", tmp_path / "selection"
    identity = capture_selection.select(
        m2m_root=roots["m2m"],
        workload_root=roots["workload"],
        worker=worker,
        venv=roots["venv"],
        schemas_root=roots["schemas"],
        run_dir=run,
        output_dir=destination,
        checkpoint=checkpoint,
        checkpoint_guest_member="weights/checkpoint.bin",
        loader_env={"MODEL_MODE": "complete"},
    )
    assert identity["schema"] == capture_selection.SCHEMA_V2
    selected = capture_selection.load(Path(identity["path"]), expected_sha256=identity["sha256"])
    assert selected["checkpoint"]["sha256"] == _file_digest(checkpoint)
    checkpoint.write_bytes(b"changed weights")
    with pytest.raises(ValueError, match="source, runtime"):
        capture_selection.issue(Path(identity["path"]), expected_sha256=identity["sha256"])
    assert not run.exists()


def test_v3_session_verifier_requires_exact_stage_membership_and_receipts(tmp_path: Path, monkeypatch):
    output = tmp_path / "capture"
    stages = output / "stages"
    for name in ("prefix_encode", "flow_denoise", "action_decode"):
        stage = stages / name
        stage.mkdir(parents=True)
        (stage / "model.mlir").write_bytes(name.encode())
        (stage / "capture_receipt.json").write_text(json.dumps({"materialized_abi": {"complete": True}}))
        (stage / "meta.json").write_text(json.dumps({"dtype": "fp32"}))
    names = ["prefix_encode", "flow_denoise", "action_decode"]
    contract = {
        "version": 2,
        "stages": names,
        "programs": [{"name": name, "bundle": f"stages/{name}"} for name in names],
    }
    (output / "session_contract.yaml").write_text(yaml.safe_dump(contract))
    report = {
        "schema": "merlin.model_session_capture.v1",
        "programs": [
            {
                "name": name,
                "materialized_abi": {"complete": True},
                "precision_selection": "untransformed",
                "receipt_sha256": _file_digest(stages / name / "capture_receipt.json"),
            }
            for name in names
        ],
        "session_contract_sha256": _file_digest(output / "session_contract.yaml"),
    }
    (output / "session-receipt.json").write_text(json.dumps(report))
    monkeypatch.setattr(
        sealed_m2m,
        "_materialized",
        lambda stage, *_args, **_kwargs: {
            "status": "verified_materialized",
            "receipt_sha256": _file_digest(stage / "capture_receipt.json"),
        },
    )
    verified = sealed_m2m._materialized_v3(output, tmp_path, output, {"dtype": "fp32", "recipe": None})
    assert verified["kind"] == "session"
    assert [row["name"] for row in verified["programs"]] == names
    (stages / "decoy").mkdir()
    with pytest.raises(sealed_m2m.SealedM2MError, match="membership"):
        sealed_m2m._materialized_v3(output, tmp_path, output, {"dtype": "fp32", "recipe": None})
    (stages / "decoy").rmdir()
    report["programs"].pop()
    (output / "session-receipt.json").write_text(json.dumps(report))
    with pytest.raises(sealed_m2m.SealedM2MError, match="roster"):
        sealed_m2m._materialized_v3(output, tmp_path, output, {"dtype": "fp32", "recipe": None})


def test_v3_single_program_accepts_complete_v1_session_contract(tmp_path: Path, monkeypatch):
    output = tmp_path / "capture"
    output.mkdir()
    for name in ("session_inputs.npz", "session_goldens.npz", "session_quality_fp32.npz"):
        (output / name).write_bytes(name.encode())
    contract = {
        "version": 1,
        "kind": "image_stream",
        "stages": ["classify"],
        "stage_schedule": [{"name": "classify", "steps": 1}],
        "inputs": "session_inputs.npz",
        "correctness": {"golden": "session_goldens.npz"},
        "quality": {"golden": "session_quality_fp32.npz"},
    }
    contract_path = output / "session_contract.yaml"
    contract_path.write_text(yaml.safe_dump(contract))
    monkeypatch.setattr(
        sealed_m2m,
        "_materialized_v2",
        lambda *_args: {
            "status": "verified_materialized",
            "receipt_sha256": "a" * 64,
        },
    )
    plan = {"dtype": "fp32", "recipe": None}
    verified = sealed_m2m._materialized_v3(output, tmp_path, output, plan)
    assert verified["kind"] == "single"
    assert verified["session_contract_sha256"] == _file_digest(contract_path)
    (output / "session_inputs.npz").unlink()
    with pytest.raises(sealed_m2m.SealedM2MError, match="absent or unsafe"):
        sealed_m2m._materialized_v3(output, tmp_path, output, plan)
    (output / "session_inputs.npz").write_bytes(b"restored")
    contract["version"] = 2
    contract_path.write_text(yaml.safe_dump(contract))
    with pytest.raises(sealed_m2m.SealedM2MError, match="invalid stage roster"):
        sealed_m2m._materialized_v3(output, tmp_path, output, plan)


@pytest.mark.parametrize("preserved", [False, True])
def test_v3_int8_session_requires_selected_recipe_and_nonzero_integer_work(tmp_path: Path, monkeypatch, preserved):
    source, output = tmp_path / "source", tmp_path / "capture"
    (source / "inputs").mkdir(parents=True)
    (source / "inputs/quant_recipe.json").write_bytes(b"selected recipe")
    names = ["integer_stage", "host_stage"]
    for name in names:
        stage = output / "stages" / name
        stage.mkdir(parents=True)
        (stage / "model.mlir").write_bytes(name.encode())
        (stage / "capture_receipt.json").write_text(json.dumps({"materialized_abi": {"complete": True}}))
    recipe = {
        "bytes": len(b"selected recipe"),
        "sha256": _file_digest(source / "inputs/quant_recipe.json"),
        "recipe_sha256": "a" * 64,
    }
    monkeypatch.setattr(sealed_m2m, "_recipe_selection", lambda *_args, **_kwargs: dict(recipe))
    monkeypatch.setattr(
        sealed_m2m,
        "_materialized",
        lambda stage, *_args, **_kwargs: {
            "status": "verified_materialized",
            "receipt_sha256": _file_digest(stage / "capture_receipt.json"),
        },
    )
    seen = 3 if preserved else 2
    remaining = int(preserved)
    agreement = {
        "status": "passed",
        "reference": "pt2e_integer",
        "atol": 0.0,
        "rtol": 0.0,
        "executed_contractions": {
            "conv2d": 0,
            "linear": 2,
            "matmul": 0,
            "total": 2,
            "selected": seen,
            "observed": seen,
        },
        "samples": 1,
        "finite": True,
        "max_abs": 0.0,
        "max_rel": 0.0,
        "outputs": [
            {"finite": True, "within_tolerance": True, "max_abs": 0.0, "max_rel": 0.0, "atol": 0.0, "rtol": 0.0}
        ],
    }
    integer_meta = {
        "dtype": "int8",
        "scheme": "int8_static_act_int8_weight",
        "recipe_sha256": recipe["recipe_sha256"],
        "recipe": {"software_numerical_engine": "integer_reference"},
        "quantization_stats": {"recipe_sha256": recipe["recipe_sha256"], "annotated_contractions": seen},
        "integerization_receipt": {
            "schema": "m2m.pt2e-integerize.v1",
            "golden_agreement": agreement,
            "exported_integer_mm_count": 2,
            "integer_mm_emitted": 2,
            "quantized_contractions_seen": seen,
            "quantized_contractions_integerized": 2,
            "quantized_contractions_remaining": remaining,
            "quantized_by_kind": {
                "linear": {"seen": seen, "integerized": 2, "remaining": remaining},
                **{kind: {"seen": 0, "integerized": 0, "remaining": 0} for kind in ("conv2d", "matmul", "unsupported")},
            },
            "refusals": [{"kind": "linear", "node": "bf16", "reason": "non-f32 dequantization"}] if preserved else [],
            "precision_decisions": [
                {"kind": "linear", "node": name, "decision": "integerized_i32"} for name in ("a", "b")
            ]
            + (
                [
                    {
                        "kind": "linear",
                        "node": "bf16",
                        "decision": "preserve_float_qdq",
                        "source_dtype": "torch.bfloat16",
                        "required_numeric_semantics": "dequantize-before-floating-contraction",
                    }
                ]
                if preserved
                else []
            ),
            "precision_decision_counts": {"integerized_i32": 2, "preserve_float_qdq": remaining, "unresolved": 0},
            "accumulator_bound_checked": True,
            "max_reduction_k": 4,
        },
    }
    (output / "stages/integer_stage/meta.json").write_text(json.dumps(integer_meta))
    (output / "stages/host_stage/meta.json").write_text(
        json.dumps(
            {
                "dtype": "fp32",
                "recipe_sha256": None,
            }
        )
    )
    contract = {
        "version": 2,
        "stages": names,
        "programs": [{"name": name, "bundle": f"stages/{name}"} for name in names],
    }
    (output / "session_contract.yaml").write_text(yaml.safe_dump(contract))
    report = {
        "schema": "merlin.model_session_capture.v1",
        "recipe_sha256": recipe["recipe_sha256"],
        "session_contract_sha256": _file_digest(output / "session_contract.yaml"),
        "programs": [
            {
                "name": name,
                "precision_selection": "recipe" if name == "integer_stage" else "no_recipe_work",
                "materialized_abi": {"complete": True},
                "receipt_sha256": _file_digest(output / "stages" / name / "capture_receipt.json"),
            }
            for name in names
        ],
    }
    (output / "session-receipt.json").write_text(json.dumps(report))
    plan = {"dtype": "int8", "recipe": recipe}
    verified = sealed_m2m._materialized_v3(output, source, output, plan)
    assert verified["integer_contractions"] == 2
    integer_stage = verified["programs"][0]
    if preserved:
        assert integer_stage["contraction_partition"] == {"seen": 3, "integerized": 2, "preserved": 1}
    else:
        assert set(integer_stage) == {"name", "model_sha256", "receipt_sha256", "precision_selection"}
    import copy

    for mutate in (
        lambda meta: meta["recipe"].clear(),
        lambda meta: meta["integerization_receipt"].update(exported_integer_mm_count=100),
        lambda meta: meta["integerization_receipt"]["golden_agreement"].update(samples=0),
        lambda meta: meta["integerization_receipt"]["golden_agreement"].update(outputs=[]),
        lambda meta: meta["integerization_receipt"]["golden_agreement"]["outputs"][0].update(max_abs=1.0),
        lambda meta: meta["integerization_receipt"]["golden_agreement"].update(atol=False),
        lambda meta: meta["integerization_receipt"]["golden_agreement"]["executed_contractions"].update(conv2d=False),
        lambda meta: meta.update(recipe=["invalid"]),
        lambda meta: meta.update(quantization_stats=["invalid"]),
        lambda meta: meta["integerization_receipt"].update(golden_agreement=["invalid"]),
    ):
        bad = copy.deepcopy(integer_meta)
        mutate(bad)
        (output / "stages/integer_stage/meta.json").write_text(json.dumps(bad))
        with pytest.raises(sealed_m2m.SealedM2MError, match="integer contractions"):
            sealed_m2m._materialized_v3(output, source, output, plan)
    if preserved:
        integer_meta["integerization_receipt"]["precision_decisions"][-1]["source_dtype"] = "torch.float32"
        (output / "stages/integer_stage/meta.json").write_text(json.dumps(integer_meta))
        with pytest.raises(sealed_m2m.SealedM2MError, match="integer contractions"):
            sealed_m2m._materialized_v3(output, source, output, plan)
        integer_meta["integerization_receipt"]["precision_decisions"][-1]["source_dtype"] = "torch.bfloat16"
    integer_meta["integerization_receipt"]["exported_integer_mm_count"] = 0
    (output / "stages/integer_stage/meta.json").write_text(json.dumps(integer_meta))
    with pytest.raises(sealed_m2m.SealedM2MError, match="integer contractions"):
        sealed_m2m._materialized_v3(output, source, output, plan)


@pytest.mark.parametrize("kind", ["single", "session"])
def test_v3_session_attestation_binds_all_output_bytes(tmp_path: Path, monkeypatch, kind: str):
    checkpoint = tmp_path / "weights.bin"
    checkpoint.write_bytes(b"pretrained checkpoint")
    bwrap = tmp_path / "bwrap"
    bwrap.write_bytes(b"bubblewrap")
    roots = {name: tmp_path / name for name in ("m2m", "workload", "merlin", "schemas", "venv", "base")}
    for root in roots.values():
        root.mkdir()
    worker = roots["merlin"] / "targetgen/worker.py"
    worker.parent.mkdir()
    worker.write_bytes(b"worker")
    plan = {
        "schema": sealed_m2m.SCHEMA_V3,
        "status": "plan_only",
        "dtype": "fp32",
        "recipe": None,
        "m2m_root": str(roots["m2m"]),
        "workload_root": str(roots["workload"]),
        "merlin_root": str(roots["merlin"]),
        "schemas_root": str(roots["schemas"]),
        "venv": str(roots["venv"]),
        "base": str(roots["base"]),
        "worker": str(worker),
        "system_libs": [],
        "execution_timeout_seconds": 3600,
        "selected_inputs": [{"role": "checkpoint", **sealed_m2m._input_selection(checkpoint, "weights.bin")}],
        "loader_env": {"MODEL_MODE": "complete"},
        "max_snapshot_bytes": 15_000_000_000,
    }
    monkeypatch.setattr(sealed_m2m, "prepare_plan", lambda **_: dict(plan))
    monkeypatch.setattr(capture_selection, "_bwrap_binary", lambda *_: bwrap)
    run = tmp_path / "run"
    identity = capture_selection.select(
        m2m_root=roots["m2m"],
        workload_root=roots["workload"],
        worker=worker,
        venv=roots["venv"],
        schemas_root=roots["schemas"],
        run_dir=run,
        output_dir=tmp_path / "selection",
        checkpoint=checkpoint,
        checkpoint_guest_member="weights.bin",
        loader_env={"MODEL_MODE": "complete"},
    )
    selected = capture_selection.load(Path(identity["path"]), expected_sha256=identity["sha256"])
    capture = run / "capture"
    model = capture / "model.mlir" if kind == "single" else capture / "stages/complete/model.mlir"
    model.parent.mkdir(parents=True)
    model.write_bytes(b"complete model")
    if kind == "single":
        (capture / "capture_receipt.json").write_bytes(b"complete receipt")
    (run / "snapshots/source").mkdir(parents=True)
    (run / "snapshots/guest-root").mkdir()
    run.chmod(0o700)
    materialized = {"kind": kind, "integer_contractions": 0}
    if kind == "session":
        materialized.update(
            session_contract_sha256="a" * 64,
            session_receipt_sha256="b" * 64,
            programs=[{"name": "complete", "model_sha256": _file_digest(model), "receipt_sha256": "c" * 64}],
        )
    monkeypatch.setattr(sealed_m2m, "_materialized_v3", lambda *_args, **_kwargs: materialized)
    monkeypatch.setattr(sealed_m2m, "_verify_staged_selection", lambda *_args, **_kwargs: {})
    pending = run / "sealed_m2m_pending.json"
    pending.write_text(
        json.dumps(
            {
                "schema": sealed_m2m.SCHEMA_V3,
                "status": "pending_replay",
                "capture_selection_sha256": identity["sha256"],
                "plan": plan,
                "policy_sha256": selected["sandbox_policy_sha256"],
                "issuer_sha256": selected["issuer_source_sha256"],
                "bwrap_sha256": selected["bwrap"]["sha256"],
                "source": sealed_m2m._snapshot_tree(run / "snapshots/source"),
                "guest_root": sealed_m2m._snapshot_tree(run / "snapshots/guest-root"),
                "output": sealed_m2m._snapshot_tree(capture),
                "materialized": materialized,
            }
        )
    )
    monkeypatch.setattr(
        sealed_m2m,
        "replay_verify",
        lambda *_args, **_kwargs: {
            "status": "verified_sandbox_replay",
            "sealed_source_closure_replayed": True,
            "receipt_sha256": _file_digest(pending),
        },
    )
    replay = capture_selection.verify(Path(identity["path"]), expected_sha256=identity["sha256"], model_path=capture)
    document = attestation.attest_sealed_m2m_v3(replay, selection_path=Path(identity["path"]), capture_path=capture)
    assert document["capture"]["kind"] == kind
    attestation.require_verified_execution(document)
    model.write_bytes(b"edited model")
    with pytest.raises(attestation.AttestationNotVerified, match="output bytes"):
        attestation.require_verified_execution(document)
