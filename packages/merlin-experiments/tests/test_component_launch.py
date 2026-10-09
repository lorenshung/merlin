"""Component transport refusal and actual local Unix broker process checks.

Synthetic lifecycle tests never qualify a model account or kernel sandbox.
"""

import hashlib
import json
import shlex
import subprocess
import sys
import tempfile
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
from merlin_experiments.phase1.providers import codex_agent as CA
from merlin_experiments.phase2 import authoring, authoring_cli
from merlin_experiments.phase2 import broker as B
from merlin_experiments.phase2 import broker_policy as BP
from merlin_experiments.phase2 import component_launch as CL
from merlin_experiments.phase2 import corpus_feedback as CF
from test_phase2_broker import make_broker


def test_native_profile_uses_exact_public_grants_and_blocks_control_home(tmp_path):
    home = tmp_path / "control-home"
    text = CA._candidate_permission_config(home, read_paths=("/component-inputs", "/perf-control", "/usr/bin/python3"))
    assert '"/component-inputs" = "read"' in text
    assert '"/scratch" = "read"' not in text and '"/scratch2" = "read"' not in text
    assert f'"{home}" = "deny"' in text and "enabled = false" in text
    with pytest.raises(ValueError, match="control credentials"):
        CA._candidate_permission_config(home, read_paths=(str(home.parent),))


@pytest.mark.parametrize("grant", [None, True, {"isolated": True, "fresh": True}])
def test_flags_cannot_admit_component_authoring(grant):
    with pytest.raises(B.StageGateError):
        authoring.admit_authoring_workflow(BP.COMPONENT_ONLY_V1, component_launch=grant)


@pytest.fixture
def short_ipc_directory():
    # AF_UNIX has a fixed path budget; pytest's test-name directory may exceed it
    # when the installed qualification lives outside a long checkout path.
    with tempfile.TemporaryDirectory(prefix="m-ipc-") as directory:
        yield Path(directory)


def test_unix_transport_executes_normal_shim_and_persists_receipts(monkeypatch, short_ipc_directory):
    monkeypatch.setattr(CF, "inspect_compiler_package", lambda _: SimpleNamespace(to_dict=lambda: {"owners": []}))
    broker = make_broker(short_ipc_directory)
    socket_path = broker.receipt_path.parent / "channel.sock"
    shim = B.stage_broker_shim(socket_path.parent, host="", port=0, token=broker.token, tool_timeout_s=10,
                               actions=tuple(broker.actions.values()), socket_path=str(socket_path))
    with broker.serving(socket_path=socket_path):
        result = subprocess.run(
            [sys.executable, str(shim), BP.INVENTORY_ACTION], capture_output=True, text=True, timeout=15
        )
        assert result.returncode == 0, result.stderr
        assert json.loads(result.stdout) == {"owners": []}
    assert not socket_path.exists()
    receipt = json.loads(broker.receipt_path.read_bytes())
    assert receipt["state"] == "complete" and receipt["returncode"] == 0


def test_networkless_tool_command_requires_captured_policy_and_no_shell(monkeypatch):
    observed = []
    policy = SimpleNamespace(network="isolated_networkless", clear_environment=True, env_prefix=None,
                             argv=("bwrap", "--unshare-all"), verify_execution=lambda: observed.append("verified"))
    assert B.inner_command(policy, object(), Path("/candidate"), ("/usr/bin/compiler", "input.mlir"), 10) == [
        "bwrap", "--unshare-all", "--", "/usr/bin/compiler", "input.mlir"
    ]
    assert observed == ["verified"]
    policy.env_prefix = "inherited credentials"
    with pytest.raises(B.StageGateError, match="environment"):
        B.inner_command(policy, object(), Path("/candidate"), ("compiler",), 10)


def test_public_projection_omits_provenance_paths_goldens_and_performance_labels():
    corpus = SimpleNamespace(manifest_sha256="1" * 64, capsules_sha256="2" * 64,
                             capsules=(SimpleNamespace(family="generated", capsule="case", source_sha256="3" * 64,
                                                       source_dir=Path("/private/answer"),
                                                       descriptor={"golden": "SECRET"}),))
    payload = CL.public_component_manifest(corpus)
    assert b"SECRET" not in payload and b"private" not in payload and b"performance" not in payload
    assert json.loads(payload)["manifest_sha256"] == corpus.manifest_sha256


def test_component_cli_requires_actual_configuration_before_qualification(monkeypatch, capsys):
    assert authoring_cli.main(["--workflow", BP.COMPONENT_ONLY_V1]) == 2
    assert "NO-GO" in capsys.readouterr().err


def test_readiness_success_exit_without_exact_expected_output_is_insufficient():
    probe = CL.ComponentReadinessProbe(
        "compiler", ("/usr/bin/probe",), hashlib.sha256(b"compiled linked executed\n").hexdigest()
    )
    probe.verify()
    with pytest.raises(B.StageGateError):
        replace(probe, stdout_sha256="caller says passed").verify()


@pytest.mark.parametrize("directory", ["long" * 40, "é" * 60])
def test_typed_launch_refuses_overlong_host_socket_before_readiness(directory):
    inputs = object.__new__(CL.ComponentLaunchInputs)
    object.__setattr__(inputs, "stage_root", Path("/private") / directory)
    object.__setattr__(inputs, "policy", SimpleNamespace(receipt_path=inputs.stage_root / "receipts/private.jsonl"))
    # Verification must refuse before touching any qualification, readiness or
    # paid-provider field. Mounting the socket under a short inner name cannot
    # repair its original host sockaddr_un path.
    with pytest.raises(B.StageGateError, match="AF_UNIX budget"):
        inputs.verify()


def test_control_mounts_only_public_ipc_files_not_private_or_public_siblings(tmp_path, monkeypatch):
    public, private = tmp_path / "public", tmp_path / "private"
    public.mkdir()
    private.mkdir()
    names = ("perf_tool.py", ".perf_broker.json", "channel.sock", "receipts.jsonl")
    for name in names:
        (public / name).write_text("public ipc")
    (public / "golden.json").write_text("MALICIOUS PUBLIC SIBLING")
    (private / "full_cost.json").write_text("PRIVATE COST DETAIL")
    inputs = SimpleNamespace(candidate=tmp_path / "candidate", view=SimpleNamespace(root=tmp_path / "view"),
                             control_runtime=(), public_control_dir=public, auth_source=private / "auth.json",
                             codex_binary=Path("/host/codex"), codex_destination="/usr/bin/codex",
                             sandbox_binary=Path("/reviewed/bwrap"), verify=lambda: None)
    monkeypatch.setattr(CL, "strict_tool_policy", lambda *_a, **_k: ("/reviewed/bwrap", "--unshare-all"))
    home = tmp_path / "control-home"
    argv = shlex.split(CL._control_command(inputs, "/host/codex exec", inputs.candidate, {},
                                          extra_binds=CL._runtime_binds(inputs, home)))
    assert str(public) not in argv and str(private) not in argv
    assert str(private / "full_cost.json") not in argv and str(public / "golden.json") not in argv
    mounts = [argv[index + 1:index + 3] for index, value in enumerate(argv) if value == "--ro-bind"]
    assert mounts == [[str(public / name), "/perf-control/" + name] for name in names]
    reads = CL._read_paths(inputs)
    assert "/perf-control" not in reads
    assert set(reads) == {"/component-inputs", *("/perf-control/" + name for name in names)}
