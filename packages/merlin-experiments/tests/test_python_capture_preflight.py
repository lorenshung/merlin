"""The Python capture inventory must never issue a sealed-execution claim."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from merlin_experiments.capture_execution.python_preflight import inspect


def _selection(tmp_path):
    m2m = tmp_path / "m2m-checkout"
    (m2m / "m2m").mkdir(parents=True)
    (m2m / "m2m/__init__.py").write_text("")
    worker = tmp_path / "worker.py"
    worker.write_text("print('not run')\n")
    workload = m2m / "workloads/example"
    workload.mkdir(parents=True)
    loader = workload / "loader.py"
    loader.write_text('import os\nINPUT = os.environ.get("MODEL_INPUT_NPZ")\n')
    venv = tmp_path / "venv"
    (venv / "bin").mkdir(parents=True)
    python = venv / "bin/python"
    python.symlink_to(sys.executable)
    version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    site = venv / "lib" / version / "site-packages"
    site.mkdir(parents=True)
    missing = tmp_path / "missing-editable-source"
    (site / "external.pth").write_text(str(missing) + "\n")
    (venv / "pyvenv.cfg").write_text(f"home = {Path(sys.executable).resolve().parent}\n")
    (workload / "capture.toml").write_text(f'venv = "{venv}"\npython = "{version[6:]}"\n')
    return worker, loader, m2m, python, missing


def test_preflight_names_missing_editable_and_env_without_execution(tmp_path):
    worker, loader, m2m, python, missing = _selection(tmp_path)
    result = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python)
    assert result["status"] == "blocked_unsealed_python_capture"
    assert result["source_closure_verified"] is False
    assert result["fresh_execution"] is False
    assert result["issued_capture"] is False
    assert str(missing) in result["missing_paths"]
    assert {row["name"] for row in result["loader_environment_reads"]} == {"MODEL_INPUT_NPZ"}
    assert result["inputs"]["venv_python"]["kind"] == "symlink"
    assert not (tmp_path / "capture").exists()
    absent_data = tmp_path / "missing-input.npz"
    selected = inspect(
        worker=worker,
        loader=loader,
        m2m_root=m2m,
        python=python,
        selected_env={"MODEL_INPUT_NPZ": str(absent_data)},
    )
    assert str(absent_data) in selected["missing_paths"]
    delegated = inspect(
        worker=worker,
        loader=loader,
        m2m_root=m2m,
        python=python,
        required_env_names=["MODEL_TOKEN_IDS"],
    )
    assert delegated["caller_required_environment_names"] == ["MODEL_TOKEN_IDS"]
    assert "MODEL_TOKEN_IDS" in delegated["unselected_loader_environment"]


def test_preflight_cli_writes_once_and_returns_blocked(tmp_path):
    worker, loader, m2m, python, _ = _selection(tmp_path)
    owner = m2m / "m2m/__init__.py"
    receipt = tmp_path / "capture_receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "source": {"path": str(loader), "sha256": hashlib.sha256(loader.read_bytes()).hexdigest()},
                "tool": {"source_sha256": {"m2m/__init__.py": hashlib.sha256(owner.read_bytes()).hexdigest()}},
                "source_closure_verified": True,
            }
        )
    )
    output = tmp_path / "evidence" / "preflight.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(Path(__file__).resolve().parents[1] / "src")
    command = [
        sys.executable,
        "-m",
        "merlin_experiments.capture_execution.python_preflight",
        "--worker",
        str(worker),
        "--loader",
        str(loader),
        "--m2m-root",
        str(m2m),
        "--python",
        str(python),
        "--capture-receipt",
        str(receipt),
        "--output",
        str(output),
    ]
    first = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    assert first.returncode == 2, first.stderr
    saved = json.loads(output.read_text())
    assert saved["source_closure_verified"] is False
    assert saved["fresh_execution"] is False
    assert saved["inputs"]["worker"]["sha256"]
    assert saved["capture_receipt_audit"]["status"] == "current_direct_sources_match"
    assert saved["capture_receipt_audit"]["historical_execution_verified"] is False
    assert subprocess.run(command, capture_output=True, text=True, env=env, check=False).returncode != 0
    assert json.loads(output.read_text()) == saved


def test_receipt_audit_reports_current_direct_source_drift_without_upgrading(tmp_path):
    worker, loader, m2m, python, _ = _selection(tmp_path)
    owner = m2m / "m2m/api.py"
    owner.write_text("VERSION = 1\n")

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    receipt = tmp_path / "old-capture-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "source": {"path": str(loader), "sha256": sha(loader)},
                "tool": {"source_sha256": {"m2m/api.py": sha(owner)}},
                "source_closure_verified": False,
            }
        )
    )
    matched = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python, capture_receipt=receipt)
    audit = matched["capture_receipt_audit"]
    assert audit["status"] == "current_direct_sources_match"
    assert audit["historical_execution_verified"] is False
    assert matched["source_closure_verified"] is False
    owner.write_text("VERSION = 2\n")
    drifted = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python, capture_receipt=receipt)
    assert drifted["capture_receipt_audit"]["status"] == "current_direct_sources_drift"
    assert drifted["capture_receipt_audit"]["tool_sources"][0]["status"] == "drift_or_missing"
    assert drifted["source_closure_verified"] is False


def test_receipt_audit_rejects_escape_and_wrong_loader(tmp_path):
    worker, loader, m2m, python, _ = _selection(tmp_path)
    receipt = tmp_path / "untrusted-receipt.json"
    receipt.write_text(
        json.dumps(
            {
                "schema": "m2m.capture-receipt.v1",
                "source": {"path": str(loader), "sha256": hashlib.sha256(loader.read_bytes()).hexdigest()},
                "tool": {"source_sha256": {"../outside.py": "0" * 64}},
            }
        )
    )
    result = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python, capture_receipt=receipt)
    assert result["capture_receipt_audit"]["status"] == "invalid_receipt"
    assert result["source_closure_verified"] is False
    payload = json.loads(receipt.read_text())
    payload["source"]["path"] = str(tmp_path / "another-loader.py")
    receipt.write_text(json.dumps(payload))
    result = inspect(worker=worker, loader=loader, m2m_root=m2m, python=python, capture_receipt=receipt)
    assert "differs from the selected loader" in result["capture_receipt_audit"]["errors"][0]
