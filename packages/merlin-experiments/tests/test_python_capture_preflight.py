"""The Python capture inventory must never issue a sealed-execution claim."""

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


def test_preflight_cli_writes_once_and_returns_blocked(tmp_path):
    worker, loader, m2m, python, _ = _selection(tmp_path)
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
        "--output",
        str(output),
    ]
    first = subprocess.run(command, capture_output=True, text=True, env=env, check=False)
    assert first.returncode == 2, first.stderr
    saved = json.loads(output.read_text())
    assert saved["source_closure_verified"] is False
    assert saved["fresh_execution"] is False
    assert saved["inputs"]["worker"]["sha256"]
    assert subprocess.run(command, capture_output=True, text=True, env=env, check=False).returncode != 0
    assert json.loads(output.read_text()) == saved
