"""Actual harmless subprocess observations, separate from correctness authority."""
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from merlin.common import invocation_record as I


def _command(tmp_path, *, exit_status=0, mutate_input=False):
    source, product = tmp_path / "source.txt", tmp_path / "product.txt"
    source.write_text("independent source")
    script = tmp_path / "owned_tool.py"
    script.write_text("from pathlib import Path\nimport sys\n" +
                      ("Path(sys.argv[1]).write_text('mutation')\n" if mutate_input else "") +
                      "Path(sys.argv[2]).write_bytes(Path(sys.argv[1]).read_bytes())\n" +
                      f"print('observed process')\nsys.exit({exit_status})\n")
    result = I.run([sys.executable, str(script), str(source), str(product)], directory=tmp_path / "evidence",
                   stage="actual_process", inputs=(source,), outputs=(product,), dependencies=(script,),
                   capture_output=True, text=True, timeout=10)
    return result, next((tmp_path / "evidence").rglob("invocation.json")), source, product, script


def test_actual_process_pins_inputs_tools_sources_products_and_full_stdout(tmp_path):
    result, path, source, product, script = _command(tmp_path)
    record = I.verify(path)
    assert result.returncode == 0 and record["kind"] == "subprocess"
    assert record["inputs"][0]["path"] == str(source) and record["outputs"][0]["path"] == str(product)
    assert record["dependencies"][0]["path"] == str(script)
    assert Path(record["stdout"]["path"]).read_bytes() == b"observed process\n"
    product.write_text("changed output")
    with pytest.raises(ValueError, match="product changed"):
        I.verify(path)


@pytest.mark.parametrize("exit_status,mutation", [(7, False), (0, True)])
def test_failed_or_input_mutating_invocation_cannot_be_verified(tmp_path, exit_status, mutation):
    _, path, *_ = _command(tmp_path, exit_status=exit_status, mutate_input=mutation)
    with pytest.raises(ValueError, match="did not complete"):
        I.verify(path)


def test_timeout_keeps_interrupted_observation_and_rethrows_process_timeout(tmp_path):
    with pytest.raises(subprocess.TimeoutExpired):
        I.run([sys.executable, "-c", "import time; time.sleep(5)"], directory=tmp_path,
              stage="actual_timeout", timeout=0.05, capture_output=True)
    path = next(tmp_path.rglob("invocation.json"))
    assert json.loads(path.read_text())["status"] == "interrupted"
    with pytest.raises(ValueError):
        I.verify(path)


def test_call_observation_is_explicitly_not_a_subprocess_command(tmp_path):
    source = tmp_path / "input"
    source.write_text("immutable")
    with I.observe_call(tmp_path / "evidence", stage="actual_callable", function=I.verify,
                        arguments={"input": str(source)}, inputs=(source,)) as observed:
        observed.returned(stdout="actual returned result")
    record = I.verify(observed.path)
    assert record["kind"] == "python_call" and record["outcome"] == "returned"
    assert "argv" not in record and "returncode" not in record


def test_scoped_package_executor_routes_actual_entrypoints_and_refuses_unowned_thread(tmp_path):
    from merlin.targetgen import package_runtime as P

    class Executor:
        def build_package(self, package, **kwargs):
            self.built = package

        def run_entrypoint(self, package, name, source, output=None, **kwargs):
            return subprocess.CompletedProcess(["owned", name], 0, stdout="owned result", stderr="")

    executor = Executor()
    package = P.Package(tmp_path, {"language": "python"}, tmp_path / "tool.py")
    with P.scoped_package_executor(executor):
        P.build_package(package)
        assert executor.built is package
        assert P.run_entrypoint(package, "parse", tmp_path / "input").stdout == "owned result"
        with ThreadPoolExecutor(max_workers=1) as pool:
            with pytest.raises(RuntimeError, match="escaped its scoped"):
                pool.submit(P.run_entrypoint, package, "parse", tmp_path / "input").result()
    assert P._scoped_executor() is None


def test_actual_child_environment_is_frozen_and_recorded_without_raw_values(tmp_path, monkeypatch):
    selected = {"PATH": "/usr/bin:/bin", "SYNTHETIC_PRIVATE_TOKEN": "private-canary", "BUILD_MODE": "original"}
    expected = dict(selected)
    output = tmp_path / "child.json"
    original_observer = I.observe

    def mutate_after_observation(*args, **kwargs):
        selected["BUILD_MODE"] = "changed"
        return original_observer(*args, **kwargs)

    monkeypatch.setattr(I, "observe", mutate_after_observation)
    child_script = "import json,os,sys;open(sys.argv[1],'w').write(json.dumps(dict(os.environ)))"
    result = I.run(
        [sys.executable, "-I", "-B", "-c", child_script,
         str(output)], directory=tmp_path / "evidence", stage="actual_selected_environment",
        outputs=(output,), env=selected, capture_output=True, timeout=10,
    )
    assert result.returncode == 0
    child = json.loads(output.read_text())
    assert all(child[key] == value for key, value in expected.items())
    path = next((tmp_path / "evidence").rglob("invocation.json"))
    record = I.require_environment(path, environment=expected)
    assert record["environment"]["keys"] == sorted(expected)
    assert "private-canary" not in path.read_text() and "original" not in path.read_text()
    with pytest.raises(ValueError, match="environment binding"):
        I.require_environment(path, environment=selected)


def test_actual_empty_environment_uses_process_default_path_not_ambient_selection(tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "nonexistent"))
    I.run(["true"], directory=tmp_path, stage="actual_empty_environment", env={}, capture_output=True, timeout=10)
    record = I.require_environment(next(tmp_path.rglob("invocation.json")), environment={})
    assert record["environment"]["keys"] == []
    assert Path(record["executable"]["path"]).name == "true"


@pytest.mark.parametrize("command", ["owned-tool", "./tools/owned-tool"])
def test_actual_child_relative_executable_selection_uses_its_working_directory(tmp_path, command):
    work = tmp_path / "child"
    tool = work / "tools" / "owned-tool"
    tool.parent.mkdir(parents=True)
    tool.write_text("#!/bin/sh\nexit 0\n")
    tool.chmod(0o700)
    selected = {"PATH": "tools"}
    I.run([command], directory=tmp_path / "evidence", stage="actual_child_relative_search",
          cwd=work, env=selected, capture_output=True, timeout=10)
    record = I.require_environment(next((tmp_path / "evidence").rglob("invocation.json")), environment=selected)
    assert record["executable"]["path"] == str(tool)


def test_actual_inherited_environment_selection_is_bound(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNTHETIC_BUILD_MODE", "before")
    selected = dict(os.environ)
    I.run([sys.executable, "-I", "-B", "-c", "pass"], directory=tmp_path,
          stage="actual_inherited_environment", capture_output=True, timeout=10)
    path = next(tmp_path.rglob("invocation.json"))
    I.require_environment(path, environment=selected)
    monkeypatch.setenv("SYNTHETIC_BUILD_MODE", "after")
    with pytest.raises(ValueError, match="environment binding"):
        I.require_environment(path, environment=dict(os.environ))


def test_legacy_completed_receipt_cannot_establish_effective_environment(tmp_path):
    _, path, *_ = _command(tmp_path)
    document = json.loads(path.read_text())
    document.pop("environment")
    path.write_text(json.dumps(document))
    assert I.verify(path)["status"] == "completed"
    with pytest.raises(ValueError, match="environment binding"):
        I.require_environment(path, environment={})


def test_posix_byte_mapping_and_string_mapping_bind_the_same_child_bytes():
    assert I.environment_identity({"A": "bc", "D": "e"}) == I.environment_identity({b"D": b"e", b"A": b"bc"})
    assert I.environment_identity({"A": "bc"}) != I.environment_identity({"Ab": "c"})
    with pytest.raises(ValueError, match="ambiguous"):
        I.environment_identity({"A": "first", b"A": b"second"})
