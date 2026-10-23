"""Submitted code that a full-call provider imports runs only in the isolated build sandbox."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

full_call_build = pytest.importorskip("merlin.targetgen.full_call_build")

# What a hostile submitted module would do on import: patch the grader, then hunt for answers.
HOSTILE = """
import json, pathlib, sys
request = json.loads(pathlib.Path(sys.argv[1]).read_text())
from merlin.targetgen import core_aten_capsules
core_aten_capsules.grade_readback = lambda *a, **k: "rewritten by the candidate"
seen = {}
for key in ("private_answer", "corpus_answer"):
    try:
        seen[key] = pathlib.Path(request[key]).read_text()
    except OSError as exc:
        seen[key] = type(exc).__name__
inputs = sorted(p.name for p in pathlib.Path(request["bundle"]).iterdir())
pathlib.Path(request["result"]).write_text(json.dumps({"error": json.dumps({"seen": seen, "inputs": inputs})}))
"""


@pytest.mark.skipif(shutil.which("bwrap") is None, reason="bubblewrap is unavailable")
def test_hostile_submitted_code_cannot_reach_private_inputs_or_the_grader(tmp_path, monkeypatch):
    from merlin.targetgen import core_aten_capsules, plugins

    run = tmp_path / "run"
    bundle, corpus = run / ".private_full_call_runtime", tmp_path / "corpus"
    for directory in (bundle, corpus / "_private", tmp_path / "support", tmp_path / "submission"):
        directory.mkdir(parents=True)
    (bundle / "model.mlir").write_text("module {}\n")
    (bundle / "golden.npz").write_text("PRIVATE ANSWER BYTES")
    (corpus / "_private" / "golden.yaml").write_text("CORPUS ANSWER BYTES")
    (tmp_path / "facts.json").write_text("{}")
    monkeypatch.setattr(plugins, "resolve_support", lambda target: SimpleNamespace(base=tmp_path / "support"))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    original = core_aten_capsules.grade_readback
    real_run = subprocess.run

    def hostile_child(command, **kwargs):
        request = json.loads((run / ".isolated" / "request.json").read_text())
        request.update(private_answer=str(bundle / "golden.npz"), corpus_answer=str(corpus / "_private/golden.yaml"))
        (run / ".isolated" / "request.json").write_text(json.dumps(request))
        tail = command.index(sys.executable)
        return real_run([*command[:tail], sys.executable, "-c", HOSTILE, command[-1]], **kwargs)

    monkeypatch.setattr(full_call_build.subprocess, "run", hostile_child)
    report = full_call_build.build_in_sandbox(
        {"target": "t"},
        bundle=bundle,
        work=run / ".isolated",
        package=tmp_path / "submission",
        provider=tmp_path / "support" / "provider.py",
        facts=tmp_path / "facts.json",
        denied=(corpus,),
        timeout=60,
    )
    observed = json.loads(report["error"])
    assert observed["seen"] == {"private_answer": "FileNotFoundError", "corpus_answer": "FileNotFoundError"}
    assert observed["inputs"] == ["model.mlir"]  # program inputs only; never the answers beside them
    assert core_aten_capsules.grade_readback is original  # the patch stayed in the sandboxed process


def test_sandbox_refuses_a_bind_that_would_expose_private_inputs(tmp_path, monkeypatch):
    from merlin.targetgen import plugins

    bundle = tmp_path / "run" / ".private_full_call_runtime"
    bundle.mkdir(parents=True)
    (tmp_path / "facts.json").write_text("{}")
    # The submission directory CONTAINS the private bundle: binding it would expose the answers.
    monkeypatch.setattr(plugins, "resolve_support", lambda target: SimpleNamespace(base=tmp_path / "run"))
    monkeypatch.delenv("MERLIN_TARGET_CONTRACT", raising=False)
    monkeypatch.setattr(full_call_build.subprocess, "run", lambda *a, **k: pytest.fail("must not launch"))
    with pytest.raises(RuntimeError, match="would expose private grading inputs"):
        full_call_build.build_in_sandbox(
            {"target": "t"},
            bundle=bundle,
            work=tmp_path / "work",
            package=tmp_path / "run",
            provider=tmp_path / "provider.py",
            facts=tmp_path / "facts.json",
            denied=(),
            timeout=60,
        )
