"""The example registry's certifier resolves to the core GSIM measurer, and that measurer's output is
what the worker's contract path turns into a verdict: MEASURED only for a quotable-correct graded run,
MEASURED_INVALID for a graded wrong one, REFUSED for one it could not grade."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from merlin_experiments.phase2.whole_model_measured import registry as REG
from merlin_experiments.phase2.whole_model_measured import worker as W
from merlin_experiments.phase2.whole_model_measured.identity import load_builder

from merlin.common.paths import repo_root
from merlin.perf import whole_model_gsim as G
from merlin.perf import whole_model_verdict as V

EXAMPLE = repo_root() / "examples" / "gemmini" / "phase2" / "whole-model-machines.yaml"


def _certifier() -> dict:
    document = REG.load(EXAMPLE)
    name = next(n for n, e in document["machines"].items() if e.get("kind") == "contract")
    return REG.resolve(document, name)


def test_the_example_certifier_is_the_core_gsim_measurer():
    spec = _certifier()
    assert spec["kind"] == "contract" and load_builder(spec["measurer"]) is G.measure
    assert REG.adjudicates(spec, "cycle_count")["status"] != "ADJUDICATED"


def _run(tmp_path: Path, monkeypatch, verdict: dict) -> dict:
    elf = tmp_path / "model.elf"
    elf.write_bytes(b"\x7fELF fake")
    sha = hashlib.sha256(elf.read_bytes()).hexdigest()
    (tmp_path / "memory_map.json").write_text(json.dumps({"groups": []}))
    (tmp_path / "oracle.json").write_text("{}")
    record = tmp_path / "build_record.json"
    record.write_text(
        json.dumps({"memory_map": str(tmp_path / "memory_map.json"), "oracle": {"path": str(tmp_path / "oracle.json")}})
    )

    def run(elf_path, memory_map, oracle, *, target, out, local=None):
        Path(out).mkdir(parents=True, exist_ok=True)
        return {"elf_sha256": sha, "emulator": {"firrtl_sha256": None}, "machine": {"registry_entry": None}, **verdict}

    monkeypatch.setattr(G, "run_gsim_whole_model", run)
    job = {"machine": _certifier(), "target": "t", "package_sha256": "p" * 64}
    build = {"elf": str(elf), "elf_sha256": sha, "parameter_header_sha256": "h" * 64}
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    return W._contract_measurement(job, job_dir, build, {"notes": {"build_record": str(record)}})


GRADED = {
    "status": "graded",
    "whole_window_cycles": 1234,
    "per_group": [{"group": 1, "kind": "matmul", "cycles": 1000, "correct": True}],
    "grade": {"agree": ["1"], "disagree": []},
    "argmax": {"from_dump": 3, "oracle": 3, "agrees_with_oracle": True},
}


@pytest.mark.parametrize(
    ("verdict", "status", "cycles"),
    [
        ({**GRADED, "quotable": True}, V.TIMING_MEASURED, 1234),
        (
            {**GRADED, "quotable": False, "grade": {"agree": [], "disagree": [{"group": 1}]}},
            V.TIMING_MEASURED_INVALID,
            None,
        ),
        ({"status": "refused", "refusal": "the dump is not complete", "quotable": False}, V.TIMING_REFUSED, None),
    ],
)
def test_the_gsim_measurer_output_becomes_the_workers_verdict(tmp_path, monkeypatch, verdict, status, cycles):
    result = _run(tmp_path, monkeypatch, verdict)
    assert result["timing_status"] == status and result["objective_cycles"] == cycles
    contract = result["run"]["contract_result"]
    assert contract["engine"] == "gsim" and contract["machine"].startswith("UNKNOWN")  # never echoed back
