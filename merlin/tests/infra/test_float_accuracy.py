"""Accuracy against the float model: required for an open model, judged at the capture's tolerance,
failing closed on anything it cannot judge. Each rule is exercised in both directions on synthetic
capsules and consoles, and the gate wiring through the one seam it adds."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
import yaml

from merlin.perf import float_accuracy as FA
from merlin.perf import whole_model_gate as G
from merlin.targetgen.golden_store import load_golden, write_golden

REFERENCE = [0.5, -1.25, 2.0, 0.0]


def _capsule(tmp_path: Path, *, policy=(0.03125, 0.02), float_outputs=None, quantized=None) -> Path:
    capsule = tmp_path / "cap"
    capsule.mkdir()
    numeric = {"compare": "tolerance_float", "dtype": "f32"}
    if policy is not None:
        numeric.update(atol=policy[0], rtol=policy[1])
    (capsule / "capsule.yaml").write_text(yaml.safe_dump({"name": "M", "numeric_policy": numeric}))
    golden = {"golden_source": "host_torch_eager", "outputs": {"Y0": quantized or [0.4, -1.2, 2.1, 0.01]}}
    if float_outputs is not None:
        golden["float_reference"] = {"source": "test", "outputs": float_outputs}
    write_golden(capsule, golden)
    return capsule


def _console(values) -> str:
    bits = np.asarray(values, np.float32).view(np.uint32)
    return "boot\nOUT {} {}\nend\n".format(len(bits), " ".join(str(int(b)) for b in bits))


def test_an_output_within_the_capture_tolerance_of_the_float_model_passes(tmp_path):
    capsule = _capsule(tmp_path, float_outputs={"Y0": REFERENCE})
    near = [v + 0.01 for v in REFERENCE]
    result = FA.check(_console(near), capsule)
    assert result["passed"] is True and result["within"] == result["of"] == 4
    assert result["policy"] == {"atol": 0.03125, "rtol": 0.02}
    assert result["max_abs"] == pytest.approx(0.01, abs=1e-6) and result["cosine"] > 0.999


def test_one_element_outside_the_tolerance_fails_and_says_how_far(tmp_path):
    capsule = _capsule(tmp_path, float_outputs={"Y0": REFERENCE})
    off = list(REFERENCE)
    off[1] += 0.5  # 0.5 > 0.03125 + 0.02 * 1.25
    result = FA.check(_console(off), capsule)
    assert result["passed"] is False and result["within"] == 3 and result["of"] == 4
    assert result["max_abs"] == pytest.approx(0.5, abs=1e-6) and result["max_rel"] == pytest.approx(0.4, abs=1e-6)


def test_the_threshold_is_the_capsules_own_declaration(tmp_path):
    loose = _capsule(tmp_path, policy=(1.0, 0.0), float_outputs={"Y0": REFERENCE})
    off = [v + 0.5 for v in REFERENCE]
    assert FA.check(_console(off), loose)["passed"] is True


@pytest.mark.parametrize("nonfinite", [float("inf"), -float("inf"), float("nan")])
@pytest.mark.parametrize("arm", ["reference", "candidate", "both"])
def test_compare_counts_nonfinite_operands_as_violations(nonfinite, arm):
    reference, candidate = [1.0, 0.0, -1.0], [1.0, 0.0, -1.0]
    if arm in ("reference", "both"):
        reference[1] = nonfinite
    if arm in ("candidate", "both"):
        candidate[1] = nonfinite
    with np.errstate(invalid="ignore"):
        result = FA.compare(candidate, reference, atol=0.03125, rtol=0.02)
    assert result["within"] == 2 and result["of"] == 3
    assert result["policy"] == {"atol": 0.03125, "rtol": 0.02}


@pytest.mark.parametrize("nonfinite", [float("inf"), -float("inf")])
def test_check_refuses_finite_candidate_against_infinite_reference(tmp_path, nonfinite):
    reference = list(REFERENCE)
    reference[1] = nonfinite
    capsule = _capsule(tmp_path, float_outputs={"Y0": reference})
    result = FA.check(_console(REFERENCE), capsule)
    assert not result["passed"] and result["within"] == 3 and result["of"] == 4
    assert result["policy"] == {"atol": 0.03125, "rtol": 0.02}


def test_finite_tolerance_boundary_is_unchanged():
    reference = [2.0, -2.0, 0.0, 0.0]
    candidate = [3.25, -3.25, 0.25, float(np.nextafter(0.25, float("inf")))]
    result = FA.compare(candidate, reference, atol=0.25, rtol=0.5)
    assert result["within"] == 3 and result["of"] == 4


@pytest.mark.parametrize(
    "build, console, why",
    [
        (lambda p: _capsule(p), _console(REFERENCE), "no float reference"),
        (lambda p: _capsule(p, policy=None, float_outputs={"Y0": REFERENCE}), _console(REFERENCE), "numeric_policy"),
        (lambda p: _capsule(p, float_outputs={"Y0": REFERENCE}), _console(REFERENCE[:3]), "3 of the 4"),
        (lambda p: _capsule(p, float_outputs={"Y0": REFERENCE}), "no output line\n", "printed no output"),
        (
            lambda p: _capsule(p, float_outputs={"Y0": REFERENCE, "Y1": REFERENCE}),
            _console(REFERENCE),
            "2 outputs",
        ),
    ],
)
def test_what_cannot_be_judged_fails_closed(tmp_path, build, console, why):
    result = FA.check(console, build(tmp_path))
    assert result["passed"] is False and why in result["note"]


def test_the_quantized_golden_is_never_taken_for_the_float_reference(tmp_path):
    """A capsule whose golden is the quantized model must not pass by agreeing with itself."""
    capsule = _capsule(tmp_path, quantized=REFERENCE)
    assert FA.check(_console(REFERENCE), capsule)["passed"] is False


def _open_model_run(tmp_path, monkeypatch, capsule: Path, console: str):
    """run_model on an open model, every seam but the new check replaced."""
    from merlin.perf import whole_model_builder as B
    from merlin.perf import whole_model_screen as S

    build_record = tmp_path / "build_record.json"
    build_record.write_text(
        json.dumps(
            {
                "attribution": {"counts": {}, "per_group": [{"group": 0, "op": "matmul", "on": "package"}]},
                "reference_identity": {"x": 1},
            }
        )
    )
    record = {
        "elf": "e",
        "elf_sha256": "0" * 64,
        "notes": {"build_record": str(build_record)},
        "expectations": {"groups": {"0": {"compare": "exact"}}, "output": {"elements": 4}},
    }
    monkeypatch.setattr(B, "build", lambda package, **kwargs: record)

    def screen(elf, *, out, **kwargs):
        out.mkdir(parents=True, exist_ok=True)
        (out / "console.txt").write_text(console)
        return {"status": "screened", "groups": [{"group": 0, "local": "correct"}], "output": [4, 4]}

    monkeypatch.setattr(S, "structure_screen", screen)
    monkeypatch.setattr(G, "_against_reference", lambda *a, **k: ({"passed": True}, {"passed": True}))
    G._READOUT_CHOICE.clear()
    model = {"name": "M", "capsule": str(capsule), "machine": "m", "header": "h", "coverage_floor": 0.0}
    G._READOUT_CHOICE[G._choice_key(model)] = {"machine": "m", "header": "h"}
    return G.run_model(tmp_path, model, target="t", roles=(), out=tmp_path / "o", keep_build=True)


def test_an_open_model_must_pass_float_accuracy_to_pass_the_gate(tmp_path, monkeypatch):
    capsule = _capsule(tmp_path, float_outputs={"Y0": REFERENCE})
    good = _open_model_run(tmp_path, monkeypatch, capsule, _console(REFERENCE))
    assert good["checks"]["float_accuracy"]["passed"] is True and good["status"] == G.PASS

    far = [v + 1.0 for v in REFERENCE]
    bad = _open_model_run(tmp_path, monkeypatch, capsule, _console(far))
    assert bad["checks"]["end_result"]["passed"] is True  # self-consistent ...
    assert bad["checks"]["float_accuracy"]["passed"] is False and bad["status"] == G.FAIL  # ... and still wrong
    lines = G.feedback_lines({"models": [bad]})
    assert any("FAIL float accuracy" in line and "0 of 4" in line for line in lines)


def test_a_capsule_without_a_float_reference_fails_the_gate_closed(tmp_path, monkeypatch):
    result = _open_model_run(tmp_path, monkeypatch, _capsule(tmp_path), _console(REFERENCE))
    assert result["checks"]["float_accuracy"]["passed"] is False and result["status"] == G.FAIL


def test_the_model_capsule_writer_carries_the_float_reference_into_the_golden(tmp_path, monkeypatch):
    from merlin.targetgen import capsule_source as CS

    workdir = tmp_path / "slot"
    workdir.mkdir()
    assert CS._float_reference_of(workdir) is None
    (workdir / "float_reference.json").write_text(json.dumps(REFERENCE))
    assert CS._float_reference_of(workdir) == REFERENCE
    # The golden written for a model keeps the quantized outputs as THE golden and adds the reference.
    capsule = _capsule(tmp_path, float_outputs={"Y0": REFERENCE})
    golden = load_golden(capsule)
    assert golden["outputs"]["Y0"] != golden["float_reference"]["outputs"]["Y0"]


_WORKER_PROBE = """
import json, random, sys
import numpy as np, torch
sys.path.insert(0, sys.argv[1])
import _m2m_capture_worker as W

class Noisy(torch.nn.Module):
    def forward(self, x):
        return x * 2.0 + torch.rand(1) * 0.0   # draws from the torch RNG
torch.manual_seed(7); random.seed(7); np.random.seed(7)
before = (torch.rand(1).item(), random.random(), float(np.random.rand()))
torch.manual_seed(7); random.seed(7); np.random.seed(7)
ref = W._float_reference(Noisy(), (torch.tensor([1.0, -2.0]),), torch)
after = (torch.rand(1).item(), random.random(), float(np.random.rand()))
print(json.dumps({"ref": ref, "same_rng": before == after}))
"""


def test_the_worker_records_the_float_model_without_moving_any_rng():
    python = os.environ.get("MERLIN_M2M_PYTHON")
    if not python:
        pytest.skip("MERLIN_M2M_PYTHON names no capture interpreter (torch is not in the test environment)")
    from merlin.targetgen import _m2m_capture_worker as worker

    done = subprocess.run(
        [python, "-c", _WORKER_PROBE, str(Path(worker.__file__).parent)],
        capture_output=True,
        text=True,
        timeout=600,
        check=True,
    )
    got = json.loads(done.stdout.strip().splitlines()[-1])
    assert got["ref"] == {"outputs": [2.0, -4.0]} and got["same_rng"] is True
