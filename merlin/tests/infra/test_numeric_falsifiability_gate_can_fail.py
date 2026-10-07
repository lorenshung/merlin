"""A real source oracle, loose policy and constant answer must fail the gate.

The mutation uses an integer program whose deterministic output has spread and
whose float comparison is explicitly declared. Raising its absolute tolerance
admits the midrange constant; the tight companion must remain clean. Missing
floating oracles cannot be replaced by integer stimuli, and an explicit partial
audit cannot suppress malformed outputs or evaluator failures.
"""

from __future__ import annotations

import importlib.util
import sys

import pytest
import yaml

from merlin.common.paths import repo_root

SCRIPTS = repo_root() / "build_tools" / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


FALSIFIABILITY = _load("check_numeric_falsifiability")


def _corpus(tmp_path, atol: float):
    """A one-capsule corpus whose golden is DERIVED, like almost every capsule in the real tree.

    Deliberately not a stored ``golden.yaml``: the defect being proved against is a check that only
    looked at capsules carrying one, so a fixture that shipped an answer key would exercise the very
    path that was never the problem.
    """
    directory = tmp_path / "capsules" / "TT0_two_element_sum"
    directory.mkdir(parents=True, exist_ok=True)
    capsule = {
        "name": "TT0_two_element_sum",
        "kind": "isa",
        "source_role": "derived_sweep",
        "source_reference": "a two-element elementwise sum, small enough that its golden is obvious",
        "label": "public",
        "interface_mlir": "capsule.interface.mlir",
        "inputs": [
            {"name": "X0", "role": "input", "shape": [1, 4], "dtype": "i8"},
            {"name": "X1", "role": "input", "shape": [1, 4], "dtype": "i8"},
        ],
        "operation": {
            "op": "residual_add",
            "attributes": {
                "lhs": "X0",
                "rhs": "X1",
                "out": "Y0",
                "epilogue": [],
                "lhs_scale": 1.0,
                "rhs_scale": 1.0,
                "bound_lsb": 0,
                "output_dtype": "i8",
            },
        },
        "numeric_policy": {"compare": "tolerance_float", "dtype": "i8", "atol": atol, "rtol": 0.0},
        "expected": {"instruction_classes": [], "modes": {"relu": False}},
        "required_oracle_tiers": ["L0"],
        "vcs": "optional",
        "firesim": "optional",
    }
    (directory / "capsule.yaml").write_text(yaml.safe_dump(capsule, sort_keys=False), encoding="utf-8")
    (directory / "capsule.interface.mlir").write_text("module {}\n", encoding="utf-8")
    return directory.parent


@pytest.fixture
def golden_spread(tmp_path):
    """The golden this corpus derives, and the half-spread a constant answer must miss it by."""
    corpus = _corpus(tmp_path, atol=0.0)
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import capsule_golden as CG

    directory = next(corpus.iterdir())
    values = CG.golden(CC.load_capsule(directory), directory)
    flat = [float(v) for output in values.values() for row in output for v in (row if isinstance(row, list) else [row])]
    if max(flat) == min(flat):
        pytest.skip("this fixture's stimulus produced a flat golden; it cannot prove a spread rule")
    return 0.5 * (max(flat) - min(flat))


def test_a_capsule_whose_tolerance_swallows_its_golden_is_reported(tmp_path, golden_spread):
    """THE MUTATION. An atol above the half-spread admits the midrange, so the capsule cannot fail."""
    corpus = _corpus(tmp_path, atol=golden_spread * 2.0)
    found = FALSIFIABILITY.offenders(corpus)
    assert {f["capsule"] for f in found} == {"TT0_two_element_sum"}
    # The MIDRANGE specifically: it is an optimal constant for this absolute-only fixture.
    # Asserting only on `zeros` would leave a band of tolerances
    # that reject the obvious wrong answer and still admit a better one.
    assert {f["answer"] for f in found} >= {"midrange"}


def test_a_capsule_whose_tolerance_is_tighter_than_its_golden_is_not_reported(tmp_path, golden_spread):
    """The clean direction. Without it, a gate that reported everything would pass the test above."""
    corpus = _corpus(tmp_path, atol=golden_spread * 0.5)
    assert FALSIFIABILITY.offenders(corpus) == []


def test_the_verdict_and_not_only_the_finding_changes(tmp_path, golden_spread):
    """A finding nothing turns into an exit code is the defect one level up: the audit existed and the
    two offenders shipped anyway. So the gate's own entry point must return 1 on the mutation and 0
    on the clean corpus."""
    assert FALSIFIABILITY.main(["--root", str(_corpus(tmp_path / "bad", atol=golden_spread * 2.0))]) == 1
    assert FALSIFIABILITY.main(["--root", str(_corpus(tmp_path / "ok", atol=golden_spread * 0.5))]) == 0


def test_missing_oracle_is_explicitly_partial_even_when_allowed(tmp_path, monkeypatch, capsys):
    corpus = _corpus(tmp_path, atol=0.0)

    def missing(*args):
        raise FALSIFIABILITY.capsule_golden.UnavailableGolden("independent oracle unavailable")

    monkeypatch.setattr(FALSIFIABILITY.capsule_golden, "golden", missing)
    assert FALSIFIABILITY.main(["--root", str(corpus), "--json"]) == 2
    import json

    report = json.loads(capsys.readouterr().out)
    assert report["complete"] is False and len(report["unmeasured"]) == 1
    assert report["coverage"]["assessed_policies"] == 0
    assert FALSIFIABILITY.main(["--root", str(corpus), "--allow-unmeasured"]) == 0
    assert "PARTIAL" in capsys.readouterr().out


def test_allow_unmeasured_cannot_hide_bad_declarations_or_empty_corpora(tmp_path, capsys):
    corpus = _corpus(tmp_path, atol=0.0)
    next(corpus.iterdir()).joinpath("capsule.yaml").write_text("[not, a, capsule]\n")
    assert FALSIFIABILITY.main(["--root", str(corpus), "--allow-unmeasured"]) == 2
    empty = tmp_path / "empty"
    empty.mkdir()
    assert FALSIFIABILITY.main(["--root", str(empty), "--allow-unmeasured"]) == 2
    assert "UNLOADABLE" in capsys.readouterr().out


def test_private_and_indirect_subtrees_are_never_entered(tmp_path):
    corpus = _corpus(tmp_path, atol=0.0)
    private = corpus / "hidden"
    private.mkdir()
    (private / "capsule.yaml").write_text("malformed: [\n")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "capsule.yaml").write_text("malformed: [\n")
    (corpus / "indirect").symlink_to(outside, target_is_directory=True)
    assert FALSIFIABILITY.offenders(corpus) == []
    assert FALSIFIABILITY.main(["--root", str(corpus / "indirect")]) == 2


def test_default_discovery_only_admits_tracked_public_declarations(tmp_path, monkeypatch):
    import subprocess

    corpus = _corpus(tmp_path, atol=0.0)
    paths = ["capsules/public/capsule.yaml", "capsules/hidden/private/capsule.yaml"]

    def tracked(argv, **kwargs):
        assert argv == ["git", "ls-files", "-z", "--", "capsules"]
        assert kwargs["check"] is True
        return subprocess.CompletedProcess(argv, 0, stdout="\0".join(paths).encode())

    monkeypatch.setattr(FALSIFIABILITY, "ROOT", tmp_path)
    monkeypatch.setattr(FALSIFIABILITY.subprocess, "run", tracked)
    assert FALSIFIABILITY._capsules(corpus, tracked_only=True) == [tmp_path / paths[0]]


def test_default_discovery_failure_is_not_a_clean_verdict(tmp_path, monkeypatch, capsys):
    import subprocess

    corpus = _corpus(tmp_path, atol=0.0)

    def failed(*args, **kwargs):
        raise subprocess.CalledProcessError(128, ["git", "ls-files"])

    monkeypatch.setattr(FALSIFIABILITY, "capsule_root", lambda: corpus)
    monkeypatch.setattr(FALSIFIABILITY.subprocess, "run", failed)
    assert FALSIFIABILITY.main(["--allow-unmeasured"]) == 2
    assert "discovery failed" in capsys.readouterr().err


@pytest.mark.parametrize(
    "outputs", [None, {}, {"Y": []}, {"Y": [[1], [2, 3]]}, {"Y": [float("nan")]}, {"Y": [float("inf")]}]
)
def test_strict_outputs_refuse_unassessed_values(outputs):
    policy = {"compare": "tolerance_float", "atol": 0.0, "rtol": 0.0}
    with pytest.raises(FALSIFIABILITY.numeric_falsifiability.UnmeasuredPolicy):
        FALSIFIABILITY.numeric_falsifiability.audit_outputs(policy, outputs, require_measurable=True)


@pytest.mark.parametrize("atol,rtol", [(-1.0, 0.0), (0.0, -1.0), (float("inf"), 0.0), (0.0, float("nan"))])
def test_strict_outputs_refuse_invalid_tolerances(atol, rtol):
    with pytest.raises(FALSIFIABILITY.numeric_falsifiability.UnmeasuredPolicy):
        FALSIFIABILITY.numeric_falsifiability.audit_outputs(
            {"compare": "tolerance_float", "atol": atol, "rtol": rtol}, {"Y": [1, 2]}, require_measurable=True
        )


def test_a_refused_output_does_not_hide_an_accepted_constant(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path, atol=1000.0)
    second = _corpus(tmp_path / "second", atol=0.0)
    import shutil

    shutil.copytree(next(second.iterdir()), corpus / "another")
    actual = FALSIFIABILITY.capsule_golden.golden

    def one_missing(capsule, directory):
        if directory.name == "another":
            raise ValueError("missing second oracle")
        return actual(capsule, directory)

    monkeypatch.setattr(FALSIFIABILITY.capsule_golden, "golden", one_missing)
    assert FALSIFIABILITY.main(["--root", str(corpus), "--allow-unmeasured"]) == 1


def test_unmeasured_output_in_same_capsule_cannot_hide_a_finding(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path, atol=1.0)
    monkeypatch.setattr(FALSIFIABILITY.capsule_golden, "golden", lambda *args: {"bad": [], "good": [0.1, 0.2]})
    assert FALSIFIABILITY.main(["--root", str(corpus), "--allow-unmeasured"]) == 1


def test_tracked_declaration_cannot_read_through_replaced_parent(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path, atol=0.0)
    outside = _corpus(tmp_path / "external", atol=0.0)
    link = corpus / "replaced"
    link.symlink_to(next(outside.iterdir()), target_is_directory=True)
    monkeypatch.setattr(FALSIFIABILITY, "_capsules", lambda *args, **kwargs: [link / "capsule.yaml"])

    def must_not_load(*args, **kwargs):
        pytest.fail("indirect capsule was opened")

    monkeypatch.setattr(FALSIFIABILITY.capsule_common, "load_capsule", must_not_load)
    assert FALSIFIABILITY.offenders(corpus)[0]["status"] == "unloadable"


def test_real_cli_rejects_the_loose_policy(tmp_path, golden_spread):
    import subprocess

    corpus = _corpus(tmp_path, atol=golden_spread * 2.0)
    result = subprocess.run(
        [sys.executable, str(SCRIPTS / "check_numeric_falsifiability.py"), "--root", str(corpus)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 1
    assert "FAIL" in result.stdout and "accepts 'midrange'" in result.stdout


@pytest.mark.parametrize("dtype", ["fp8_e4m3", "bf16", "f32", "unknown_format"])
def test_missing_source_format_oracle_never_becomes_an_integer_surrogate(tmp_path, dtype, monkeypatch, capsys):
    corpus = _corpus(tmp_path, atol=1000.0)
    path = next(corpus.iterdir()) / "capsule.yaml"
    capsule = yaml.safe_load(path.read_text())
    capsule["inputs"][0]["dtype"] = dtype
    path.write_text(yaml.safe_dump(capsule))

    def forbidden(*args, **kwargs):
        pytest.fail("integer stimulus/arithmetic used for a floating or unknown format")

    monkeypatch.setattr(FALSIFIABILITY.capsule_golden, "_recompute_golden", forbidden)
    # Unknown formats are schema-refused; declared float formats are unmeasured.
    if dtype == "unknown_format":
        assert FALSIFIABILITY.main(["--root", str(corpus), "--allow-unmeasured"]) == 2
    else:
        assert FALSIFIABILITY.main(["--root", str(corpus)]) == 2
        assert "UNMEASURED" in capsys.readouterr().out
        assert FALSIFIABILITY.main(["--root", str(corpus), "--allow-unmeasured"]) == 0


def test_matching_float_oracle_is_still_assessed(tmp_path, monkeypatch):
    corpus = _corpus(tmp_path, atol=1000.0)
    directory = next(corpus.iterdir())
    path = directory / "capsule.yaml"
    capsule = yaml.safe_load(path.read_text())
    capsule["inputs"][0]["dtype"] = "bf16"
    path.write_text(yaml.safe_dump(capsule))
    (directory / "golden.yaml").write_text(
        yaml.safe_dump({"golden_source": "independent_test", "outputs": {"Y0": [[0.25, 0.5]]}})
    )

    def forbidden(*args, **kwargs):
        pytest.fail("selected independent output was replaced by an integer recomputation")

    monkeypatch.setattr(FALSIFIABILITY.capsule_golden, "_recompute_golden", forbidden)
    assert FALSIFIABILITY.main(["--root", str(corpus), "--allow-unmeasured"]) == 1


@pytest.mark.parametrize(
    "outputs", [None, {}, {"Y": []}, {"Y": [[1], [2, 3]]}, {"Y": [float("nan")]}, {"Y": [float("inf")]}]
)
def test_partial_opt_out_cannot_hide_malformed_stored_outputs(tmp_path, outputs, capsys):
    corpus = _corpus(tmp_path, atol=0.0)
    directory = next(corpus.iterdir())
    (directory / "golden.yaml").write_text(yaml.safe_dump({"golden_source": "independent_test", "outputs": outputs}))
    assert FALSIFIABILITY.main(["--root", str(corpus), "--allow-unmeasured"]) == 2
    assert "UNLOADABLE" in capsys.readouterr().out


def test_partial_opt_out_cannot_hide_malformed_golden_bytes(tmp_path, capsys):
    corpus = _corpus(tmp_path, atol=0.0)
    directory = next(corpus.iterdir())
    (directory / "golden.yaml").write_text("golden_source: independent_test\noutputs: [\n")
    assert FALSIFIABILITY.main(["--root", str(corpus), "--allow-unmeasured"]) == 2
    assert "UNLOADABLE" in capsys.readouterr().out


def test_partial_opt_out_cannot_hide_evaluator_defects(tmp_path, monkeypatch, capsys):
    corpus = _corpus(tmp_path, atol=0.0)

    def failed(*args):
        raise RuntimeError("internal evaluator failure")

    monkeypatch.setattr(FALSIFIABILITY.capsule_golden, "golden", failed)
    assert FALSIFIABILITY.main(["--root", str(corpus), "--allow-unmeasured"]) == 2
    assert "UNLOADABLE" in capsys.readouterr().out
