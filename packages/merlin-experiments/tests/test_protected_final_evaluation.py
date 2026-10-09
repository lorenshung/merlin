"""Actual numerical reconstruction with a synthetic host execution capability.

The injected verifier only exercises lifecycle joins and refusals. Its fixture
cycles and qualification are not hardware observations or semantic qualification.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from dataclasses import replace
from types import ModuleType

import pytest
from merlin_experiments.phase2 import protected_final_evaluation as F
from merlin_experiments.phase2 import protected_verifier_qualification as Q
from merlin_experiments.phase2.contracts import StageGateError, sha256_file
from merlin_experiments.phase2.numerical_readback import ReadbackFiles
from test_numerical_readback import _rebind
from test_numerical_readback import native_case as _native_case

native_case = _native_case


VERIFIER = """
from dataclasses import replace
from pathlib import Path
import json
from merlin_experiments.phase2.protected_final_evaluation import FinalExecutionWitness, REQUIRED_WITNESSES
from merlin_experiments.phase2.contracts import sha256_file
from merlin_experiments.phase2.protected_verifier_qualification import ProtectedArmRefusal

def verify(*, binding, arm, files, original_reference):
    root = Path(__file__).parent
    config = json.loads((root / "options.json").read_text())
    controlled = binding.member.startswith("control/")
    if controlled:
        path = root / "controls" / (binding.member.split("/")[-1] + ".json")
        control = json.loads(path.read_text())
        if control["defect"]:
            if config.get("control_fault") == "crash":
                raise RuntimeError("synthetic verifier crashed; no diagnosed refusal")
            if config.get("control_fault") == "accept":
                control["defect"] = None
        if control["defect"]:
            raise ProtectedArmRefusal(witness_kind=control["defect"], binding_sha256=binding.identity(),
                                      arm=arm, evidence=((path, sha256_file(path)),),
                                      reason="synthetic control has explicit defective authority")
    evidence = tuple((root / (role + ".json"), sha256_file(root / (role + ".json"))) for role in REQUIRED_WITNESSES)
    result = FinalExecutionWitness(
        binding.identity(), arm,
        binding.reference_compiler_sha256 if arm == "reference" else binding.candidate_compiler_sha256,
        sha256_file(files.executable), sha256_file(files.executable), sha256_file(files.console),
        binding.hardware_sha256, binding.inputs_sha256, binding.timer_scope_sha256,
        (100 if arm == "reference" else 99) if controlled else config[arm + "_cycles"],
        sha256_file(original_reference.console),
        tuple((role, digest) for role, (_path, digest) in zip(REQUIRED_WITNESSES, evidence)), evidence,
    )
    mode = None if controlled else config.get("mode")
    if arm == "candidate" and mode == "hardware":
        result = replace(result, hardware_sha256="9" * 64)
    elif arm == "candidate" and mode == "staged":
        result = replace(result, staged_executable_sha256="9" * 64)
    elif arm == "candidate" and mode == "missing":
        result = replace(result, witnesses=result.witnesses[:-1])
    elif arm == "candidate" and mode == "unbound":
        result = replace(result, witnesses=tuple((role, "9" * 64) for role, _digest in result.witnesses))
    elif arm == "candidate" and mode == "mutate":
        files.console.write_text(files.console.read_text() + "altered")
    elif arm == "candidate" and mode == "dict":
        return vars(result)
    return result
"""


def _json(path, record):
    path.write_text(json.dumps(record))


def _request(case, *, mode=None, candidate_cycles=99, reference=None, original_reference=None,
             control_fault=None, omit_control=False):
    root = case.private / "execution"
    root.mkdir()
    owner = root / "verifier.py"
    owner.write_text(VERIFIER)
    options = root / "options.json"
    _json(options, dict(reference_cycles=100, candidate_cycles=candidate_cycles, mode=mode,
                       control_fault=control_fault))
    records = []
    for role in F.REQUIRED_WITNESSES:
        record = root / (role + ".json")
        _json(record, dict(test_fixture_only=True, role=role))
        records.append(record)
    module = ModuleType("synthetic_protected_final_verifier")
    module.__file__ = str(owner)
    sys.modules[module.__name__] = module
    exec(compile(owner.read_bytes(), str(owner), "exec"), module.__dict__)
    code_sha = F.callable_code_sha256(module.verify)
    dependencies = tuple((path, sha256_file(path)) for path in (owner, options, *records))
    verifier = F.ProtectedExecutionVerifier(
        module.verify, owner, dependencies, None, None, code_sha,
    )
    oracle = original_reference or case.arms[0]
    binding = F.FinalExecutionBinding(
        member="independent-final",
        reference_compiler_sha256="1" * 64,
        candidate_compiler_sha256="2" * 64,
        program_sha256="3" * 64,
        inputs_sha256="4" * 64,
        quality_budget_sha256=sha256_file(case.private / "budget.json"),
        original_reference_console_sha256=sha256_file(oracle.console),
        runtime_sha256="5" * 64,
        toolchain_sha256="6" * 64,
        hardware_sha256="7" * 64,
        timer_scope_sha256="8" * 64,
    )
    binding_file = root / "binding.json"
    _json(binding_file, vars(binding))
    _json(
        root / "campaign.json",
        dict(
            schema="merlin.protected_final_campaign.v1",
            members=[dict(member=binding.member, binding_sha256=binding.identity())],
        ),
    )
    controls = []
    control_root = root / "controls"
    control_root.mkdir()
    for kind in F.REQUIRED_WITNESSES:
        for expected in ("observe", "refuse"):
            name = kind + "_" + expected
            control_file = control_root / (name + ".json")
            _json(control_file, dict(test_fixture_only=True, defect=kind if expected == "refuse" else None))
            control_binding = replace(binding, member="control/" + name)
            original = control_root / (name + "_binding.json")
            _json(original, vars(control_binding))
            controls.append(Q.ProtectedVerifierControl(
                name, kind, expected, control_binding, original, "candidate", case.arms[1], oracle,
                tuple((path, sha256_file(path)) for path in (*records, control_file)),
            ))
    if omit_control:
        controls.pop()
    plan = root / "control_plan.json"
    _json(plan, dict(schema=Q.CONTROL_SCHEMA, scope="synthetic joins only; no actual source or hardware qualification",
                     dependencies=[[str(path), digest] for path, digest in dependencies],
                     controls=[row.record() for row in controls]))
    inputs = case.freeze()
    verifier = Q.qualify_protected_execution_verifier(
        selection=verifier, original_plan=plan, controls=tuple(controls), run_dir=inputs["run_dir"],
        environment=inputs["environment"], receipt=inputs["run_dir"] / "private-qualification/receipt.json",
    )
    if reference is not None:
        inputs["reference"] = reference
    return F.ProtectedFinalMember(
        **inputs,
        binding=binding,
        original_binding=binding_file,
        verifier=verifier,
        original_reference=oracle,
        original_reference_build=case.service,
    )


def test_final_lifecycle_reconstructs_all_native_values(native_case):
    request = _request(native_case)
    row = F.admit_protected_final_comparison(**vars(request))
    assert row.accuracy_passed and row.final_executable_passed
    assert (row.reference_cycles, row.candidate_cycles) == (100, 99)
    verdict = F.evaluate_protected_final_campaign(
        (request,),
        original_campaign=request.original_binding.parent / "campaign.json",
        expected_members=(row.member,),
    )
    assert verdict["status"] == "pass"
    assert verdict["convergence"]["status"] == "unknown"


@pytest.mark.parametrize(
    "mode,pattern",
    [
        ("hardware", "binding"),
        ("staged", "binding"),
        ("missing", "roster"),
        ("unbound", "evidence record"),
        ("mutate", "changed during"),
        ("dict", "typed arm"),
    ],
)
def test_incomplete_or_changed_execution_cannot_make_comparison(native_case, mode, pattern):
    request = _request(native_case, mode=mode)
    with pytest.raises(StageGateError, match=pattern):
        F.admit_protected_final_comparison(**vars(request))


def test_cycle_boolean_is_not_hardware_cycles(native_case):
    request = _request(native_case, candidate_cycles=True)
    with pytest.raises(StageGateError, match="positive measured"):
        F.admit_protected_final_comparison(**vars(request))


def test_selected_binding_cannot_be_replaced_after_freeze(native_case):
    request = _request(native_case)
    request = replace(request, binding=replace(request.binding, timer_scope_sha256="a" * 64))
    with pytest.raises(StageGateError, match="original evaluator freeze"):
        F.admit_protected_final_comparison(**vars(request))


def test_original_console_cannot_drift_from_frozen_numerical_bytes(native_case):
    request = _request(native_case)
    request.candidate.console.write_text(request.candidate.console.read_text() + "new bytes")
    with pytest.raises(StageGateError, match="absent, indirect or changed"):
        F.admit_protected_final_comparison(**vars(request))


def test_public_verifier_selection_is_not_a_private_capability(native_case, tmp_path):
    request = _request(native_case)
    public_binding = tmp_path / "public-binding.json"
    public_binding.write_bytes(request.original_binding.read_bytes())
    request = replace(request, original_binding=public_binding)
    with pytest.raises(StageGateError, match="private host input"):
        F.admit_protected_final_comparison(**vars(request))


def test_campaign_does_not_accept_precomputed_comparisons(native_case):
    request = _request(native_case)
    row = F.admit_protected_final_comparison(**vars(request))
    with pytest.raises(StageGateError, match="lifecycle inputs"):
        F.evaluate_protected_final_campaign(
            (row,),
            original_campaign=request.original_binding.parent / "campaign.json",
            expected_members=(row.member,),
        )
    with pytest.raises(StageGateError, match="roster"):
        F.evaluate_protected_final_campaign(
            (request, request),
            original_campaign=request.original_binding.parent / "campaign.json",
            expected_members=(row.member,),
        )


def _recompile(case, arm):
    source = arm.harness.parent / "kernel.c"
    source.write_text(source.read_text().replace("h[0]=", "f[3] += 0.03f; h[0]="))
    subprocess.run(
        case.recipe.compile_command(source=source, output=arm.kernel_object), check=True, capture_output=True
    )
    subprocess.run(
        case.recipe.link_command(
            objects=[arm.kernel_object, arm.harness.parent / "harness.o"],
            output=arm.executable,
            link_script=case.recipe.link_script,
        ),
        check=True,
        capture_output=True,
    )
    arm.console.write_bytes(subprocess.run([arm.executable], check=True, capture_output=True).stdout)


def test_two_approximations_do_not_double_original_tolerance(native_case):
    case = native_case
    approximate_dir = case.private / "approximate-reference"
    shutil.copytree(case.arms[0].harness.parent, approximate_dir)
    approximate = ReadbackFiles(*(approximate_dir / path.name for path in case.arms[0].paths()[:6]))
    _recompile(case, approximate)
    _recompile(case, case.arms[1])
    # A second independent increment puts candidate at .06 versus oracle zero.
    _recompile(case, case.arms[1])
    case.arms.append(approximate)
    _rebind(case, 2)
    _rebind(case, 1)
    request = _request(case, reference=approximate, original_reference=case.arms[0])
    pair_only = F.admit_protected_numerical_readback(
        run_dir=request.run_dir,
        environment=request.environment,
        original_budget=request.original_budget,
        budget=request.budget,
        reference=approximate,
        candidate=request.candidate,
        reference_build=case.service,
        candidate_build=case.service,
    )
    assert dict(pair_only.quality.values)["elementwise_violation_count"] == 0
    row = F.admit_protected_final_comparison(**vars(request))
    assert not row.accuracy_passed
    assert (
        F.evaluate_protected_final_campaign(
            (request,),
            original_campaign=request.original_binding.parent / "campaign.json",
            expected_members=(row.member,),
        )["status"]
        == "fail"
    )


def test_campaign_cannot_exclude_an_original_member(native_case):
    request = _request(native_case)
    campaign = request.original_binding.parent / "campaign.json"
    _json(
        campaign,
        dict(
            schema="merlin.protected_final_campaign.v1",
            members=[
                dict(member=request.binding.member, binding_sha256=request.binding.identity()),
                dict(member="omitted-member", binding_sha256="9" * 64),
            ],
        ),
    )
    with pytest.raises(StageGateError, match="original member/binding freeze"):
        F.evaluate_protected_final_campaign(
            (request,),
            original_campaign=campaign,
            expected_members=(request.binding.member,),
        )
