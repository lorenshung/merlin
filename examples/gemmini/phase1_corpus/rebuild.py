"""Rebuild an additive, unsealed device draft from retained operator-owned corpora."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.capsule_common import load_capsule
from merlin.targetgen.core_aten_capsules import load_private
from merlin.targetgen.core_aten_capture import case_capture_name

DEVICE_OPS = {"aten.mm.default", "aten.addmm.default", "aten.convolution.default"}
HOST_OPS = {
    "aten._fft_c2r.default",
    "aten._fft_r2c.default",
    "aten.convolution_backward.default",
    "aten.upsample_bilinear2d.vec",
    "aten.upsample_nearest2d.vec",
}


def write(path, document):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(document, indent=2, sort_keys=True) + "\n"
        if path.suffix == ".json"
        else yaml.safe_dump(document, sort_keys=False)
    )
    path.chmod(0o600)


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in ("previous-release", "float-draft", "provider", "facts", "oracle-timing", "overlay", "work", "release"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    root = repo_root().resolve()
    if args.work.exists() or args.release.exists() or not args.release.name.endswith("-v3-draft"):
        raise ValueError("fresh work and a fresh ...-v3-draft release required")
    args.work.mkdir(parents=True, mode=0o700)
    import merlin_experiments

    import merlin

    assert Path(merlin.__file__).is_relative_to(root / "src")
    assert Path(merlin_experiments.__file__).is_relative_to(root / "packages/merlin-experiments/src")
    write(
        args.work / "import-resolution.json",
        dict(
            merlin=merlin.__file__,
            experiments=merlin_experiments.__file__,
            python=os.sys.executable,
            pythonpath=os.environ["PYTHONPATH"],
        ),
    )
    cohorts = {
        c: [dict(retained_corpus=str(args.previous_release / "payload/corpus" / c))]
        for c in ("public", "hidden", "host_guard")
    }
    roster = []
    # Reuse original floating capture bytes. The numeric fixture owner performs
    # fresh capture-time TorchAO preparation under the selected trunk and recipe.
    for cohort in ("public", "hidden"):
        for declaration in sorted((args.float_draft / "payload/corpus" / cohort).glob("*/capsule.yaml")):
            capsule = load_capsule(declaration.parent)
            record = load_private(capsule)["batch_map"]["cases"][0]
            case = record.get("fp32_case", record["case"])
            op = case["overload"]
            if op not in DEVICE_OPS | HOST_OPS:
                raise ValueError("unexpected draft operator: " + op)
            destination = cohort if op in DEVICE_OPS or cohort == "hidden" else "host_guard"
            lane = "device" if op in DEVICE_OPS else "host-guard"
            case_path = args.work / "cases" / (capsule["name"] + ".json")
            write(case_path, dict(cases=[case]))
            capture = Path(record["capture_directory"])
            assert capture.name == case_capture_name(case)
            cohorts[destination].append(
                dict(cases=str(case_path), captures=str(capture.parent), select=[case.get("case_id", op)], lane=lane)
            )
            roster.append(
                dict(
                    name=capsule["name"],
                    source_cohort=cohort,
                    category=destination,
                    lane=lane,
                    case=case,
                    original_capture=str(capture),
                )
            )
    write(args.work / "private-roster.json", roster)
    flow = dict(
        policy="spec_int8",
        software_spec=str(root / "examples/gemmini/target/software-spec.yaml"),
        host_capabilities=str(root / "examples/gemmini/target/host-capabilities.yaml"),
        execution_provider=str(args.provider / "merlin-support/core_aten/execution_provider.py"),
        device_package=str(args.provider),
        rtl_facts=str(args.facts),
    )
    write(
        args.work / "recipe.yaml",
        dict(
            capsule_policy="derived_only", core_aten=dict(overlay=str(args.overlay), numeric_flow=flow, cohorts=cohorts)
        ),
    )
    old = args.previous_release / "payload/experiment"
    contract = yaml.safe_load((old / "contracts/target_contract.yaml").read_text())
    contract["runner"]["full_call_provider"] = flow["execution_provider"]
    contract["runner"]["full_call_instructions"] = contract["runner"].get("full_call_instructions", "") + (
        "\nRetained int8-native calls preserve their original ATen result dtype and full-call contract. "
        "Float device calls use spec_int8 capture-time TorchAO preparation with static scales; "
        "mandatory additional results expose exact i32 accumulators and f32 readouts. "
        "Original FP32 error is diagnostic. Host guards are unscored and forbid on-mesh execution.\n"
    )
    write(args.work / "target_contract.yaml", contract)
    os.environ["MERLIN_TARGET_CONTRACT"] = str(args.work / "target_contract.yaml")
    descriptor = yaml.safe_load((old / "target_experiment.yaml").read_text())
    task = args.work / "task"
    task.mkdir()
    for filename in ("software-spec.yaml", "host-capabilities.yaml"):
        shutil.copyfile(root / "examples/gemmini/target" / filename, task / filename)
    descriptor.update(task_root=str(task), resources_root=str(old), capsule_corpus=str(args.work / "ungraded/public"))
    descriptor["hardware_spec"]["target_contract"] = str(args.work / "target_contract.yaml")
    descriptor["grading"] = dict(
        release_admission="derive_from_corpus_v1", resource_bound=dict(policy="representative_l3_capstones_v1")
    )
    descriptor["answer_surfaces"]["prior_backends"].extend(
        [str(args.provider), str(root / "examples/gemmini/phase1_corpus")]
    )
    descriptor["workload_spec"]["phase1"] = dict(schedule="continuous", round_timeout=43200, max_wall_s=43200)
    write(args.work / "descriptor.yaml", descriptor)
    authored = dict(
        schema_version=1,
        id="gemmini-additive-device-corpus",
        target="gemmini",
        description="Retained native int8 calls plus captured float device calls and host guards",
        phases={
            0: dict(
                adapter="capsule_derivation",
                config=dict(
                    descriptor=str(args.work / "descriptor.yaml"),
                    recipe=str(args.work / "recipe.yaml"),
                    performance_template=str(root / "experiments/templates/phase0/performance.yaml"),
                    capability_contract=str(args.work / "target_contract.yaml"),
                    evidence_mode="diagnostic",
                    rtl_facts=str(args.facts),
                ),
            )
        },
    )
    write(args.work / "authored.yaml", authored)

    def cli(*arguments, output):
        with output.open("w") as log:
            subprocess.run(
                [str(root / ".venv/bin/merlin"), "experiment", *map(str, arguments)],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    cli(
        "corpus",
        "derive",
        args.work / "authored.yaml",
        "--output",
        args.work / "derive",
        output=args.work / "derive.log",
    )
    config = authored["phases"][0]["config"]
    config.update(
        recipe=str(args.work / "derive/recipe.yaml"),
        conformance_spec=str(args.work / "derive/requirements.yaml"),
        synth_profile=str(args.work / "derive/synth.yaml"),
    )
    write(args.work / "derived.yaml", authored)
    cli(
        "run",
        args.work / "derived.yaml",
        "--phase",
        "0",
        "--run-dir",
        args.work / "phase0-run",
        output=args.work / "phase0.log",
    )
    cli(
        "corpus",
        "prepare",
        args.work / "phase0-run",
        "--generated-only",
        "--output",
        args.release,
        output=args.work / "prepare.json",
    )
    cli("corpus", "inspect", args.release, output=args.work / "inspect.json")
    released = args.release / "payload/experiment"
    reviewed = dict(
        schema_version=1,
        id="gemmini-additive-device-reviewed",
        target="gemmini",
        description="Additive int8-native and capture-time spec_int8 device corpus",
        inputs=dict(target=str(released / "target_experiment.yaml")),
        policy=dict(prohibited_instruction_roles=["loop_descriptor"]),
        phases={
            1: dict(
                adapter="capsule_bench",
                config=dict(
                    require_reviewed_corpus=True,
                    descriptor=str(released / "target_experiment.yaml"),
                    level="EL4",
                    bundle="merlin_assisted_rtlchecks_public_v0",
                    bundle_manifest=str(
                        released / "input_bundles/merlin_assisted_rtlchecks_public_v0/input_bundle_manifest.yaml"
                    ),
                    oracle_timing=str(args.oracle_timing),
                    driver="codex",
                    model="gpt-6.1-sol",
                    effort="high",
                    provider="subscription",
                    max_wall_s=43200,
                    round_timeout=43200,
                    grade_interval=900,
                    sim_max_jobs=2,
                ),
            )
        },
    )
    write(args.work / "reviewed-definition.yaml", reviewed)
    print(json.dumps(dict(release=str(args.release), state="unsealed", added=len(roster)), indent=2))


if __name__ == "__main__":
    main()
