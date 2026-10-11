"""Prepare an operator-owned device corpus; never seal or start a candidate session."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path

import yaml

from merlin.common.paths import repo_root
from merlin.targetgen.core_aten_capture import capture_case_corpus, case_capture_name

OPS = tuple(
    "aten." + name
    for name in (
        "mm.default",
        "addmm.default",
        "convolution.default",
        "convolution_backward.default",
        "_fft_c2r.default",
        "_fft_r2c.default",
        "upsample_bilinear2d.vec",
        "upsample_nearest2d.vec",
    )
)
GUARDS = tuple(
    "aten." + name
    for name in (
        "bmm.default",
        "_cdist_forward.default",
        "add.Tensor",
        "relu.default",
        "sin.default",
        "sum.dim_IntList",
        "permute.default",
        "cat.default",
        "clone.default",
        "where.self",
    )
)


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n"
        if path.suffix == ".json"
        else yaml.safe_dump(value, sort_keys=False)
    )


def lift_overlay(overlay_path, float_path):
    # Float restatements retain overlay geometry/layout, with fresh native float goldens.
    # Native i8 wrapping and the non-Core widening bridge are different contracts.
    overlay = json.loads(overlay_path.read_text())
    import torch

    from merlin.targetgen.core_aten_cases import case_document_from_arguments, decode_arguments

    float_cases = []

    def lift(value):
        if isinstance(value, torch.Tensor):
            out = torch.empty_strided(value.shape, value.stride(), dtype=torch.float32)
            out.copy_(value.to(torch.float32) / 8)
            return out
        if isinstance(value, tuple):
            return tuple(lift(v) for v in value)
        if isinstance(value, list):
            return [lift(v) for v in value]
        if isinstance(value, dict):
            return {k: lift(v) for k, v in value.items()}
        return value

    for original in overlay["selected_cases"]:
        if original["overload"] not in ("aten.mm.default", "aten.convolution.default"):
            continue
        positional, keywords = decode_arguments(original["arguments"])
        case = case_document_from_arguments(original["overload"], lift(positional), lift(keywords))
        case["case_id"] = original["case_id"] + "::float32-spec-chain"
        case["source_overlay_case_id"] = original["case_id"]
        case["source_overlay_sha256"] = hashlib.sha256(overlay_path.read_bytes()).hexdigest()
        float_cases.append(case)
    write(float_path, {"cases": float_cases})


def count_classes(release):
    from collections import Counter

    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.kernels.shapes import observe_contractions
    from merlin.targetgen.capsule_common import load_capsule
    from merlin.targetgen.core_aten_capsules import load_private

    result = {}
    for cohort in ("public", "hidden"):
        tally = Counter()
        sites = 0
        for declaration in (release / "payload/corpus" / cohort).rglob("capsule.yaml"):
            capsule = load_capsule(declaration.parent)
            if not capsule.get("scored", True):
                continue
            golden = load_private(capsule)
            plan = golden["batch_map"]["numeric_preparation"]
            # Geometry is the provider's proved padded primitive, from its verified ISA header.
            dim = plan["schedule_proof"]["kernels"][0]["padded_primitive_k"]
            classes = set()
            for _, shape in observe_contractions(parse_mlir_file(declaration.parent / "capsule.interface.mlir")):
                if tuple(shape.dtypes) != ("i8", "i8", "i32"):
                    continue
                sites += 1
                m, n = shape.parallel[-2:]
                k = math.prod(shape.reduction)
                if m < dim and n < dim and k < dim:
                    classes.add("sub_tile")
                if all(v % dim == 0 for v in (m, n, k)):
                    classes.add("tile_aligned")
                for axis, extent in (("M", m), ("N", n), ("K", k)):
                    if extent % dim:
                        classes.add(axis + "_tail")
                if k > dim:
                    classes.add("K_split")
            tally.update(classes)
        result[cohort] = dict(
            contraction_sites=sites, capsule_classes=dict(sorted(tally.items())), class_counts_overlap=True
        )
    return result


def main():
    parser = argparse.ArgumentParser(__doc__)
    for name in (
        "canonical",
        "bounded",
        "overlay",
        "canonical-captures",
        "bounded-captures",
        "provider",
        "facts",
        "oracle-timing",
        "previous-release",
        "work",
        "release",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--reuse-overlay-captures", type=Path)
    parser.add_argument("--numeric-policy", default="spec_int8")
    args = parser.parse_args()
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    root = repo_root()
    work = args.work.resolve()
    if work.exists() or args.release.exists() or not args.release.name.endswith("-draft"):
        raise ValueError("fresh work and a fresh ...-draft release are required")
    work.mkdir(parents=True, mode=0o700)
    os.chmod(work, 0o700)
    assert Path(__import__("merlin").__file__).is_relative_to(root / "src")
    assert Path(__import__("merlin_experiments").__file__).is_relative_to(root / "packages/merlin-experiments/src")
    write(
        work / "import-resolution.json",
        dict(
            merlin=__import__("merlin").__file__,
            experiments=__import__("merlin_experiments").__file__,
            python=os.sys.executable,
            pythonpath=os.environ["PYTHONPATH"],
        ),
    )
    canonical = json.loads(args.canonical.read_text())
    bounded = json.loads(args.bounded.read_text())
    float_path = work / "overlay-float.json"
    from merlin.llvmlower.toolchain import m2m_python

    subprocess.run([str(m2m_python()), __file__, "--lift-overlay", str(args.overlay), str(float_path)], check=True)
    float_cases = json.loads(float_path.read_text())["cases"]
    capture_root = args.reuse_overlay_captures or work / "captures-overlay-float"
    if args.reuse_overlay_captures is None:
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            reports = list(pool.map(lambda c: capture_case_corpus({"cases": [c]}, capture_root), float_cases))
        write(work / "capture-summary.json", reports)
    # Screen membership by declared source-body and rank admission, not reference results.
    from merlin.frontends.linalg_mlir import parse_mlir_file
    from merlin.targetgen.core_aten_device import selected_facts
    from merlin.targetgen.core_aten_spec_int8 import admission_plan, select_spec

    with selected_facts("gemmini", args.facts) as facts:
        selection = select_spec(
            "gemmini",
            facts,
            root / "examples/gemmini/target/software-spec.yaml",
            root / "examples/gemmini/target/host-capabilities.yaml",
        )

    def eligible(case):
        assignment = case["partition_assignment"]
        if assignment["dtype"] not in ("float32", "complex64") or assignment["values"] not in (
            "ordinary",
            "sign_mix",
            "ties",
        ):
            return False
        capture = args.bounded_captures / case_capture_name(case)
        if not (capture / "capture.json").is_file():
            return False
        receipt = json.loads((capture / "capture.json").read_text())
        if receipt["status"] != "captured_exact" or not receipt["capture_meta"].get("result_contract"):
            return False
        try:
            plan = admission_plan(parse_mlir_file(capture / "capsule.linalg.mlir"), selection)
            return any(s["status"] == "admitted" for s in plan["sites"])
        except Exception:
            return False

    pools = {op: [c for c in bounded["selected_cases"] if c["overload"] == op and eligible(c)] for op in OPS}
    chosen_public = []
    for op in OPS[:2]:
        chosen_public.append(
            next(c for c in pools[op] if c["partition_assignment"]["layout"] == "noncontiguous_padded")
        )
    public_ids = {c["case_id"] for c in chosen_public}
    hidden = []
    for op in OPS:
        count = 2 if op in OPS[:3] else 1
        choices = [c for c in pools[op] if c["case_id"] not in public_ids]
        choices.sort(key=lambda c: (c["partition_assignment"]["values"] != "ordinary", c["case_id"]))
        if len(choices) < count:
            raise ValueError("insufficient independently admitted held-out cases for " + op)
        hidden.extend(choices[:count])
    cohorts = {
        "public": [
            dict(cases=str(args.canonical), captures=str(args.canonical_captures), select=list(OPS)),
            dict(cases=str(float_path), captures=str(capture_root), select=[c["case_id"] for c in float_cases]),
            dict(cases=str(args.bounded), captures=str(args.bounded_captures), select=sorted(public_ids)),
        ],
        "hidden": [
            dict(cases=str(args.bounded), captures=str(args.bounded_captures), select=[c["case_id"] for c in hidden])
        ],
        "host_guard": [dict(cases=str(args.canonical), captures=str(args.canonical_captures), select=list(GUARDS))],
    }
    flow = dict(
        policy=args.numeric_policy,
        software_spec=str(root / "examples/gemmini/target/software-spec.yaml"),
        host_capabilities=str(root / "examples/gemmini/target/host-capabilities.yaml"),
        execution_provider=str(args.provider / "merlin-support/core_aten/execution_provider.py"),
        device_package=str(args.provider),
        rtl_facts=str(args.facts),
    )
    write(
        work / "recipe.yaml",
        dict(
            capsule_policy="derived_only", core_aten=dict(overlay=str(args.overlay), numeric_flow=flow, cohorts=cohorts)
        ),
    )
    old = args.previous_release / "payload/experiment"
    contract = yaml.safe_load((old / "contracts/target_contract.yaml").read_text())
    contract["runner"]["full_call_provider"] = flow["execution_provider"]
    instructions = contract["runner"].get("full_call_instructions", "")
    contract["runner"]["full_call_instructions"] = (
        instructions
        + "\nSelected numerical contract: "
        + args.numeric_policy
        + ". The prepared public source contains only declared static quantization, "
        "i8 contractions, i32 readout and f32 scaling. Additional result slots expose exact i32 "
        "accumulators and f32 readouts; all are mandatory. Original FP32 error is diagnostic only. "
        "Catalog kernels must include schedule metadata accepted by the selected provider numerical schedule check.\n"
    )
    write(work / "target_contract.yaml", contract)
    os.environ["MERLIN_TARGET_CONTRACT"] = str(work / "target_contract.yaml")
    descriptor = yaml.safe_load((old / "target_experiment.yaml").read_text())
    public_task = work / "task"
    public_task.mkdir()
    for filename in ("software-spec.yaml", "host-capabilities.yaml"):
        shutil.copyfile(root / "examples/gemmini/target" / filename, public_task / filename)
    descriptor["task_root"] = str(public_task)
    descriptor["resources_root"] = str(old)
    descriptor["hardware_spec"]["target_contract"] = str(work / "target_contract.yaml")
    descriptor["grading"] = dict(
        release_admission="derive_from_corpus_v1", resource_bound=dict(policy="representative_l3_capstones_v1")
    )
    descriptor["capsule_corpus"] = str(work / "ungraded/public")
    descriptor["answer_surfaces"]["prior_backends"].extend(
        [str(args.provider), str(root / "examples/gemmini/phase1_corpus")]
    )
    descriptor["workload_spec"]["phase1"] = dict(schedule="continuous", round_timeout=43200, max_wall_s=43200)
    write(work / "descriptor.yaml", descriptor)
    authored = dict(
        schema_version=1,
        id="gemmini-spec-int8-device-corpus",
        target="gemmini",
        description="Diagnostic preparation of the declared int8 device corpus",
        phases={
            0: dict(
                adapter="capsule_derivation",
                config=dict(
                    descriptor=str(work / "descriptor.yaml"),
                    recipe=str(work / "recipe.yaml"),
                    performance_template=str(root / "experiments/templates/phase0/performance.yaml"),
                    capability_contract=str(work / "target_contract.yaml"),
                    evidence_mode="diagnostic",
                    rtl_facts=str(args.facts),
                ),
            )
        },
    )
    write(work / "authored.yaml", authored)

    def cli(*arguments, output):
        with output.open("w") as log:
            subprocess.run(
                [str(root / ".venv/bin/merlin"), "experiment", *map(str, arguments)],
                cwd=root,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    cli("corpus", "derive", work / "authored.yaml", "--output", work / "derive", output=work / "derive.log")
    config = authored["phases"][0]["config"]
    config.update(
        recipe=str(work / "derive/recipe.yaml"),
        conformance_spec=str(work / "derive/requirements.yaml"),
        synth_profile=str(work / "derive/synth.yaml"),
    )
    write(work / "derived.yaml", authored)
    cli("run", work / "derived.yaml", "--phase", "0", "--run-dir", work / "phase0-run", output=work / "phase0.log")
    cli(
        "corpus",
        "prepare",
        work / "phase0-run",
        "--generated-only",
        "--output",
        args.release,
        output=work / "prepare.json",
    )
    cli("corpus", "inspect", args.release, output=work / "inspect.json")
    released = args.release / "payload/experiment"
    reviewed = dict(
        schema_version=1,
        id="gemmini-spec-int8-device-reviewed",
        target="gemmini",
        description="Declared int8 device lane with unscored host guards",
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
    write(work / "reviewed-definition.yaml", reviewed)
    roster = dict(
        public=[*filter(lambda c: c["overload"] in OPS, canonical["cases"]), *float_cases, *chosen_public],
        hidden=hidden,
        host_guard=[c for c in canonical["cases"] if c["overload"] in GUARDS],
    )
    write(work / "private-roster.json", roster)
    from collections import Counter

    counts = {
        cohort: dict(
            total=len(cases),
            by_op=dict(Counter(c["overload"] for c in cases)),
            by_source=dict(
                Counter(
                    "overlay_float" if "source_overlay_case_id" in c else "bounded" if "case_id" in c else "canonical"
                    for c in cases
                )
            ),
            by_layout=dict(
                Counter(
                    c.get("partition_assignment", {}).get(
                        "layout", "overlay" if "source_overlay_case_id" in c else "canonical"
                    )
                    for c in cases
                )
            ),
        )
        for cohort, cases in roster.items()
    }
    counts["device_classes"] = count_classes(args.release)
    write(work / "counts.json", counts)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    import sys

    if sys.argv[1] == "--lift-overlay":
        lift_overlay(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
