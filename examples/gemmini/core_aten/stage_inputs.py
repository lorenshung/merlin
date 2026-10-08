"""Stage operator inputs and diagnostic bundles; never seal or launch an experiment."""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from merlin.common.paths import module_source_path, out_dir, python_source_dir, repo_root
from merlin.targetgen.generate_bundles import materialize_bundles
from merlin.targetgen.sandbox.toolchain import ToolchainPaths
from merlin.targetgen.target_experiment import load_target_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, required=True)
    parser.add_argument("--hidden", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, required=True, help="fresh diagnostic staging root beneath out/artifacts"
    )
    parser.add_argument("--oracle-timing", type=Path, required=True, help="selected genuine timing path; may be absent")
    parser.add_argument("--rtl-facts", type=Path, required=True)
    parser.add_argument("--isa-header", type=Path, action="append", required=True, help="selected ordinary ISA header")
    args = parser.parse_args()
    public = args.public.resolve(strict=True)
    hidden = args.hidden.resolve(strict=True)
    if not public.is_dir() or not hidden.is_dir() or hidden != public.parent / "hidden":
        parser.error("public and hidden must be ordinary sibling categories, with the private category named hidden")
    if args.public.is_symlink() or args.hidden.is_symlink():
        parser.error("source categories must not be symlinks")
    output = args.output.absolute()
    if output.exists() or output.is_symlink():
        parser.error("choose a fresh staging output")
    if not output.resolve().is_relative_to(out_dir().resolve() / "artifacts"):
        parser.error("generated inputs must be beneath the configured out/artifacts root")
    if output.is_relative_to(public.parent) or public.parent.is_relative_to(output):
        parser.error("staging output must not overlap the source corpus")
    source = repo_root() / "examples/gemmini/core_aten"
    document = yaml.safe_load((source / "target/descriptor.yaml").read_text())
    document["capsule_corpus"] = str(public)
    resources = output / "payload/experiment"
    resources.mkdir(parents=True)
    document["resources_root"] = str(resources)
    # This declaration selects the exact source corpus. Genuine corpus prepare
    # will instead copy the corpus and resources into its own reviewed payload.
    header_root = resources / "hardware_spec/isa_headers"
    header_root.mkdir(parents=True)
    headers = []
    for header in args.isa_header:
        source_header = header.resolve(strict=True)
        if header.is_symlink() or not source_header.is_file():
            parser.error("ISA headers must be ordinary files")
        staged_header = header_root / header.name
        if staged_header.exists():
            parser.error("ISA headers must have distinct filenames")
        shutil.copy2(source_header, staged_header)
        headers.append(str(staged_header))
    document["hardware_spec"]["isa_headers"] = headers
    facts_root = resources / "rtl_facts"
    facts_root.mkdir()
    shutil.copy2(args.rtl_facts.resolve(strict=True), facts_root / "facts.json")
    shim = resources / "scripts/agent_selfcheck.py"
    shim.parent.mkdir()
    shutil.copy2(module_source_path("merlin_experiments.phase1.tools.selfcheck"), shim)
    descriptor = resources / "target_experiment.yaml"
    descriptor.write_text(yaml.safe_dump(document, sort_keys=False))
    private_inputs = [str(public.parent / "_private")]
    toolchain = ToolchainPaths.from_checkout()
    materialize_bundles(
        load_target_experiment(descriptor),
        resources / "input_bundles",
        variants=("public_v0",),
        arms=("merlin_rtlchecks",),
        host_inputs=tuple(private_inputs),
        python_source_root=python_source_dir(),
        llvm_toolchain_root=Path(toolchain.llvm).resolve(strict=True),
        rtl_facts_root=facts_root,
    )
    definition = yaml.safe_load((source / "experiment.yaml").read_text())
    definition["inputs"]["target"] = str(descriptor)
    config = definition["phases"][1]["config"]
    config["descriptor"] = str(descriptor)
    config["bundle_manifest"] = str(resources / "input_bundles" / config["bundle"] / "input_bundle_manifest.yaml")
    config["oracle_timing"] = str(args.oracle_timing.absolute())
    selected = output / "experiment.yaml"
    selected.write_text(yaml.safe_dump(definition, sort_keys=False))
    (output / "UNREVIEWED.md").write_text(
        "Diagnostic input staging only. No completed Phase 0 receipt, release preparation, "
        "review, seal, numerical certification or hardware qualification is asserted. "
        "Use corpus prepare on a genuine completed producer before Phase 1.\n"
    )
    print(selected)


if __name__ == "__main__":
    main()
