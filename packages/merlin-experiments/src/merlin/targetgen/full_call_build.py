"""Build a full-call image from submitted code in an isolated process.

A Core ATen execution provider binds device calls by importing modules from the SUBMITTED package.
That code must never share a process with the grader: it could patch the readback grader or read the
private runtime bundle. The grader therefore decides the route from trusted inputs
(:func:`core_aten_device.source_route`) and runs only the provider/candidate build here, under a
deny-by-default bubblewrap sandbox that sees the selected interpreter, Merlin, the support provider,
the toolchains, the submission (read-only) and a fresh work directory holding copies of the public
program inputs. The grader alone executes the returned image and grades its readback.

    python -m merlin.targetgen.full_call_build <request.json>     (inside the sandbox)
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

#: Program inputs a build reads; nothing else of the private runtime bundle is copied out.
PROGRAM_INPUTS = (
    "model.mlir",
    "inputs.npz",
    "input_order.json",
    "semantic_io.json",
    "weights.safetensors.manifest.json",
    "weights.safetensors",
)
ADMISSION = "spec-int8-admission.json"


def _write(path: Path, document: dict) -> None:
    path.write_text(json.dumps(document, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")


def main(argv=None) -> int:
    """Child: route and build one image, then report the build and its routing record."""
    request = json.loads(Path((argv or sys.argv[1:])[0]).read_text(encoding="utf-8"))
    out = Path(request["result"])
    try:
        from merlin.llvmlower.device_build import load_region_offload
        from merlin.llvmlower.device_offload import load_sidecar
        from merlin.runtime.backends import spike_model
        from merlin.targetgen.core_aten_device import bind_source_route, load_execution_provider, selected_facts

        target, bundle, package = request["target"], Path(request["bundle"]), Path(request["package"])
        build_dir = Path(request["build"])
        provider = load_execution_provider(target, Path(request["provider"]))
        routing, reason = None, request["reason"]
        with selected_facts(target, Path(request["facts"])) as facts:
            prepare = getattr(provider, "prepare_bundle", None)
            if request["prepare"] and prepare is not None:
                prepare(bundle, target=target, facts=facts)
            if request["validate"]:
                validate = getattr(provider, "validate_package", None)
                if not callable(validate):
                    raise ValueError("optional-device fallback requires independent provider package validation")
                if validate(bundle, target=target, package=package, facts=facts) is not None:
                    raise ValueError("provider package validation must return None or raise")
            if request["route"] is not None:
                routing, reason = bind_source_route(request["route"], reason, bundle, target, package, provider, facts)
            if routing is None and request["require_device"]:
                raise ValueError("submitted backend did not route this capsule: " + reason)
            build = spike_model.build(
                bundle,
                build_dir,
                inputs_npz=bundle / "inputs.npz",
                arena_mb=request["arena_mb"],
                dump_all_outputs=True,
                backend=request["backend"],
                device=routing,
                **request["layout"],
            )
        # A bound route reports its bindings (possibly none, which the grader refuses); no route reports None.
        routed = (load_region_offload(build_dir) or load_sidecar(build_dir)).get("routed") or [] if routing else None
        keys = ("elf", "mem_bytes", "build_hash", "vlen")
        _write(out, {"build": {k: build.get(k) for k in keys}, "routed": routed, "reason": reason, "error": None})
    except Exception as exc:  # noqa: BLE001 -- a refusal of the submitted build is a verdict input
        _write(out, {"error": f"{type(exc).__name__}: {exc}"[:2000]})
    return 0


def _covering(paths, *, excluded) -> list[Path]:
    """The given existing paths, minus any containing an ``excluded`` path or inside another one."""
    roots: list[Path] = []
    candidates = {Path(p).resolve() for p in paths if p and Path(p).exists()}
    for path in sorted(candidates, key=lambda p: len(p.parts)):
        if any(path == e or e.is_relative_to(path) for e in excluded):
            continue
        if not any(path.is_relative_to(root) for root in roots):
            roots.append(path)
    return roots


def build_in_sandbox(
    request: dict, *, bundle: Path, work: Path, package: Path, provider: Path, facts: Path, denied, timeout
) -> dict:
    """Parent: copy the public program inputs into ``work``, run :func:`main` sandboxed, return its report.

    ``denied`` are paths the sandbox must not expose (the private runtime bundle, the capsule corpus with
    its answers); the command is refused if any of them is reachable. ``bundle`` is never mounted.
    """
    from merlin.llvmlower import toolchain
    from merlin.runtime.backends.spike import gcc_path
    from merlin.targetgen.plugins import resolve_support
    from merlin.targetgen.sandbox import bwrap as BW
    from merlin.targetgen.sandbox.answer_surfaces import AnswerSurface

    if work.exists():
        shutil.rmtree(work)
    inputs, contract = work / "in", work / "contract"
    for directory in (inputs, contract, work / "tmp", work / "out"):
        directory.mkdir(parents=True)
    for name in PROGRAM_INPUTS:
        if (bundle / name).is_file():
            shutil.copyfile(bundle / name, inputs / name)
    if (bundle / ADMISSION).is_file():  # only the selected numeric policy, never its references
        selection = json.loads((bundle / ADMISSION).read_text(encoding="utf-8"))["selection"]
        _write(inputs / ADMISSION, {"selection": {"spec": selection["spec"]}})
    shutil.copyfile(facts, work / "facts.json")
    environment = {"MERLIN_RTL_FACTS": str(work / "facts.json")}
    if os.environ.get("MERLIN_TARGET_CONTRACT"):
        shutil.copyfile(os.environ["MERLIN_TARGET_CONTRACT"], contract / "target_contract.yaml")
        environment["MERLIN_TARGET_CONTRACT"] = str(contract / "target_contract.yaml")
    request = {
        **request,
        "bundle": str(inputs),
        "build": str(work / "out" / "build"),
        "package": str(package),
        "provider": str(provider),
        "facts": str(work / "facts.json"),
        "result": str(work / "out" / "result.json"),
    }
    _write(work / "request.json", request)
    compiler = os.environ.get("MERLIN_COMPILER_PYTHON")
    excluded = [Path(p).resolve() for p in (work, bundle, *denied)]
    required = [Path(package).resolve(), Path(resolve_support(request["target"]).base).resolve()]
    if any(e == r or e.is_relative_to(r) for r in required for e in excluded):
        raise RuntimeError("full-call build sandbox would expose private grading inputs")
    roots = _covering(
        [
            sys.prefix,
            sys.base_prefix,
            *(entry for entry in sys.path if entry and Path(entry).is_dir()),
            resolve_support(request["target"]).base,
            package,
            toolchain.llvm_install(),
            gcc_path().resolve().parents[1],
            Path(compiler).parents[1] if compiler else None,
            os.environ.get("MERLIN_REPO_ROOT"),
        ],
        excluded=excluded,
    )
    argv = BW.base_argv(work, {}, include_claude_home=False, inherit_environment=False)
    for root in roots:
        argv += ["--ro-bind", str(root), str(root)]
    environment.update(
        MERLIN_RISCV_GCC=str(gcc_path()),  # the bound cross compiler, not a toolchain root left outside
        HOME=str(work),
        TMPDIR=str(work / "tmp"),
        MERLIN_OUT_ROOT=str(work / "merlin-out"),
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONPATH=os.environ.get("PYTHONPATH", ""),
    )
    # Every other MERLIN_* selection passes only when the path it names is visible in the sandbox.
    for key, value in os.environ.items():
        if key.startswith("MERLIN_") and key not in environment:
            if not value.startswith("/") or BW.is_exposed(argv, Path(value)):
                environment[key] = value
    environment["PATH"] = os.pathsep.join(
        p for p in os.environ.get("PATH", "/usr/bin:/bin").split(os.pathsep) if p and BW.is_exposed(argv, Path(p))
    )
    for key, value in sorted(environment.items()):
        argv += ["--setenv", key, value]
    surfaces = [
        AnswerSurface("full-call private input", Path(p).resolve(), "dir" if Path(p).is_dir() else "file", "private")
        for p in (bundle, *denied)
    ]
    if BW.coverage_gap(argv, surfaces):
        raise RuntimeError("full-call build sandbox would expose private grading inputs")
    command = [*argv, sys.executable, "-m", "merlin.targetgen.full_call_build", str(work / "request.json")]
    done = subprocess.run(command, capture_output=True, text=True, timeout=timeout, stdin=subprocess.DEVNULL)
    result = work / "out" / "result.json"
    if not result.is_file() or result.is_symlink():
        raise RuntimeError(f"isolated full-call build produced no report: {(done.stderr or done.stdout)[-1500:]}")
    return json.loads(result.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
