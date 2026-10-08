"""FP32 Gemmini-owned source catalog and Spike extension selection for Core ATen."""

from pathlib import Path

TARGET = "gemmini_fp32"


def routing(directory, *, target, package, facts, eligible):
    from merlin.targetgen.core_aten_device import submitted_catalog_routing

    # The shared adapter scopes catalog imports to the canonical submission root,
    # keeping repeated graded snapshots isolated in a long-lived worker.
    return submitted_catalog_routing(directory, target=target, package=package, facts=facts, eligible=eligible)


def runner_options():
    from merlin.runtime.backends.spike import spike_path

    binary = Path(spike_path()).resolve()
    # Extension and tool search roots are properties of the selected tool installation.
    import os

    from merlin.common.paths import build_dir

    prefix = binary.parent.parent
    library = Path(os.environ.get("MERLIN_FP32_SPIKE_EXTLIB") or build_dir() / "fp32-gemmini" / "lib" / "libgemmini.so")
    if not library.is_file():
        raise ValueError(f"isolated FP32 Spike extension is absent: {library}")
    return {
        "spike_binary": binary,
        "extension": "gemmini",
        "extlib": library.resolve(),
        "path_prepend": [prefix.parent / "bin"],
    }


def full_call_oracle(target):
    """Use submitted source-bound device catalogs at the full-call L2 boundary."""
    import os

    from merlin.common.paths import artifacts_dir
    from merlin.targetgen.core_aten_device import selected_facts
    from merlin.targetgen.rtl.facts import ensure_facts
    from merlin.targetgen.spike_full_call import full_call_adapter
    from merlin.targetgen.target_experiment import load_capability_manifest

    if target != TARGET:
        raise ValueError("execution provider names a different target")
    contract = load_capability_manifest(target).contract
    selected = (
        os.environ.get("MERLIN_RTL_FACTS") or artifacts_dir() / "targets" / target / contract["runner"]["rtl_facts"]
    )
    facts_path = ensure_facts(target, explicit=selected)
    with selected_facts(target, facts_path):
        pass
    isa = contract["isa"]["march"]
    return full_call_adapter(
        target,
        isa=isa,
        runner_options=runner_options(),
        execution_provider=Path(__file__),
        rtl_facts=facts_path,
    )
