"""Gemmini-owned source catalog and Spike extension selection for Core ATen."""

from pathlib import Path

TARGET = "gemmini"


def prepare_bundle(directory, *, target, facts):
    from merlin.llvmlower.modular_contractions import prepare_bundle as prepare
    from merlin.system.offload import device_dtype_triples

    return prepare(Path(directory), device_dtype_triples(target))


def routing(directory, *, target, package, facts, eligible):
    from merlin.targetgen.core_aten_device import submitted_catalog_routing

    return submitted_catalog_routing(directory, target=target, package=package, facts=facts, eligible=eligible)


def runner_options():
    from merlin.runtime.backends.spike import spike_path

    binary = Path(spike_path()).resolve()
    # Extension and tool search roots are properties of the selected tool installation.
    prefix = binary.parent.parent
    return {
        "spike_binary": binary,
        "extension": "gemmini",
        "extlib": prefix / "lib" / "libgemmini.so",
        "path_prepend": [prefix.parent / "bin"],
    }


def full_call_oracle(target):
    """Use submitted source-bound device catalogs at the full-call L2 boundary."""
    from merlin.targetgen.rtl.facts import ensure_facts
    from merlin.targetgen.spike_full_call import full_call_adapter
    from merlin.targetgen.target_experiment import load_capability_manifest

    isa = load_capability_manifest(target).contract["isa"]["march"]
    return full_call_adapter(
        target,
        isa=isa,
        runner_options=runner_options(),
        execution_provider=Path(__file__),
        rtl_facts=ensure_facts(target),
    )
