"""Gemmini-owned source catalog and Spike extension selection for Core ATen."""

from pathlib import Path

from merlin.llvmlower import toolchain
from merlin.llvmlower.device_build import DeviceRouting
from merlin.targetgen.plugins import load_module

TARGET = "gemmini"


def prepare_bundle(directory, *, target, facts):
    from merlin.llvmlower.modular_contractions import prepare_bundle as prepare
    from merlin.system.offload import device_dtype_triples

    return prepare(Path(directory), device_dtype_triples(target))


def routing(directory, *, target, package, facts, eligible):
    from merlin.system.offload import device_dtype_triples

    triples = {tuple(shape.dtypes) for _, shape in eligible}
    if len(triples) != 1 or not triples.issubset(set(device_dtype_triples(target))):
        raise ValueError("ambiguous or underivable device datapath")
    operand, weight, accum = next(iter(triples))
    if operand != weight:
        raise ValueError("catalog requires identical operand storage types")
    backend = load_module(package, "mlir_oot.golden_device_catalog", package_name="core_aten_backend")
    # The provider owns semantic matching. Unsupported operations remain host work.
    source = (Path(directory) / "model.mlir").read_text()
    _, inventory = backend.build_catalog(source)
    if not inventory["covered_contractions"]:
        return None
    return DeviceRouting(
        device=target,
        package_dir=package,
        operand_dtype=operand,
        accum_dtype=accum,
        catalog_builder=backend.merlin_builder(toolchain.llvm_install() / "bin"),
    )


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
