"""SpecRefSource: a capsule sourced from a specir verification spec. The spec supplies the PROGRAM (its own
command buffer), the bit-exact golden its refmodel computes over deterministic operands, and the declared
coverage. Program emission exists for the RoCC/command-buffer families (e.g. gemmini); a gen without an
emitter fails closed (never a faked program). The spec capture is skipped when specir is absent.

Target-agnostic: the gen is a parameter (the ``spec_ref`` a profile entry declares), never a code literal
in library scope — this test file is a legitimate edge that names gens as data under test."""

from __future__ import annotations

import sys
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

from merlin.targetgen import capsule_source as CSrc

_SPEC = CSrc.SpecRefSource()
_needs_spec = pytest.mark.skipif(not _SPEC.available(), reason="specir unavailable; set SPECIR_ROOT")


def test_parse_spec_ref():
    assert CSrc._parse_spec_ref("gemmini:op.matmul") == ("gemmini", "op.matmul")
    with pytest.raises(ValueError):
        CSrc._parse_spec_ref("nocolon")


@_needs_spec
def test_spec_capture_gemmini_program_and_golden():
    """The gemmini spec yields a command-buffer program + a bit-exact golden that equals an independent
    int matmul over the SAME deterministic operands (self-consistent, no live oracle needed)."""
    art = _SPEC.capture("gemmini:op.matmul", workload=(16, 16, 16), tile_dim=16)
    assert art.gen == "gemmini" and art.command_buffer["target"] == "gemmini"
    assert set(art.operands) == {"lhs", "weight"} and "out" in art.golden and art.compare == "exact_int"
    assert art.instructions and art.opcode_backing  # RoCC issue sequence + spec-RoCC backing present
    A0 = np.array(art.operands["lhs"], dtype=np.int64)
    W = np.array(art.operands["weight"], dtype=np.int64)
    assert np.array_equal(np.array(art.golden["out"], dtype=np.int64), A0 @ W)


@_needs_spec
def test_spec_capture_fails_closed_unknown_gen():
    with pytest.raises(CSrc.SpecProgramUnavailable):
        _SPEC.capture("not_a_gen:op.matmul")


@_needs_spec
@pytest.mark.parametrize(
    "spec_ref,workload,td,contraction",
    [
        ("atlas-npu:op.matmul_mxu0", (32, 32, 32), 32, True),  # fp8 MXU command sequence
        ("radiance:op.matmul", (16, 16, 16), 16, True),  # SIMT warp schedule
    ],
)
def test_spec_capture_float_families(spec_ref, workload, td, contraction):
    """Atlas (MXU) and radiance (SIMT warp) programs: decoded role-keyed operands + a float golden that
    equals a matmul over those operands (self-consistent, no live oracle needed)."""
    recipe = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / ("atlas/phase0/regression-seeds.yaml" if spec_ref.startswith("atlas-") else "radiance/phase0/recipe.yaml")
    )
    entries = yaml.safe_load(recipe.read_text(encoding="utf-8"))["capsules"]
    selected = next(entry["spec_program_emitter"] for entry in entries if entry.get("spec_ref") == spec_ref)
    art = _SPEC.capture(spec_ref, workload=workload, tile_dim=td, program_emitter=selected)
    assert art.compare == "tolerance_float"
    assert art.program_emitter == selected
    assert set(art.operands) == {"lhs", "weight"} and "out" in art.golden
    A = np.array(art.operands["lhs"], dtype=np.float64)
    W = np.array(art.operands["weight"], dtype=np.float64)
    G = np.array(art.golden["out"], dtype=np.float64)
    rel = np.abs(G - A @ W).max() / (np.abs(A @ W).max() + 1e-9)
    assert rel < 0.25  # golden reproduces the spec matmul over the decoded operands


@_needs_spec
def test_spec_capture_fails_closed_no_program():
    """A gen no emitter authors a matmul program for (a vector unit) fails closed, never a faked program."""
    with pytest.raises(CSrc.SpecProgramUnavailable):
        _SPEC.capture("saturn:op.matmul")


def test_explicit_spec_program_emitter_and_capsule_provenance(tmp_path, monkeypatch):
    """Synthetic SpecIR program: selection is explicit, scoped and recorded in both capsule records."""
    from merlin.integrations import specir as scoped_specir
    from merlin.targetgen import corpus_spec

    def module(name, **members):
        obj = ModuleType(name)
        obj.__dict__.update(members)
        monkeypatch.setitem(sys.modules, name, obj)
        return obj

    module("specir", __path__=[])
    module("specir.interface", __path__=[])
    module("specir.oracle", __path__=[])
    module("specir.gate", load_targets=lambda root: [{"id": "synthetic", "spec": "synthetic.spec"}])
    module("specir.registry", _SPEC_ROOT=tmp_path)
    node = SimpleNamespace(name="spec.op")
    module("specir.loading", parse_spec_file=lambda path: [node])
    module("specir.graph", all_nodes=lambda mod: mod, name_of=lambda n: "op.matmul", attrs_of=lambda n: {})
    module("specir.interface.emit_capsule", emit_command_buffer=lambda *a, **kw: (None, {}))
    module("specir.interface.rocc_lower", RoccLoweringError=RuntimeError, lower_buffer=lambda *a, **kw: None)
    module("specir.oracle.dtypes", float_format=lambda dtype: dtype, decode_float=lambda bits, fmt: float(bits))

    program = {
        "target": "synthetic",
        "tensors": {
            "A": {"role": "lhs", "dtype": "f32", "bits": [[1.0]]},
            "W": {"role": "weight", "dtype": "f32", "bits": [[2.0]]},
        },
        "golden": {"out": {"values": [[2.0]]}},
        "commands": [{"opcode": "synthetic.matmul"}],
    }
    selected = {"module": "specir.interface.synthetic_program", "function": "emit_synthetic_program"}
    module(selected["module"], emit_synthetic_program=lambda *a, **kw: (program, {}))
    monkeypatch.setattr(scoped_specir, "importable", lambda root: nullcontext())
    monkeypatch.setattr(CSrc.SpecRefSource, "available", lambda self: True)
    monkeypatch.setattr(corpus_spec, "build", lambda entry, binding: ({"expected": {}}, "module {}"))

    source = CSrc.SpecRefSource(root=str(tmp_path))
    with pytest.raises(CSrc.SpecProgramUnavailable, match="emitter"):
        source.capture("synthetic:op.matmul")
    with pytest.raises(CSrc.SpecProgramUnavailable, match="emitter"):
        source.capture("synthetic:op.matmul", program_emitter={"module": "os", "function": "system"})
    with pytest.raises(CSrc.SpecProgramUnavailable, match="module"):
        source.capture(
            "synthetic:op.matmul",
            program_emitter={"module": "specir.interface.absent", "function": "emit_absent"},
        )

    # The selector-free route remains a generic RoCC class, not a name-based target exception.
    sys.modules["specir.interface.emit_capsule"].emit_command_buffer = lambda *a, **kw: (
        {"tensors": {"A": {"role": "lhs"}, "W": {"role": "weight"}}},
        {"opcode_backed_by_spec_rocc": {"synthetic.matmul": ["COMPUTE"]}},
    )
    sys.modules["specir.interface.rocc_lower"].lower_buffer = lambda *a, **kw: SimpleNamespace(
        operands={"A": [[1]], "W": [[2]]}, golden={"Y0": [[2]]}, instructions=["COMPUTE"]
    )
    rocc = source.capture("synthetic:op.matmul")
    assert rocc.compare == "exact_int"
    assert rocc.program_emitter == {"module": "specir.interface.emit_capsule", "function": "emit_command_buffer"}

    entry = {
        "name": "synthetic_spec",
        "cat": "isa",
        "op": "matmul",
        "spec_ref": "synthetic:op.matmul",
        "spec_program_emitter": selected,
    }
    out = CSrc.write_spec_capsule(entry, SimpleNamespace(tile_dim=1), tmp_path / "corpus", source=source)
    golden = yaml.safe_load((out / "golden.yaml").read_text(encoding="utf-8"))
    capsule = yaml.safe_load((out / "capsule.yaml").read_text(encoding="utf-8"))
    assert golden["oracle_provenance"]["program_emitter"] == selected
    assert capsule["spec_program_emitter"] == selected

    # A requested composite is legal only when the recipe names an actually declared leaf
    # and the selected emitter's output confirms that exact composition.
    sys.modules["specir.graph"].name_of = lambda node: "op.warp_fma"
    program["composed_from"] = "op.warp_fma"
    program["kernel"] = [{"op": "op.warp_fma"}]
    selected_composite = {**selected, "composed_from": "op.warp_fma"}
    sys.modules[selected["module"]].emit_synthetic_program = lambda *a, **kw: (
        program,
        {"composed_from": "op.warp_fma"},
    )
    composed = source.capture("synthetic:op.matmul", program_emitter=selected_composite)
    assert composed.program_emitter == selected_composite
    assert composed.command_buffer["composed_from"] == "op.warp_fma"
    program["composed_from"] = "op.other"
    with pytest.raises(CSrc.SpecProgramUnavailable, match="composed_from"):
        source.capture("synthetic:op.matmul", program_emitter=selected_composite)
