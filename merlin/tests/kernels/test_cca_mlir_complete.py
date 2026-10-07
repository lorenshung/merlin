"""CCA artifacts must retain the facts that route whole-program optimization."""

import dataclasses

import pytest

from merlin.kernels import cca, cca_mlir


def complete_record():
    return cca.CCA(
        op="program",
        backend=["array", "host"],
        scope="program",
        compute=cca.ComputeFacet(
            op="contraction", register_block=(8, "runtime_width"), accumulator_dtype="f32", mr_adapts_to_m=False
        ),
        vector=cca.VectorFacet(sew=32, lmul=1.0, tail="tu"),
        memory=cca.MemoryFacet(
            capacity_fit=True,
            onchip_bytes_required=4096,
            banks_used=2,
            spill_reason="none",
            dma_pattern="burst",
            onchip_resident="both",
        ),
        dispatch=cca.DispatchFacet(
            n_dispatches=521, config_fraction=0.125, loop_offloaded=False, double_buffered_banks=2, dma_overlap=True
        ),
        layout=cca.LayoutFacet(transpose_materialized=False, operand_major="k_major"),
        envelope=cca.EnvelopeFacet(
            calls_in_loop=0, runtime_calls=("copy", "helper"), work_ins_per_mac=4.25, overhead_ins_per_output=1.5
        ),
        spatial=cca.SpatialFacet(pe_rows=7, pe_cols=11, dataflow="stationary", accumulator_resident=True),
        simt=cca.SimtFacet(warps=2, threads_per_warp=32, smem_resident=False, barriers_in_loop=0),
        coverage=cca.CoverageFacet(
            claimed_mac_fraction=0.75, unclaimed_op_classes=("norm", "activation"), non_contraction_op_fraction=0.25
        ),
        communication=cca.CommunicationFacet(
            host_device_bytes=1536, intermediate_materialized=False, resident_across_calls=True, fences=3
        ),
        provenance={
            "source": "compiler",
            "level": "schedule",
            "confidence": "derived",
            "evidence": {"sha256": "a" * 64, "count": 0},
        },
    )


def test_complete_program_record_survives_artifact_boundary():
    original = complete_record()
    restored = cca_mlir.from_mlir(cca_mlir.to_mlir(original))
    assert dataclasses.asdict(restored) == dataclasses.asdict(original)
    assert not restored.scope_problems()


@pytest.mark.parametrize("name", list(cca_mlir._facet_classes()))
def test_all_actual_facet_field_sets_are_serialized(name):
    original = complete_record()
    restored = cca_mlir.from_mlir(cca_mlir.to_mlir(original))
    assert getattr(restored, name) == getattr(original, name)


def test_unknown_is_distinct_from_zero_false_and_empty():
    original = cca.CCA(
        op="eltwise",
        backend=[],
        compute=cca.ComputeFacet(op=None),
        dispatch=cca.DispatchFacet(n_dispatches=0, loop_offloaded=False, dma_overlap=None),
        envelope=cca.EnvelopeFacet(calls_in_loop=0, runtime_calls=()),
        provenance={"reason": None},
    )
    assert cca_mlir.from_mlir(cca_mlir.to_mlir(original)) == original


def test_legacy_typed_view_disagreement_refuses():
    text = cca_mlir.to_mlir(complete_record())
    text = text.replace('accumulator_dtype = "f32"', 'accumulator_dtype = "f64"', 1)
    with pytest.raises(ValueError, match="view disagrees"):
        cca_mlir.from_mlir(text)


@pytest.mark.parametrize("mutation", ["unknown_kind", "missing_field", "bad_type", "duplicate"])
def test_malformed_complete_facet_refuses(mutation):
    from xdsl.dialects.builtin import DictionaryAttr, StringAttr
    from xdsl.parser import Parser

    from merlin.xdsl_dialects import cca as D
    from merlin.xdsl_dialects._common import make_context, text

    module = Parser(make_context(D.get_dialect()), cca_mlir.to_mlir(complete_record())).parse_module()
    kernel = next(iter(module.body.block.ops))
    facet = next(op for op in kernel.body.block.ops if isinstance(op, D.FacetOp) and op.facet.data == "dispatch")
    if mutation == "unknown_kind":
        facet.properties["facet"] = StringAttr("unrecognized")
    elif mutation == "duplicate":
        kernel.body.block.add_op(facet.clone())
    else:
        values = dict(facet.values.data)
        if mutation == "missing_field":
            del values["loop_offloaded"]
        else:
            values["n_dispatches"] = StringAttr("true")
        facet.properties["values"] = DictionaryAttr(values)
    with pytest.raises(ValueError):
        cca_mlir.from_mlir(text(module))


def test_legacy_subset_artifact_remains_readable():
    text = """builtin.module {
      "cca.kernel"() <{op = "matmul", backend = "array", source = "old", level = "asm"}> ({
        "cca.compute"() <{accumulator_dtype = "i32", register_block_mr = "4"}> : () -> ()
      }) : () -> ()
    }"""
    restored = cca_mlir.from_mlir(text)
    assert restored.compute.register_block == (4, None)
    assert restored.provenance == {"source": "old", "level": "asm"}
    assert restored.dispatch is None and restored.scope == "kernel"
