"""Whole-model routing must enumerate the contractions in the parsed graph, not only its tags."""

from __future__ import annotations

import pytest

from merlin.common.paths import merlin_dir
from merlin.targetgen.capsule_source import ModelDemandIncomplete, model_op_demands_checked

pytestmark = pytest.mark.target("gemmini")


def test_real_model_demands_match_the_structural_inventory_and_refuse_a_lost_tag():
    root = merlin_dir() / "contract/capsules/model"
    expected = {"SY_model_resnet50": 54, "SY_model_smolvla": 391}
    for name, count in expected.items():
        path = root / name / "capsule.interface.mlir"
        text = path.read_text(encoding="utf-8")
        demands = model_op_demands_checked(text, "int8")
        assert sum(d.family == "contraction" for d in demands) == count, name

        if name == "SY_model_resnet50":
            lines = text.splitlines(keepends=True)
            index = next(i for i, line in enumerate(lines) if "linalg.matmul" in line and 'prov.op = "' in line)
            before, _, after = lines[index].partition('prov.op = "')
            tag, _, tail = after.partition('"')
            lines[index] = before + f'prov.unattributed = "{tag}"' + tail
            with pytest.raises(ModelDemandIncomplete, match="untagged=.*linalg.matmul"):
                model_op_demands_checked("".join(lines), "int8")


def test_generic_mlir_serialization_keeps_the_same_contraction_demands():
    """A valid generic-form capture must not erase its contraction carriers."""
    from merlin.common import mlir_query
    from merlin.targetgen.capsule_source import model_op_demands
    from merlin.xdsl_dialects._common import text as module_text

    path = merlin_dir() / "contract/capsules/model/M2_microvit_gemmini/capsule.interface.mlir"
    original = path.read_text(encoding="utf-8")
    generic = module_text(mlir_query.parse(path), generic=True)
    expected = [d for d in model_op_demands(original, "int8") if d.family == "contraction"]
    observed = [d for d in model_op_demands_checked(generic, "int8") if d.family == "contraction"]
    assert [(d.op, d.batch, d.m, d.k, d.n, d.elem_fmt, d.region_id) for d in observed] == [
        (d.op, d.batch, d.m, d.k, d.n, d.elem_fmt, d.region_id) for d in expected
    ]
    assert all(d.region_id for d in observed)

    from merlin.targetgen.coverage_certificate import denominator_completeness

    complete = denominator_completeness(generic, demands=observed)
    assert complete["inventory_status"] == "verified_against_parsed_contractions"
    assert complete["n_unmatched_contractions"] == 0


def test_compile_routing_refuses_a_missing_contraction_tag_before_target_lookup():
    from merlin.compile_cli import _route_before_build

    path = merlin_dir() / "contract/capsules/model/M2_microvit_gemmini/capsule.interface.mlir"
    text = path.read_text(encoding="utf-8")
    text = text.replace('prov.op = "matmul"', 'prov.unattributed = "matmul"', 1)
    with pytest.raises(ModelDemandIncomplete, match="untagged=.*linalg.matmul"):
        _route_before_build("no_target_lookup", text, datapath="int8")
