"""One binding for a captured group: the accumulator is stated by the target, never assumed."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from merlin.targetgen import corpus_spec as CS
from merlin.targetgen import group_capsule_entries as G
from merlin.xdsl_dialects.lowering import group_command as GC

_TE = SimpleNamespace(target="synthetic", sim_via="sim")
_DATAPATH = {"required_oracle_tiers": ["L0"], "compare": "exact_int"}


def _contract(**unit) -> dict:
    return {
        "name": "synthetic",
        "compute_units": [{"name": "array", "kind": "systolic", "dtypes": ["int8"], **unit}],
        "capabilities": {"mesh": {"rows": 4, "cols": 4}},
        "encoding": {
            "semantic_class": {"a": "MVIN", "b": "COMPUTE", "c": "MVOUT"},
            "corpus_issue_order": ["MVIN", "COMPUTE", "MVOUT"],
        },
    }


def _facts(input_dtype: str = "i8", accumulator: str = "i32") -> dict:
    return {
        "facts": {"datapaths": [{"name": "input", "dtype": input_dtype}, {"name": "accumulator", "dtype": accumulator}]}
    }


def test_accumulator_comes_from_the_contract_before_the_facts():
    stated = _contract(accumulate=[{"in": "int8", "weight": "int8", "acc": "i32"}])
    assert CS.derived_accumulator(stated, "int8", _facts(accumulator="int32")) == "i32"


def test_accumulator_comes_from_the_rtl_datapath_of_the_same_format():
    assert CS.derived_accumulator(_contract(), "int8", _facts()) == "i32"
    # A facts pair about another element format says nothing about this operand.
    assert CS.derived_accumulator(_contract(), "fp8_e4m3", _facts()) is None
    assert CS.derived_accumulator(_contract(), "int8", None) is None
    assert CS.derived_accumulator(_contract(), "int8", {"facts": {"datapaths": []}}) is None


def test_group_binding_refuses_an_underived_accumulator():
    with pytest.raises(G.UnderivedAccumulator, match="assumed accumulator"):
        G.group_binding(_TE, _DATAPATH, contract=_contract(), facts={"facts": {}}, taxonomy={})


def test_group_binding_is_the_corpus_binding():
    facts = _facts()
    binding = G.group_binding(_TE, _DATAPATH, contract=_contract(), facts=facts, taxonomy={})
    corpus = CS.derive_binding(_TE, dict(_DATAPATH), contract=_contract(), facts=facts, taxonomy={})
    assert binding.accum_dtype == corpus.accum_dtype == "i32"
    assert (binding.tile_dim, binding.operand_dtype, binding.tiers) == (
        corpus.tile_dim,
        corpus.operand_dtype,
        corpus.tiers,
    )
    # A declared accumulator is the recipe's own statement and is honoured as such.
    declared = G.group_binding(_TE, {**_DATAPATH, "accum_dtype": "i32"}, contract=_contract(), facts={}, taxonomy={})
    assert declared.accum_dtype == "i32"


def test_interface_capsule_is_the_builder_output_under_the_routed_binding():
    binding = G.group_binding(_TE, _DATAPATH, contract=_contract(), facts=_facts(), taxonomy={})
    entry = {"name": "G_t", "op": "matmul", "M": 4, "K": 8, "N": 4, "epilogue": [], "operand_dtype": "int8"}
    capsule, iface = G.interface_capsule(entry, binding)
    stated = G.interface_entry(entry)
    assert (capsule, iface) == CS.build(stated, CS.entry_binding(stated, binding)[1])
    assert capsule["source_role"] == G.SOURCE_ROLE and capsule["kind"] == "op"
    assert "!merlin_iface.acc<i32>" in iface
    assert capsule["expected"]["instruction_classes"] == ["MVIN", "COMPUTE", "MVOUT"]
    with pytest.raises(ValueError, match="needs a name"):
        G.interface_entry({"op": "matmul"})


def test_device_output_shape_is_positions_by_features():
    conv = {
        "op": "conv2d",
        "N": 64,
        "Himg": 56,
        "Wimg": 56,
        "kh": 3,
        "kw": 3,
        "stride": [1, 1],
        "padding": [1, 1, 1, 1],
    }
    assert GC.device_output_shape(conv) == [3136, 64]
    stem = {**conv, "Himg": 224, "Wimg": 224, "kh": 7, "kw": 7, "stride": [2, 2], "padding": [3, 3, 3, 3]}
    assert GC.device_output_shape(stem) == [112 * 112, 64]
    pooled = {**stem, "epilogue": ["maxpool"], "pool_size": [3, 3], "pool_stride": [2, 2], "pool_padding": [1, 1, 1, 1]}
    assert GC.device_output_shape(pooled) == [56 * 56, 64]
    assert GC.device_output_shape({"op": "matmul", "M": 49, "K": 2048, "N": 1}) == [49, 1]
    with pytest.raises(GC.NoDeviceShape):
        GC.device_output_shape({**pooled, "pool_size": None})
    with pytest.raises(GC.NoDeviceShape):
        GC.device_output_shape({"op": "matmul", "K": 4})
