"""A group capsule is written by the Phase 0 writer, with the interface a model build would request."""

from __future__ import annotations

from types import SimpleNamespace

import yaml
from merlin_experiments.phase0.group_forms import write_group_capsule

from merlin.targetgen import group_capsule_entries as G

_TE = SimpleNamespace(target="synthetic", sim_via="sim")
_CONTRACT = {
    "name": "synthetic",
    "compute_units": [{"name": "array", "kind": "systolic", "dtypes": ["int8"]}],
    "capabilities": {"mesh": {"rows": 4, "cols": 4}},
    "encoding": {
        "semantic_class": {"a": "MVIN", "b": "COMPUTE", "c": "MVOUT"},
        "corpus_issue_order": ["MVIN", "COMPUTE", "MVOUT"],
    },
}
_FACTS = {"facts": {"datapaths": [{"name": "input", "dtype": "i8"}, {"name": "accumulator", "dtype": "i32"}]}}


def _binding():
    return G.group_binding(
        _TE, {"required_oracle_tiers": ["L0"], "compare": "exact_int"}, contract=_CONTRACT, facts=_FACTS, taxonomy={}
    )


def test_written_group_capsule_matches_the_requested_interface(tmp_path):
    entry = {
        "name": "G_m4k8n4_raw",
        "cat": "layers",
        "op": "matmul",
        "M": 4,
        "K": 8,
        "N": 4,
        "epilogue": [],
        "operand_dtype": "int8",
    }
    binding = _binding()
    _capsule, requested = G.interface_capsule(entry, binding)
    written = write_group_capsule(entry, binding, tmp_path)
    assert written == tmp_path / "layers" / "G_m4k8n4_raw"
    assert (written / "capsule.interface.mlir").read_text(encoding="utf-8") == requested
    capsule = yaml.safe_load((written / "capsule.yaml").read_text(encoding="utf-8"))
    assert capsule["source_role"] == G.SOURCE_ROLE
    assert capsule["expected"]["instruction_classes"] == ["MVIN", "COMPUTE", "MVOUT"]
    golden = yaml.safe_load((written / "golden.yaml").read_text(encoding="utf-8"))
    assert golden["golden_source"] == "merlin_tensor_int" and golden["outputs"]
