"""Authored admission extensions must not imply operator approval."""

from merlin.common.paths import repo_root
from merlin.common.yaml import load_yaml


def test_authored_family_and_rank_extensions_record_pending_review():
    root = repo_root() / "examples/gemmini/target"
    software = load_yaml(root / "software-spec.yaml")["operations"]
    for name in ("movement_bitwise", "component_linear", "component_bilinear", "host_quantize_exact", "host_dequantize_per_tensor"):
        assert software[name]["evidence"]["review_status"] == "authored, pending operator review"
    host = {row["id"]: row for row in load_yaml(root / "host-capabilities.yaml")["operations"]}
    for name in ("host_exact_native_boundary", "host_quantize_exact", "host_dequantize_per_tensor"):
        assert host[name]["numerical_contract"]["evidence"]["review_status"] == "authored, pending operator review"
