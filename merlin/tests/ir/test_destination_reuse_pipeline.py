from merlin.llvmlower.impr_features import _reuse_tensor_destination


def test_destination_forwarding_is_after_fusion_and_before_bufferization():
    source = [
        "canonicalize",
        "func.func(linalg-fuse-elementwise-ops)",
        "one-shot-bufferize{bufferize-function-boundaries}",
        "cse",
    ]
    result = _reuse_tensor_destination(source)
    assert result == source[:2] + ["eliminate-empty-tensors"] + source[2:]
    assert _reuse_tensor_destination(result) == result
    assert "eliminate-empty-tensors" not in source


def test_destination_forwarding_does_not_change_other_pipeline_stages():
    source = ["canonicalize", "convert-scf-to-cf"]
    assert _reuse_tensor_destination(source) == source
