"""Late host legalization preserves original artifacts and build input identity."""

import hashlib

import pytest

from merlin.runtime.backends.spike_model import SpikeModelError, _transform_host_ir


def test_unselected_stage_preserves_default_path_without_io(tmp_path):
    source = tmp_path / "not_created.ll"
    work = tmp_path / "not_created"
    assert _transform_host_ir(source, work, None) == (source, None)
    assert not work.exists()


def test_explicit_legalization_retains_both_inputs(tmp_path):
    source = tmp_path / "source.ll"
    source.write_text("define i32 @value() { ret i32 1 }\n")
    before = source.read_bytes()

    def transform(original, work):
        result = work / "selected.ll"
        result.write_bytes(original.read_bytes().replace(b"i32 1", b"i32 2"))
        return result

    selected, receipt = _transform_host_ir(source, tmp_path / "stage", transform)
    assert source.read_bytes() == before
    assert b"i32 2" in selected.read_bytes()
    assert receipt["source_sha256"] == hashlib.sha256(before).hexdigest()
    assert receipt["selected_sha256"] == hashlib.sha256(selected.read_bytes()).hexdigest()
    assert receipt["source_sha256"] != receipt["selected_sha256"]


def test_source_mutation_refuses(tmp_path):
    source = tmp_path / "source.ll"
    source.write_text("original")

    def transform(original, work):
        original.write_text("changed")
        return original

    with pytest.raises(SpikeModelError, match="modified its source"):
        _transform_host_ir(source, tmp_path / "stage", transform)


def test_unchanged_source_can_be_explicitly_selected(tmp_path):
    source = tmp_path / "source.ll"
    source.write_text("original")
    selected, receipt = _transform_host_ir(source, tmp_path / "stage", lambda original, _: original)
    assert selected == source
    assert receipt["source_sha256"] == receipt["selected_sha256"]


@pytest.mark.parametrize("external", [False, True])
def test_missing_or_unretained_output_refuses(tmp_path, external):
    source = tmp_path / "source.ll"
    source.write_text("original")
    unretained = tmp_path / "unretained.ll"
    unretained.write_text("different")

    def transform(_, work):
        return unretained if external else work / "missing.ll"

    with pytest.raises(SpikeModelError, match="workdir|no LLVM"):
        _transform_host_ir(source, tmp_path / "stage", transform)
