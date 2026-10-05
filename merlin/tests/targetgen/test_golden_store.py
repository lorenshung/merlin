"""Goldens are a YAML document plus a digest-bound npz of their arrays; every reader sees the same values."""

from __future__ import annotations

import copy
import zipfile

import numpy as np
import pytest
import yaml

from merlin.targetgen import golden_store as GS


def _golden() -> dict:
    hexes = [f"0x{i % 256:02x}" for i in range(4096)]
    decoded = [float(i % 17) / 4.0 for i in range(4096)]
    return {
        "golden_source": "specir_refmodel_float",
        "oracle_provenance": {
            "engine": "independent reference",
            "grade_policy": {"compare": "tolerance_float", "atol": 0.25, "rtol": 0.02},
            "inputs": {
                "A0": {"shape": [64, 64], "fp8_raw_hex": hexes, "decoded": decoded},
                "B": {"shape": [4], "decoded": [0.5, -1.0, 2.0, 0.0]},
            },
        },
        "outputs": {"Y0": [[1.5, -2.25], [0.0, 3.0]], "Y1": [[7]]},
        "integer_partial_sum_bound": {"status": "proven_safe", "bound": [1, 2]},
    }


def test_round_trip_is_value_identical_and_moves_only_arrays(tmp_path):
    original = _golden()
    GS.write_golden(tmp_path, copy.deepcopy(original))
    assert GS.load_golden(tmp_path) == original
    document = GS.read_document(tmp_path)
    provenance = document["oracle_provenance"]["inputs"]["A0"]
    # Big operand provenance and every output live in the archive; metadata and small lists stay inline.
    assert provenance["shape"] == [64, 64]
    assert provenance["fp8_raw_hex"][GS.REFERENCE] == GS.ARRAYS and provenance["decoded"]["shape"] == [4096]
    assert document["oracle_provenance"]["inputs"]["B"]["decoded"] == [0.5, -1.0, 2.0, 0.0]
    assert document["outputs"]["Y0"][GS.REFERENCE] == GS.ARRAYS and document["outputs"]["Y1"]["dtype"] == "<i8"
    assert document["integer_partial_sum_bound"]["bound"] == [1, 2]
    assert isinstance(GS.load_golden(tmp_path)["outputs"]["Y1"][0][0], int)


def test_the_archive_is_deterministic(tmp_path):
    first, second = tmp_path / "a", tmp_path / "b"
    first.mkdir(), second.mkdir()
    GS.write_golden(first, _golden())
    GS.write_golden(second, _golden())
    for name in GS.FILES:
        assert (first / name).read_bytes() == (second / name).read_bytes()


def test_an_altered_or_missing_archive_is_refused(tmp_path):
    GS.write_golden(tmp_path, _golden())
    archive = tmp_path / GS.ARRAYS
    with zipfile.ZipFile(archive) as zf:
        members = {name: zf.read(name) for name in zf.namelist()}
    with np.load(archive) as data:
        swapped = {key: data[key] for key in data.files}
    swapped["a0000"] = swapped["a0000"].copy()
    swapped["a0000"][0] = "0xff"
    np.savez(archive, **swapped)
    with pytest.raises(GS.GoldenStoreError, match="digest"):
        GS.load_golden(tmp_path)
    archive.unlink()
    with pytest.raises(GS.GoldenStoreError, match="absent"):
        GS.load_golden(tmp_path)
    assert members  # the archive really held the arrays before it was altered


def test_an_inline_golden_still_loads_and_a_rewrite_drops_a_stale_archive(tmp_path):
    inline = {"golden_source": "hand", "outputs": {"Y": [[1, 2], [3, 4]]}}
    (tmp_path / GS.DOCUMENT).write_text(yaml.safe_dump(inline))
    assert GS.load_golden(tmp_path) == inline
    GS.write_golden(tmp_path, _golden())
    assert (tmp_path / GS.ARRAYS).is_file()
    GS.write_golden(tmp_path, {"golden_source": "hand"})
    assert not (tmp_path / GS.ARRAYS).exists()
    assert GS.load_golden(tmp_path / "absent") is None


def test_update_keeps_the_arrays(tmp_path):
    GS.write_golden(tmp_path, _golden())
    GS.update_golden(tmp_path, qualification="checked")
    loaded = GS.load_golden(tmp_path)
    assert loaded["qualification"] == "checked" and loaded["outputs"] == _golden()["outputs"]


def test_the_grader_and_input_readers_resolve_archived_arrays(tmp_path):
    from merlin.targetgen import capsule_golden as CGOLD
    from merlin.targetgen import capsule_inputs as CI
    from merlin.targetgen import golden_provenance as GP

    GS.write_golden(tmp_path, _golden())
    assert CGOLD._load_golden_yaml(tmp_path) == _golden()
    assert CI._input_provenance(tmp_path)["A0"]["decoded"] == _golden()["oracle_provenance"]["inputs"]["A0"]["decoded"]
    assert GP._declared_source(tmp_path) == "specir_refmodel_float"


def test_both_golden_files_are_answer_surfaces():
    from merlin_experiments.phase1.workspace_transport import _is_answer_file

    from merlin.targetgen.contract import materialize

    for name in GS.FILES:
        assert _is_answer_file(__import__("pathlib").Path(name))
        assert name in materialize._CAPSULE_FILES
