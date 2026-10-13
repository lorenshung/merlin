"""Portable logical payloads cannot authorize unrecorded raw statistic reads."""

import copy
import importlib.util
import itertools
import json
import sys
from argparse import Namespace
from types import SimpleNamespace

import pytest

from merlin.common.paths import repo_root
from merlin.targetgen.capture_refusals import InputStorageUnavailable, M2MUnavailable, NumericCaptureRefused
from merlin.targetgen.core_aten_capture import case_capture_name, loader_source
from merlin.targetgen.core_aten_input_storage import require_native_input_storage

OVERLOAD = "aten.native_layer_norm_backward.default"


def tensor(shape, stride, offset=0):
    def values(dims):
        return [values(dims[1:]) for _ in range(dims[0])] if dims else 1.0

    return dict(
        kind="tensor",
        dtype="float32",
        shape=shape,
        stride=stride,
        storage_offset=offset,
        requires_grad=False,
        values=values(shape),
    )


def case(shape, stride, *, rows=2, offset=0):
    x = tensor([rows, 3], [3, 1])
    stats = tensor(shape, stride, offset)
    return dict(
        overload=OVERLOAD,
        case_id="independent-storage-case",
        rng_seed=0,
        arguments=dict(
            args=dict(kind="tuple", items=[x, x, [3], stats, copy.deepcopy(stats), None, None, [True] * 3]), kwargs={}
        ),
        expected=dict(kind="tuple", items=[]),
    )


@pytest.mark.parametrize("shape,stride", [([2], [1]), ([2, 2], [1, 2]), ([2, 2], [4, 1]), ([], [])])
@pytest.mark.parametrize("offset", [0, 5])
def test_dense_or_partially_read_storage_is_covered(shape, stride, offset):
    document = case(shape, stride, rows=1 if not shape else 2, offset=offset)
    before = copy.deepcopy(document)
    compile(loader_source(document), "<covered input storage>", "exec")
    assert document == before


@pytest.mark.parametrize("stride,cause", [([0], "raw_read_out_of_bounds"), ([2], "unrecorded_storage_hole")])
def test_unavailable_statistics_are_structural_typed_refusals(tmp_path, stride, cause):
    from merlin.targetgen.capture_refusals import read_refusal, write_exception_refusal

    document = case([2], stride, offset=4)
    # Refusal happens from source arguments before any output document is read.
    del document["expected"]
    with pytest.raises(InputStorageUnavailable) as caught:
        loader_source(document)
    refusal = caught.value
    assert isinstance(refusal, M2MUnavailable) and not isinstance(refusal, NumericCaptureRefused)
    assert refusal.code == "input_storage_unavailable"
    assert refusal.details["row_count"] == 2
    assert [entry["argument"] for entry in refusal.details["unavailable"]] == ["mean", "rstd"]
    assert all(
        entry["cause"] == cause and entry["first_unrecorded_address"] == 5 for entry in refusal.details["unavailable"]
    )
    write_exception_refusal(tmp_path, refusal, fallback_code="unexpected")
    assert read_refusal(tmp_path / "refusal.json") is None  # Never a numeric host-fallback grant.


def test_keyword_and_tuple_normalized_arguments_preserve_coverage():
    document = case([2], [1])
    args = document["arguments"]["args"]["items"]
    document["arguments"] = dict(
        args=dict(kind="tuple", items=[args[0]]),
        kwargs=dict(input=args[1], normalized_shape=dict(kind="tuple", items=[3]), mean=args[3], rstd=args[4]),
    )
    require_native_input_storage(OVERLOAD, document["arguments"])


def test_device_metadata_cannot_bypass_the_actual_cpu_loader():
    document = case([2], [0])
    document["arguments"]["args"]["items"][1]["device"] = "cuda"
    with pytest.raises(InputStorageUnavailable):
        loader_source(document)
    require_native_input_storage("aten.alias.default", {})


def test_coverage_matches_independent_address_enumeration():
    # Enumeration is deliberately independent of the production interval proof.
    for shape in ([], [0], [1], [2], [3], [2, 2], [2, 3], [1, 2, 2]):
        for stride in itertools.product(range(4), repeat=len(shape)):
            addresses = {
                sum(i * s for i, s in zip(index, stride)) for index in itertools.product(*(range(n) for n in shape))
            }
            for rows in range(6):
                document = case(shape, list(stride), rows=rows)
                if set(range(rows)) <= addresses:
                    require_native_input_storage(OVERLOAD, document["arguments"])
                else:
                    with pytest.raises(InputStorageUnavailable):
                        require_native_input_storage(OVERLOAD, document["arguments"])


def test_native_batch_provenance_binds_input_qualification_sources(tmp_path):
    from merlin.targetgen.core_aten_provenance import batch_provenance

    receipt = batch_provenance(runner_options={"spike_binary": str(tmp_path / "absent")})
    for filename in ("core_aten_capture.py", "core_aten_input_storage.py", "capture_refusals.py"):
        assert str(repo_root() / "src/merlin/targetgen" / filename) in receipt["sources"]


@pytest.mark.parametrize("runner", [False, True])
@pytest.mark.parametrize("stride", [[0], [2]])
def test_each_capture_path_records_refusal_and_processes_the_next_case(tmp_path, monkeypatch, runner, stride):
    from merlin.targetgen import capsule_source, core_aten_capture, core_aten_cover

    calls = []

    class Source:
        m2m_dir = tmp_path

        def __init__(self, **kwargs):
            pass

        def capture_loader(self, loader, *args, **kwargs):
            calls.append(loader)
            return SimpleNamespace(linalg_mlir="builtin.module {}", meta={}, inputs=[], golden=[])

    monkeypatch.setattr(capsule_source, "PytorchRefSource", Source)
    monkeypatch.setattr(core_aten_capture, "PytorchRefSource", Source)
    monkeypatch.setattr(core_aten_capture, "_git_revision", lambda _: None)
    for module in (core_aten_capture, core_aten_cover):
        monkeypatch.setattr(module, "_provenance_ops", lambda _: {"aten.alias.default"})
    bad = case([2], stride)
    good = dict(
        overload="aten.alias.default",
        case_id="next-case",
        rng_seed=0,
        arguments=dict(args=dict(kind="tuple", items=[tensor([2], [1])]), kwargs={}),
        expected=dict(kind="tensor"),
    )
    if runner:
        spec = importlib.util.spec_from_file_location(
            "storage_capture_runner", repo_root() / "build_tools/scripts/run_bounded_core_aten_suite.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        args = Namespace(
            output=tmp_path,
            m2m_dir=tmp_path,
            m2m_python=sys.executable,
            capture_timeout=1,
            label="track",
            numeric_policy="fp32",
        )
        monkeypatch.setenv("PYTHONPATH", "unchanged test environment")
        for document in (bad, good):
            module.capture(args, document, {})
        root = tmp_path / "captures"
        module.execute(args, bad, {})
        observed = json.loads((tmp_path / "track/shards" / case_capture_name(bad) / "result.json").read_text())[
            "observation"
        ]
        assert observed["status"] == "capture_unavailable"
        assert observed["evidence"]["refusal_code"] == "input_storage_unavailable"
    else:
        result = core_aten_capture.capture_case_corpus(dict(cases=[bad, good], pytorch_version="fixture"), tmp_path)
        assert result["case_count"] == 2 and result["status_counts"] == {"capture_failed": 1, "captured_exact": 1}
        root = tmp_path
    failed = json.loads((root / case_capture_name(bad) / "capture.json").read_text())
    assert failed["status"] == "capture_failed" and failed["refusal_code"] == "input_storage_unavailable"
    assert failed["refusal_details"]["row_count"] == 2
    assert not (root / case_capture_name(bad) / "capsule.pytorch.py").exists()
    assert json.loads((root / case_capture_name(good) / "capture.json").read_text())["status"] == "captured_exact"
    assert calls == [root / case_capture_name(good) / "capsule.pytorch.py"]
