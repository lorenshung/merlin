"""Descriptor folding retains exact capture authority after its query disappears."""

from types import SimpleNamespace

import pytest

from merlin.targetgen import core_aten_capture as capture


@pytest.mark.parametrize(
    "proof,expected",
    [
        (
            {
                "authority": "static_capture_input_descriptors",
                "queries": [{"operator": "aten.sym_numel.default", "value": 6}],
            },
            "captured_exact",
        ),
        (
            {"authority": "unknown", "queries": [{"operator": "aten.sym_numel.default", "value": 6}]},
            "captured_lost_exact_provenance",
        ),
        ({}, "captured_lost_exact_provenance"),
    ],
)
def test_constant_capture_requires_descriptor_specialization_proof(tmp_path, monkeypatch, proof, expected):
    artifact = SimpleNamespace(
        linalg_mlir="builtin.module {}",
        inputs={},
        golden={},
        meta={"static_metadata_specialization": proof},
    )
    monkeypatch.setattr(
        capture,
        "PytorchRefSource",
        lambda **kwargs: SimpleNamespace(capture_loader=lambda *args, **kwargs: artifact, m2m_dir=tmp_path),
    )
    monkeypatch.setattr(capture, "_git_revision", lambda path: None)
    corpus = dict(
        pytorch_version="test",
        cases=[
            dict(
                overload="aten.sym_numel.default",
                arguments={"args": []},
                expected=6,
            )
        ],
    )
    result = capture.capture_case_corpus(corpus, tmp_path)
    assert result["results"]["aten.sym_numel.default"]["status"] == expected
