"""Drivers shared by the native component admission tests (merlin/tests/targetgen)."""

import json

import numpy as np

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.targetgen.core_aten_components import prepare_integer_components


class RefusingProvider:
    """A provider that fails the test if admission asks it for a device proof."""

    def admit_numeric_schedule(self, *args, **kwargs):
        raise AssertionError("a refused source must not request a device proof")


class DecliningProvider:
    """A provider that records each schedule request and grants no device implementation."""

    def __init__(self):
        self.called = []

    def admit_numeric_schedule(self, source, **kwargs):
        self.called.append(source)
        return dict(status="declined", reason="test grants no device implementation")


def eager_record(abi, output_bytes, post_input_bytes, post_indices):
    """A bundled case carrying an untransformed native CPU eager boundary (hex words)."""
    return dict(
        status="bundled",
        input_abi=abi,
        output_indices=[0],
        semantic_boundary=dict(post_indices=post_indices),
        native_eager_boundary=dict(
            authority="untransformed_native_cpu_eager", output_bytes=output_bytes, post_input_bytes=post_input_bytes
        ),
    )


def prepare_components(tmp_path, module, cases, provider=None):
    return prepare_integer_components(
        tmp_path,
        dict(cases=cases),
        dict(spec={}, host={}),
        provider or RefusingProvider(),
        None,
        module,
        limb_attribute="test.limb",
    )


def assert_host_refusal(tmp_path, source, inputs, cases, code, provider=None):
    """Admission keeps ``source`` (module or MLIR text) on the host with refusal ``code``.

    Writes model.mlir and inputs.npz (``inputs`` become in0, in1, ...), checks that admission
    returns None, leaves model.mlir byte-identical and records ``code``; returns the refusal.
    """
    text = source if isinstance(source, str) else str(source) + "\n"
    module = parse_mlir_text(source) if isinstance(source, str) else source
    (tmp_path / "model.mlir").write_text(text)
    np.savez(tmp_path / "inputs.npz", **{f"in{i}": value for i, value in enumerate(inputs)})
    assert prepare_components(tmp_path, module, cases, provider) is None
    assert (tmp_path / "model.mlir").read_text() == text
    refusal = json.loads((tmp_path / "exact-dtype-refusal.json").read_text())
    assert refusal["code"] == code
    return refusal
