"""Do not infer native reduced-precision NaN policy from IEEE negf."""

import json

import numpy as np
from mlir_fixtures import pointwise

from merlin.targetgen.core_aten_components import prepare_integer_components


def test_bf16_nan_negation_needs_explicit_source_policy(tmp_path):
    module = pointwise(["tensor<1xbf16>"], "tensor<1xbf16>", "%v=arith.negf %a : bf16")
    original = str(module) + "\n"
    (tmp_path / "model.mlir").write_text(original)
    np.savez(tmp_path / "inputs.npz", in0=np.array([float("nan")], np.float32))

    class Provider:
        def admit_numeric_schedule(self, *args, **kwargs):
            raise AssertionError("undefined native NaN bits cannot request a device proof")

    assert (
        prepare_integer_components(
            tmp_path,
            dict(cases=[dict(input_abi=[dict(dtype="bf16")])]),
            dict(spec={}, host={}),
            Provider(),
            None,
            module,
            limb_attribute="test.limb",
        )
        is None
    )
    assert (tmp_path / "model.mlir").read_text() == original
    assert json.loads((tmp_path / "exact-dtype-refusal.json").read_text())["code"] == (
        "native_cpu_word_layout_unproved"
    )
