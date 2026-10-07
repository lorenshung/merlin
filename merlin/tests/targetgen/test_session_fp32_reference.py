"""Joined staged-precision, recipe and session reference ownership checks."""

import json
import os
import subprocess
from pathlib import Path

import numpy as np
import pytest
from test_fp32_capture_staging import LOADER_OWNED_CALIBRATION, MULTI_LOADER, _static_recipe

from merlin.targetgen import _m2m_capture_worker as worker


@pytest.mark.slow
@pytest.mark.parametrize("multi", [False, True])
def test_staged_quantized_session_keeps_independent_fp32_trajectory(tmp_path, multi):
    python = os.environ.get("MERLIN_M2M_PYTHON")
    root = os.environ.get("MERLIN_M2M_DIR")
    if not python or not root:
        pytest.skip("an explicit trace-capable capture interpreter is required")
    if multi:
        text = MULTI_LOADER.replace(
            'ExternalRuntimeProgram("second", core, (context,), 1)',
            """ExternalRuntimeProgram("second", core, (context,), 2,
                dict(kind="generic_recurrent", paper_ready=True, steps=2,
                     stages=["second"], streams=[],
                     states=[dict(name="state", input_index=0, output_index=0)],
                     quality=dict(output_index=0, reference="eager_fp32")))""",
        )
        bundle = tmp_path / "capture" / "stages" / "second"
        text = text.replace("name=p.name, steps=1,", "name=p.name, steps=p.steps,")
        expected = np.asarray([[[1.0, 2.0, 3.0, 4.0]]] * 2, dtype=np.float32)
    else:
        text = (
            LOADER_OWNED_CALIBRATION
            + """
def get_session_spec(model, inputs):
    return dict(kind="image_sequence", paper_ready=True, steps=2,
                stages=["forward"], states=[],
                streams=[dict(name="images", input_index=0, values=model.calibration_images)],
                quality=dict(output_index=0, reference="eager_fp32"))
"""
        )
        bundle = tmp_path / "capture"
        expected = np.asarray([[[1.0, 2.0]], [[3.0, 4.0]]], dtype=np.float32)
    loader = tmp_path / "loader.py"
    loader.write_text(text)
    recipe = tmp_path / "recipe.json"
    recipe.write_text(json.dumps(_static_recipe()))
    result = subprocess.run(
        [
            python,
            str(Path(worker.__file__)),
            "--m2m-dir",
            root,
            "--loader",
            str(loader),
            "--dtype",
            "int8",
            "--recipe",
            str(recipe),
            "--stage-fp32",
            "--materialize-bundle",
            "--out",
            str(tmp_path / "capture"),
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=180,
        env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"},
    )
    assert result.returncode == 0, result.stderr[-6000:]
    import yaml

    contract = yaml.safe_load((bundle / "session_contract.yaml").read_text())
    quality, correctness = contract["quality"], contract["correctness"]
    assert quality["reference"] == "eager_fp32"
    with np.load(bundle / quality["golden"]) as values:
        trajectory = values[quality["key"]]
    np.testing.assert_array_equal(trajectory, expected)
    with np.load(bundle / correctness["golden"]) as values:
        compiled_reference = values[correctness["key"]]
    assert not np.array_equal(trajectory, compiled_reference)
    assert quality["reference_sha256"] != correctness["reference_sha256"]
