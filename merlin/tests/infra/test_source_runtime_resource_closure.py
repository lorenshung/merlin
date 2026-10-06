"""Installed source-certificate emission retains its complete C dependency tree."""

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from merlin.common.paths import repo_root
from merlin.llvmlower import source_attention_frontier as source


@pytest.mark.parametrize("prepared", [False, True])
def test_declared_installed_runtime_emits_and_compiles(tmp_path, monkeypatch, prepared):
    compiler = shutil.which("clang") or shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler required")
    root = repo_root()
    files = json.loads((root / "build_tools/package_resources.json").read_text())["files"]
    bundled = tmp_path / "installed_data"
    for name in files:
        if name.startswith("merlin/runtime/c/"):
            destination = bundled / Path(name).relative_to("merlin")
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(root / name, destination)
    monkeypatch.setattr(source, "data_path", lambda *parts: bundled.joinpath(*parts))
    plan = source.SourceAttentionFrontierPlan(
        1,
        2,
        4,
        8,
        3,
        2,
        0.125,
        -87.3365478515625,
        1.4426950216293335,
        (-0.079204238951206207, -0.22433836758136749, 0.30354261398315430, 0.00010703434963943437),
        8388608.0,
        1065353216.0,
        127.0,
        float.fromhex("0x1.5p-17"),
        -127,
        127,
    )
    options = (
        dict(
            prepare_endpoint_rows=True,
            word_interval_enclosure=True,
            prepare_product_domain=True,
            prepare_required_norms=True,
            retain_certified_rows=True,
            separable_source_radius=True,
            prepare_softmax_domain=True,
            integer_reconstruction=True,
            prepare_probability_bins=True,
            prepare_encoded_rows=True,
            prepare_softmax_spans=True,
            polynomial_batch_four=True,
        )
        if prepared
        else {}
    )
    generated = tmp_path / "provider.c"
    generated.write_text(source.emit_source_attention_frontier(plan, symbol="test", **options))
    subprocess.run(
        [compiler, "-std=c11", "-fsyntax-only", "-I", str(bundled / "runtime/c"), str(generated)],
        check=True,
        capture_output=True,
        text=True,
    )
