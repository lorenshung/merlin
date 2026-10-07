"""Actual bare-metal build observations and failed invocation isolation."""

from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from merlin.llvmlower import toolchain
from merlin.llvmlower.compilation_recipe import FILENAME, CompilationRecipe
from merlin.runtime.backends import spike, spike_model


def _read(work):
    return json.loads((work / FILENAME).read_text())


def test_failed_compiler_keeps_old_object_unqualified(tmp_path):
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("native compiler unavailable")
    source, obj = tmp_path / "source.c", tmp_path / "source.o"
    source.write_text("int value(void) { return 7; }\n")
    recipe = CompilationRecipe(tmp_path, producer=Path(spike_model.__file__))
    cmd = [compiler, "-c", source, "-o", obj]

    def run(argv):
        return subprocess.run(argv, capture_output=True)

    recipe.run(cmd, runner=run, inputs=[source], output=obj)
    old_bytes = obj.read_bytes()
    source.write_text("not valid C\n")
    with pytest.raises(RuntimeError, match="compilation command returned"):
        recipe.run(cmd, runner=run, inputs=[source], output=obj)
    record = _read(tmp_path)
    assert record["status"] == "invoked"
    assert record["commands"][-1]["status"] == "invoked"
    assert "output" not in record["commands"][-1]
    assert obj.read_bytes() == old_bytes
    assert "executable" not in record
    with pytest.raises(ValueError, match="complete successful command sequence"):
        recipe.completed(obj)


def test_normal_model_compile_runtime_and_link_are_observed(tmp_path, monkeypatch):
    if not spike.available() or not toolchain.m2m_python().is_file() or not toolchain.clang().is_file():
        pytest.skip("bare-metal and upstream toolchain unavailable")
    monkeypatch.setenv("MERLIN_TEST_API_TOKEN", "private-compiler-canary")
    bundle, work = tmp_path / "bundle", tmp_path / "build"
    bundle.mkdir()
    (bundle / "model.mlir").write_text("""module {
      func.func @forward(%arg0: tensor<5xf32>) -> tensor<5xf32> {
        %init = tensor.empty() : tensor<5xf32>
        %out = linalg.generic {
          indexing_maps = [affine_map<(i) -> (i)>, affine_map<(i) -> (i)>],
          iterator_types = ["parallel"]
        } ins(%arg0 : tensor<5xf32>) outs(%init : tensor<5xf32>) {
          ^bb0(%value: f32, %unused: f32):
            %two = arith.constant 2.0 : f32
            %sum = arith.addf %value, %two : f32
            linalg.yield %sum : f32
        } -> tensor<5xf32>
        return %out : tensor<5xf32>
      }
    }""")
    (bundle / "weights.safetensors.manifest.json").write_text('{"0":{"kind":"input","name":"values"}}')
    values = np.array([-5, -0.75, 0, 1.25, 7], dtype=np.float32)
    np.savez(bundle / "inputs.npz", in0=values)
    built = spike_model.build(
        bundle,
        work,
        arena_mb=1,
        cflags_override=["-march=rv64gc", "-mabi=lp64d", "-mcmodel=medany", "-O2", "-ffreestanding", "-fno-builtin"],
        host_math_policy="expf_via_double",
    )
    record = _read(work)
    assert record["status"] == "completed"
    assert record["executable"]["sha256"] == hashlib.sha256(built["elf"].read_bytes()).hexdigest()
    assert "private-compiler-canary" not in json.dumps(record)
    commands = record["commands"]
    assert all(command["status"] == "returned" for command in commands)
    by_output = {Path(command["output"]["path"]).name: command for command in commands}
    assert {"model.o", "mlir_rt.o", "host_math.o", "model_main.o", "weights_blob.o", "model.elf"} <= by_output.keys()
    assert by_output["mlir_rt.o"]["executable"] == by_output["model.o"]["executable"]
    assert by_output["mlir_rt.o"]["executable"] != by_output["model_main.o"]["executable"]
    assert "-march=rv64gc" in by_output["mlir_rt.o"]["argv"]
    assert by_output["weights_blob.o"]["cwd"] == str((work / "cgen").resolve())
    assert "-Wl,--wrap=expf" in by_output["model.elf"]["argv"]
    link_inputs = by_output["model.elf"]["inputs"]
    assert [Path(item["path"]).name for item in link_inputs] == [
        "model_link.ld",
        "model_call.o",
        "merlin_model.o",
        "model_main.o",
        "mlir_rt.o",
        "crt.o",
        "console.o",
        "libc_min.o",
        "malloc.o",
        "model.o",
        "weights_blob.o",
        "host_math.o",
        "libm.a",
    ]
    for command in commands:
        for identity in [command["executable"], *command["inputs"], command["output"]]:
            assert identity["sha256"] == hashlib.sha256(Path(identity["path"]).read_bytes()).hexdigest()
    # The recorded ordinary link is itself sufficient to reproduce these exact bytes.
    subprocess.run(commands[-1]["argv"], cwd=commands[-1]["cwd"], check=True, capture_output=True)
    assert hashlib.sha256(built["elf"].read_bytes()).hexdigest() == record["executable"]["sha256"]
    ran = spike_model.run(built["elf"], mem_bytes=built["mem_bytes"], isa="rv64gc", timeout=60)
    np.testing.assert_array_equal(ran["outputs"], values + np.float32(2))

    with pytest.raises(ValueError, match="output_dump_cap"):
        spike_model.build(bundle, work, output_dump_cap=0)
    assert not (work / FILENAME).exists()
