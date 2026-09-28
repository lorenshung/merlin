"""A renderer's exact bytes survive the target-neutral harness sidecar seam."""

from __future__ import annotations

import subprocess
import shutil

import pytest

from merlin.targetgen.contract.harness_blobs import stage_harness_blobs


def test_blob_sidecar_assembles_to_exact_aligned_bytes(tmp_path):
    if not shutil.which("cc") or not shutil.which("objcopy"):
        pytest.skip("requires a host C toolchain")
    payload = bytes(range(32))
    (source,) = stage_harness_blobs(tmp_path, {"T_W": {"bytes": payload, "align": 16, "elems": 32}})
    assert (tmp_path / "harness_blob_T_W.bin").read_bytes() == payload
    assert '.incbin "harness_blob_T_W.bin"' in source.read_text()
    result = subprocess.run(["cc", "-c", source.name, "-o", "blob.o"], cwd=tmp_path,
                            capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    objcopy = subprocess.run(["objcopy", "--dump-section", ".rodata=blob.raw", "blob.o"],
                             cwd=tmp_path, capture_output=True, text=True)
    assert objcopy.returncode == 0, objcopy.stderr
    assert (tmp_path / "blob.raw").read_bytes() == payload


@pytest.mark.parametrize("declaration", [
    {"T/escape": {"bytes": b"x", "align": 1, "elems": 1}},
    {"T_W": {"bytes": b"x", "align": 3, "elems": 1}},
    {"T_W": {"bytes": b"x", "align": 1, "elems": 2}},
    {"T_W": {"bytes": b"x", "align": 1, "elems": 0}},
])
def test_invalid_blob_declarations_fail_closed(tmp_path, declaration):
    with pytest.raises(ValueError):
        stage_harness_blobs(tmp_path, declaration)


def test_generic_linker_passes_renderer_blobs_to_the_executable(tmp_path, monkeypatch):
    if not shutil.which("cc"):
        pytest.skip("requires a host C compiler")
    from merlin.runtime.backends import base
    from merlin.targetgen.contract import compile as compiler
    from merlin.targetgen import runtime_build

    class Recipe:
        load_address = 0
        link_script = tmp_path / "unused.ld"
        support_sources = ()
        error_cls = RuntimeError

        @staticmethod
        def compile_command(*, source, output):
            return ["cc", "-c", str(source), "-o", str(output)]

        @staticmethod
        def link_command(*, objects, output, link_script):
            return ["cc", *(str(item) for item in objects), "-o", str(output)]

    def render(_cb, *, target, blobs):
        assert target == "synthetic"
        blobs["T_W"] = {"bytes": b"\x07\x08\x09\x0a", "align": 4, "elems": 4}
        return "extern const unsigned char T_W[4]; int main(void) { return T_W[0] != 7 || T_W[3] != 10; }\n"

    monkeypatch.setattr(base, "harness_build_recipe", lambda target: Recipe())
    monkeypatch.setattr(base, "harness_renderer", lambda target: render)
    monkeypatch.setattr(runtime_build, "derived_link_script", lambda *args: tmp_path / "unused.ld")
    kernel = tmp_path / "kernel.o"
    subprocess.run(["cc", "-c", "-x", "c", "-", "-o", str(kernel)], input="\n",
                   text=True, check=True)
    elf = compiler.link_elf({}, kernel, tmp_path, target="synthetic")
    assert subprocess.run([str(elf)], check=False).returncode == 0
    assert (tmp_path / "harness_blob_T_W.bin").read_bytes() == b"\x07\x08\x09\x0a"
