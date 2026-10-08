"""A newly compiled native image must never execute a cached predecessor."""

from __future__ import annotations

import ctypes
import hashlib
import shutil
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from merlin.llvmlower.abi import HostModel, PrivateHostImagePolicy


def _compile(source: Path, output: Path, value: int, nargs: int = 1) -> None:
    source.write_text(
        "typedef struct { void *allocated; long *aligned; long offset; } Desc;\n"
        f"void _mlir_ciface_forward({', '.join(f'void *p{i}' for i in range(nargs))}) "
        f"{{ Desc *d=p0; d->aligned[d->offset]={value}; }}\n"
    )
    subprocess.run(["cc", "-fPIC", "-shared", str(source), "-o", str(output)], check=True, capture_output=True)


@pytest.mark.parametrize("nargs", [None, 1, 1025])
def test_replaced_source_path_runs_each_loaded_image(tmp_path, nargs):
    if shutil.which("cc") is None:
        pytest.skip("native C compiler unavailable")
    path = tmp_path / "model.so"
    source = tmp_path / "model.c"
    _compile(source, path, 17, nargs or 1)
    first_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    policy = PrivateHostImagePolicy(tmp_path.resolve())
    first = HostModel.load(str(path), n_args=nargs, image_policy=policy)
    successor = tmp_path / "successor.so"
    _compile(source, successor, 29, nargs or 1)
    successor.replace(path)
    second_digest = hashlib.sha256(path.read_bytes()).hexdigest()
    second = HostModel.load(str(path), n_args=nargs, image_policy=policy)
    assert (first.image_sha256, second.image_sha256) == (first_digest, second_digest)
    output = ctypes.c_long()
    args = [(ctypes.addressof(output), [])] * (nargs or 1)
    for model, expected in ((first, 17), (second, 29), (first, 17), (second, 29)):
        model(args)
        assert output.value == expected
    assert not list(tmp_path.glob(".merlin-host-image-*"))


def test_relative_dependency_lookup_survives_image_loading(tmp_path):
    if shutil.which("cc") is None:
        pytest.skip("native C compiler unavailable")
    helper = tmp_path / "helper.c"
    helper.write_text("long selected_value(void) { return 43; }\n")
    subprocess.run(["cc", "-fPIC", "-shared", str(helper), "-o", str(tmp_path / "libselected.so")], check=True)
    source = tmp_path / "relative.c"
    source.write_text(
        "typedef struct { void *allocated; long *aligned; long offset; } Desc;\n"
        "extern long selected_value(void);\n"
        "void _mlir_ciface_forward(void *p) { Desc *d=p; d->aligned[d->offset]=selected_value(); }\n"
    )
    path = tmp_path / "relative.so"
    subprocess.run(
        [
            "cc",
            "-fPIC",
            "-shared",
            str(source),
            "-L" + str(tmp_path),
            "-lselected",
            "-Wl,-rpath,$ORIGIN",
            "-o",
            str(path),
        ],
        check=True,
    )
    model = HostModel.load(str(path), image_policy=PrivateHostImagePolicy(tmp_path.resolve()))
    output = ctypes.c_long()
    model([(ctypes.addressof(output), [])])
    assert output.value == 43
    assert not list(tmp_path.glob(".merlin-host-image-*"))


@pytest.mark.parametrize("nargs", [None, 1, 1025])
def test_default_load_preserves_readonly_artifact_compatibility(tmp_path, nargs):
    if shutil.which("cc") is None:
        pytest.skip("native C compiler unavailable")
    source = tmp_path / "readonly.c"
    path = tmp_path / "readonly.so"
    _compile(source, path, 37, nargs or 1)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    path.chmod(0o400)
    tmp_path.chmod(0o500)
    try:
        model = HostModel.load(str(path), n_args=nargs)
        output = ctypes.c_long()
        model([(ctypes.addressof(output), [])] * (nargs or 1))
        assert output.value == 37 and model.image_sha256 is None
        assert hashlib.sha256(path.read_bytes()).hexdigest() == before
        assert not list(tmp_path.glob(".merlin-host-image-*"))
    finally:
        tmp_path.chmod(0o700)


def test_policy_cannot_select_an_unrelated_staging_directory(tmp_path):
    directory = tmp_path / "owned"
    directory.mkdir()
    with pytest.raises(ValueError, match="selected build directory"):
        HostModel.load(str(tmp_path / "model.so"), image_policy=PrivateHostImagePolicy(directory))
    with pytest.raises(ValueError, match="exact absolute"):
        PrivateHostImagePolicy(Path("relative"))


def test_private_policy_refuses_an_aliased_source_parent(tmp_path):
    directory = tmp_path / "owned"
    directory.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(directory, target_is_directory=True)
    path = directory / "model.so"
    _compile(directory / "model.c", path, 47)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="selected build directory"):
        HostModel.load(str(alias / path.name), image_policy=PrivateHostImagePolicy(directory))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not list(directory.glob(".merlin-host-image-*"))


@pytest.mark.parametrize("failure", ["invalid-image", "missing-entry"])
def test_private_stage_cleanup_on_load_failure(tmp_path, failure):
    path = tmp_path / "invalid.so"
    if failure == "invalid-image":
        path.write_bytes(b"not an ELF image")
    else:
        source = tmp_path / "missing.c"
        source.write_text("long different_entry(void) { return 5; }\n")
        subprocess.run(["cc", "-fPIC", "-shared", str(source), "-o", str(path)], check=True)
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises((OSError, ValueError)):
        HostModel.load(str(path), image_policy=PrivateHostImagePolicy(tmp_path.resolve()))
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before
    assert not list(tmp_path.glob(".merlin-host-image-*"))


@pytest.mark.parametrize("directory_kind", ["existing", "missing", "alias"])
def test_fresh_compile_host_owner_selects_policy_after_successful_build(tmp_path, monkeypatch, directory_kind):
    from xdsl.dialects.builtin import ModuleOp

    from merlin.llvmlower import kernel_backend, lower

    canonical = tmp_path / "owned"
    if directory_kind != "missing":
        canonical.mkdir()
    selected = canonical
    if directory_kind == "alias":
        selected = tmp_path / "alias"
        selected.symlink_to(canonical, target_is_directory=True)
    calls = []

    def compile_native_fixture(_source, directory, *, targets):
        assert targets == ("host",)
        assert directory == canonical
        directory.mkdir(parents=True, exist_ok=True)
        value = (17, 29)[len(calls)]
        path = directory / "model_host.so"
        _compile(directory / "model.c", path, value)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        calls.append((path, digest))
        return SimpleNamespace(host_so=path)

    # The real owner compiles and loads two native images. This substitutes
    # only upstream lowering, so it proves owner composition, not MLIR lowering.
    monkeypatch.setattr(lower, "lower_model", compile_native_fixture)
    first = kernel_backend.compile_host(ModuleOp([]), selected)
    second = kernel_backend.compile_host(ModuleOp([]), selected)
    assert (first.image_sha256, second.image_sha256) == (calls[0][1], calls[1][1])
    output = ctypes.c_long()
    for model, value in ((first, 17), (second, 29), (first, 17)):
        model([(ctypes.addressof(output), [])])
        assert output.value == value
    assert hashlib.sha256(calls[-1][0].read_bytes()).hexdigest() == calls[-1][1]
    assert not list(canonical.glob(".merlin-host-image-*"))


def test_failed_compile_does_not_attempt_private_load(tmp_path, monkeypatch):
    from xdsl.dialects.builtin import ModuleOp

    from merlin.llvmlower import kernel_backend, lower

    def refused_compile(*_args, **_kwargs):
        raise RuntimeError("upstream compilation failed")

    def unexpected_load(*_args, **_kwargs):
        pytest.fail("failed compilation must not attempt image loading")

    monkeypatch.setattr(lower, "lower_model", refused_compile)
    monkeypatch.setattr(HostModel, "load", unexpected_load)
    with pytest.raises(RuntimeError, match="upstream compilation failed"):
        kernel_backend.compile_host(ModuleOp([]), tmp_path)
