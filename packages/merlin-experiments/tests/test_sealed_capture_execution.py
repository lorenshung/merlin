"""Process-level proof of the static-only capture boundary and replay gate."""

import json
import shutil
import subprocess

import pytest
from merlin_experiments.capture_execution.sealed_static import SealedCaptureError, issue, replay_verify

_SOURCE = r"""
#include <fcntl.h>
#include <stdio.h>
#include <unistd.h>
int main(void) {
    if (access("/home", F_OK) == 0 || access("/etc", F_OK) == 0 ||
        access("/checkout", F_OK) == 0 || access("/no-cache", F_OK) == 0 ||
        access("/proc", F_OK) == 0)
        return 41;
    if (open("/source/loader.txt", O_WRONLY) >= 0 ||
        open("/runtime/bin/capture", O_WRONLY) >= 0)
        return 40;
    char data[64];
    int in = open("/source/loader.txt", O_RDONLY);
    if (in < 0) return 42;
    int count = read(in, data, sizeof(data));
    close(in);
    if (count < 1) return 43;
    if (data[0] == 'F') {
        fputs("declared failure\n", stderr);
        return 46;
    }
    int out = open("/capture-out/model.mlir", O_CREAT | O_EXCL | O_WRONLY, 0644);
    if (out < 0) return 44;
    if (write(out, data, count) != count) return 45;
    close(out);
    puts("capture-ok");
    fputs("capture-note\n", stderr);
    return 0;
}
"""


def _runtime(tmp_path):
    if not shutil.which("gcc") or not shutil.which("bwrap"):
        pytest.skip("static C compiler and bubblewrap are required for the process test")
    cfile = tmp_path / "capture.c"
    cfile.write_text(_SOURCE)
    runtime = tmp_path / "runtime"
    (runtime / "bin").mkdir(parents=True)
    proc = subprocess.run(
        ["gcc", "-static", "-no-pie", str(cfile), "-o", str(runtime / "bin/capture")], capture_output=True, text=True
    )
    if proc.returncode:
        pytest.skip(f"static libc toolchain unavailable: {proc.stderr[:200]}")
    return runtime


def test_fresh_static_capture_isolated_and_replayed(tmp_path):
    runtime = _runtime(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "loader.txt").write_bytes(b"module { captured }\n")
    run = tmp_path / "run"
    receipt = issue(source, runtime, ("/runtime/bin/capture",), run)
    document = json.loads(receipt.read_text())
    assert document["issuer"] == "merlin.sealed-static-capture.v1"
    assert document["scope"] == "static_elf_process_only"
    assert document["source_closure_verified"] is False
    assert document["static_source_closure_observed"] is True
    assert document["process"]["returncode"] == 0
    assert document["process"]["stdout"]["bytes"] == len(b"capture-ok\n")
    assert document["process"]["stderr"]["bytes"] == len(b"capture-note\n")
    assert (run / "capture/model.mlir").read_bytes() == b"module { captured }\n"
    replay = replay_verify(run)
    assert replay["status"] == "replay_verified_static"
    assert replay["historical_execution_verified"] is False
    assert replay["source_closure_verified"] is False
    (source / "loader.txt").write_bytes(b"changed live source")
    assert replay_verify(run)["static_replay_source_closure_verified"] is True
    with pytest.raises(FileExistsError):
        issue(source, runtime, ("/runtime/bin/capture",), run)
    (run / "capture/model.mlir").write_bytes(b"tampered")
    with pytest.raises(SealedCaptureError, match="output bytes differ"):
        replay_verify(run)


def test_dynamic_python_and_self_edited_receipt_cannot_qualify(tmp_path):
    runtime = _runtime(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "loader.txt").write_bytes(b"original")
    run = tmp_path / "run"
    receipt = issue(source, runtime, ("/runtime/bin/capture",), run)
    original = json.loads(receipt.read_text())
    document = dict(original, command=["/runtime/bin/other"])
    receipt.chmod(0o644)
    receipt.write_text(json.dumps(document))
    with pytest.raises(SealedCaptureError, match="policy differs"):
        replay_verify(run)
    receipt.write_text(json.dumps(dict(original, status="verified_sealed_execution", source_closure_verified=True)))
    with pytest.raises(SealedCaptureError, match="no supported, current static execution issuer"):
        replay_verify(run)

    dynamic = tmp_path / "dynamic"
    (dynamic / "bin").mkdir(parents=True)
    shutil.copy2("/usr/bin/python3", dynamic / "bin/python")
    with pytest.raises(SealedCaptureError, match="dynamic ELF"):
        issue(source, dynamic, ("/runtime/bin/python",), tmp_path / "no-python-run")
    assert not (tmp_path / "no-python-run").exists()


def test_failed_payload_and_snapshot_mutation_never_admit(tmp_path):
    runtime = _runtime(tmp_path)
    source = tmp_path / "source"
    source.mkdir()
    (source / "loader.txt").write_bytes(b"FAIL")
    with pytest.raises(SealedCaptureError, match="exited 46: declared failure"):
        issue(source, runtime, ("/runtime/bin/capture",), tmp_path / "failed")
    assert not (tmp_path / "failed/capture_execution_attestation.json").exists()
    (source / "loader.txt").write_bytes(b"valid")
    run = tmp_path / "run"
    issue(source, runtime, ("/runtime/bin/capture",), run)
    (run / "snapshots/source/loader.txt").write_bytes(b"mutated snapshot")
    with pytest.raises(SealedCaptureError, match="sealed source or runtime bytes differ"):
        replay_verify(run)
    linked = tmp_path / "source-link"
    linked.symlink_to(source, target_is_directory=True)
    with pytest.raises(SealedCaptureError, match="symlink component"):
        issue(linked, runtime, ("/runtime/bin/capture",), tmp_path / "linked-run")
