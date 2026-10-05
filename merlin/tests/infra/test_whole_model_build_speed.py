"""What makes a whole-model build fast must not change what it builds.

Every speed-up here is a cache keyed by the exact bytes it stands for, or an overlap of work that does
not depend on each other; each test pins that the result is the same one the slow path gives.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml


class _Package:
    manifest = {"commands": {"emit_analysis_bundle": {"argv": ["{tool}", "{input_mlir}", "{output_json}"]}}}


def _invoker(log, *, target_rc=0, delay=0.2):
    def invoke(pkg, name, source, output=None, *, timeout):
        log.append(("start", name, time.monotonic()))
        if name != "parse":
            time.sleep(delay)
        if name == "emit_analysis_bundle":
            Path(output).write_text('{"abi_version": "0.1", "commands": []}')
        log.append(("end", name, time.monotonic()))
        rc = target_rc if name == "lower_interface_to_target" else 0
        return SimpleNamespace(returncode=rc, stdout="artifact" if rc == 0 else "", stderr="refused")

    return invoke


def test_the_two_independent_entrypoints_overlap_and_the_target_plane_is_still_judged_first(tmp_path):
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import oot_runner as OR

    interface = tmp_path / "g.iface.mlir"
    interface.write_text("module {}\n")
    log: list = []
    with pytest.raises(OR.CertFailure) as refused:
        CC.lower_interface(
            _Package(), interface, tmp_path / "gen", contract=None, timeout=5,
            invoke=_invoker(log, target_rc=1), overlap=True,
        )  # fmt: skip
    assert refused.value.plane == "interface_to_target"
    starts = {name: at for kind, name, at in log if kind == "start"}
    ends = {name: at for kind, name, at in log if kind == "end"}
    assert starts["emit_analysis_bundle"] < ends["lower_interface_to_target"], "the two did not overlap"
    assert "emit_analysis_bundle" in ends, "the buffer command was left running behind the refusal"


def test_an_interface_lowered_once_is_answered_from_the_memo_for_every_other_group(tmp_path, monkeypatch):
    """Groups that state the same interface get the same accepted lowering without the package's
    replies being read again; each still gets its own generated files, and a refusal is never kept."""
    from merlin.targetgen import artifact_scale
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import oot_runner as OR

    monkeypatch.setattr(CC.schemas, "validate_command_buffer", lambda cb, contract=None: None)
    monkeypatch.setattr(CC, "validate_interface_tensor_dtypes", lambda cb, text: None)
    monkeypatch.setattr(artifact_scale, "refusal", lambda artifact: None)
    interface = tmp_path / "g.iface.mlir"
    interface.write_text("module {}\n")
    memo: dict = {}
    log: list = []
    first = CC.lower_interface(
        _Package(), interface, tmp_path / "g1", contract=None, timeout=5, invoke=_invoker(log, delay=0), memo=memo
    )
    asked = len(log)
    second = CC.lower_interface(
        _Package(),
        interface,
        tmp_path / "g2",
        contract=None,
        timeout=5,
        invoke=_invoker(log, delay=0),
        memo=memo,
        artifact_name="g2.artifact.txt",  # a group files its artifact under its own name; still a hit
    )
    assert len(log) == asked, "the second group read the package's replies again"
    assert second == first and second[0] is not first[0], "a memo hit must be the same value, never the same object"
    for generated in (tmp_path / "g1", tmp_path / "g2"):
        assert (generated / "command_buffer.json").read_text() == (tmp_path / "g1" / "command_buffer.json").read_text()
    assert (tmp_path / "g1" / "lowered.llvm.mlir").read_text() == "artifact"
    assert (tmp_path / "g2" / "g2.artifact.txt").read_text() == "artifact"
    other = tmp_path / "h.iface.mlir"
    other.write_text("module { }\n")
    with pytest.raises(OR.CertFailure):
        CC.lower_interface(
            _Package(),
            other,
            tmp_path / "h",
            contract=None,
            timeout=5,
            invoke=_invoker(log, target_rc=1, delay=0),
            memo=memo,
        )
    assert len(memo) == 1, "a refusal was recorded"


def test_without_overlap_the_entrypoints_run_in_their_declared_order(tmp_path):
    from merlin.targetgen import capsule_common as CC
    from merlin.targetgen import oot_runner as OR

    interface = tmp_path / "g.iface.mlir"
    interface.write_text("module {}\n")
    log: list = []
    with pytest.raises(OR.CertFailure):
        CC.lower_interface(
            _Package(), interface, tmp_path / "gen", contract=None, timeout=5, invoke=_invoker(log, target_rc=1)
        )
    assert [name for kind, name, _ in log if kind == "start"] == ["parse", "lower_interface_to_target"]


def test_the_asker_starts_each_ask_when_its_interface_is_written_and_the_second_pass_waits_for_it(tmp_path):
    from merlin.perf import whole_model_replies as R

    asked, compiled = [], []
    gate = threading.Event()

    class Cache:
        def __init__(self):
            self.recorded = set()

        def replay(self, name, source, output=None):
            key = (name, Path(source).read_bytes())
            return SimpleNamespace(returncode=0, stdout="x", stderr="") if key in self.recorded else None

        def invoke(self, package, name, source, output=None, *, timeout=600):
            reply = self.replay(name, source, output)
            if reply is not None:
                return reply
            gate.wait(5)
            asked.append(name)
            self.recorded.add((name, Path(source).read_bytes()))
            return SimpleNamespace(returncode=0, stdout="x", stderr="")

    cache = Cache()
    asker = R.Asker(object(), cache, tmp_path, timeout=5, jobs=2, compile_artifact=compiled.append)

    def fake_lower(package, interface, generated, *, contract, timeout, invoke, overlap):
        for name in ("parse", "lower_interface_to_target", "emit_analysis_bundle"):
            invoke(package, name, interface, timeout=timeout)
        return {}, "artifact-text"

    import merlin.targetgen.capsule_common as CC

    original = CC.lower_interface
    CC.lower_interface = fake_lower
    try:
        source = tmp_path / "input.interface.mlir"
        source.write_text("module { g1 }\n")
        refused = asker.recorded(None, "parse", source)
        assert refused.returncode == 1 and asker.missing == ["parse"]
        assert len(asker.pending) == 1, "the ask did not start when the interface was written"
        gate.set()
        reply = asker.invoke(None, "parse", source)
        assert reply.returncode == 0 and asked == ["parse", "lower_interface_to_target", "emit_analysis_bundle"]
        asker.close()
        assert compiled == ["artifact-text"]
        assert not (tmp_path / "prefetch").exists()
    finally:
        CC.lower_interface = original


def test_a_prewarmed_object_is_the_entry_the_object_stage_reads(tmp_path, monkeypatch):
    from merlin.perf import whole_model_object_cache as OC
    from merlin.targetgen.contract import compile as compile_mod

    calls = []

    def compile_one(text, work, *, target):
        calls.append(text)
        work.mkdir(parents=True, exist_ok=True)
        (work / "kernel.o").write_bytes(b"obj:" + text.encode())
        return work / "kernel.o"

    monkeypatch.setattr(compile_mod, "llvm_mlir_to_object", compile_one)
    monkeypatch.setattr(OC, "object_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setitem(OC._FINGERPRINTS, "t", "fingerprint")
    assert OC.prewarm("a", target="t", work=tmp_path / "w1") == "miss"
    assert OC.prewarm("a", target="t", work=tmp_path / "w2") == "hit" and calls == ["a"]
    key = OC.object_cache_key("a", target="t", fingerprint="fingerprint")
    assert OC.load(tmp_path / "cache" / key[:2] / f"{key}.o").read_bytes() == b"obj:a"
    monkeypatch.setattr(OC, "object_cache_root", lambda: None)
    assert OC.prewarm("b", target="t", work=tmp_path / "w3") == "off" and calls == ["a"]


def test_a_background_prewarm_compiles_a_text_once_and_the_object_stage_waits_for_it(tmp_path, monkeypatch):
    """The statement starts a group's compile and moves on; asking again for the same text joins the
    running compile, and the object stage waits for it rather than compiling a second time."""
    from merlin.perf import whole_model_object_cache as OC
    from merlin.targetgen.contract import compile as compile_mod

    calls, release = [], threading.Event()

    def compile_one(text, work, *, target):
        calls.append(text)
        release.wait(10)
        work.mkdir(parents=True, exist_ok=True)
        (work / "kernel.o").write_bytes(b"obj:" + text.encode())
        return work / "kernel.o"

    monkeypatch.setattr(compile_mod, "llvm_mlir_to_object", compile_one)
    monkeypatch.setattr(OC, "object_cache_root", lambda: tmp_path / "cache")
    monkeypatch.setitem(OC._FINGERPRINTS, "t", "fingerprint")
    first = OC.prewarm_async("a", target="t", jobs=2)
    assert OC.prewarm_async("a", target="t", jobs=2) is first, "a running compile of the same text is joined"
    key = OC.object_cache_key("a", target="t", fingerprint="fingerprint")
    waited = threading.Thread(target=OC.await_inflight, args=(key,))
    waited.start()
    time.sleep(0.2)
    assert waited.is_alive(), "the object stage returned before the running compile finished"
    release.set()
    waited.join(10)
    assert not waited.is_alive() and calls == ["a"]
    assert OC.load(tmp_path / "cache" / key[:2] / f"{key}.o").read_bytes() == b"obj:a"
    assert OC.prewarm_async("a", target="t", jobs=2) is None, "a cached text starts nothing"


def test_a_recorded_oracle_is_served_only_for_its_exact_key(tmp_path, monkeypatch):
    import subprocess as SP

    import numpy as np

    from merlin.perf import whole_model_oracle as O

    monkeypatch.setenv("MERLIN_OUT_ROOT", str(tmp_path / "out"))
    keys = iter(["k" * 64, "k" * 64, "j" * 64])
    monkeypatch.setattr(O, "oracle_cache_key", lambda capsule, target: next(keys))
    computed = []

    class Child:
        """The child process, run in place: it writes the oracle where the job told it to."""

        def __init__(self, argv, **kwargs):
            computed.append(argv)
            entry = np.arange(4, dtype=np.int8)
            O._write_oracle(Path(argv[-1]), {"argmax": 21}, entry.tobytes(), "int8", [4])

        def wait(self):
            return 0

    monkeypatch.setattr(SP, "Popen", Child)
    capsule = SimpleNamespace(directory=tmp_path, interface=tmp_path / "i.mlir")
    first = O.OracleJob(capsule, target="t")
    assert first.state == "miss" and first.result()[0] == {"argmax": 21}
    assert computed[0][1:3] == ["-m", "merlin.perf.whole_model_oracle"]
    again = O.OracleJob(capsule, target="t")
    oracle, entry = again.result()
    assert again.state == "hit" and oracle == {"argmax": 21} and entry.tolist() == [0, 1, 2, 3]
    assert len(computed) == 1, "a recorded oracle was recomputed"
    assert O.OracleJob(capsule, target="t").state == "miss", "a different key served a recorded oracle"


def test_the_pin_registry_keeps_every_machine_a_build_names_live():
    """A stray top-level section once moved every later artifact out of `artifacts`, so the builder's
    machine lookup refused machines that exist. The registry's sections are the ones the loaders read."""
    from merlin.common import provenance as PROV

    doc = yaml.safe_load(PROV.pins_path().read_text(encoding="utf-8"))
    assert set(doc) == {"version", "pins", "artifacts"}
    for machine in (
        "gemmini_gsim_emulator",
        "gemmini_gsim_model_testharness",
        "firesim_gemmini_rocket_u250_30mhz",
        "gemmini_shuttle_opu_u250_firesim_bitstream",
    ):
        assert machine in doc["artifacts"], machine


def test_a_marked_stage_runs_from_where_the_previous_one_ended():
    from merlin.perf import whole_model_build as WMB

    ticks = iter([1.0, 3.0, 4.0, 10.0, 10.0])
    clock = WMB._StageClock()
    clock._clock = lambda: next(ticks)
    clock._started = clock._last = 0.0
    with clock("block"):  # 1.0 -> 3.0
        pass
    clock.mark("after")  # 3.0 -> 4.0
    clock.mark("later")  # 4.0 -> 10.0
    assert clock.record()["stages"] == {"block": 2.0, "after": 1.0, "later": 6.0}
