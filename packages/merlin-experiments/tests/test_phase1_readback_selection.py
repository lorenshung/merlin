"""Explicit frozen output transport at real grading and trusted-child seams."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import pytest
import yaml
from merlin_experiments.phase1 import context as C
from merlin_experiments.phase1.feedback import formal, loop_grading, selfcheck
from merlin_experiments.phase1.options import build_parser


def invocation(root: Path, selection: str | None = None) -> C.InvocationContext:
    return C.InvocationContext(
        root, root / "descriptor.yaml", root, "neutral", root, root, root, (), readback_policy=selection
    )


@pytest.mark.parametrize("transport", ["out_b64_v1", "out_bin_v1"])
def test_context_and_worker_roundtrip_keep_explicit_policy(tmp_path, monkeypatch, transport):
    original = invocation(tmp_path, transport)
    command = selfcheck.worker_command(original, tmp_path / "capsules", tmp_path / "contract")
    assert command[command.index("--readback-policy") + 1] == transport
    parser = build_parser()
    # Children use the shared context parser, not the authoring option parser.
    import argparse

    child = argparse.ArgumentParser()
    C.add_context_arguments(child)
    args = child.parse_args(C.context_argv(original))
    monkeypatch.setattr(
        C, "load_context", lambda descriptor, **kwargs: invocation(kwargs["repo"], kwargs["readback_policy"])
    )
    assert C.resolve_context(args, child).readback_policy == original.readback_policy
    assert parser.parse_args(["--run-id", "neutral"]).readback_policy == ""


def test_default_context_preserves_legacy_arguments(tmp_path):
    context = invocation(tmp_path)
    assert C.context_argv(context) == ["--descriptor", str(context.descriptor), "--repo", str(tmp_path)]
    assert C.readback_kwargs(context) == {}
    assert C.readback_record(context) is None
    C.verify_readback_record(context, None)


def test_generic_codec_header_is_available_from_selected_runtime_resources():
    from merlin.common.paths import runtime_dir

    header = runtime_dir() / "baremetal" / "out_b64.h"
    assert header.is_file()
    assert b"merlin_out_b64_finish" in header.read_bytes()


@pytest.mark.parametrize("transport", ["out_b64_v1", "out_bin_v1"])
def test_selected_policy_is_typed_and_records_exactly(tmp_path, transport):
    context = invocation(tmp_path, transport)
    record = {"schema": "merlin_readback_policy_v1", "transport": transport}
    assert C.readback_record(context) == record
    assert C.readback_kwargs(context)["readback_policy"].record() == record
    C.verify_readback_record(context, record)
    for altered in (None, {}, {**record, "transport": "digest_only"}, {**record, "extra": True}):
        with pytest.raises(RuntimeError, match="readback policy"):
            C.verify_readback_record(context, altered)
    with pytest.raises(RuntimeError, match="readback policy"):
        C.verify_readback_record(dataclasses.replace(context, readback_policy=None), record)


def test_formal_policy_cannot_be_selected_or_removed_after_admission(tmp_path):
    selected = invocation(tmp_path, "out_b64_v1")
    with pytest.raises(RuntimeError, match="readback policy"):
        formal._verify_formal_readback_policy(tmp_path, selected)
    path = tmp_path / "environment.yaml"
    path.write_text(yaml.safe_dump({"readback_policy": C.readback_record(selected)}))
    formal._verify_formal_readback_policy(tmp_path, selected)
    with pytest.raises(RuntimeError, match="readback policy"):
        formal._verify_formal_readback_policy(tmp_path, invocation(tmp_path))
    path.write_text("{}\n")
    with pytest.raises(RuntimeError, match="readback policy"):
        formal._verify_formal_readback_policy(tmp_path, selected)


def test_selfcheck_selection_reaches_both_native_tiers(tmp_path, monkeypatch):
    policy = C.readback_kwargs(invocation(tmp_path, "out_b64_v1"))["readback_policy"]
    calls = []
    monkeypatch.setattr(selfcheck, "_sim_policy_error", lambda *_: None)
    monkeypatch.setattr(
        selfcheck.CR, "simulator_adapter", lambda sim, target, **kwargs: calls.append((sim, target, kwargs)) or object()
    )
    adapters, sim = selfcheck._adapters("gsim", "neutral", "chipyard", readback_policy=policy)
    assert sim == "gsim" and set(adapters) == {"L2", "L3"}
    assert calls == [
        ("spike", "neutral", {"readback_policy": policy}),
        ("gsim", "neutral", {"readback_policy": policy}),
    ]
    with pytest.raises(ValueError, match="does not support"):
        selfcheck._adapters("vcs", "neutral", "chipyard", readback_policy=policy)


def test_explicit_transport_failure_cannot_remove_certification_tiers(tmp_path, monkeypatch):
    def unavailable(_):
        raise ValueError("selected transport unavailable")

    monkeypatch.setattr(loop_grading, "load_target_experiment", unavailable)
    kwargs = {"policy_roots": (), "public_roots": lambda: tmp_path}
    with pytest.raises(ValueError, match="transport unavailable"):
        loop_grading.cert_tiers_beyond_loop(context=invocation(tmp_path, "out_b64_v1"), **kwargs)
    assert loop_grading.cert_tiers_beyond_loop(context=invocation(tmp_path), **kwargs) == (set(), set())
