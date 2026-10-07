"""Spike must enable standard ISA extensions declared by a non-vector ELF too."""

from __future__ import annotations

from subprocess import CompletedProcess

from merlin.runtime.backends import spike_model


def test_nonvector_elf_isa_keeps_declared_bitmanip_and_counter_extensions(monkeypatch) -> None:
    arch = "TagName: arch\nValue: rv64i2p1_m2p0_a2p1_f2p2_d2p2_c2p0_b1p0_zicsr2p0_zifencei2p0_zba1p0_zbb1p0_zbs1p0\n"
    monkeypatch.setattr(
        spike_model.subprocess,
        "run",
        lambda *_args, **_kwargs: CompletedProcess([], 0, arch),
    )
    isa = spike_model.declared_isa("nonvector.elf")
    assert isa == "rv64imafdcb_zicntr_zihpm_zicsr_zifencei_zba_zbb_zbs"


def test_vector_elf_retains_the_maximum_declared_zvl(monkeypatch) -> None:
    monkeypatch.setattr(
        spike_model,
        "arch_extensions",
        lambda _elf: ["rv64i", "m", "a", "f", "d", "c", "v", "zvl128b", "zvl256b"],
    )
    assert spike_model.declared_isa("vector.elf") == "rv64imafdcv_zicntr_zihpm_zvl256b"


def test_elf_without_arch_attribute_does_not_invent_an_isa(monkeypatch) -> None:
    monkeypatch.setattr(spike_model, "arch_extensions", lambda _elf: [])
    assert spike_model.declared_isa("unknown.elf") is None
