from dataclasses import replace

import pytest

from merlin.llvmlower.constant_fma_packet import (
    analyze,
    rewrite,
    validate_packet_source,
)

POLICY = dict(ordinary_nontrapping=True, exception_flags_unobserved=True)


def source(*, lanes=2, mode="constant_addend"):
    calls = []
    for lane in range(lanes):
        operands = ("%a", "%b", "1.0")
        if mode == "constant_rhs":
            operands = ("%a", "1.0", "%b")
        elif mode == "constant_lhs_addend":
            operands = ("1.0", "%a", "2.0")
        calls.append(
            f"  %r{lane} = call float @llvm.fma.f32(" + ", ".join("float " + value for value in operands) + ")"
        )
    return (
        "define float @probe(float %a, float %b) {\nentry:\n"
        + "\n".join(calls)
        + "\n  %live = fadd float %a, %r0\n"
        + "  ret float %live\n}\n"
        + "declare float @llvm.fma.f32(float,float,float)\n"
    )


@pytest.mark.parametrize("width", [2, 4])
@pytest.mark.parametrize("mode", ["constant_rhs", "constant_addend", "constant_lhs_addend"])
def test_actual_typed_modes_and_tail(width, mode):
    text = source(lanes=width + 1, mode=mode)
    packets = analyze(text, width=width, **POLICY)
    assert len(packets) == 1
    assert len(packets[0].calls) == width
    assert packets[0].mode == mode
    validate_packet_source(text, packets[0])
    seen = []

    def provider(packet, temporary):
        seen.append(temporary)
        return "\n  ".join(
            f"{call.result} = call float @llvm.fma.f32(" + ", ".join("float " + value for value in call.operands) + ")"
            for call in packet.calls
        )

    result, report = rewrite(text, width=width, emitter=provider, **POLICY)
    assert len(report["packets"]) == 1
    assert seen == ["%constant.fma.packet.0"]
    assert f"%r{width} = call float" in result
    assert "%live = fadd float %a, %r0" in result
    assert "ret float %live" in result


def test_default_has_no_parse_or_provider_effect():
    text = "this is intentionally not LLVM"
    assert rewrite(text)[0] == text
    assert not analyze(source(), width=2)
    assert not analyze(source(), width=2, ordinary_nontrapping=True)


@pytest.mark.parametrize(
    "change",
    [
        lambda s: s.replace(" {", " strictfp {", 1),
        lambda s: s.replace("call float", "call fast float", 1),
        lambda s: s.replace("float 1.0)", "float 1.0) #0", 1),
        lambda s: s.replace("float 1.0)", "float 1.0), !test !0", 1),
        lambda s: s.replace("call float", "call double", 1),
        lambda s: s.replace("%r1 = call float @llvm.fma.f32(float %a", "%r1 = call float @llvm.fma.f32(float %r0"),
        lambda s: s.replace("  %r1", "  store float %a, ptr null\n  %r1"),
        lambda s: s.replace("  %r1", "  call void @unknown()\n  %r1"),
        lambda s: s.replace("  %r1", "  %extra = fadd float %a,%b\n  %r1"),
        lambda s: s.replace("  %r1", "other:\n  %r1"),
        lambda s: s.replace(
            "  %r1 = call float @llvm.fma.f32(float %a, float %b, float 1.0)",
            "  %r1 = call float @llvm.fma.f32(float %a, float %b, float 2.0)",
        ),
        lambda s: s.replace("float 1.0", "float 0x7FF0000000000000"),
        lambda s: s.replace("float 1.0", "float -0.0"),
        lambda s: s.replace("float %b, float 1.0", "float %missing, float 1.0"),
        lambda s: s + '\n@env = private constant [4 x i8] c"frm\\00"\n',
        lambda s: s + '\nattributes #0 = { "unsafe-fp-math"="true" }\n',
        lambda s: s + '\nattributes #0 = { "denormal-fp-math-f32"="preserve-sign" }\n',
    ],
)
def test_meaningful_refusals_preserve_source(change):
    text = change(source())
    assert not analyze(text, width=2, **POLICY)
    called = []
    result, report = rewrite(text, width=2, emitter=lambda *args: called.append(args), **POLICY)
    assert result == text and not report["packets"] and not called


def test_complete_ancestor_and_packet_mutation_refuse():
    text = source()
    (packet,) = analyze(text, width=2, **POLICY)
    with pytest.raises(ValueError, match="context changed"):
        validate_packet_source(text.replace(" {", " strictfp {", 1), packet)
    with pytest.raises(ValueError, match="context changed"):
        validate_packet_source(text, replace(packet, constant_words=(0x40000000,)))


def test_ssa_name_collision_and_comments():
    text = source().replace("  %r0", '  %"constant.fma.packet.0" = fadd float %a,%b\n  %r0', 1)
    text = text.replace("  %r1", "  ; strictfp in a comment grants no effect\n  %r1")
    seen = []

    def provider(packet, temporary):
        seen.append(temporary)
        return "\n  ".join(
            f"{call.result} = call float @llvm.fma.f32(" + ", ".join("float " + value for value in call.operands) + ")"
            for call in packet.calls
        )

    result, _ = rewrite(text, width=2, emitter=provider, **POLICY)
    assert seen == ["%constant.fma.packet.1"]
    assert '%"constant.fma.packet.0" = fadd' in result


def test_unsupported_width_and_generated_name():
    with pytest.raises(ValueError, match="widths"):
        analyze(source(), width=3, **POLICY)
    with pytest.raises(ValueError, match="prefix"):
        rewrite(source(), emitter=lambda *args: "", temporary_prefix="bad\nname", **POLICY)
