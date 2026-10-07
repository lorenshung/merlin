from merlin.llvmlower.constant_float_clamp import _finite_f32, rewrite


def source(lo="-8.700000e+01", hi="8.800000e+01", extra=""):
    return f"""declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
define float @clamp(float %x) {{
entry:
  %low = call float @llvm.maximum.f32(float %x, float {lo})
  %high = call float @llvm.minimum.f32(float %low, float {hi})
{extra}  ret float %high
}}
"""


def test_one_original_input_guard_and_original_nan_path():
    selected, proof = rewrite(source())
    assert len(proof["routes"]) == 1
    assert selected.count("fcmp uno float %frozen, %frozen") == 1
    assert "  %frozen = freeze float %x" in selected
    assert "number_lo = call float @llvm.maxnum.f32" in selected
    assert "original_lo = call float @llvm.maximum.f32" in selected
    assert "original_hi = call float @llvm.minimum.f32" in selected
    assert "  %low = call float @llvm.maximum" not in selected
    assert "  ret float %high" in selected
    assert rewrite(selected)[0] == selected


def test_possible_poison_input_is_frozen_before_every_guard_path_use():
    original = source().replace("entry:\n", "entry:\n  %poison = bitcast i32 poison to float\n")
    original = original.replace("float %x, float -", "float %poison, float -")
    selected, proof = rewrite(original)
    assert len(proof["routes"]) == 1
    assert "(float %poison)" in selected
    helper = selected[selected.index("define internal float @") :]
    assert helper.count("%x") == 2  # Argument definition and freeze operand only.
    assert "%isnan = fcmp uno float %frozen, %frozen" in helper
    assert "maximum.f32(float %frozen," in helper
    assert "maxnum.f32(float %frozen," in helper


def test_another_maximum_consumer_retains_original_definition():
    selected, proof = rewrite(source(extra="  %other = fadd float %low, %high\n"))
    assert proof["routes"][0]["original_maximum_retained_for_other_uses"]
    assert "  %low = call float @llvm.maximum.f32" in selected
    assert "  %other = fadd float %low, %high" in selected


def test_independent_interleaved_clamps_have_same_block_proof():
    original = (
        source()
        .replace(
            "  %high = call",
            "  %low2 = call float @llvm.maximum.f32(float %x, float -8.700000e+01)\n  %high = call",
        )
        .replace(
            "  ret float %high",
            "  %high2 = call float @llvm.minimum.f32(float %low2, float 8.800000e+01)\n  ret float %high",
        )
    )
    selected, proof = rewrite(original)
    assert len(proof["routes"]) == 2
    assert [r["intervening_independent_ieee_calls"] for r in proof["routes"]] == [1, 1]
    assert "  %low = call" not in selected and "  %low2 = call" not in selected
    assert "  %high = call float @__merlin_ieee_const_clamp_" in selected
    assert "  %high2 = call float @__merlin_ieee_const_clamp_" in selected


def test_intervening_dependent_intrinsic_keeps_definition_and_refuses_motion():
    original = source().replace(
        "  %high = call", "  %dependent = call float @llvm.maximum.f32(float %low, float 1.000000e+00)\n  %high = call"
    )
    assert rewrite(original)[0] == original


def test_equal_typed_endpoints_share_helper_across_literal_spellings():
    first = source()
    second = source("0xC055C00000000000", "0x4056000000000000").replace("@clamp(", "@second(")
    selected, proof = rewrite(first + "\n".join(second.splitlines()[2:]))
    assert len(proof["routes"]) == 2
    assert len({r["helper"] for r in proof["routes"]}) == 1
    assert selected.count("define internal float @__merlin_ieee_const_clamp_") == 1


def test_bounds_are_integer_decoded_even_for_gradual_underflow():
    assert _finite_f32("-8.700000e+01")[1] == "c2ae0000"
    assert _finite_f32("8.800000e+01")[1] == "42b00000"
    assert _finite_f32("0x36A0000000000000")[1] == "00000001"
    assert _finite_f32("1.401298464324817070923729583289916131280e-45")[1] == "00000001"
    assert _finite_f32("0xB6A0000000000000")[1] == "80000001"
    assert _finite_f32("0x3690000000000000") is None  # Exactly half minimum subnormal: RNE zero.
    for invalid in ["++1.0", "--1.0", "1.0e++2", "1.0e", "1.0.0", "1", "."]:
        assert _finite_f32(invalid) is None


def test_zero_nan_infinite_reversed_and_untyped_bounds_refuse():
    for lo, hi in [
        ("0.000000e+00", "1.000000e+00"),
        ("-1.000000e+00", "-0.000000e+00"),
        ("0x7FF0000000000000", "1.000000e+00"),
        ("0x7FF8000000000000", "1.000000e+00"),
        ("2.000000e+00", "1.000000e+00"),
        ("%bound", "1.000000e+00"),
    ]:
        original = source(lo, hi)
        assert rewrite(original)[0] == original


def test_no_instruction_motion_across_memory_or_unknown_call():
    for instruction in [
        "  store float %x, ptr null\n",
        "  %unknown = call float @unknown(float %x)\n",
        "  %arithmetic = fadd float %x, %x\n",
        "  br label %next\nnext:\n",
    ]:
        original = source().replace("  %high =", instruction + "  %high =")
        assert rewrite(original)[0] == original


def test_strict_constrained_or_explicit_environment_refuses():
    for scope in [
        "attributes #0 = { strictfp }\n",
        "declare float @llvm.experimental.constrained.fadd.f32(float, float, metadata, metadata)\n",
        "declare i32 @fetestexcept(i32)\n",
    ]:
        original = source() + scope
        selected, proof = rewrite(original)
        assert selected == original and "refusal" in proof


def test_fastmath_and_other_types_remain_unchanged():
    for original in [
        source().replace("call float", "call nnan float"),
        source().replace("float", "double").replace(".f32", ".f64"),
    ]:
        assert rewrite(original)[0] == original


def test_structural_tokens_accept_whitespace_and_quoted_ssa_identity():
    original = source().replace("%low = call float", '%"low" =  call   float')
    original = original.replace("@clamp(", '@"clamp function"(')
    selected, proof = rewrite(original)
    assert len(proof["routes"]) == 1
    assert proof["routes"][0]["source_function"] == '"clamp function"'
    assert '%"low" =  call' not in selected
    assert "  ret float %high" in selected


def test_comment_mentions_do_not_create_other_ssa_uses_or_environment_scope():
    original = source().replace("  ret float", "  ; %low fflags strictfp\n  ret float")
    selected, proof = rewrite(original)
    assert len(proof["routes"]) == 1
    assert not proof["routes"][0]["original_maximum_retained_for_other_uses"]
    assert "  ; %low fflags strictfp" in selected


def test_quoted_numeric_name_is_distinct_from_unnamed_numeric_slot():
    original = """declare float @llvm.maximum.f32(float, float)
declare float @llvm.minimum.f32(float, float)
define float @different(float %x, float %0) {
entry:
  %"0" = call float @llvm.maximum.f32(float %x, float -8.700000e+01)
  %high = call float @llvm.minimum.f32(float %0, float 8.800000e+01)
  ret float %high
}
"""
    selected, proof = rewrite(original)
    assert selected == original and not proof["routes"]
