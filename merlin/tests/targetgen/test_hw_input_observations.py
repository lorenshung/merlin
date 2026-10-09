"""Local HW equality observations preserve bit paths and stop at opaque state."""

import pytest

from merlin.targetgen.rtl.hw_graph import parse_generic_hw
from merlin.targetgen.rtl.hw_observations import input_observations


def _hardware(body, *, signature="input cmd : i12, input other : i12"):
    return (
        """builtin.module {
      "hw.module"() ({
      ^bb0(%cmd: i12, %other: i12):
      """
        + body
        + """
        "hw.output"() : () -> ()
      }) {sym_name = "Unit", module_type = !hw.modty<"""
        + signature
        + """>,
          parameters = [#hw.param.decl<"W" = 12 : i32> : i32]} : () -> ()
    }"""
    )


def _observe(body):
    return input_observations(parse_generic_hw(_hardware(body)))


def test_exact_nested_extract_and_reassembled_contiguous_bits():
    result = _observe("""
      %a = "comb.extract"(%cmd) {lowBit = 2 : i32} : (i12) -> i6
      %lo = "comb.extract"(%a) {lowBit = 0 : i32} : (i6) -> i3
      %hi = "comb.extract"(%a) {lowBit = 3 : i32} : (i6) -> i3
      %joined = "comb.concat"(%hi, %lo) : (i3, i3) -> i6
      %k = "hw.constant"() {value = 17 : i6} : () -> i6
      %is_k = "comb.icmp"(%joined, %k) {predicate = 0 : i64} : (i6, i6) -> i1
    """)
    assert result["scope"] == "local_module_input" and result["complete_isa"] is False
    found = [o for o in result["modules"][0]["observations"] if o["equality_constants"]]
    assert found == [{"input": "cmd", "offset": 2, "width": 6, "equality_constants": [17]}]


@pytest.mark.parametrize(
    "expression",
    [
        '"seq.firreg"(%cmd) : (i12) -> i12',
        '"comb.xor"(%cmd, %other) : (i12, i12) -> i12',
    ],
)
def test_opaque_and_nontransparent_operations_do_not_become_input_fields(expression):
    result = _observe(f"""
      %state = {expression}
      %k = "hw.constant"() {{value = 17 : i12}} : () -> i12
      %is_k = "comb.icmp"(%state, %k) {{predicate = 0 : i64}} : (i12, i12) -> i1
    """)
    assert result["modules"][0]["observations"] == []


def test_instance_result_comparison_keeps_input_and_effects_unknown():
    result = _observe("""
      %state = "hw.instance"(%cmd) {moduleName = @Buffer, instanceName = "buffer",
        argNames = ["input"], resultNames = ["output"]} : (i12) -> i12
      %k = "hw.constant"() {value = -5 : i12} : () -> i12
      %is_k = "comb.icmp"(%state, %k) {predicate = 0 : i64} : (i12, i12) -> i1
    """)
    unit = result["modules"][0]
    assert unit["observations"] == []
    assert unit["instance_output_observations"] == [
        {
            "scope": "opaque_instance_result",
            "instance": "buffer",
            "module": "Buffer",
            "output": "output",
            "width": 12,
            "equality_constants": [4091],
            "input_correspondence": "unknown",
            "instance_effects": "unknown",
        }
    ]


def test_instance_result_without_complete_source_names_refuses():
    with pytest.raises(ValueError, match="complete explicit"):
        _observe("""
          %state = "hw.instance"(%cmd) {moduleName = @Buffer} : (i12) -> i12
          %k = "hw.constant"() {value = 17 : i12} : () -> i12
          %is_k = "comb.icmp"(%state, %k) {predicate = 0 : i64} : (i12, i12) -> i1
        """)


def test_noncontiguous_or_multiple_input_concat_does_not_report_one_field():
    result = _observe("""
      %lo = "comb.extract"(%cmd) {lowBit = 1 : i32} : (i12) -> i3
      %hi = "comb.extract"(%other) {lowBit = 4 : i32} : (i12) -> i3
      %joined = "comb.concat"(%hi, %lo) : (i3, i3) -> i6
      %k = "hw.constant"() {value = 17 : i6} : () -> i6
      %is_k = "comb.icmp"(%joined, %k) {predicate = 0 : i64} : (i6, i6) -> i1
    """)
    assert all(not row["equality_constants"] for row in result["modules"][0]["observations"])


def test_non_equality_predicate_is_not_a_decode_constant():
    result = _observe("""
      %k = "hw.constant"() {value = 17 : i12} : () -> i12
      %is_k = "comb.icmp"(%cmd, %k) {predicate = 1 : i64} : (i12, i12) -> i1
    """)
    assert result["modules"][0]["observations"] == []


def test_complete_input_signature_is_required():
    text = _hardware("", signature="input cmd : i12")
    with pytest.raises(ValueError, match="exactly match arguments"):
        input_observations(parse_generic_hw(text))


def test_typed_parameter_attribute_is_preserved_in_lossless_analysis():
    module = parse_generic_hw(_hardware(""))
    parameter = module.body.block.ops.first.attributes["parameters"].data[0]
    assert parameter.attr_name.data == "hw.param.decl"
    assert parameter.value.data.endswith(" : i32")
