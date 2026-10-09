"""Physical capacities derive equally from source and lowered FIRRTL memories."""

import io
import json

import pytest

from merlin.targetgen.rtl.firrtl_memory_lines import observed_memory_lines
from merlin.targetgen.rtl.introspect import census_facts
from merlin.targetgen.rtl.source_selection import derive_hierarchy


def test_lowered_memory_stream_preserves_modules_and_complete_row_shape():
    text = (
        "  module Unit :\n"
        "    mem values : @[generators/test_unit/Storage.scala 3:1]\n"
        "      data-type => UInt<8>[4]\n"
        "      depth => 9\n"
        "      reader => load\n"
        "      writer => store\n"
        "      read-latency => 1\n"
        "      write-latency => 1\n"
        "    input valid : UInt<1>\n"
    )
    output = "".join(observed_memory_lines(io.StringIO(text)))
    assert output == (
        "  module Unit :\n"
        "    mem values  : UInt<8>[4] [9]  @[generators/test_unit/Storage.scala 3:1]\n"
        "    input valid : UInt<1>\n"
    )


@pytest.mark.parametrize("typ,row_elems,elem_bits,capacity", [("UInt<8>[4]", 4, 8, 64), ("UInt<128>", 1, 128, 256)])
def test_lowered_memory_specialized_banks_are_counted_once_per_unit(tmp_path, typ, row_elems, elem_bits, capacity):
    fir = tmp_path / "design.fir"
    fir.write_text(
        "FIRRTL version 3.2.0\ncircuit Harness :\n"
        "  module Harness :\n    inst first of Unit\n    inst second of Unit\n"
        "  module Unit : @[generators/test_unit/Unit.scala 1:1]\n"
        "    inst bankA of BankA\n    inst bankB of BankB\n"
        "  module BankA : @[generators/test_unit/Bank.scala 1:1]\n"
        "    mem words : @[generators/test_unit/Bank.scala 8:1]\n"
        f"      data-type => {typ}\n      depth => 8\n      reader => read\n      writer => write\n"
        "  module BankB : @[generators/test_unit/Bank.scala 1:1]\n"
        "    mem words : @[generators/test_unit/Bank.scala 8:1]\n"
        f"      data-type => {typ}\n      depth => 8\n      reader => read\n      writer => write\n"
    )
    hierarchy = tmp_path / "hierarchy.json"
    hierarchy.write_text(json.dumps(derive_hierarchy(fir)))
    facts = census_facts(fir, hierarchy, generator="test_unit")
    assert facts["census"]["units"] == 2
    assert len(facts["memories"]) == 1
    memory = facts["memories"][0]
    assert memory["banks"] == 2
    assert memory["bytes"] == capacity
    assert memory["row_elems"] == row_elems
    assert memory["elem_bits"] == elem_bits
    assert "copies" not in memory


@pytest.mark.parametrize(
    "properties",
    [
        "      depth => 3\n",
        "      data-type => UInt<8>\n      depth => 0\n",
        "      data-type => UInt<8>\n      depth => 3\n      depth => 4\n",
    ],
)
def test_unclosed_lowered_memory_refuses(properties):
    with pytest.raises(ValueError, match="lowered FIRRTL memory"):
        list(observed_memory_lines(["    mem words :\n", *properties.splitlines(keepends=True)]))


def test_bundle_row_remains_unknown_without_inventing_element_width(tmp_path):
    fir = tmp_path / "design.fir"
    fir.write_text(
        "FIRRTL version 3.2.0\ncircuit Unit :\n"
        "  module Unit : @[generators/test_unit/Unit.scala 1:1]\n"
        "    mem words : @[generators/test_unit/Unit.scala 8:1]\n"
        "      data-type => {valid: UInt<1>, data: UInt<8>}\n      depth => 3\n"
    )
    hierarchy = tmp_path / "hierarchy.json"
    hierarchy.write_text(json.dumps(derive_hierarchy(fir)))
    facts = census_facts(fir, hierarchy, generator="test_unit")
    assert facts["memories"] == []
    assert facts["memories_undeterminable"][0]["type"].startswith("{valid")
