"""Normal lowering preserves the target layout after performance hooks are removed."""

import json

import pytest

from merlin.llvmlower import int_softmax_table, pipeline, toolchain

SOURCE = """module {
  func.func @add(%a: f32, %b: f32) -> f32 {
    %c = arith.addf %a, %b : f32
    return %c : f32
  }
}"""
LAYOUT = "e-p:64:64-i64:64-f64:64-n8:16:32:64-S128"


def test_actual_runner_layout_slot(tmp_path):
    if not toolchain.m2m_python().is_file():
        pytest.skip("selected upstream lowering toolchain unavailable")
    llvm = pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path, vectorize=False, data_layout=LAYOUT)
    record = json.loads((tmp_path / "lowering_recipe.json").read_text())
    # argv[0] is the Python executable; the runner's sys.argv starts at argv[1].
    runner_argv = record["commands"][0]["argv"][1:]
    assert runner_argv[17] == LAYOUT
    # The retained integer-softmax gate follows the layout and stays off unless selected.
    assert len(runner_argv) == int_softmax_table.ARGV_INDEX + 1 and runner_argv[int_softmax_table.ARGV_INDEX] == "0"
    assert f'target datalayout = "{LAYOUT}"' in llvm
    assert record["status"] == "returned"


def test_empty_copy_selection_keeps_default_lowering_bytes(tmp_path):
    if not toolchain.m2m_python().is_file():
        pytest.skip("selected upstream lowering toolchain unavailable")
    ordinary = pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path / "ordinary", vectorize=False)
    empty = pipeline.lower_to_llvm_ir(SOURCE, workdir=tmp_path / "empty", vectorize=False, features=frozenset())
    assert ordinary == empty
    record = json.loads((tmp_path / "empty/lowering_recipe.json").read_text())
    assert record["commands"][0]["argv"][1:][17:] == ["", "0"]
