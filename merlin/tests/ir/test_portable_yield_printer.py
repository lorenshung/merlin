"""Attributed yields retain provenance in assembly accepted by upstream MLIR."""

import subprocess
from pathlib import Path

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.toolchain import m2m_python
from merlin.xdsl_dialects._common import text

SOURCE = """builtin.module {
  func.func @forward(%a: tensor<4xf32>) -> tensor<4xf32> {
    %b = linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> (d0)>],
      iterator_types = ["parallel"]
    } ins(%a : tensor<4xf32>) outs(%a : tensor<4xf32>) {
    ^bb0(%x: f32, %y: f32):
      "linalg.yield"(%x) {prov.source_node_ids = ["node:1"]} : (f32) -> ()
    } -> tensor<4xf32>
    func.return %b : tensor<4xf32>
  }
}"""


def test_attributed_yield_roundtrips_through_upstream():
    module = parse_mlir_text(SOURCE)
    printed = text(module)
    assert '"linalg.yield"' in printed
    assert 'prov.source_node_ids = ["node:1"]' in printed
    parse_mlir_text(printed).verify()
    if not Path(m2m_python()).is_file():
        pytest.skip("upstream MLIR Python is unavailable")
    proc = subprocess.run(
        [
            str(m2m_python()),
            "-c",
            "import sys; from torch_mlir import ir; "
            "ctx=ir.Context(); "
            "module=ir.Module.parse(sys.stdin.read(), ctx); "
            "module.operation.verify()",
        ],
        input=printed,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr


def test_external_function_preserves_bufferization_access():
    from xdsl.dialects.builtin import ArrayAttr, DictionaryAttr, ModuleOp, StringAttr, TensorType, i8
    from xdsl.dialects.func import FuncOp
    from xdsl.ir import Region

    from merlin.llvmlower.declaration_access import unpatched_declarations

    typ = TensorType(i8, [4])
    declaration = FuncOp(
        "external",
        ([typ], [typ]),
        Region(),
        visibility="private",
        arg_attrs=ArrayAttr([DictionaryAttr({"bufferization.access": StringAttr("read")})]),
    )
    printed = text(ModuleOp([declaration]))
    assert '"func.func"' in printed
    reparsed = parse_mlir_text(printed)
    fn = reparsed.body.block.first_op
    assert fn.arg_attrs.data[0].data["bufferization.access"].data == "read"
    assert unpatched_declarations(text(reparsed), ["external"]) == ()
    assert unpatched_declarations(text(reparsed), ["missing"]) == ("missing",)


def test_attributed_reduce_preserves_provenance_through_upstream():
    source = """builtin.module {
      func.func @forward(%a: tensor<2x4xf32>, %init: tensor<2xf32>) -> tensor<2xf32> {
        %sum = "linalg.reduce"(%a, %init) <{dimensions = array<i64: 1>}> ({
        ^bb0(%x: f32, %acc: f32):
          %next = arith.addf %x, %acc : f32
          linalg.yield %next : f32
        }) {prov.op = "layer_norm", prov.source_node_ids = ["node:reduce"]}
          : (tensor<2x4xf32>, tensor<2xf32>) -> tensor<2xf32>
        func.return %sum : tensor<2xf32>
      }
    }"""
    printed = text(parse_mlir_text(source))
    reparsed = parse_mlir_text(printed)
    reduction = next(op for op in reparsed.walk() if op.name == "linalg.reduce")
    assert reduction.attributes["prov.op"].data == "layer_norm"
    assert reduction.attributes["prov.source_node_ids"].data[0].data == "node:reduce"
    reparsed.verify()
    if not Path(m2m_python()).is_file():
        pytest.skip("upstream MLIR Python is unavailable")
    proc = subprocess.run(
        [
            str(m2m_python()),
            "-c",
            "import sys; from torch_mlir import ir; "
            "ctx=ir.Context(); "
            "module=ir.Module.parse(sys.stdin.read(), ctx); "
            "module.operation.verify()",
        ],
        input=printed,
        text=True,
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr
