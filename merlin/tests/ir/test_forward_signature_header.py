"""A function's signature read from its header alone is the one a full parse of the module reads."""

from __future__ import annotations

import subprocess

import pytest

from merlin.common import mlir_query as mq
from merlin.common.paths import repo_root


def _full(text: str, name: str = "forward"):
    module = mq.parse_mlir_text(text)
    fn = next(op for op in module.walk() if op.name == "func.func" and mq._sym_name(op) == name)
    ftype = fn.function_type
    return [mq.type_shape_dtype(t) for t in ftype.inputs.data], [mq.type_shape_dtype(t) for t in ftype.outputs.data]


def _tracked_forwards():
    root = repo_root()
    listed = subprocess.run(["git", "ls-files", "*.mlir"], cwd=root, capture_output=True, text=True).stdout.split()
    picked = []
    for rel in listed:
        path = root / rel
        if path.stat().st_size > 2_000_000:
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        if "func.func @forward(" in text:
            picked.append(path)
    return picked


def test_the_header_reader_agrees_with_a_full_parse_on_every_tracked_forward():
    paths = _tracked_forwards()
    assert len(paths) >= 5
    checked = 0
    for path in paths:
        text = path.read_text(encoding="utf-8")
        try:
            want = _full(text)
        except Exception:  # noqa: BLE001 -- a module the parser cannot read is not this reader's case
            continue
        assert mq.forward_signature(text) == want, path
        assert mq._signature_skeleton(text, "forward") is not None, path
        checked += 1
    assert checked >= 5


@pytest.mark.parametrize(
    "header",
    [
        '%0: tensor<2x3xf32> {foo = "a{b)"}, %1: i64) -> (tensor<2xf32>, i1) attributes {x = "}"}',
        "%0: memref<4xf32, strided<[1], offset: ?>>) -> memref<4xf32, strided<[1], offset: ?>>",
        "%0: tensor<4xi8>)",
    ],
)
def test_attributes_strings_and_layouts_in_a_header_are_read_exactly(header):
    text = f"builtin.module {{\n  func.func @forward({header} {{\n    func.return\n  }}\n}}\n"
    try:
        want = _full(text)
    except Exception:  # noqa: BLE001
        pytest.skip("the parser itself does not read this header")
    assert mq.forward_signature(text) == want


def test_a_header_it_cannot_isolate_falls_back_to_the_full_parse():
    generic = (
        '"builtin.module"() ({\n  "func.func"() <{sym_name = "forward", function_type = (i64) -> i64}> ({\n'
        '  ^bb0(%a: i64):\n    "func.return"(%a) : (i64) -> ()\n  }) : () -> ()\n}) : () -> ()\n'
    )
    assert mq._signature_skeleton(generic, "forward") is None
    assert mq.forward_signature(generic) == ([([], "i64")], [([], "i64")])
