"""A tensor supplies shape/type, not a memref layout constraint."""

import pytest
from xdsl.utils.exceptions import VerifyException

from merlin.frontends.linalg_mlir import parse_mlir_text


@pytest.mark.parametrize("layout", ["", ", strided<[?, ?], offset: ?>", ", strided<[0, 1], offset: 2>"])
def test_to_buffer_keeps_declared_layout(layout):
    source = f'''builtin.module {{
      func.func @forward(%x: tensor<2x3xbf16>) {{
        %b = "bufferization.to_buffer"(%x) : (tensor<2x3xbf16>) -> memref<2x3xbf16{layout}>
        func.return
      }}
    }}'''
    module = parse_mlir_text(source)
    module.verify()
    parse_mlir_text(str(module)).verify()
    assert f"memref<2x3xbf16{layout}>" in str(module)


@pytest.mark.parametrize("result", ["memref<3x2xbf16>", "memref<2x3xf16>"])
def test_to_buffer_still_checks_shape_and_element_type(result):
    module = parse_mlir_text(f'''builtin.module {{
      func.func @forward(%x: tensor<2x3xbf16>) {{
        %b = "bufferization.to_buffer"(%x) : (tensor<2x3xbf16>) -> {result}
        func.return
      }}
    }}''')
    with pytest.raises(VerifyException, match="matching tensor/memref"):
        module.verify()


def test_unranked_buffer_correspondence_is_preserved():
    source = '''builtin.module {
      func.func @forward(%x: tensor<*xf32>) {
        %b = "bufferization.to_buffer"(%x) : (tensor<*xf32>) -> memref<*xf32>
        func.return
      }
    }'''
    module = parse_mlir_text(source)
    module.verify()
    parse_mlir_text(str(module)).verify()
