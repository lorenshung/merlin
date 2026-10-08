"""General result allocation aliasing must never crash upstream lowering."""

import pytest

from merlin.llvmlower.pipeline import lower_to_llvm_ir
from merlin.llvmlower.toolchain import m2m_python


@pytest.mark.parametrize("extent", [0, 7])
def test_aliased_allocated_results_lower_without_erasing_twice(tmp_path, extent):
    if not m2m_python().is_file():
        pytest.skip("native MLIR interpreter unavailable")
    source = f"""builtin.module {{
      func.func @forward() -> (tensor<{extent}xf32>, tensor<{extent}xf32>) {{
        %a = tensor.empty() : tensor<{extent}xf32>
        func.return %a, %a : tensor<{extent}xf32>, tensor<{extent}xf32>
      }}
    }}"""
    text = lower_to_llvm_ir(source, workdir=tmp_path)
    assert "define void @forward" in text


def test_repeated_borrowed_argument_is_not_an_allocation(tmp_path):
    if not m2m_python().is_file():
        pytest.skip("native MLIR interpreter unavailable")
    source = """builtin.module {
      func.func @forward(%x: tensor<3xi64>) -> (tensor<3xi64>, tensor<3xi64>) {
        func.return %x, %x : tensor<3xi64>, tensor<3xi64>
      }
    }"""
    assert "define void @forward" in lower_to_llvm_ir(source, workdir=tmp_path)
