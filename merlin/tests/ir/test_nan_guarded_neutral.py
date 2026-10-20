"""A NaN-class guard cannot license an observable NaN payload change."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_text
from merlin.llvmlower.nan_guarded_neutral import fold_nan_guarded_neutral


def source(operation, literal, *, escape=False, bits=False, predicate="uno", reverse=False):
    no, yes = ("%qnan", "%v") if reverse else ("%v", "%qnan")
    result = "%v" if escape else "%guard"
    extra = "%raw=arith.bitcast %v : f32 to i32" if bits else ""
    return parse_mlir_text(f"""module {{func.func @forward(%x:f32) -> f32 {{
      %literal=arith.constant {literal} : f32
      %v=arith.{operation} %x,%literal : f32
      %qnan=arith.constant 0x7fc01234 : f32
      %p=arith.cmpf {predicate},%v,%v : f32
      %guard=arith.select %p,{yes},{no} : f32
      {extra}
      return {result} : f32
    }} }}""")


@pytest.mark.parametrize(
    "operation,literal", [("maximumf", "0xff800000"), ("minimumf", "0x7f800000"), ("mulf", "1.0"), ("divf", "1.0")]
)
@pytest.mark.parametrize("predicate,reverse", [("uno", False), ("ord", True)])
def test_every_use_observes_only_the_non_nan_word_or_nan_class(operation, literal, predicate, reverse):
    module = source(operation, literal, predicate=predicate, reverse=reverse)
    assert fold_nan_guarded_neutral(module) == 1
    assert not any(op.name == "arith." + operation for op in module.walk())
    assert any(op.name == "arith.cmpf" for op in module.walk())
    assert any(op.name == "arith.select" for op in module.walk())
    module.verify()


@pytest.mark.parametrize("options", [dict(escape=True), dict(bits=True), dict(predicate="ord"), dict(reverse=True)])
def test_a_nan_payload_escape_or_nan_word_selection_refuses(options):
    module = source("maximumf", "0xff800000", **options)
    original = str(module)
    assert fold_nan_guarded_neutral(module) == 0
    assert str(module) == original


@pytest.mark.parametrize(
    "operation,literal", [("maximumf", "0.0"), ("minimumf", "0.0"), ("mulf", "2.0"), ("divf", "0.5")]
)
def test_non_neutral_numeric_work_is_retained(operation, literal):
    assert fold_nan_guarded_neutral(source(operation, literal)) == 0
