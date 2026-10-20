"""Literal mask endpoints retain typed scaling and only discard private state."""

import pytest

from merlin.frontends.linalg_mlir import parse_mlir_file
from merlin.targetgen.source_mask_endpoints import specialize


def fixture(tmp_path, *, probability=0, training=True, private=True, unknown=False):
    text = f'''module {{
      "ml_program.global"() <{{sym_name="rng",sym_visibility="{"private" if private else "public"}",is_mutable,
        type=tensor<1xi64>,value=dense<1>:tensor<1xi64>}}> : ()->()
      func.func @forward(%x:tensor<4xf32>) -> (tensor<4xf32>,tensor<4xi1>,tensor<4xf32>)
        attributes {{"prov.aten"="aten.native_dropout.default"}} {{
        %state="ml_program.global_load"() <{{global=@rng}}> : ()->tensor<1xi64>
        "ml_program.global_store"(%state) <{{global=@rng}}> : (tensor<1xi64>)->()
        {'"fixture.effect"() : ()->()' if unknown else ""}
        %data=tensor.empty() : tensor<4xf32>
        %mask=tensor.empty() : tensor<4xi1>
        return %data,%mask,%x : tensor<4xf32>,tensor<4xi1>,tensor<4xf32>
      }}
    }}'''
    (tmp_path / "model.mlir").write_text(text)
    return dict(
        cases=[
            dict(
                status="bundled",
                case=dict(
                    overload="aten.native_dropout.default",
                    arguments=dict(
                        args=dict(items=[dict(kind="tensor", values="MUST NOT READ"), probability, training])
                    ),
                ),
                output_indices=[0, 1],
                semantic_boundary=dict(post_indices=[2]),
            )
        ]
    )


@pytest.mark.parametrize("probability,training,mask", [(0, True, 1), (1, True, 0), (0.3, False, 1)])
def test_complete_source_endpoint_and_private_state_scope(tmp_path, probability, training, mask):
    report = fixture(tmp_path, probability=probability, training=training)
    proof = specialize(tmp_path, report)
    assert proof and proof["mask"] == mask
    module = parse_mlir_file(tmp_path / "model.mlir")
    module.verify()
    assert not any(op.name == "builtin.unregistered" for op in module.walk())
    assert sum(op.name == "arith.mulf" for op in module.walk()) == int(training)
    returned = next(op for op in module.walk() if op.name == "func.return")
    assert returned.operands[2] is next(op for op in module.walk() if op.name == "func.func").body.block.args[0]


@pytest.mark.parametrize("options", [dict(probability=0.5), dict(private=False), dict(unknown=True)])
def test_stochastic_or_external_effects_are_not_erased(tmp_path, options):
    report = fixture(tmp_path, **options)
    original = (tmp_path / "model.mlir").read_bytes()
    assert specialize(tmp_path, report) is None
    assert (tmp_path / "model.mlir").read_bytes() == original
