"""Ineligible and empty sources remain clean host candidates."""

import pytest
from targetgen_fixtures import run_m2m_script as _torch

from merlin.targetgen.capsule_source import NoEligibleCaptureOperations, _require_integer_contraction


def test_plain_module_is_not_an_integerizable_pt2e_graph():
    _torch('''
import torch
from merlin.capture.integerization import integerize_static_pt2e
class Model(torch.nn.Module):
 def forward(self,x): return x
try:
 integerize_static_pt2e(Model(),(torch.ones(2,dtype=torch.int8),))
except ValueError as exc:
 assert "converted FX GraphModule" in str(exc)
else: raise AssertionError("plain module was accepted as PT2E")
''')


def test_folded_readout_without_dequantize_has_typed_refusal():
    _torch('''
import torch
from merlin.targetgen._recipe_quantizer import restore_floating_readout_widths, NoEligibleRecipeOperations
graph=torch.fx.Graph()
source=graph.placeholder("x")
output=graph.call_function(torch.ops.aten._to_copy.default,(source,),{"dtype":torch.float64})
output.meta["merlin_readout_width"]=True
graph.output(output)
model=torch.fx.GraphModule({},graph)
try: restore_floating_readout_widths(model)
except NoEligibleRecipeOperations as exc: assert "direct dequantization" in str(exc)
else: raise AssertionError("unproved dequantization was accepted")
''')


def test_agreement_records_empty_samples_without_reducing_them():
    _torch('''
import torch
from merlin.targetgen._recipe_quantizer import agreement
identity=lambda x:x
empty=torch.empty(2,0,3)
assert agreement(identity,identity,[(empty,)]) == {"samples":1,"empty_samples":1}
mixed=agreement(identity,identity,[(empty,),(torch.tensor([1.,2.]),)])
assert mixed["samples"] == 2 and mixed["empty_samples"] == 1
assert mixed["max_abs_error"] == 0 and mixed["argmax_agreement"] == 1
try:
 agreement(identity,lambda x:torch.ones(1),[(empty,)])
except ValueError as exc: assert "shapes" in str(exc)
else: raise AssertionError("shape mismatch was ignored")
''')


def test_constant_integer_program_without_quantize_has_typed_refusal():
    with pytest.raises(NoEligibleCaptureOperations, match="no int8 quantize"):
        _require_integer_contraction(
            "builtin.module {}",
            scheme="int8_static_act_int8_weight",
            recipe=None,
            meta={"integerization_receipt": {
                "integer_program": "non_contraction",
                "non_contraction_annotations": {
                    "annotated_sums": 0, "annotated_means": 0,
                    "annotated_affine_maps": 0, "annotated_movements": 1,
                },
                "quantized_contractions_seen": 0,
                "quantized_contractions_integerized": 0,
                "quantized_contractions_remaining": 0,
                "quantized_by_kind": {}, "refusals": [],
            }},
        )
