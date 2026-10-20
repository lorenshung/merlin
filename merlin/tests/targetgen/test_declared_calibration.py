"""A declared observer domain grants no access to graded input values."""
import copy
import pytest


def test_program_qparams_and_integer_kernel_are_payload_independent():
    torch=pytest.importorskip('torch')
    from merlin.targetgen._recipe_quantizer import apply_recipe
    from merlin.capture.integerization import integerize_static_pt2e

    class Model(torch.nn.Module):
        def forward(self,x,y):return x @ y

    recipe=dict(schema='quant_recipe_v1',status='derived',families=['contraction'],
        weight=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-127,quant_max=127),
        activation=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-128,quant_max=127,mode='static'),
        calibration_domain=dict(schema='merlin.declared_observer_domain.v1',minimum=-1.,maximum=1.,evaluation_inputs='never_read'))
    graphs=[];qparams=[]
    class ForbiddenStream:
        def __iter__(self):raise AssertionError('evaluation calibration stream was inspected')
    for number in (0.0,1.e30):
        inputs=(torch.full((2,2),number),torch.full((2,2),number))
        model=apply_recipe(Model(),recipe,example_inputs=inputs,calibration_inputs=ForbiddenStream())
        stats=model._recipe_quantization_stats
        assert stats['framework_capture_policy']['calibration_source']=='declared_observer_domain'
        qparams.append([(str(n.target),n.args[1:]) for n in model.graph.nodes if 'quantized_decomposed' in str(n.target)])
        model,_=integerize_static_pt2e(model,inputs)
        graphs.append(model.code.encode())
    assert qparams[0]==qparams[1]
    assert graphs[0]==graphs[1]  # Complete executable integer graph, not a near-value comparison.
