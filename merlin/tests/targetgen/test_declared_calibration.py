"""A declared observer domain grants no access to graded input values."""
import copy
import pytest


def test_program_qparams_and_integer_kernel_are_payload_independent():
    torch=pytest.importorskip('torch')
    from merlin.targetgen._recipe_quantizer import apply_recipe
    from merlin.capture.integerization import integerize_static_pt2e

    class Model(torch.nn.Module):
        def forward(self,x,y):
            return x @ y

    recipe=dict(schema='quant_recipe_v1',status='derived',families=['contraction'],
        weight=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-127,quant_max=127),
        activation=dict(dtype='int8',granularity='tensor',symmetric=True,quant_min=-128,quant_max=127,mode='static'),
        quantization_nonfinite='byte_storage_zero_nan_saturating_infinity',
        calibration_domain=dict(schema='merlin.declared_observer_domain.v1',minimum=-1.,maximum=1.,evaluation_inputs='never_read',scale_policy='enclosing_power_of_two'))
    graphs=[];qparams=[]
    class ForbiddenStream:
        def __iter__(self):raise AssertionError('evaluation calibration stream was inspected')
    for number in (0.0,1.e30,float("inf"),float("nan")):
        inputs=(torch.full((2,2),number),torch.full((2,2),number))
        model=apply_recipe(Model(),recipe,example_inputs=inputs,calibration_inputs=ForbiddenStream())
        stats=model._recipe_quantization_stats
        assert stats['framework_capture_policy']['calibration_source']=='declared_observer_domain'
        qparams.append([(str(n.target),n.args[1:]) for n in model.graph.nodes if 'quantized_decomposed' in str(n.target)])
        from merlin.targetgen.exact_quantized_reference import run
        independent=run(model,inputs).output
        model,_=integerize_static_pt2e(model,inputs)
        actual=model(*inputs)
        torch.utils._pytree.tree_map(lambda a,b: torch.testing.assert_close(a,b,rtol=0,atol=0), actual, independent)
        graphs.append(model.code.encode())
    assert all(params==qparams[0] for params in qparams)
    assert all(graph==graphs[0] for graph in graphs)  # Complete executable integer graph, not a near-value comparison.


@pytest.mark.parametrize('token,low,high',[('int8',-128,127),('uint8',0,255),('int8',-8,7)])
def test_declared_nonfinite_byte_boundary_matches_every_half_nan_word(token,low,high):
    torch=pytest.importorskip('torch')
    import numpy as np
    from merlin.targetgen.exact_arithmetic import quantize
    words=np.arange(65536,dtype=np.uint16)
    values=torch.from_numpy(words.view(np.float16).copy()).float()
    special=values[~torch.isfinite(values)]
    import torch.ao.quantization.fx._decomposed
    dtype=getattr(torch,token)
    expected=torch.ops.quantized_decomposed.quantize_per_tensor.default(special,0.015625,0,low,high,dtype)
    got=quantize(special,0.015625,0,low,high,nonfinite='byte_storage_zero_nan_saturating_infinity')
    assert got.tolist()==expected.tolist()
    with pytest.raises(ValueError,match='finite rational'):
        quantize(special,0.015625,0,low,high)


def test_rational_aggregate_can_feed_a_declared_nan_selector():
    torch=pytest.importorskip('torch')
    import torch.ao.quantization.fx._decomposed
    from merlin.targetgen.exact_quantized_reference import run
    graph=torch.fx.Graph()
    source=graph.placeholder('x')
    quant=graph.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default,(source,.015625,0,-128,127,torch.int8))
    real=graph.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default,(quant,.015625,0,-128,127,torch.int8))
    pooled=graph.call_function(torch.ops.aten.avg_pool2d.default,(real,[2,2]))
    mask=graph.call_function(torch.ops.aten.isnan.default,(pooled,))
    fill=graph.call_function(torch.ops.aten.full_like.default,(pooled,0.))
    selected=graph.call_function(torch.ops.aten.where.self,(mask,fill,pooled))
    quant=graph.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default,(selected,.015625,0,-128,127,torch.int8))
    result=graph.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default,(quant,.015625,0,-128,127,torch.int8))
    graph.output(result)
    model=torch.fx.GraphModule(torch.nn.Module(),graph)
    model._recipe_quantization_stats={'framework_capture_policy':{'quantization_nonfinite':'byte_storage_zero_nan_saturating_infinity'}}
    for value in (0.,1.e30,float('inf'),float('nan')):
        inputs=(torch.full((1,1,2,2),value),)
        expected=run(model,inputs).output
        assert torch.equal(expected.view(torch.uint8),model(*inputs).view(torch.uint8))


@pytest.mark.parametrize('value',[0.,float('nan'),float('inf'),-float('inf')])
def test_pool_adjoint_reads_only_primal_shape(value):
    torch=pytest.importorskip('torch')
    import numpy as np
    from fractions import Fraction
    from merlin.targetgen.exact_arithmetic import ExactTensor
    from merlin.targetgen.exact_quantized_reference import evaluate_operation
    gradient=ExactTensor(np.full((1,1,2,2),Fraction(1,7),dtype=object),torch.float32)
    primal=torch.full((1,1,3,3),value)
    for target,args in [
        (torch.ops.aten._adaptive_avg_pool2d_backward.default,(gradient,primal)),
        (torch.ops.aten.avg_pool2d_backward.default,(gradient,primal,[2,2],[1,1],[0,0],False,True,None)),
    ]:
        actual=evaluate_operation(target,args,{})
        expected=np.array([[1,2,1],[2,4,2],[1,2,1]],dtype=object)*Fraction(1,28)
        assert np.array_equal(actual.values[0,0],expected)
