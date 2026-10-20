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


@pytest.mark.parametrize('mask',[(True,False,False),(False,True,False),(False,False,True)])
def test_convolution_adjoint_uses_only_selected_value_dependencies(mask):
    torch=pytest.importorskip('torch')
    import numpy as np
    from merlin.targetgen.exact_arithmetic import exact_array,ExactTensor
    from merlin.targetgen.exact_quantized_reference import convolution_backward
    gradient=ExactTensor(exact_array(torch.ones(1,1,2,2)),torch.float32)
    source=torch.full((1,1,3,3),1. if mask[1] else float('nan'))
    weight=torch.full((1,1,2,2),1. if mask[0] else float('inf'))
    args=(gradient,source,weight,[1],[1,1],[0,0],[1,1],False,[0,0],1,list(mask))
    actual=convolution_backward(args[:3],args,'adjoint',[])
    baseline=(gradient,torch.ones_like(source),torch.ones_like(weight),*args[3:])
    expected=convolution_backward(baseline[:3],baseline,'adjoint',[])
    for wanted,left,right in zip(mask,actual,expected,strict=True):
        if wanted:assert np.array_equal(left.values,right.values)
        else:assert left is None and right is None


@pytest.mark.parametrize('beta',[-2.,0.,2.])
@pytest.mark.parametrize('relu',[False,True])
def test_readout_special_overlay_is_only_the_broadcast_bias_dependence(beta,relu):
    torch=pytest.importorskip('torch')
    pytest.importorskip('m2m.capture.integer_readout')
    from fractions import Fraction
    from merlin.targetgen.declared_readout_specials import retain_special_readouts
    graph=torch.fx.Graph();acc=graph.placeholder('acc');bias=graph.placeholder('bias')
    result=graph.call_function(torch.ops.m2m_q.exact_readout.default,(acc,.015625,[.015625],1.,bias,beta,.015625,3,-128,127,relu,False))
    graph.output(result);model=torch.fx.GraphModule(torch.nn.Module(),graph)
    assert retain_special_readouts(model,'byte_storage_zero_nan_saturating_infinity')==1
    values=torch.tensor([[64,-64,0,128]],dtype=torch.int32)
    residual=torch.tensor([float('nan'),float('inf'),-float('inf'),1.25])
    got=model(values,residual)
    expected=[]
    for index,(integer,b) in enumerate(zip(values[0].tolist(),residual.tolist(),strict=True)):
        if beta and index<3:code=0 if index==0 else 127 if (index==1)==(beta>0) else max(-128,3) if relu else -128
        else:
            value=Fraction(integer,4096)+(Fraction(5,4)*Fraction(beta) if index==3 else 0)
            code=min(127,max(3 if relu else -128,round(value/Fraction(1,64))+3))
        expected.append(code)
    assert got.tolist()==[expected]
    original=values.clone();model(values,torch.zeros_like(residual));assert torch.equal(values,original)


def test_declared_special_bias_is_discarded_only_at_the_final_quantizer():
    torch=pytest.importorskip('torch')
    import torch.ao.quantization.fx._decomposed
    from merlin.targetgen.exact_quantized_reference import run
    graph=torch.fx.Graph();x=graph.placeholder('x');bias=graph.placeholder('bias')
    q=graph.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default,(x,.015625,0,-128,127,torch.int8))
    real=graph.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default,(q,.015625,0,-128,127,torch.int8))
    result=graph.call_function(torch.ops.aten.add.Tensor,(real,bias))
    result=graph.call_function(torch.ops.aten.relu.default,(result,))
    q=graph.call_function(torch.ops.quantized_decomposed.quantize_per_tensor.default,(result,.015625,0,-128,127,torch.int8))
    result=graph.call_function(torch.ops.quantized_decomposed.dequantize_per_tensor.default,(q,.015625,0,-128,127,torch.int8))
    graph.output(result);model=torch.fx.GraphModule(torch.nn.Module(),graph)
    model._recipe_quantization_stats={'framework_capture_policy':{'quantization_nonfinite':'byte_storage_zero_nan_saturating_infinity'}}
    inputs=(torch.ones(4),torch.tensor([float('nan'),float('inf'),-float('inf'),.25]))
    actual=run(model,inputs).output;expected=model(*inputs)
    assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
    # Removing the Q boundary must not manufacture an IEEE NaN payload claim.
    output=next(n for n in graph.nodes if n.op=='output');output.args=(next(n for n in graph.nodes if n.target==torch.ops.aten.relu.default),)
    graph.eliminate_dead_code();model.recompile()
    with pytest.raises(ValueError,match='final byte quantizer'):run(model,inputs)
