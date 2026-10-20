"""Initialize static observers from a declared domain, never evaluation payloads."""


def initialize_observers(prepared, declaration, observer_type):
    import math
    import torch

    if declaration.get('schema') != 'merlin.declared_observer_domain.v1':
        raise ValueError('unknown declared calibration contract')
    lower, upper = declaration['minimum'], declaration['maximum']
    if not all(type(v) in (int,float) and math.isfinite(v) for v in (lower,upper)) or not lower < upper:
        raise ValueError('declared calibration endpoints must be finite and ordered')
    if declaration.get('evaluation_inputs') != 'never_read':
        raise ValueError('declared calibration must exclude evaluation inputs')
    count = 0
    for node in prepared.graph.nodes:
        if node.op != 'call_module':
            continue
        observer = prepared.get_submodule(node.target)
        if not isinstance(observer, observer_type):
            continue
        axis = getattr(observer, 'ch_axis', None)
        shape = [2]
        varying_axis = 0
        if axis is not None:
            metadata = node.args[0].meta.get('val')
            if metadata is None:
                raise ValueError('per-channel declared range needs static observer input metadata')
            axis = axis % len(metadata.shape)
            shape = [1] * max(2, len(metadata.shape))
            shape[axis] = int(metadata.shape[axis])
            varying_axis = next(i for i in range(len(shape)) if i != axis)
            shape[varying_axis] = 2
        sample_shape = [1] * len(shape)
        sample_shape[varying_axis] = 2
        sample = torch.tensor([lower,upper],dtype=torch.float32).reshape(sample_shape).expand(shape)
        observer(sample)
        if declaration.get('scale_policy') == 'enclosing_power_of_two':
            from types import MethodType

            original = observer.calculate_qparams

            def qparams(self, original=original):
                scales, zero = original()
                def enclose(value):
                    fraction, exponent = math.frexp(value)
                    return value if fraction == .5 else math.ldexp(1., exponent)
                rounded = torch.tensor([enclose(v) for v in scales.reshape(-1).tolist()],
                                       dtype=scales.dtype,device=scales.device).reshape(scales.shape)
                return rounded, zero

            observer.calculate_qparams = MethodType(qparams,observer)
        count += 1
    return count
