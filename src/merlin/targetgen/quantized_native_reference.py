"""Independent rational evaluation of original source cases and byte boundaries.

Never imports lowered components, provider schedules or simulator outputs.
"""

import importlib.util
import json
import sys
from fractions import Fraction
from pathlib import Path


def main():
    import numpy as np
    import torch

    from merlin.targetgen.core_aten_capture import loader_source
    from merlin.targetgen.exact_arithmetic import ExactTensor, quantize, round_binary
    from merlin.targetgen.exact_quantized_reference import convolution_backward, evaluate_operation

    directory = Path(sys.argv[1])
    request = json.loads((directory / "quantized-native-request.json").read_text())
    p = request["contract"]
    scale = Fraction(p["scale_numerator"], p["scale_denominator"])
    import merlin.targetgen.exact_quantized_reference as oracle

    original_rational = oracle.rational
    oracle.rational = lambda x: Gaussian.of(x) if isinstance(x, (Gaussian, complex)) else original_rational(x)
    outputs = {}
    for record in request["records"]:
        case = record["case"]
        loader = directory / "quantized-reference-source.py"
        loader.write_text(loader_source(case))
        spec = importlib.util.spec_from_file_location("quantized_reference_source", loader)
        source = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(source)
        _, inputs = source.get_model_and_inputs()
        exact = []
        for value in inputs:
            if value.is_complex():
                real = quantize(value.real, scale, nonfinite=p["nonfinite"]) * scale
                imag = quantize(value.imag, scale, nonfinite=p["nonfinite"]) * scale
                values = np.vectorize(Gaussian, otypes=[object])(real, imag)
            else:
                values = quantize(value, scale, nonfinite=p["nonfinite"]) * scale
            exact.append(ExactTensor(values, torch.float32))
        arguments = source._decode(source._DOCUMENT["arguments"], exact)
        # Dtypes govern only the final required dequantization, not accumulation.
        args, kwargs = arguments["args"], arguments["kwargs"]
        kwargs = dict(kwargs)
        if "dtype" in kwargs and kwargs["dtype"] is not None:
            kwargs["dtype"] = torch.float32
        op = source._op(case["overload"])
        if str(op) == "aten.convolution_backward.default":
            result = convolution_backward(args[:3], args, "source", [])
        else:
            result = evaluate_operation(op, args, kwargs)
        leaves = torch.utils._pytree.tree_leaves(result)
        if len(leaves) != len(record["output_indices"]):
            raise ValueError("quantized oracle output count differs")
        for leaf, index, abi in zip(leaves, record["output_indices"], record["output_abi"], strict=True):
            if not isinstance(leaf, ExactTensor):
                raise ValueError("non-rational source output")
            dtype = abi["dtype"]
            if dtype == "complex<f32>":
                real = np.vectorize(lambda v: Gaussian.of(v).real, otypes=[object])(leaf.values)
                imag = np.vectorize(lambda v: Gaussian.of(v).imag, otypes=[object])(leaf.values)
                values = np.stack([quantize(real, scale), quantize(imag, scale)], axis=-1) * scale
                dtype = "f32"
            else:
                values = quantize(leaf, scale) * scale
            if dtype not in {"f16", "bf16", "f32"}:
                raise ValueError("non-floating quantized output")
            rounded = [
                round_binary(v, 16 if dtype in {"f16", "bf16"} else 32, bfloat16=dtype == "bf16") for v in values.flat
            ]
            tensor = torch.tensor(
                rounded, dtype={"f16": torch.float16, "bf16": torch.bfloat16, "f32": torch.float32}[dtype]
            )
            outputs[str(index)] = bytes(tensor.contiguous().view(torch.uint8).tolist()).hex()
        # Nonmutating source arguments are also exported by the batch ABI.
        for index, original in zip(record.get("native_eager_indices", {}).get("posts", ()), inputs):
            outputs[str(index)] = bytes(original.detach().contiguous().reshape(-1).view(torch.uint8).tolist()).hex()
    (directory / "quantized-native-reference.json").write_text(
        json.dumps(dict(expected_output_bytes=outputs, engine="independent_source_exact_rational")) + "\n"
    )


class Gaussian:
    """Two rational components; no binary floating complex arithmetic."""

    def __init__(self, real=0, imag=0):
        self.real, self.imag = Fraction(real), Fraction(imag)

    @staticmethod
    def of(value):
        return (
            value
            if isinstance(value, Gaussian)
            else Gaussian(value.real, value.imag)
            if isinstance(value, complex)
            else Gaussian(value)
        )

    def __add__(self, other):
        other = self.of(other)
        return Gaussian(self.real + other.real, self.imag + other.imag)

    __radd__ = __add__

    def __neg__(self):
        return Gaussian(-self.real, -self.imag)

    def __sub__(self, other):
        return self + -self.of(other)

    def __rsub__(self, other):
        return self.of(other) + -self

    def __mul__(self, other):
        other = self.of(other)
        return Gaussian(
            self.real * other.real - self.imag * other.imag, self.real * other.imag + self.imag * other.real
        )

    __rmul__ = __mul__

    def __truediv__(self, other):
        other = self.of(other)
        d = other.real**2 + other.imag**2
        return Gaussian(
            (self.real * other.real + self.imag * other.imag) / d, (self.imag * other.real - self.real * other.imag) / d
        )
