# Integer-grid exact convolution reference

`targetgen.exact_quantized_reference._convolution` factors each rational source
tensor into an integer grid and a rational scale, using a common denominator
and the gcd of the resulting numerators. Weight grids are per output channel,
so unequal per-channel scales do not inflate all integer codes. This recovers
quantized codes from the values themselves; metadata is not trusted as an
arithmetic shortcut.

For each output, let K be input channels per group times kernel volume, and
X/W the maximum absolute integer grid codes. If K*X*W <= 2^63-1, every product
and every prefix sum fits int64, regardless of summation order. The factors
must separately fit int64, including when the other factor is zero. Otherwise
object matmul uses Python integers. Padding object zeros are explicitly Python
integers too: NumPy scalar zeros can otherwise coerce large products to int64.

Sliding window views and grouped integer matrix products implement the same
index map as the scalar oracle, including stride, padding and dilation. Only
after all MACs is the exact source scale times the exact channel weight scale
applied, followed by the exact rational bias. Thus distributivity proves
identity with every Fraction product and sum in `_convolution_fraction`.
There is no floating multiply, accumulation, scale product or extra rounding.

Tests compare with the independent scalar Fraction loop on randomized 1D/2D/3D
convolutions, groups, dilation, stride, asymmetric final quantizers, exact ties,
int64 boundaries, Python integers above 64 bits, zero factors, unrelated rational
denominators and extreme dyadic exponents. A complete 625-input small domain
is checked as well. The scalar path remains available as the independent oracle.

Paired qualification in the worktree environment, including integer-grid
construction and rational result construction:

| Source / weights | Nominal MACs | Scalar Fraction | Integer grid | Speedup |
|---|---:|---:|---:|---:|
| 1x16x16x16 / 32x16x3x3 | 1,179,648 | 6.04 s | 0.0132 s | 459x |
| 1x64x56x56 / 64x64x1x1 | 12,845,056 | 76.34 s | 0.4288 s | 178x |

Every rational output is identical. These are individual convolution reference
measurements, with other jobs sharing the machine, not whole-model timings.
Driver and raw measurements: `out/codex-dev/measure_reference.py` and
`out/codex-dev/reference-measurements.json` in the qualification worktree.
