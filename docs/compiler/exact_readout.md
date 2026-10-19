# Exact readout lowering

The xDSL preprocessing stage `xdsl-exact-readouts` specializes the frontend's
exact readout scalar bodies before LLVM legalization. It checks the complete
scalar body, maps and iterators against the independent structural specification
in `exact_readout_source.py`. Changed or unsupported bodies retain the original
exact arithmetic. Provenance alone never authorizes a replacement. Three scalar trace
annotations (source IDs, origin IDs and trace role) are ignored in a cloned
comparison; arithmetic properties and every other attribute remain checked.

For a signed int32 accumulator and rational coefficient num/den, choose
M=ceil(abs(num)*2^s/den), E=M*den-abs(num)*2^s, A=2^31.
When A*E<2^s, floor(a*M/2^s) is the exact floor quotient for every 0<=a<=A:
the error is less than 1/den and a rational remainder is at most (den-1)/den.
When only A*E<den*2^s holds, the estimate is q or q+1. Compare its product
with den to the exact a*abs(num), and subtract one precisely when needed.
Both products are proved to fit uint128; narrower arithmetic is used only
when its full-domain bound fits. Multiplier and accumulator fit uint64.
LLVM can implement the wide product using RV64 mul/mulhu.

After recovering q, r=a*abs(num)-q*den is exact. Increment iff r>den-r,
or r=den-r and q is odd. Signed symmetry covers every negative tie.
Clamp before adding the zero point, in the shifted code domain, to avoid
addition overflow. All visible rounding/clamp boundaries and both int32
extremes are checked independently with Fraction at compile time. The tests
also exhaust a 65,537-value domain. The formal certificate covers all 2^32
int32 values, independently of calibration samples.

Certificates, bounds, magic constants, inequalities, boundary coverage and
fallback reasons are serialized in `merlin.exact_readout_proof` on the lowered
operation and retained in the preprocessing audit IR. Per-channel parameters
use indexed constant tables rather than one arithmetic body per channel.

Floating readouts require one IEEE rounding rather than byte requantization.
For unbiased dyadics, the normal/finite certificate proves that an exact u128
integer product can become a u64 value with a guard/sticky representation:
when the high word is nonzero, shift right 32 and set the low bit iff any
lost bit was nonzero. At least 32 significant bits remain, so all 24 output
bits, the guard and sticky bits survive. One u64-to-f32 RNE then makes the
only rounding decision. Multiplication by the appropriate power of two is
proved exact and normal/finite over the complete int32 domain. Boundary
checks cover powers of two and the switch between 64-bit representations.
This uses the ordinary LLVM/default IEEE RNE host floating environment.

Other floating readouts bound the exact dyadic product plus every finite f32
bias, find the leading bits using native-width limb searches, and construct
the IEEE result from the exact quotient and remainder. Nonfinite bias retains
the original assertion. Quantized readouts with runtime float bias and
unsupported rational widths retain the original exact implementation.

## Qualification

Isolated scalar host epilogues, at representative ResNet tensor sizes. These
are synthetic concrete fp32 scale classes, not a full-model speedup. Spike
minstret includes identical loop, indexing and call overhead. All 1,024 distinct
inputs per class are checked against independent Fraction arithmetic before
timing; old and new implementations use identical inputs. No Verilator run.

| Epilogue | Elements | Old instructions/element | New instructions/element |
|---|---:|---:|---:|
| conv_readout | 200,704 | 904.81 | 461.76 |
| conv_readout_unbiased | 200,704 | 185.11 | 32.02 |
| mm_readout | 1,000 | 177.29 | 35.68 |
| conv_requantize | 200,704 | 102.07 | 34.51 |
| mean_dim | 2,048 | 82.60 | 36.51 |
| add_Tensor | 802,816 | 69.59 | 22.53 |

The byte paths approach the requested ~20-instruction target; the general
floating path, especially runtime bias, remains above it. No ~20-instruction
claim is made for that path. The measurement driver, LLVM IR, linked ELF,
certificates and raw Spike output are in `out/codex-dev/measure-b9/` and
`out/codex-dev/measure_readout.py` in the qualification worktree.
