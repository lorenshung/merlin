# Exact readout lowering

The xDSL preprocessing stage `xdsl-exact-readouts` specializes the frontend's
exact readout scalar bodies before LLVM legalization. It checks the complete
scalar body, maps and iterators against the independent structural specification
in `exact_readout_source.py`. Changed or unsupported bodies retain the original
exact arithmetic. Provenance alone never authorizes a replacement. Three scalar trace
annotations (source IDs, origin IDs and trace role) are ignored in a cloned
comparison; arithmetic properties and every other attribute remain checked.

Byte readouts default to the shared [requant-A-v1 contract](requant_A_contract.md).
The compiler solves an int64 clamp/multiply/add/arithmetic-shift map against
every exact rational output transition. It validates the integer immediately
below and at each transition, both domain endpoints, both band endpoints,
and separate int64 bounds for multiplication and addition. Both maps are
monotone; agreement at the ends of every constant segment proves equality
throughout the full reachable int32 interval. Zero point and frozen rational
bias participate in the exact boundaries, before solving the constants.

If no affine candidate passes, a binary search over at most 255 exact integer
thresholds implements the same map, including alternating even/odd ties.
Negative coefficients normalize the accumulator sign in int64; zero coefficients
become constant maps. Per-channel parameters use indexed constant tables and
independent certificates, with identical classes shared. No failed constants
are emitted, and no reciprocal approximation is used.

Certificates are serialized in `merlin.exact_readout_proof`. Imported A bodies
are rebuilt from the exact source parameters and compared structurally, including
all constants, channel tables and guards. The certificate is revalidated as
well. Region-offload catalogs carrying the shared schema receive the same
boundary validation when Merlin accepts their link artifacts.

A finite frozen bias may come from an SSA constant or the capture's parameter
file and manifest. File-bound parameters remain callable operands: a per-element
bitwise guard selects the fast candidate only when the runtime bias matches.
Otherwise the original exact scalar body runs. This preserves parameter updates
and does not assume that a capture makes a function argument immutable.

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

Frozen floating biases use their actual dyadic grid (zero bias also enables
the normal/finite specialization). Other floating readouts bound the exact
dyadic product plus every finite f32 bias, find the leading bits using native-width limb searches, and construct
the IEEE result from the exact quotient and remainder. Nonfinite bias retains
the original assertion. Quantized readouts with runtime float bias and
unsupported rational widths retain the original exact implementation.

## Earlier B9 qualification (before requant-A)

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

## Scheduling and remaining frontend boundaries

The provider region-offload route invokes its fence-checked two-slot readout
schedule before emitting kernels. Its shared certificate and pipeline metadata
are retained in the linked catalog; see the overlap contract. A retained-source
integer-output call has no tile-ready callback ABI, so its separate host loop
cannot overlap that call without a provider/compiler ABI extension.

An observed f32 readout still denotes one exact IEEE rounding. It has far more
than 255 steps, and cannot honestly claim the byte requant-A proof. A later
standalone f32-to-byte quantizer retains its exact ordered-IEEE threshold
search. Frontend fusion must preserve or explicitly remove an IEEE boundary
according to the source semantic contract, rather than silently changing double
rounding. The same applies to a runtime floating bias without frozen candidates:
it retains the original exact implementation. Qualification gaps and minimal
frontend requirements are recorded in the worktree's `out/codex-dev/gaps.md`.

## Requant-A qualification

Spike scalar epilogues, identical inputs checked against independent exact
Fraction arithmetic before timing. The biased floating fixture uses an SSA
frozen bias. Counts include identical scalar loop and call overhead.

| Epilogue | Original source instructions/element | Requant-A / exact IEEE instructions/element |
|---|---:|---:|
| conv_readout | 904.81 | 131.52 |
| conv_readout_unbiased | 185.11 | 32.02 |
| mm_readout | 177.29 | 35.68 |
| conv_requantize | 102.07 | 22.98 |
| mean_dim | 82.60 | 24.98 |
| add_Tensor | 69.59 | 22.99 |

The byte kernels use the A map; the floating kernels retain one exact IEEE
rounding. These scalar counts do not imply every floating boundary meets the
~20-instruction byte target. The qualification lane passed all 193 cases, with
44 cases on device and all 38 required support cases on device. Unit tests
cover random/adversarial ratios, alternating ties, negative and zero factors,
per-channel frozen bias, parameter mutation, imported-body tampering and
provider certificate round trips.

The full frozen ResNet-50 model was rebuilt twice against the same provider
snapshot, once with Merlin 101f4aaa8 and once with the A lowering. Both passed
Spike byte-for-byte against the model golden (FNV `f31781bd5dafe1ad`, top-1 850).
The host profiler uses `minstret` through a local runtime overlay; its windows
include identical mark/Gantt overhead. Provider overlap is enabled in both
builds, isolating the LLVM lowering change.

| ResNet readouts | Elements | Before instructions | After instructions | Before / element | After / element |
|---|---:|---:|---:|---:|---:|
| i8 | 3,286,528 | 1,594,495,876 | 102,583,958 | 485.16 | 31.21 |
| f32 | 7,828,456 | 3,707,767,523 | 1,033,793,006 | 473.63 | 132.06 |
| all | 11,114,984 | 5,302,263,399 | 1,136,376,964 | 477.04 | 102.24 |

Combined host readout instructions fall 4.67x. Total profiled top-level
instructions fall from 6,894,943,297 to 2,729,036,167 (2.53x). The model's 32
byte readouts all carry A proofs; its 22 observed floating readouts carry exact
frozen-dyadic IEEE proofs. The guarded per-channel byte average of 31.21 remains
above the approximately 20-instruction target. These are Spike instruction
counts, not RTL cycle measurements or a wall-clock speedup claim.

The provider catalog has 34 eligible pipelined kernels out of 61 distinct
kernels, spanning 2,206 readout tiles. Merlin revalidated the two shared semantic
classes across that catalog. Retained int32 outputs continue to use the serial
pointer ABI described above. No Verilator was run. Raw profiles, model outputs,
source snapshot hashes and the full results are in
`out/codex-dev/requant-a/` in the qualification worktree.
