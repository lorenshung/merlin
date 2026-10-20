---
title: General native dtype component proofs
kind: design
status: current
owner: core
last_verified: 2026-10-20
related: [host_cpu_arithmetic, core_aten_bitwise_movement]
code_refs: [src/merlin/targetgen/native_ieee_certificate.py, src/merlin/targetgen/native_typed_semantics.py, src/merlin/llvmlower/ieee_dyadic_ir.py, src/merlin/llvmlower/significand_products.py, src/merlin/llvmlower/integer_division_components.py]
---

# General native dtype component proofs

These are full storage-domain proofs. Tensor examples, eager outputs for a
captured case, zero digits observed in data and input hashes grant no admission.
The CPU reference profile is the actual selected AVX2 PyTorch library, recorded
by content hash. The compiler retains source index/thread-slice selectors and
every typed rounding boundary. Unknown primitives and aggregate NaN/order
semantics refuse. This document supplies no target instruction or cost facts.

Numerical proofs do not override the native-unit construction policy. A parallel
product of two runtime payloads remains host, including complex component
products. The significand-product producer requires a compile-time operand.
Its multiplication theorem serves affine scalar/constant forms only.

## Storage conversions

A finite binary word is exactly `(-1)^s m 2^e`, with the implicit bit present
only for a normal exponent field. The integer decoder returns that signed
significand and its runtime exponent. A product of two binary32 significands
has magnitude below `2^48`; binary16 and bfloat16 are smaller. A bounded
significand never needs a data-dependent limb plan.

For conversion, normalize the integer's leading bit. Discard the larger of
the target-significand excess and the target subnormal-quantum requirement.
Integer quotient/remainder decide ties to even. Renormalize a carry, encode
gradual underflow, and saturate exponent overflow to signed infinity. If more
than 61 bits must be dropped from a significand below `2^60`, its magnitude
is strictly less than half a target quantum. Clamp that otherwise out-of-range
shift and return signed zero. All executed shifts are in range. Negative zero
uses the source sign bit independently of the integer coefficient zero.

NaNs and infinities use integer bit readout. Widening bfloat16 to binary32
preserves even signaling payloads. Widening half quiets the payload; dense-copy
scalar tails instead select the positive all-payload word. Half narrowing keeps
the high payload/sign and sets the quiet bit. Bfloat16 tensor-copy narrowing
selects all ones, whereas its scalar opmath constructor selects positive quiet
NaN. These are distinct certified forms. Original source guards determine the
applicable path; no runtime data selects placement.

`native_ieee_certificate --write ...` exhausts all 65,536 source storage words
for the widening forms against scalar and vector eager paths. For binary32
narrowing it enumerates every destination rounding-cell transition and the
adjacent binary32 words, both signs, the normal/subnormal seam, zero and overflow,
and every retained-payload/sign class with discarded-bit boundary classes.
The finite preimage of each destination value is an interval: the exact RNE
midpoint and tie parity determine both endpoints. This interval partition proves
the unsampled interior, rather than extrapolating from random examples.
NaN classes are bit predicates, independent of finite rounding.

Each check has three independent representations: the unbounded integer
specification, CPU eager storage bits, and the actual emitted conversion IR.
The last is translated structurally to native unsigned-integer C, including
SCF branches, and executed on the complete domain/partition. Certificates record
version, form key, source hashes, executable IR hash and output hash. Changed
sources, library bytes, malformed keys or inadequate precision refuse reuse.
Tests deliberately corrupt NaN quieting and midpoint rounding and require refusal.

The CPU loop certificate observes vector widths and the parallel grain from the
installed headers, and verifies the vector loop's boundary/update form. Declared
shape and stride metadata then select its scalar/vector readout. Dense physical
orders, rank-one strides and fully broadcast layouts below that grain have closed
profiles; unknown compound layouts stay host. No tensor payload is inspected.
Unary storage and copy profiles require one original tensor argument. Additional
destination tensors can change iteration order even when functional capture drops
them from the arithmetic SSA. Such contexts remain host. Binary profiles require
the complete original tensor-layout roster and compatible iteration profiles.
Binary64 arithmetic and conversions are outside these certificates. In particular,
a binary64-to-reduced conversion must not silently assume the CPU omits an
intermediate binary32 rounding. Existing bit-preserving storage proofs are separate.
Uniform source-literal expressions may fold with exact rational arithmetic and
RNE at each original typed boundary, including zero signs. This is ordinary
compile-time constant propagation, independent of runtime input values.

## Opmath and each typed arithmetic boundary

An unflagged IEEE binary32 add, subtraction, product or fused multiply-add on
finite operands returns RNE of the exact mathematical operation, including
gradual underflow and overflow to infinity. Exact integer components followed
by integer RNE therefore implement that boundary. A fused operation has one
rounding; separate operations keep their separate source roundings. No compiler
reassociation is licensed. Integer conversion preceding true division is also
retained, so a large integer is rounded before the division exactly as in opmath.

For a *single* operation on same-storage finite operands, double rounding is
harmless at intermediate precision at least `2p+2`. Here half has `p=11`,
bfloat16 has `p=8`, and binary32 has 24 significant bits. The normal-range
result follows the midpoint separation argument in
[Figueroa, chapter 6](https://cs.nyu.edu/media/publications/figueroa_sam.pdf).
The bound covers addition/subtraction, multiplication, division and square root;
it does not apply to arbitrary wider scalar coefficients or a chain of operations.

The exponent edges need separate arguments. Half operands' products and
quotients stay within the normal binary32 exponent range until well below a
half rounding transition. For bfloat16, a subnormal sum/difference is an exact
multiple of its input quantum and is exactly representable in binary32.
A product near a bfloat16 subnormal midpoint has at most 16 significand bits,
which binary32 represents exactly there. Division near the smallest midpoint
has at least 16 effective binary32 bits, meeting the `2p` division bound;
smaller values round to zero in either route. At the overflow edge, the wider
overflow threshold lies above the narrower format's last finite rounding cell.
These edge arguments concern finite numbers only. The implementation preserves
opmath boundaries even where the single-operation corollary would remove one.

Zero signs follow IEEE RNE: exact additive cancellation is positive zero,
except addition of two negative zeros (or the equivalent subtraction signs).
Product and quotient signs are XOR, including zero/underflow. Specials do not
pass through the finite integer producer: replace them by zero in packing and
overlay only the source element's exact dependence set. Input NaN priority,
quieting, invalid operations and scalar/vector alternatives remain the original
explicit source bit selections. Arithmetic NaN payloads are never delegated to
the accelerator or inferred from one observed output. A reduction without a
general CPU order/payload theorem remains host for its entire input domain.

## Integer division

For a fixed positive divisor `d`, choose `s=w+ceil(log2(d))` and
`M=ceil(2^s/d)`. Then `0 <= Md-2^s < d`. For every `0 <= n < 2^w`,
the error of `nM/2^s` relative to `n/d` is in `[0,1)`. Thus the provisional
quotient is either the exact floor quotient or one too high. The device computes
the exact reciprocal product with full dtype-derived limbs. Compare `q*d` to
`n` and decrement once when needed. Sign reconstruction gives truncation;
nonzero remainder and differing signs give the extra floor decrement. Remainder
is `x-q*d`. Every intermediate is widened before arithmetic; final storage wrap
is explicit. The checker exhausts the entire signed-byte numerator/divisor domain
and checks the universal error inequalities for wider dtypes.

Divisor zero is an error domain rather than a numerical value theorem. Signed
`INT_MIN/-1` can trap in the reference CPU kernel at native widths. Runtime
divisors or trapping literal domains require a declared exception/range contract;
they stay host. The signed-byte promoted case is representable in its wider
compute type and wraps only at the final storage boundary. Floating true division
retains explicit integer-to-float promotion and every following typed RNE.
For a captured floor quotient, the certificate separately checks the singleton
`INT_MIN/-1` overflow class at each native width. Where eager returns wrapped
`INT_MIN`, the widened exact quotient followed by storage wrap covers that class;
the universal quotient argument covers all remaining numerators. This exception
does not license a truncating source that traps. A missing or changed exception
certificate refuses the whole domain.

## Complex component composition

A complex64 source is an explicit pair of binary32 component programs. Apply
the preceding theorem to each typed real primitive and retain its source
recombination, fused/separate choices and index/path guards. This proves a closed
pointwise component program, not an algebraic rewrite of arbitrary complex
arithmetic. In particular, replacing multiple typed operations by one exact
Gaussian expression would change rounding and is forbidden. Complex contractions
retain the aggregate order/payload refusal until separately proved.

## Integer aggregate and carrier closure

A closed integer program uses the storage/modular theorem independently of any
IEEE certificate. A positive literal signed divisor excludes both division by
zero and signed minimum divided by minus one. Exact limb accumulation first
reconstructs the original storage ring; its signed quotient then truncates toward
zero. Moving that quotient before the storage wrap would change the program and
is forbidden. Named additive reductions expose their dimension maps to the same
limb sum producer as generic reductions. No floating proof is needed.

A compile-time integer coefficient too wide for one bounded accumulator product
is expanded into signed radix-256 literal planes. Each coefficient-plane/input-
limb product fits the checked full signed-byte product domain, and the host
recombines modulo the original result width. Terms whose exponent reaches that
width vanish in the ring. Source constants, not runtime values, choose these planes.

Literal-zero elimination is optional. If it would erase the complete device
schedule, the compiler retains the original exact zero linear producers and
unchanged typed/sign readout. In particular, multiplying a promoted signed integer
by floating zero must preserve the input-dependent zero sign. It is neither an
unsigned constant fill nor permission to drop the promotion boundary.

Payload tracing recognizes a defined positive-literal quotient as arithmetic:
it produces new values, propagates UNKNOWN from every input, and cannot certify
an opaque carrier. This provenance rule grants no numerical or schedule proof;
the separate modular/division gate remains mandatory. Boolean union stays a
construction exclusion, including named reductions. Boolean storage carriers
remain ordinary bitwise movement.

IEEE float-to-float RNE narrowing is an explicit storage conversion. An affine
addend before that boundary is not itself an offset-rounding implementation.
The detector therefore does not treat typed narrowing as float-to-integer
rounding. Integer rounding after a bias, magic-bias add/subtract constructions,
and unknown storage forms retain their independent exclusions and proof gates.
