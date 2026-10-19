# requant-A-v1: shared Merlin / Gemmini host boundary

The default is `clamp_mul_add_shift`, proved separately for every concrete
parameter class (including each channel and frozen bias). If no candidate passes,
use `threshold_table`. Invalid or tampered constants are rejected, never emitted.
The shared implementation is `merlin.runtime.requant_a`; the provider prototype
is `mlir_oot/core_aten/requant_a.py`. This contract lets both compiler lanes emit the same form
without depending on concurrent, unmerged changes.

All arithmetic parameters and bounds are decimal integer **strings** in JSON;
`S`, zero point, output bounds, counts and flags use JSON integers/booleans.
Rationals have positive denominators and reduced numerator/denominator pairs.
The canonical semantic record has these exact fields:

```
{
  "schema": "requant-A-v1",
  "algorithm": "clamp_mul_add_shift" | "threshold_table",
  "domain": {"min": "...", "max": "..."},
  "ratio": {"numerator": "...", "denominator": "..."},
  "offset": {"numerator": "...", "denominator": "..."},
  "zero_point": 0, "quant_min": -128, "quant_max": 127,
  "relu": false, "rounding": "half_to_even",
  "constants": {"L": "...", "H": "...", "M": "...", "B": "...",
                "S": 0, "out_clamp": false, "qlo": -128, "qhi": 127},
  "proof": {...}
}
```

For `threshold_table`, omit `constants`; provide `thresholds` (at most 255
nondecreasing decimal strings) and a `refusal` string. Threshold `j` is the first
integer `x` with output at least `quant_min+j+1`, clipped to `[min,max+1]`.
Duplicates are intentional. Evaluate `quant_min + upper_bound(thresholds,x)`.
No per-element rational multiply/divide is needed on the fallback path.

Semantics: `clamp(RNE(x*ratio+offset)+zero_point, quant_min,quant_max)`.
With `relu`, replace the real rational value by `max(0,value)` before RNE.
The fast form is `v=clamp(x,L,H); q=(v*M+B) arithmetic_shift_right S`.
If `out_clamp` is true, finish with `clamp(q,qlo,qhi)` (the reference solver's
second candidate family). Prefer the form without this extra output clamp.
A constant map may use `M=0,S=0,L=H=0,B=code`.

`M>=0`, `0<=S<=62`. `L,H,M,B`, the products at both band endpoints and both
product-plus-bias endpoints must fit signed int64. Check products separately:
cancellation by `B` does not license a wrapping intermediate. Clamp in the
original exact integer width **before** narrowing to int64. Emit no FP, rounding
correction, reciprocal division or second quantization.

Proof: both maps are monotone. Compute every exact RNE output transition with
rational arithmetic; test `t-1,t` inside the domain for all transitions, both
domain endpoints, and in-domain band endpoints. Agreement at both ends of each
constant segment proves agreement everywhere, including every ties-to-even
boundary. The proof record is:

```
{"method":"all_monotone_step_boundaries", "step_count":255,
 "checked_points":0, "checked_points_sha256":"...", "thresholds_sha256":"...",
 "monotone":true, "integer_arithmetic":"signed_i64" | "exact_integer_comparisons",
 "product_min":"...", "product_max":"...", "sum_min":"...", "sum_max":"..."}
```

Omit product/sum fields for threshold records. A hash is SHA-256 of the UTF-8
concatenation `decimal_integer + "\n"` for each value. Checked points are sorted
and deduplicated; thresholds preserve order and duplicates. Reconstruct and
validate the boundary proof when importing a certificate. Never trust a marker.

For multi-term sums, compute an exact integer normal form first. Over a common
denominator `D`, let the variable coefficients be `a[j]`, and `g=gcd(abs(a[j]))`
(or 1 when all are zero). Then `x=sum((a[j]/g)*acc[j])`, `ratio=g/D`,
`offset=constant/D`. Negative weights remain exact integer terms. Frozen bias is
in `offset`, never rounded into an integer seed. Bound each product and partial
sum from the recorded tile intervals; choose a sufficiently wide integer type.
Single-term positive factors reduce to the original accumulator without an
extra runtime multiply. Parameter broadcasting follows output tensor shape;
channel indices and zero points must use the same channel axis as the scales.

Gemmini records the semantic record plus `output`, `elements`, `output_indices` (`{start,stop,step}` or `{indices:[...]}`),
`normal_form_width` (signed formation bit width), `normal_form` (ordered tile/decimal-weight pairs) and `accumulator_bounds` under
`kernels[].meta.host_requantization_proofs`. These binding annotations do not
change the semantic record or its hashes. Per-channel classes with identical
parameters may share one proof and emitted body; independent constants never
inherit another channel's proof.


## Gemmini overlap contract

`kernels[].meta.readout_pipeline` records `status=enabled`, `buffers=2`, the
private int32 `source`, `slot_elements`, `original_elements`, and ordered `tiles`.
Each tile has `tile`, `slot` (alternating 0/1), logical min/max, element count,
`ready_after` and `host_after` instruction boundaries (counts, not ordinals).
The static order is:

```
issue tile 0 and full-i32 MVOUT 0; fence
issue tile 1 and full-i32 MVOUT 1
host requant tile 0
fence
issue tile 2 into slot 0
host requant tile 1
fence
...
host requant final tile after its fence
```

The first fence establishes readiness of slot 0. Future commands only write the
other slot; slot reuse follows the previous CPU consumer. The final epilogue is
fenced and drained. RoCC issue asm and fences retain their memory clobbers, so
LLVM cannot move CPU reads before the ready fence or after reuse. Original local
scratchpad/accumulator allocation and internal ordering fences remain intact.
Static hazard lint runs on the transformed stream as well.

Eligibility derives from byte addresses, buffer ownership and dependencies:
one quantized output; a last-produced private identity-layout int32 component;
all other rational components ready before the first tile; no device readers
of the reused buffer; full-width stores partition the exact output; no later
device side effects; and two slots fit within the original scratch extent.
Other components may include immutable input bias or already completed int32
sums, with their existing broadcast/permutation maps. Multi-component cases that
fail these conditions retain serial host execution and record the reason.

Partial `Readout` objects bind global output `indices` to `source_offsets`
(element offsets applied after any layout map). `parameters_compact=true` means
term factors, offset, zero point and quant bounds are scalar or indexed by the
partial index list; otherwise they broadcast to the logical output shape.
`axis` selects the channel dimension for one-dimensional qparam arrays, default
last dimension. Compact parameter slices prevent duplicating full tensor
metadata once per pipeline tile. These fields are execution/binding metadata;
the `requant-A-v1` semantic certificate is unchanged.

## Merlin LLVM binding

`merlin.exact_readout_proof` wraps `schema`, `method="requant_A"`, `axis=-1`
and `channels` (semantic records with channel and integer-normal-form bindings).
The full int32 storage interval is used, even when a kernel could prove a tighter
bound. A negative coefficient uses weight -1 and the negated int32 interval;
a zero coefficient uses weight 0, domain [0,0] and ratio 1.

An optional `frozen_bias` binding records its source, exact f32 `bit_patterns`
as unsigned decimal strings, and `guarded`. File-bound candidates also record
the argument, weight key and payload hash. `guarded=true` requires a bitwise
runtime check followed by the exact original body on mismatch. An unguarded
candidate must be an SSA constant. Imported bodies are reconstructed and checked
against these bindings, so changing a record, bias or emitted constant cannot
bypass validation by retaining a proof marker.
