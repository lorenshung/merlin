# Prepared positive softmax interval domain

`emit_source_attention_frontier(..., prepare_softmax_domain=True)` selects a
private positive-lane implementation. The default generated source is unchanged.
The word-space polynomial policy is currently refused in combination with this
option. Signed PV endpoint arithmetic remains checked.

The prepared domain requires the existing immutable RNE environment and
nontrapping, unobserved exception flags. It admits finite ordered active score
spans once before the source schedule. Inactive mask entries need no finite
value; malformed masks refuse. The source maximum is still validated by the
existing source replay and bound checks. Unsupported preparation delegates to
the original checked implementation before writing state or counters.

For the source-q polynomial policy, preparation bounds every original Horner
stage over the fractional domain [0,1], then bounds source q over [cutoff,0].
The maximum prepared source-q error budget bounds all local buckets; outward
binary64 and binary32 conversion covers the emitted enclosure. Positive finite
IEEE word monotonicity yields a uniform bound for the returned interval, not
only the underlying mathematical polynomial. Cutoff crossings include zero.
Point intervals use the unchanged original source evaluation. Invalid/nonfinite
polynomial results still refuse at the use site.

Preparation applies the original RNE lane additions, balanced tree, and chunk
FMA to this nonnegative upper value, checking every prefix for overflow. Short
lanes are padded with upper values for the proof only. Alpha is checked once per
row and chunk in [0,1]. Monotonicity then proves finite ordered runtime lane
endpoints without repeating public interval-constructor checks. Each runtime
source addition, multiplication, subtraction and FMA retains its original order.
The denominator operation itself remains checked.

The private two-field interval is not a public permission to omit checks. Its
consumers require the admitted source schedule, immutable arrays and plan,
stable rounding environment, and the derived finite maximum. This proof does
not authorize signed PV additions or arbitrary caller-created intervals.

Independent tests cover unrelated polynomial plans, cutoff/floor crossings,
underflow, signed zero, masks, invalid spans, overflow and non-RNE refusal.
Generated executor tests compare the original source oracle on ordinary and
strided inputs, all-masked rows, dirty workspace reuse, and unchanged public
output on refusal. A full original consumer/whole-model qualification and
complete cost measurement are separate requirements before performance use.
