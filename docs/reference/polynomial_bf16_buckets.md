---
title: Exact polynomial BF16 buckets
kind: reference
status: current
owner: core
last_verified: 2026-10-08
related: [lowering_pipeline]
code_refs: [src/merlin/llvmlower/polynomial_bf16_buckets.py]
---

# Exact polynomial BF16 buckets

This optional source representation derives ordered binary32 input intervals
whose original rounded polynomial values have one BF16 observation. It changes
no default preparation, numeric policy, workload gate or target implementation.

`prepare_polynomial_bf16_buckets` consumes the existing complete
`RoundedPolynomialMonotonicity` theorem, its exact eight source plan words and
source-header identity, and `SourceNumericContract` effects. The theorem is
supplied evidence, not inferred from the preparation evaluations. Its native
and target evaluator equivalence remains a caller obligation.

An independent integer evaluator implements binary32 RNE with gradual
underflow, separate multiply/subtract, finite floor, three source Horner FMAs,
separate subtraction, the final encoded FMA, signed integer conversion and
float bitcast. It retains cutoff branches and signed zeros. Dyadic subdivision
then derives a complete finite negative-cutoff through positive-zero partition.
Both endpoints of each cell are evaluated exactly; the complete rounded
monotonicity theorem covers every interior word. Exact BF16 ties are retained.
Quotas refuse incomplete tables before publication.

The emitted membership consumer admits the exact prepared plan, zero word
budget and RNE mode once in a private immutable source epoch. Unknown plans,
input domains, changed mode or cells spanning different BF16 observations use
the existing checked source. Effects, plan and memory must remain unchanged
through that epoch; a Boolean parameter does not prove these properties.
Finite input admission reads the binary32 representation under the existing
copy contract; it introduces no interposed classification call.

A hit provides the original BF16 probability and a conservative enclosure of
the **original unrounded f32 probability**. The denominator keeps its original
f32 operations. The enclosure can be wider than an exact endpoint evaluation;
this matters when a later quantizer needs a narrow denominator. Do not infer a
performance benefit from membership hit rate. Price original reconstruction,
metadata, consumer observations, table admission and all extra source replay.

Production selection must derive from the current typed source grammar and
effects. Shapes, target names, model names, source IDs and measured answers do
not choose these cells. Static table memory and cache cost are additional
resource obligations. No source word or intermediate floating observation may
be removed without the actual closed-use contract; an i8/scale consumer may
permit different unobserved BF16 carrier words after exact observation proof.

## Current experiment

The complete first 16-query, 12-head source callback preserves all 12,288
original i8 observations and 16 escaping BF16 scales. A table of 16,129 cells
uses 258,064 bytes. Returning its coarse f32 boxes regresses complete native
median time by 10.1796% and increases denominator source replay from two to
157 head rows. This candidate is rejected. The optional source API is qualified independently of this rejected policy.
No normal preparation hook selects bucket membership or coarse output boxes.
Automatic selection and successful complete producer/consumer cost remain
separate obligations; no whole-model or hardware speedup is claimed.
