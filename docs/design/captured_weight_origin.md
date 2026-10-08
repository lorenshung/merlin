---
title: "Captured weight origin for integer contractions: an unlanded prototype"
kind: design
status: draft
owner: targetgen
last_verified: 2026-10-08
related: [capture_execution_attestation, quantized_affine_pair]
code_refs:
  - src/merlin/targetgen/application_inventory.py
  - src/merlin/targetgen/performance_basis.py
---

# Captured weight origin for integer contractions

A prototype (September 2026) tried to say, for each captured integer contraction, which
original model parameter its right-hand operand came from, and whether the frozen int8
buffer is a faithful quantization of that parameter. It was not ported to the current
compiler. This note keeps what it established and why it stopped.

## What was tried

Three layers, each narrower than the one before it:

1. **Captured buffer.** Follow the contraction's RHS through layout-only SSA operations
   (`linalg.transpose`, `tensor.collapse_shape`, `tensor.expand_shape`, at most 32 steps,
   every hop i8) back to a function argument, and require the receipt-bound model2MLIR
   argument manifest to name it as an int8 `param` or `buffer` of the same shape.
2. **Declared source identity.** Compose model2MLIR's producer-declared operation lineage
   and operand graphs to name the original floating-point parameter. This is the
   producer's declaration, not a numerical relation.
3. **Numerical lineage.** Replay the frozen int8 tensor from a pre-quantization float
   archive bound by a lineage receipt (`m2m.quant_weight_lineage.v1`) and byte-compare.

The evidence was attached to application-inventory rows (`weight_buffer_evidence`,
`source_weight_identity`, `weight_transform_gap`) and copied into the Phase 0
performance basis.

## What it established

Layers 1 and 2 worked on unit fixtures and on the development captures of the time:
the RHS path to a frozen int8 buffer was exact, and the declared parameter name was
recoverable. Layer 3 never ran on a real capture. The recorded `weight_transform_gap`
stated why, precisely: the frozen int8 tensor is present, but the effective
pre-quantization float tensor (after any Conv/BatchNorm fold) is not retained; PT2E's
scale and zero-point receipts are an aggregate list with no source or frozen tensor
identities; and no per-weight rounding or saturation formula is recorded. Pairing a
qparam row with a weight by list position is not a proof.

## Why it was not landed

- The numerical layer depends on a model2MLIR module (`m2m.capture.weight_lineage`)
  and receipt schema that model2MLIR never gained. Without it the prototype can only
  report the gap.
- Nothing decided anything from the first two layers. Their only consumers were a
  field copied into the performance basis and a Gemmini prefix-materialization probe;
  the exact-form match in Phase 2 already requires byte-bound source integerization
  (`capture_integerization`, `verified_static_integerization`) instead.
- A declared identity that cannot be checked numerically reads like coverage it does
  not provide.

## What would reopen it

model2MLIR would have to capture the effective float tensor and emit a direct
original-tensor to frozen-buffer relation, per-tensor qparams and versioned conversion
semantics. Merlin could then recompute the frozen bytes and compare them, and the
result could gate the exact-form match rather than sit beside it.
