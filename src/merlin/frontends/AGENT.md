# AGENT.md — merlin/python/merlin/frontends

## Purpose

Frontends that ingest external IR into the Merlin pipeline. Today: linalg-on-tensors MLIR as produced by **model2MLIR** (`/path/to/model2MLIR`) — smolVLA and the other VLA workloads.

## What belongs here

- `linalg_mlir.py` — xDSL parsing of linalg-on-tensors artifacts + matmul inventory (shapes, dtypes, weights resolved via the safetensors manifest, `prov.*` provenance).
  Its standard memref and bufferization dialect registrations admit prepared
  fresh borrowed writer wrappers without an unregistered custom-syntax fallback.
- `facts.py` — lifting the inventory to contract-level reuse facts, driving the core pipeline with real model shapes, residency-variant DSE measurement.
- `linalg_math_patterns.py` — source-only, closed unary f32 sine/cosine scalar-region checks;
  no host admission or numerical-equivalence claim. Scalar-literal power remains refused.
- `linalg_integer_reductions.py` — closed static i64 sum and masked prefix-sum source checks,
  with an explicit index-width premise; no host admission or linked-build verdict.
- `bucketize_source.py` — a source-only, trace-bound proof of the closed f32
  bucketize counting reduction. It does not grant host placement, prove sorted
  boundaries, or certify generated code and numerical equivalence.

## What does not belong here

- Model capture/conversion (that is model2MLIR's job — don't vendor it).
- Lowering or dialect definitions (`merlin/python/merlin/xdsl_dialects/`).

## Interfaces

- Input: `workloads/<model>/<model>.mlir` + `<model>.safetensors.manifest.json` from model2MLIR.
- Output: `MatmulRecord`/`WeightReuseFact` inventories; `LoweringResult` via the existing pipeline; `dse` IR + `dse_result`-shaped dicts.

## Invariants

- Parse the **full** model artifact, not `sections/*.mlir`: the model2MLIR section splitter currently emits use-before-def SSA references (invalid SSACFG IR; e.g. `%2034` in `sections/smolvla.model.mlir`). Fix upstream before relying on sections.
- `PAREN_RESULTS` exists because xDSL 0.65's linalg parser rejects `} -> (T1, T2)`; remove it once xDSL accepts the parenthesized multi-result form.
- The integer pipeline executes a layer's **i8 deployment GEMM** with the real (M,K,N); the capture dtype is preserved as provenance, never silently claimed as executed.

## Testing expectations

`merlin/python/tests/test_frontend_linalg.py` — synthetic-text tests run everywhere; smolVLA-artifact and spike tests auto-skip when the artifact/toolchain are absent.

## Notes for future agents

smolVLA reuse structure: capture unit = one denoise step; `select_action` runs N steps per tick, so every weight is `reused_across_invocations` — the repeated-RHS pattern with real shapes.
