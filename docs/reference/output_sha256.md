---
title: "Compact whole-output evidence"
kind: reference
status: current
owner: core
last_verified: 2026-10-06
related: [ordered_fma_certificates, quantized_host_optimizations]
code_refs: [src/merlin/llvmlower, merlin/runtime/c]
---

# Compact whole-output evidence

`spike_model.build_app(output_sha256=True, output_dump_cap=1)` prints the
existing first-value `OUT` record and a compact record:

```
OUT_SHA256 f32le <element-count> <byte-count> <64-lowercase-hex-digits>
```

The digest covers every element of the **first model output**, in contiguous
row-major order, encoded as raw little-endian IEEE float32 bits. It preserves
signed zero and NaN payloads. It does not summarize additional model outputs.
The byte count must equal four times the element count. Hashing occurs after
the forward interval's final cycle read, so hash work is excluded from the
reported forward cycles. Existing ARGMAX/SUM diagnostics remain unchanged.
The option defaults off and is included in build identity when enabled.

Use `merlin.runtime.output_digest.verify_output_sha256(text, reference)` to
check the digest and both counts against a validated float32 array. Record
reference file hash, originating build identity, target ELF hash and UART hash.
First establish the same artifact's native/Spike agreement and separately
record its comparison to the captured model oracle with explicit tolerances.
A digest match establishes byte equality to that reference; it cannot establish
model correctness by itself. Keep a full-output validated baseline as reference.

The actual C helper is tested against SHA-256 known vectors (empty, `abc`,
multi-block, one million `a` bytes), every relevant padding boundary, streaming
updates, and float bit patterns. Record parsing tests reject missing/duplicate
records, mismatched byte/element counts and a mismatched digest.
