---
title: Native full-value output readback
kind: reference
status: current
owner: targetgen
last_verified: 2026-10-08
related: [runtime, experiment_abi]
code_refs:
  - src/merlin/targetgen/contract/readback_policy.py
  - src/merlin/targetgen/contract/compile.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/native_memory_readback.py
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/native_output_readback.py
---

# Native full-value output readback

Readback is an invocation choice, not a candidate command-buffer field. Phase 1
accepts `--readback-policy out_b64_v1`, `out_bin_v1`, or `coherent_dump_v1`.
Omitting the flag preserves the existing provider behavior. A frozen run keeps
its selected policy; changing transport requires a new run/tooling identity.

| Policy | Output mechanism | Build receipt |
|---|---|---|
| `out_b64_v1` | Complete framed serial values | `merlin_readback_build_v1` |
| `out_bin_v1` | Complete framed binary values | `merlin_readback_build_v1` |
| `coherent_dump_v1` | Admitted output storage exported by the selected backend | `merlin_readback_build_v2` |

Coherent readback stages no serial codec. The OOT harness provider must explicitly
accept the policy and declare its memory export protocol. The trusted evaluator
binds each output's logical shape, physical strides and writable ELF storage
before launch. Current readers support fixed-address ELF64 little-endian
executables, coherent `GSIMDMP1` exports, and single-output Spike HTIF signatures
with exact `begin_signature`/`end_signature` aliases and byte granularity.
Missing provider support refuses; serial output is not a substitute.

The execution path requires normal simulator exit, exactly one `DONE`, no serial
output records, complete logical values, and unchanged command-buffer, caller,
provider, ELF, selected facts/FIRRTL and engine identities. Sources and engine
bytes are revalidated before output decoding. Truncated exports, unknown layouts,
backing-store-only dumps and mixed serial/memory results cannot pass.

`readback_build` records the selected build bytes; `readback_memory` records the
full-value admission and export artifact identity. The oracle's `memory_engine`
records the independently selected engine citation. Native model receipts label
the prepared command `native_base_command` when memory-export arguments are
separately carried in the closed readback request. Partial consoles remain
diagnostic artifacts, never completed numerical results.

Readback admission establishes transport and attribution, not numerical
correctness, accelerator placement, performance improvement, or universal model
coverage. The ordinary independent golden comparison, host-compute audit,
mandatory tiers and private full-model gates remain separate obligations.
