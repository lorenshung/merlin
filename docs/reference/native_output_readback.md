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
  - packages/merlin-experiments/src/merlin_experiments/phase1/feedback/native_packet_readback.py
  - src/merlin/runtime/out_packet.py
---

# Native full-value output readback

Readback is an invocation choice, not a candidate command-buffer field. Phase 1
accepts `--readback-policy out_b64_v1`, `out_bin_v1`, `coherent_dump_v1`, or
`coherent_packet_v1`.
Omitting the flag preserves the existing provider behavior. A frozen run keeps
its selected policy; changing transport requires a new run/tooling identity.

| Policy | Output mechanism | Build receipt |
|---|---|---|
| `out_b64_v1` | Complete framed serial values | `merlin_readback_build_v1` |
| `out_bin_v1` | Complete framed binary values | `merlin_readback_build_v1` |
| `coherent_dump_v1` | Admitted output storage exported by the selected backend | `merlin_readback_build_v2` |
| `coherent_packet_v1` | All actual logical values packed losslessly in admitted memory | `merlin_readback_build_v3` |

Coherent readback stages no serial codec. The OOT harness provider must explicitly
accept the policy and declare its memory export protocol. The trusted evaluator
binds each output's logical shape, physical strides and writable ELF storage
before launch. Current readers support fixed-address ELF64 little-endian
executables, coherent `GSIMDMP1` exports, and Spike HTIF signatures with exact
`begin_signature`/`end_signature` aliases and byte granularity. Multiple outputs
must exactly tile that signature window by their verified ELF addresses: unknown
gaps, overlaps or linker-order assumptions refuse. The reader splits the physical
bytes at those proven offsets and decodes every declared logical tensor. GSim
exports each output as a separate region, ordered by its verified address.
Missing provider support refuses; serial output is not a substitute.

The opt-in packet policy stages and pins the existing range helper, binary
word codec, and bounded memory sink. A selected provider must explicitly support
the policy; core does not infer support from the old physical-dump capability.
Every actual logical output word is packed, with runtime range narrowing only
when its value fits exactly. FP32 carries its unsigned source bits, including
signed zero and NaN payloads. No sampling, digest-only result or reference value
is substituted. Output padding is not included in the packet's evidence claim.

Before launch the trusted reader independently derives the complete arena bound
from the admitted output shapes and storage widths, and proves exact, nonoverlapping
writable ELF symbols for the arena, publication word and source outputs. Current
packet admission retains the physical reader's static non-scalar i8/i16/i32/i64/f32
scope. The selected native exporter reads the published length, exactly that many
bytes through live coherent memory, then rechecks the length before publishing
the complete `GSIMPKT1` envelope. All output frames, shapes, widths, checksums,
typed value bounds and wire closure must validate. Packet `DONE` is only a wire
terminator; it never substitutes for normal native exit and console `DONE`.
Spike keeps its complete raw output-only signature under this policy, so the
same linked program can be checked independently without the packet decoder.
This interface is opt-in: source and installed checks alone do not qualify a
provider/engine combination or establish a large-case speedup.

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

Single-output Spike bounds/value records retain their v1 schema. Multi-output
records use `htif_signature_bounds_v2` and `htif_signature_output_values_v2`,
binding the complete sized-symbol roster. Historical records are not restamped.

Readback admission establishes transport and attribution, not numerical
correctness, accelerator placement, performance improvement, or universal model
coverage. The ordinary independent golden comparison, host-compute audit,
mandatory tiers and private full-model gates remain separate obligations.
