---
title: Selectable placement profitability
kind: reference
status: current
owner: core
last_verified: 2026-10-18
code_refs: [src/merlin/targetgen/core_aten_profitability.py, src/merlin/targetgen/placement_reference.py, src/merlin/targetgen/placement_cost.py]
---

# Selectable placement profitability

The selected software specification's `placement_cost_model.default_policy` selects
`reference_scale` or `strict_per_shape`. An explicit `MERLIN_PLACEMENT_POLICY`
environment variable overrides that selection for a run. Unknown names fail rather
than silently choosing another policy. A declaration without `default_policy`
retains the legacy strict policy. An absent cost declaration retains existing
numerical preparation behavior.

`reference_scale` qualifies a semantic family at the declaration's reference
workload. It compares host scalar throughput and memcpy bandwidth with integer
MAC throughput, DMA bandwidth, command issue, synchronization/dispatch, and
amortized host preparation/readout. The specification declares all reference
sizes, element/row widths, throughput rates, and the scalar-work cutoff. These
are authored assumptions pending operator review, not timing measurements or
numerical proofs. Workloads use contraction depth, reduction-window work,
pointwise streams, or materialized movement bytes. They contain no overload list.
A source whose reachable tensor data work has at most one element remains host
according to the declared scalar cutoff; a reduction of multiple elements to one
scalar still qualifies its arithmetic family.
Empty boundaries retain `empty_tensor_no_device_work`.

Movement inspects typed index maps and slice strides. Contiguous selectors use
reference logical rows. Element gathers pay a full selected RTL memory row per
element; they do not inherit contiguous DMA throughput. Host selection work
counts affine binary operations; a strided merge also pays lower/upper bounds,
stride congruence, and base/update selection. Reversed innermost selectors can
therefore lose while a row copy or a different indexed merge wins. Outer-axis
nonunit slice strides are conservatively treated as gathers as well. Unsupported
or unavailable cost evidence retains host execution with a precise reason.

`strict_per_shape` estimates the actual candidate dimensions, traffic, integer
producer count, emitted F1 commands/fences, and full retained host work. It is
available for paper-model runs and unchanged by reference-scale placement.
Every prepared admission records `placement_cost.policy` and
`placement_cost.strict_policy`. The strict verdict is informational under the
reference policy. With no numerically admitted candidate, its status is
`not_applicable` and its reason records that refusal. Case records and their
numeric grading evidence retain the same admission document.

Neither policy admits partial computation as placement. Complete source producer
coverage, byte-selector lineage, native result-bit gates, exact quantized
references, resource proofs, and full integer readout obligations remain in
force. A conversion before host transcendental arithmetic cannot qualify its
operation. A cost rejection restores original program, inputs, and result ABI.
Reference-scale profitability makes no per-case latency claim; use the strict
observation and independent timing measurements for that purpose.

The reference declaration uses a common producer/traffic profile for each
family. Actual dtype-specific limb multiplicity, integer widths, and complete
retained epilogue instruction counts belong to the strict observation. A
profitable reference family is a coverage-policy decision, not an assertion
that every instantiated dtype or sample runs faster.

Unqualified source comparisons and maxima/minima after a conversion also fail
complete-operation coverage. Generated Boolean truth decoding and closed
sign/clamp readouts carry their own exact host arithmetic proof markers; these
readouts do not grant permission to retain arbitrary source predicates.
