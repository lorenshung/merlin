---
title: Prepared private endpoint DAG
kind: reference
status: current
owner: compiler
last_verified: 2026-10-07
related: []
code_refs:
  - src/merlin/llvmlower/prepared_endpoint_dag.py
---

# Prepared private endpoint DAG

`prepared_endpoint_dag.c_header` is an explicit, default-off producer/consumer
mechanism. It changes no numerical permission. It operates on finite ordered
intervals supplied by the existing policy, and makes no claim that a heuristic
interval is an exact enclosure of the original computation.

The producer copies complete `[part,row,column]` spans in ordinal order. In the
same pass it validates finite endpoints, candidate membership and derives a
maximum absolute endpoint word for each row. Input/output spans, metadata and
caller scratch are disjoint private owners. The producer context binds their
shape and a lexical epoch; partial failure never publishes a public result.

Before evaluating a row, admission follows the same nonnegative scale/add
sequence on its magnitude bound. RNE monotonicity and sign symmetry prove that
this recurrence bounds every signed intermediate and candidate. Every bound
prefix must be finite. The reciprocal interval is checked, and the final bound
must not exceed the largest finite BF16 value. Only then may the same ordered
source operations run without repeated interval constructors. The source
midpoint reciprocal, sign-selected products, BF16 conversion and final clamp
are unchanged. Unsupported range, epoch or environment retains checked code.

Any refinement invalidates that row **before** writing a partial or denominator.
The prepared path then refuses it for the remainder of the epoch; it does not
reuse a certificate across mutation. The original checked row path recomputes
its outputs. Source fallback and original final accuracy gates remain active.
No previously certified row may be mutated. No external alias may change a
producer span or metadata, and no asynchronous use may outlive the epoch.

Preparation/maxima, metadata initialization and stack lifetime are real costs.
The mechanism does not imply a hardware speedup. Quantizer validation remains
unchanged in the first source-bound experiment; it is not bypassed using an
untyped assertion about arbitrary arrays.
