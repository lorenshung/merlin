# Phase 0 (functional, all native ops)

The deterministic `merlin.targetgen.functional_truth.expectation` is vendored from
`/scratch/loren/core-aten-snap/phase0-release/truth/truth_rules.py`, including the
2026-10-20 forced-zero-division ruling, scalar threshold promotion and quantized
float widening. Reporting rosters, payload checking and run13 placement I/O are
excluded. The rules inspect schema, scalar arguments, dtype and shapes only.
The specification remains primitive families and constraints; operation dispatch
in the semantic classifier implements the user-reviewed truth derivations.

Strict Phase 0 commits device/must-accelerate, no-work, or host-guard lanes before
compiler admission and fails mismatches in either direction. A correct result
on the wrong lane does not satisfy the commitment.

Future formal grading can represent the operator's deferred private full-model
gate with `--defer-private-full-models REASON`. This records deferred, not passed;
public/hidden obligations remain unchanged. Installing this option and the Codex
agent/VCS timing patches does not execute experiment Phase 1.
