# Scalar RISC-V full-call authoring

EL1 uses `raw_baseline_public_v0` and plain rv64gc/lp64d Spike at L2. The candidate
starts with an empty `submission/` and implements the public four-entrypoint backend
contract. Only executed submitted LLVM emission can earn a full-call pass, including
user results, post-call arguments, mutation, alias and layout metadata. Public and
hidden full-call passes are scored; no device lane or RTL claim applies.

Prepare a fresh retained Core ATen recipe with `lane_expectation: full_call`, explicit
public and hidden selections, and `host_guard: []`. Run `corpus derive`, select its
recipe, conformance_spec and synth_profile in Phase 0, execute the real Phase 0 runner,
then use `corpus prepare --generated-only` and `corpus inspect`. Bundles belong to that
release. An operator seals after reviewing the release and known reference failures.

Select an external support provider carrying this target contract through
`MERLIN_TARGET_PATH`. It supplies execution facts, never a candidate implementation.
The raw bundle exposes public ABI/call inputs and ordinary toolchains; Merlin compiler
sources, hidden inputs, private answers and support sources stay outside agent grants.
Continuous authoring uses 43200-second wall/round budgets, 900-second grading interval,
and two grading jobs. Spike is functional evidence only.
