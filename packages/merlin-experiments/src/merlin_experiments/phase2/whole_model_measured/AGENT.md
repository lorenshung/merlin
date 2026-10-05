# packages/merlin-experiments/src/merlin_experiments/phase2/whole_model_measured

This package owns the optional, installed whole-model measured experiment mode.
It prepares and resumes runs, builds exact candidate snapshots, schedules bounded
measurement jobs, and records evidence and agent feedback. Shared compiler,
runtime, and measurement primitives remain in `src/merlin/`.

Keep target machines, builders, input capsules, and instruction policies explicit
data selected by the experiment. Do not discover a source checkout or infer a
target from a simulator name. Candidate-facing feedback must not expose private
answers or host-only input snapshots.

Preserve the distinction between a functional-model screen, board measurements,
and certifying native evidence. A process exit, cached result, partial output,
or candidate-provided claim is not a whole-model verdict. Resume must verify
the frozen inputs and exact candidate/build/engine identities before reuse;
infrastructure failures must not be scored as candidate failures.

Tests use synthetic builders and process fixtures. Do not launch agents or
simulators while validating package structure or imports.
