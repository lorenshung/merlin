---
title: Component-driven compiler convergence and experiment isolation
kind: design
status: draft
owner: experiments
last_verified: 2026-10-08
related: [agentic_experiment_integrity, capsule_phase_split, perf_phase2_wiring, phase0_specification]
code_refs:
  - src/merlin/targetgen/compiler_library.py
  - src/merlin/targetgen/package_runtime.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_experiment.py
---

# Component-driven compiler convergence

The experiment asks whether a fresh target compiler can reach an independently
frozen handwritten reference's quality through generated component tests, with
substantially lower authoring and measurement cost. It uses reviewed shared
compiler infrastructure. That infrastructure's prior development history must
be disclosed; upstream publication alone never grants experimental access.

## Implemented boundary primitives

`CompilerLibraryContract` freezes exact reviewed Python modules, namespace
initializers, resources, direct shared-library imports and their bytes. Public
APIs are leaf modules; approving a module does not approve sibling or child
modules. Withheld evaluation identities remain forbidden. The host supplies the
contract and root explicitly to `integrity_scan`; a candidate manifest cannot
grant itself access. Legacy scans retain their existing input-grammar exception.

This is direct-import and byte admission, not proof of semantic generality,
dynamic dependency closure or complete interpreter isolation. A reviewer must
establish the public implementation's generality and dependencies. Runtime
isolation is required independently.

`materialize_component_view` copies explicit members into a fresh minimal view.
Its public manifest contains logical roles and content hashes, not original
private source paths. Only approved compiler members, contracts, generated
inputs and toolchain files are selected. Repository history, session state,
reports and test trees are excluded. Reverification rejects changed bytes,
symlinks, special files and added files or directories.

The Phase 0 generation digest binds an independently admitted receipt; a hash
alone does not prove that inputs were generated. Keep the existing Phase 0
derivation/evidence verifier authoritative.

`strict_tool_policy` constructs a networkless, clear-environment subprocess
boundary with explicit runtime-file mounts, a read-only minimal view and one
writable candidate. It does not mount a whole checkout, home or system tree.
`run_isolation_probe` refuses failed/unavailable execution. A successful probe
does not qualify a model-client authoring transport or establish that every
private surface was checked. A formal campaign needs actual sandbox probes,
fresh-session admission and independent protected-evaluation execution.

## Publication and experiment admission

Publish reusable generic compiler/runtime changes in the shared compiler;
publish capture fidelity changes in the importer; publish target instructions,
physical schedules and ABI code in target support. Preserve rejected policies
and their evidence without enabling them as defaults.

Then review an explicit experimental dependency subset. Do not expose the
handwritten target compiler, reference programs, validation captures, weights,
goldens, layer census, tuning transcripts or performance journey to authors.
Do not expose Git history or enable network retrieval of excluded publications.
Start fresh agents with no inherited investigation context.

Keep graders, reference engines, holdouts and private receipts outside agent
read/edit authority. Diagnostic output from those owners must not expose
expected values, private IR or workload-specific selection guidance. Prove the
boundary from inside the actual sandbox, including denied reads/imports,
escaping links and network retrieval. No unavailable probe is a pass.

## Phase 0 generates the experiment

Derive training, calibration and hidden tests from pinned hardware/software
contracts and independent derivation workloads before backend authoring. Use
the existing capsule generators and commit/reveal protocol. Do not create
handwritten workload-layer fixtures or fit generation weights to validation
layer frequencies.

Required families include contractions, convolution, batching, layout changes,
tails, resource limits, quantization, mixed precision, ordered reductions,
multi-consumer observation, nonlinear operations, producer-consumer composition,
immutable reuse, effects, ownership, lifetime, publication and fallback. Include
zero-MAC work with explicit falsifiable cost objectives.

A trusted auditor may inspect validation graphs after generation. Its findings
cannot revise the corpus, supply exact validation shapes to authors, or choose
optimization actions. Frozen source coverage and numerical qualification are
different obligations; a small coverage witness basis proves neither arithmetic
nor a complete compiler.

## Phase 1 establishes a functional compiler

Declare the supported domain and refuse unsupported cases. Join every source
compute/support operation to its lowering, placement, transfers, linked symbols
and executed artifacts. Include host nonlinear and support work in the census.

Compile and execute through normal installed APIs outside the source checkout.
Qualify numerical order, scales, effects, ownership, resources, complete writes,
host/device composition and final linking. Inspect all executable sections for
prohibited target instructions. Frozen public and hidden generated obligations
must pass without skipped members disappearing from the denominator.

Record effective ordered compiler stages and emission witnesses. Replacing a
transform must not discard a previously selected legalizer. A feature name or
constructor toggle is not evidence that the transformation reached LLVM/object
code. Generated tests establish a bounded supported domain, not correctness for
every future graph.

## Phase 2 uses complete components

Give authors separately reviewed shared-compiler and target-support edit
surfaces. Protect hardware facts, numerical gates and evaluation owners. Use
typed schedule/resource records, complete CCA comparison and explicit unknowns.
Expose actual emitted stages, correctness/refusal diagnostics, complete cost,
uncertainty, resource pressure and concrete editable owners.

Progress from static legality to functional tests, calibrated analytical
comparison, and small matching-configuration RTL measurements when ranking is
unresolved. Cache by complete dependency identity, reuse compilation, batch
independent candidates and stop rejected candidates early. Keep whole-model
analysis and simulation outside the component optimization loop.

Price preparation, device work, commands, transfers, CPU arithmetic, tables,
cache states, packing, readout, dispatch, refinement and fallback. Separate
preparation from invocation and warm from cold reuse. Calibrate independently
of validation workloads and reference totals. Instruction retirement is a work
feature, not hardware timing; one global instruction-to-cycle multiplier is
insufficient. Different configurations cannot share calibration silently.

## Final comparison and accounting

Freeze source, installed packages, runtime, toolchains and selected options
before private full-workload compilation and final hardware evaluation. Match
hardware, inputs, accuracy criteria and timer boundaries to the frozen reference.
Every workload must independently stay within five percent of reference cycles;
an aggregate improvement cannot hide one regressing member.

The convergence objective is at least twenty times lower Phase 1 plus Phase 2
wall time using a matched historical method. Report reasoning, compilation,
simulation, queue waits, token telemetry, candidate count and cache reuse.
Report reusable setup/calibration separately and also report total cost including
them. Missing historical records remain unknown. Fix seeds, budgets, stopping
and measurement policy before authoring.

`final_component_campaign_gate` checks this comparison arithmetic on already
verified observations. It does not authenticate hardware flags, arbitrary
receipt hashes or historical telemetry. Independent hardware/capture/accuracy
receipt admission and a formal campaign completion gate are still required.

Final holdout results are not tuning feedback within that campaign. Disclosing
failures for repair ends the campaign; later work records the exposure and a new
campaign. Existing frozen receipts are never rewritten to acquire stronger claims.

## Remaining implementation and qualification

These primitives are not a completed Phase 0-to-2 launch. Mandatory remaining
work includes the protected fresh authoring transport, complete approved shared
dependency selection, generated mixed-precision/frontier coverage, normal
compiler stage witnesses, independently calibrated component providers, and
content-bound final hardware admission. A campaign must remain unavailable
until those obligations and actual runtime isolation are established.

Keep a structured optimization inventory containing semantic applicability,
owner, source/package/object identity, independent cases, complete costs,
numerical results, positive/negative evidence, promotion state, missing phase
tooling and measured token attribution. Workload-specific inventory remains
excluded from the agent environment.
