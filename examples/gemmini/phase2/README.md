# Phase 2: performance optimization

Use the reviewed derivation's `phase_corpora.phase2` selection, not Phase 1's
functional or Phase 0-only diagnostic members. Carry the same selected SW-spec
and hardware-evidence identities with the exact frozen functional compiler.
Changing operation admission, formats or hardware selection requires renewed
functional qualification; it is not merely another performance knob.

Phase 2 consumes a frozen functional compiler; it does not tune Merlin's shared
implementation for Gemmini. Target-specific generated compiler code stays in the
run's candidate payload and may later be published to the target's OOT repository.
Its workloads answer a different question from Phase 1's network-compile bar:
they measure placement, cycles and model-level cost after functional capability
has been established. A fast Phase 2 objective is not a substitute for a
successful ResNet-50 or SmolVLA whole-model compile and correctness receipt.

After the Phase 1 compiler is frozen, an owner-only form audit must compare a
held-out capture with the iteration-derived form scope. It reports both the
share in known operation-form classes and the share whose three GEMM extents
are dominated by one actual iteration group. The latter detects a scale gap
that a matching operation name alone conceals. Neither number proves cycles,
correctness or placement; held-out extents never feed capsule generation.
Before a model-scale performance claim, also compare each held-out group with
the generated performance members on **all three** GEMM extents at once; a
single-axis K or M/N sweep cannot witness their interaction. A zero joint-size
share is a diagnostic failure of extrapolation, not proof that a compiler cannot
run the model. Generate any additional stress members from hardware capacities
and independent iteration mechanisms, never held-out dimensions. Measure
candidate/vendor pairs on those independent members, then use occasional
whole-model runs to test whether their bounded measurements predict real
bottlenecks. Keep the audit and its model-specific dimensions owner-private.

Start from the [Phase 1 workflow](../phase1/README.md). There are three distinct
experiment modes, each with one shared definition template:

The EL1–EL4 labels name **Phase 1 information treatments**, not Phase 2 modes.
Older Phase 2 records use `arm4` as a compatibility key for a compiler produced
by the EL4 treatment; it is not a separate experiment level.

| Mode | Template | Required handoff |
| --- | --- | --- |
| Measured claims | [`measured-claims-template.yaml`](../../../experiments/definitions/measured-claims-template.yaml) | Frozen functional run ID and submission hash, descriptor, RTL facts, performance profile, both declared GSIM certificates and their exact hashes |
| Model portfolio | [`model-portfolio-template.yaml`](../../../experiments/definitions/model-portfolio-template.yaml) | Deployment record, frozen campaign configuration, writable candidate checkout and explicit authoring budgets |
| Whole model, measured | [`whole-model-measured-template.yaml`](../../../experiments/definitions/whole-model-measured-template.yaml) | Phase 1's frozen submission and `oot/`, the model capsule, an objective config naming machines from [`whole-model-machines.yaml`](whole-model-machines.yaml), a launch profile and a round driver |

Do not interpret a model-portfolio estimate as measured hardware performance.
The measured mode's admission gates validate the functional handoff and timing
evidence; a zero exit status from Phase 1 is not a substitute for those records.

The measured-claims catalog adapter selects the installed managed Chia envelope
and checkpoint coordinator. The model-portfolio adapter selects the installed
portfolio launcher with explicit deployment inputs. The model-portfolio adapter
retains its native campaign and checkpoint semantics; the shared catalog only
freezes and dispatches its declared inputs. See the [installed explicit-resource guide](installed-measured-claims.md)
for deployment inputs and direct coordinator diagnostics; neither route alone
establishes a qualified deployment. The global
model-portfolio and corpus measured-claims handoffs are different schemas; do not
substitute one for the other.

## Author one explicit definition

Copy the selected template into your own experiment input directory, not into
`out/` as an execution script. Keep the template's `kind: template` while editing.
Set a unique `id`, set `target: gemmini`, and supply every operator-owned input
and budget. Resolve paths relative to the new definition's location, or use
explicit absolute paths. There is no placeholder substitution or template
inheritance: edit the YAML values themselves.

For measured claims, select the descriptor associated with the frozen functional
run, not the mutable example descriptor merely because its target name matches.
Take submission and certificate hashes from their recorded evidence; never
rehash edited payloads and present them as the original certification. New verified
execution requires the current frozen-input provenance, including source ownership
and private support classification. Preserve older runs for inspection.
Also select the provisioned managed supervisor endpoint, installed package roots,
contract resources, holdout catalog, functional-run root and new stage/measurement
destinations. The template keeps these separate from scientific evidence. Never
use a generated output directory as an immutable input root containing its own
future outputs. The catalog selects the installed wrapper and current interpreter.

For model portfolio, select the exact campaign inputs, candidate checkout and
`merlin.portfolio-deployment.v1` record described in the
[catalog deployment guide](../../../experiments/README.md#definitions-and-execution).
Declare additional immutable roots in `inputs` when orchestration must also pin
them. The installed engine validates its nested input closure. Set explicit
agent budgets; do not inherit a long-running study's budget accidentally.

After completing the definition, change `kind` to `experiment`. Inspect and
preflight it through the existing interface:

```sh
merlin experiment inspect /absolute/inputs/gemmini-performance.yaml --phase 2
merlin experiment preflight /absolute/inputs/gemmini-performance.yaml --phase 2
```

Templates intentionally cannot execute. Preflight checks local configuration,
not live simulator readiness. Only when inputs, managed execution resources and
costs are approved, launch explicitly:

```sh
merlin experiment run /absolute/inputs/gemmini-performance.yaml --phase 2
```

The runner allocates a fresh directory beneath the configured run root and records
resolved inputs and process attempts. Do not create launch scripts in generated
output folders. See the shared [execution and reporting guide](../../../experiments/README.md#definitions-and-execution)
for each mode's deployment requirements, evidence reporting and checkpoint/resume rules.
Compiler payloads remain immutable; certification and publication records belong
alongside them, not inside an edited certified manifest.

The legacy FireSim GEMM-window study has its queue budget, contention and failure
policy in [firesim_loop.py](firesim_loop.py). This is example-owned policy for that
board and workload, not a Merlin core default. It submits nothing on import;
the study's launcher opts into it when running a measured batch.

The [primitive probe](primitive_probe.py) is likewise an example-owned Gemmini
diagnostic for splitting a decoded RoCC short kernel into setup, compute and
readback. It is not part of Merlin's shared Phase 2 API or the measured-claims
runner; use the target-neutral provider interface for new accelerators.

## Whole model, measured

The `whole_model_measured` example prefers the full U250 `FireSimGemminiRocketConfig` board
for screening each correct candidate (several per board job, the vendor reference as the in-batch control),
the functional model grades every group locally first, and the elaborated-RTL emulator certifies each
new best. The experiment's `prohibited_instruction_roles` is enforced over every candidate's whole
linked ELF before any machine time. [`whole-model-machines.yaml`](whole-model-machines.yaml) says how
each machine is run (host locations are environment references); which device each one is stays the
pin registry's. [`whole-model-objective.json`](whole-model-objective.json) is an example objective
config; replace each `/ABSOLUTE/...` placeholder with artifacts measured on that *same full design*.
The Lean board remains an explicitly named historical option, not an interchangeable fallback: it
lacks full-width accumulator readout. The stock `FireSimGemminiRocketConfig` board
(`stock_u250_board`, batched as `stock_batched_board`) has that readout; its hw-config
resolves to the pinned bitstream that declares its generated ABI header, and its chipyard
is the private tree its host driver was built in (`MERLIN_CHIPYARD_GEMMINI_STOCK`). Merely selecting the full board here does not establish that
its queue configuration is currently available, that it matches a particular Phase 0 RTL snapshot,
or that a new run passed qualification. Large inputs such as the model capsule are frozen by
content into each run, never copied.

The example leaves `builder` and `store` unset: run preparation selects Merlin's
shared builder and creates a target-scoped artifact store. Its `chunk_ops: "auto"`
build option lets an open model's host forward be cut into bounded functions
when it is large (an unchunked SmolVLA forward compiled for over two hours,
against about eleven minutes cut at 1,000 ops); a forward that fits in one chunk,
and every closed model, builds exactly as without it. Its `mechanism_policy`
derives whether verified package passes and fused regions are available from the
frozen model capsule's host/accelerator closure. Closed models can use both;
open models can use neither. The prepared, read-only objective records the exact
decision. This enables mechanisms for the Phase 2 agent, not hand-authored
Gemmini transformations, and does not make a diagnostic or unreviewed capsule
eligible for a verified run.
