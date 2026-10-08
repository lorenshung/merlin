---
title: "Component-only Phase 2 feedback and launch boundaries"
kind: design
status: current
owner: merlin-experiments
last_verified: 2026-10-08
related: [beam_cca_architecture]
code_refs:
  - packages/merlin-experiments/src/merlin_experiments/phase2/component_workflow.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/broker_policy.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/stage_inputs.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/stage_prompt.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/authoring.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/authoring_cli.py
  - packages/merlin-experiments/tests/test_component_workflow.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/component_generation.py
  - packages/merlin-experiments/src/merlin_experiments/phase0/generation.py
  - packages/merlin-experiments/src/merlin_experiments/phase2/corpus.py
  - packages/merlin-experiments/tests/test_component_generation.py
---

# Component-only Phase 2

`component-only-v1` is an explicit scientific action profile for improving a
compiler library using generated development components. Historical
`corpus-feedback-v1` and `whole-model-v1` keep their existing identities, required
actions and receipt rules. A historical receipt is never relabelled as a new
component experiment.

The new registry includes candidate manifest commands, a compiler edit-surface
inventory, optional command-buffer structural analysis, optional analytical
feedback and optional component RTL feedback. It excludes target-descriptor shell
probes, complete-model graph/analysis actions and all legacy global context,
source-pair and mechanism-probe providers. Passing those services to the policy
is an admission error. Registry inspection without providers advertises the
feedback tiers as unavailable.

## Independent Phase 0 inputs

The ordinary `phase0.generate_target(..., component_only=True)` API and
`python -m merlin_experiments.phase0 --component-only` reuse the existing sweep
expansion, builders, independent golden engines and written-program admission.
The normal `capsule_derivation` adapter forwards the explicit option. This mode
requires a fresh output directory, explicit descriptor/recipe/shared template,
and `evidence_mode: diagnostic`. The descriptor must contain no workload,
claim-boundary or grading selectors. Captures, conformance/application sidecars,
synthesis/solver/hidden sidecars, old evidence bundles and legacy profile-directory
discovery are refused. The public recipe supplies numerical/target choices;
all component members come from the shared independent sweeps.

Every selected shared template must itself be independent. A template containing
`requires_form_scope` is refused in this mode; its capture-derived families are
not silently removed. Reusing the legacy shared template therefore requires a
separately reviewed independent template selection. Existing direct builders
and their numerical engines are not a generated domain-coverage certificate.
In particular, the interface, generated operand bytes and golden output extents
must agree at each concrete sweep point, including rectangular contractions,
tails and the separate row extents of resident-weight consumers. Bare extents
resolved by a sweep take precedence over tile-count aliases; an omitted attention
key length preserves the query-length square default.

The selected reviewed software spec may declare non-MAC performance objectives:

```yaml
component_performance:
  schema: merlin.component_performance.v1
  status: reviewed
  hardware:
    contract_sha256: <selected EvidenceSelection.derivation_identity contract digest>
    raw_facts_sha256: <selected raw hardware facts byte digest>
  objectives:
    - family: <declared shared sweep family>
      operations: [<reviewed software operation declaration id>]
      objective:
        metric: complete_component_cycles
        unit: cycles
        direction: min
        basis: all preparation, device work, readout and output publication
```

The declaration explicitly commits to the selected HW contract/facts; its source
bytes and reviewed operation owners supply objective provenance. The generator
records the actual recipe, template and generator-owner source bytes, and binds
the resulting declaration identity to every generated member. Sweeps cannot
author their own generated identity or objective provenance. A named objective
owner must declare the generated operation/family. Every emitted program still
needs the existing concrete reviewed placement screen. Unknown work remains
refused; a real zero-MAC member needs its explicit typed objective.

`corpus.performance_objectives(live_or_frozen_corpus)` returns the checked
name-to-`PerformanceObjective` map for the existing `phase_policy.split_report`
and `anchors` APIs. Corpus selection, discovery and freezing preserve these
bindings. Objective admission does not supply measured cost, cycle certification,
utilization, an anchor or a target rate. Absent objectives retain the legacy
zero-MAC refusal. The recorded source digests are host-issued selection metadata,
not cryptographic proof of review authority or a fresh source-closure check when
a frozen corpus is later used. Fresh runtime/library isolation is separate.

Global application coverage remains diagnostic/unknown. Since this input mode
does not supply application conformance or a Phase 1 certification, its guard
obligation is explicitly `not_established`, with no functional guard or candidate
numerical acceptance inferred. Legacy generation keeps its original coverage and
guard-link gates. Existing generated goldens, graders, component certification
and final hidden/whole-model gates retain their independent roles. No fresh
experimental agent launch, hardware measurement or final convergence follows
from generating and freezing this corpus.

## Host-owned selection

The trusted host calls `broker_policy.select_workflow(COMPONENT_ONLY_V1, ...)`
with a verified `FrozenPerformanceCorpus`, candidate root, target descriptor,
receipt destination and explicit providers. The corpus is admitted through the
existing generated development-corpus path and retains its exact manifest,
generation and source identities. The component policy rechecks capsule source
bytes and descriptors and rejects explicit model-kind members and forged
in-memory descriptor changes. This is component input admission; it does not
certify Phase 0 derivation or approve an agent-visible filesystem view.

Candidate metadata cannot select a provider or grant a host capability. Resolve
any provider descriptor at the trusted launch edge, before creating the policy.
Source file pins bind the executing provider owner, not its entire transitive
dependency graph. Compiler-library/runtime import admission must separately bind
dependencies and resources. No implicit current-checkout or installed-target
fallback is authorized by this profile.

An active analytical policy also snapshots its selected callable and Python code
object. Replacing that callable with another function in the same pinned file or
changing its code object refuses execution. This guards active selection drift;
it does not prove arbitrary closure state or dynamic dependency integrity. The
normal fresh runtime admission remains a separate prerequisite.

## Fast and measured tiers

`ComponentAnalyticalProvider` binds an explicit callable's source file and an
adapter accepted by `prepare_phase2_calibration`. Each use revalidates the adapter,
its referenced controlled evidence and the exact resulting calibration document.
The calibration must name the current target descriptor digest. The callback
receives the sealed candidate, frozen generated corpus and remaining tool budget.
It returns exactly one baseline/candidate typed `CycleInterval` pair per member.
Resolved intervals require finite bounds and provenance. Missing costs remain
unresolved with nonempty reasons. Analytical rows are estimates; this schema
does not certify the callback's model accuracy or claim measured cycles.

`component-rtl-feedback` accepts the existing `DevelopmentGsimFeedback` evaluator
with its explicit host executor, exact frozen corpus, baseline compiler, target,
RTL configuration and workload-equivalence certificate. These identities and
the executor source are rechecked. The existing certificate/evaluation gate and
redacted feedback validator remain unchanged. This optional action is limited
to two invocations per broker round. A skipped member retains the existing
unmeasured status and null cycle fields. No fabricated measurements fill gaps.

Neither tier requires a GSIM measurement on every iteration. Neither grants final
accuracy, hardware parity or convergence acceptance. Spike functional evidence
cannot be presented as RTL or FPGA timing. A provider must bound its own execution;
the policy checks elapsed time on return but does not supervise an arbitrary
in-process callback that hangs. A future isolated runner must own cancellation
before this profile can support a bounded scientific campaign.

## Prompt and normal launch

`stage_inputs.prepare_component_prompt_inputs` binds the selected policy to
explicit paths inside a separately approved materialized view. It calls no
complete-model selector and inherits no functional bundle or host-lane grant.
`stage_prompt.render_component_prompt` produces only the component declaration,
available actions, exact member identities and budgets. These functions prepare
input; they do not approve the view or start an agent.

Normal `authoring.run_stage` and `authoring_cli --workflow component-only-v1`
currently refuse before descriptor loading, executable lookup, telemetry,
legacy GSIM flags or model selection. There is no qualified fresh-session,
network-restricted authoring transport in this owner. The future launch edge
must verify an approved minimal view, content-bound runtime isolation and a
zero-history participant session, then create the explicit providers and broker.
A library already developed by an implementation agent is not evidence of a
fresh participant's convergence.

## Receipts and final validation

New component broker rows require an explicit profile identity. The qualifier
joins action/binding digests to the transcript, validates canonical private
feedback bytes, checks stdout identity and the action's evidence tier, and
requires successful candidate manifest commands. Final component feedback must
bind the sealed candidate digest when that check is requested. Its result still
states `final_acceptance: NOT_ESTABLISHED`.

Final unseen-component, numeric, complete-composition, compiler delivery and
whole-model hardware gates belong to the trusted final evaluator. They must use
the original fixed accuracy and performance contracts after the participant is
sealed, without exposing answers through inner-loop services. This owner does
not run those gates, change their thresholds, submit FireSim jobs, or certify a
complete-model speedup.

The structural command-buffer action does not yet expose a complete CCA gap
report. A future component CCA route must use the existing all-facet comparator
and `uncomparable_axes`, preserve scope and provenance, and report missing
communication or coverage evidence as UNKNOWN. A partial agreement predicate
cannot stand in for that report.
