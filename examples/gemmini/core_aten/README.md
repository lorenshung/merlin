# Gemmini Core ATen experiment

Catalog ID: `gemmini-core-aten-functional`. This is a new EL4 Phase 1
comparison using the existing `capsule_bench` adapter, which selects
`--schedule continuous --sandbox bwrap`. Budgets are 12 hours with grading
every 900 seconds. The original Gemmini experiment is unchanged.

The source policy selects `out/artifacts/probes/phase1-corpus/public` and
its sibling `hidden`. Each public member is a single Core ATen full-call
capsule; the hidden cohort varies its arguments within the bounded suite.
The operator stages external corpus paths explicitly. Source capsules and
answers are generated artifacts and are never example content.

Use `stage_inputs.py --help` to stage an operator definition and diagnostic
EL4 bundle beneath the configured `out/artifacts` root. Supply ordinary ISA
headers from the selected RTL checkout, its exact extracted `facts.json`,
and the intended genuine timing-record path. The script grants `public/`
only; `hidden/` and `_private/` are host inputs. It copies headers, RTL facts,
and the installed self-check shim into the staging resource directory.
It neither runs a producer nor copies private answers into public resources.
Its `UNREVIEWED.md` identifies the output as diagnostic input staging.

For launch, use a genuine completed Phase 0 producer and `merlin experiment
corpus prepare`; that workflow regenerates the bundle **inside the same
release** as the corpus and descriptor. Inspect the release and let the user
seal it. Select its `private/seal.json` and
`payload/experiment/input_bundles/merlin_assisted_rtlchecks_public_v0/input_bundle_manifest.yaml`
together using `--corpus-seal` and `--bundle-manifest`. The controller then
selects the release descriptor automatically. A diagnostic staged bundle
cannot substitute for that release bundle.

Driver selections are flags on `inspect`, `preflight`, and `run`:

```sh
# Codex
--phase1-driver codex --phase1-model gpt-6.1-sol \
  --phase1-effort high --phase1-provider subscription
# Claude Code
--phase1-driver claudecode --phase1-model claude-opus-4-8 \
  --phase1-effort high --phase1-provider subscription
```

L2 is the fast functional feedback tier, and L3 is mandatory certification.
The descriptor selects `certification_floor: L3`; it checks declarations,
not engine availability, and does not add L3 to L2-only source capsules.
The target producer must declare direct L3 obligations before preparation.
The existing ladder promotes passing candidate bytes per capsule and keeps
missing mandatory tiers incomplete. Do not remove L3 to obtain a success.

A reviewed L2-only start is currently blocked by the Chipyard admission
contract: even initial authoring requires a genuine L3 timing receipt bound
to the selected Verilator simulator bytes. Also, the complete full-call
producer and private-sidecar release integration must be installed before
these capsules can enter reviewed admission. Ordinary capsule directories
alone do not satisfy the immutable Phase 0 receipt required by preparation.
The original matrix/model Phase 0 recipe generates a different corpus and
must not be used to claim production of these 193 cases. Consequently this
definition declares Phase 1 only; it does not mislabel the old recipe as the
Core ATen producer.

Select an executable OOT support provider with `MERLIN_TARGET_PATH` and the
matching contract with `MERLIN_TARGET_CONTRACT`. Select the actual RTL
simulator tree with `MERLIN_EXT_CHIPYARD`. Keep Spike/libgemmini toolchain
selection distinct if those tools reside in a different Chipyard checkout.
A GSIM engine is usable at L3 only when its program execution, result
readback, build receipt and FIRRTL identity match the selected hardware.
Configuration preflight does not establish numerical or engine readiness.
