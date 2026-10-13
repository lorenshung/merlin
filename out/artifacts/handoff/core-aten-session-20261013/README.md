# Core ATen session delivery and Phase 1 journal

This snapshot records the reviewed sources, evidence locations, and qualification
still required before extending Phase 1. Paths beginning `out/` are
relative to the workspace root and name ignored local evidence, not files shipped
with the source repository. Detailed run artifacts remain local. Publication and experiment execution are separate
actions; pushing reviewed source does not authorize a Phase 1 or FireSim launch.

## Preserve the source and handoff

The clean prepared baseline is Merlin
`2ba0d942d7cb69fb5289953af54de6043f125472` and model2MLIR
`47943a48af164e1ad079c7ba475200313418c198`. Preserve that Merlin baseline
when updating the model2MLIR fork URL and preparing the delivery commit. The
software pin must bind the selected commit and all required source bytes; a URL
change alone establishes neither publication nor qualification. Target support
remains an explicit OOT dependency, separate from these two source repositories.

Preserve the exact unfinished starting checkpoint
`b5a7cd49f19ebadd473ce7f72be4a8f3bc8b60b6` on an archive reference.
It is not a production ancestor and carries no validation claim; merging it would
duplicate or supersede finished work. The original narrative is
`out/artifacts/handoff/codex-handoff-20261012.md`. Commit inventories, lineage,
exclusions and pin audits live under
`out/artifacts/verification/core-aten/remaining-20261012/upstream-preparation/`.

Merlin delivers full-call capture and grading, strided inputs, borrowed/fresh
storage identities, mutation and result-layout contracts, complex scalar precision,
safe resize growth, authenticated legacy ABI recovery, typed capture refusals,
integer device qualification and source/target provenance. Its storage guard refuses
unrecorded raw LayerNorm statistics without consulting expected values, and capture
continues with the next case. model2MLIR delivers upstream integration, static/traced
tuple projection and RNG lineage, source-only metadata geometry, full-spectrum FFT,
and reviewed CPU numeric/embedding dispatch repairs. Their detailed audits are
`merlin-delivery-review.md`, `model2mlir-audit.md` and the companion JSON files there.

## Read qualification at its actual source revision

The fully audited pair is Merlin `007900f69f8b9111bca7f3a364e44db224c767cf`
and model2MLIR `e9e553c73dc52ebaf68f507df01e9c72a6a26518`:

| Corpus | Scalar track | Gemmini track |
| --- | --- | --- |
| Canonical 193 | 193 observed-semantic passes, all host | 193 passes: 3 device, 190 host |
| Bounded 10,160 | 10,067 observed-semantic passes | 10,067 passes: 48 device, 10,019 host |

Each bounded track retains 93 nonpasses: 71 mismatches, ten compile/lowering
failures, eight capture-unavailable cases and four ungradable cases. Observed
semantics covers checked values, mutation and metadata; it is not an unrestricted
full-semantics claim. The 193 frontend traces include 188 complete traces and five
preexisting diagnostics, while all 193 prepared-to-MLIR mappings resolve. See
`out/artifacts/verification/core-aten/final1/integrated-core193/final-campaign-audit.json`
and `out/artifacts/verification/core-aten/final1/integrated-final-report.txt`.

The prepared 2ba0d942/47943a4 line has 591 model2MLIR tests, 89 native layout
tests, and 132 Merlin tests passing (two skipped, one deselected), plus one
Torch-enabled loader test. Evidence is under
`out/artifacts/verification/core-aten/remaining-20261012/integration/` and
`out/artifacts/verification/core-aten/remaining-20261012/storage-growth/`.
These focused repairs do not establish a new 193 or 10,160 total. Normal gates
retain exactly the independently checked 17 inherited wiring diagnostics; keep
that exception explicit. The experimental compensated-addmm commit `a86ec0d`
is excluded: independent review found four new eager-equivalence regressions.

## Keep placement and quantization claims precise

The unchanged software spec admits the existing TorchAO/PT2E W8A8 flow with
int8 device operands and i32 accumulation. Device qualification checks an
independent integer reference, raw accumulators and declared f32 readout; error
against the original float result is separate. Host preprocessing and fallback
are allowed and counted separately. Of the 48 bounded device passes, 25 use
capture quantization and 23 retain the qualification **“device path conforms;
host int8 narrowing not declared.”** No specification amendment is implied.

The canonical three-device count measures the current selector, not hardware
capability: 187 calls have no selected contraction, one BMM is blocked by a rank2
host-cast declaration, and two FFTs need undeclared complex boundary operations.
Broader elementwise, pooling and movement opportunities require source-derived
selection and qualification. See `integration/baseline193-routing-audit.md`
under the remaining-work evidence root. model2MLIR catalogs 22 TorchAO schemes;
that catalog does not certify INT4, FP8 or MX execution on this target.

## Resume Phase 1 only after joint review

The first two entries in `out/artifacts/handoff/phase1-journal.md` record the
clean 2ba0d942/47943a4 baseline and correct the three-device routing interpretation.
The retained old38 reference has 28 device cases and ten host guards; it does not
define the maximum eligible corpus. The next plan is to validate the committed compiler pair, run Phase1 on193
then10,160 cases, and derive the Gemmini coverage set and minimum gap set
together with Loren under the unchanged software spec, keeping host placement
visible. That derivation and all agent launches are held pending joint review.
Preserve private grading isolation, corpus denominators, original answers and
receipt bindings. No Phase 1 agent, new quantization method, FireSim/AWS job or
performance claim follows from this delivery; Spike evidence is functional only.

## Journal snapshot

- 2026-10-13: committed baseline2ba0d942/47943a4; paired193/10,160 validation pending.
- 2026-10-13: corrected routing interpretation; three current device cases are not a hardware limit. Gemmini derivation remains joint work with Loren.
- 2026-10-13: fork publication authorized, without AI coauthor trailers. Preserve upstream-ready2ba0d942 and unfinishedb5a7 archives; fork delivery selects lorenshung/model2MLIR in the content-checked pin.

## BatchNorm follow-up

The completed paired 10,160-case baseline at Merlin `3325d93d` / model2MLIR `47943a4` records 10,136 observed-semantic passes, 11 mismatches and 13 input-storage refusals per track. It repairs 83 prior failures but exposes five BatchNorm reduction-order regressions; nine earlier passes also become explicit missing-storage refusals. Two inherited trace-mapping gaps remain strict evidence limits. Gemmini has 48 verified device passes and 10,088 host passes. The original run receipts remain unchanged.

This pin selects model2MLIR `587133ffd7befa88b2793140b9c2d06557ffd6ed`, which restricts spatial-then-batch mean reduction to the CPU strided fallback in both BatchNorm paths. The isolated native sweep passes all 94 sealed BatchNorm cases and 124 regression controls. Compiled replay, independent review and fresh paired validation are separate pending qualification; these native checks do not replace the completed full-suite result. Broader inherited dtype/geometry limitations remain recorded locally.
